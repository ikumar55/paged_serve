"""The Hugging Face static-batching baseline server, on a tiny random-weight model (CPU)."""

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest
import torch
from fastapi.testclient import TestClient
from transformers import LlamaForCausalLM

from bench.baselines.batching import PendingRequest
from bench.baselines.hf_server import HFGenerator, build_app
from bench.loadgen import main
from bench.results_io import read_results
from engine.config import LlamaConfig
from tests.fake_server import UvicornServer
from tests.hf_reference import build_reference_model
from tests.test_batching import drain

CFG = LlamaConfig.tiny()


@pytest.fixture(scope="module")
def model() -> LlamaForCausalLM:
    return build_reference_model(CFG, seed=0)


def reference_greedy(model: LlamaForCausalLM, prompt: list[int], count: int) -> list[int]:
    """Ground truth: recompute the whole sequence each step, no cache, no generate()."""
    ids = torch.tensor([prompt])
    out: list[int] = []
    with torch.no_grad():
        for _ in range(count):
            token = int(model(ids).logits[0, -1].argmax())
            out.append(token)
            ids = torch.cat([ids, torch.tensor([[token]])], dim=1)
    return out


def prompt_of(length: int, seed: int) -> list[int]:
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(3, CFG.vocab_size, (length,), generator=generator).tolist()


async def serve(
    generator: HFGenerator, specs: list[tuple[list[int], int, bool]]
) -> tuple[int, list[tuple[list[int], str]]]:
    loop = asyncio.get_running_loop()
    requests = [PendingRequest(p, m, ignore, loop) for p, m, ignore in specs]
    steps = await loop.run_in_executor(None, generator, requests)
    results = []
    for r in requests:
        events, _, _ = await drain(r)
        tokens = [payload for kind, payload in events if kind == "token"]
        results.append((tokens, events[-1][1]))
    return steps, results


def test_batched_output_matches_single_request_greedy(model: LlamaForCausalLM) -> None:
    """Left-padding a mixed batch must not change any request's tokens."""
    prompts = [prompt_of(3, 1), prompt_of(7, 2), prompt_of(12, 3)]
    limits = [4, 9, 6]
    specs = [(p, m, True) for p, m in zip(prompts, limits, strict=True)]
    steps, results = asyncio.run(serve(HFGenerator(model), specs))

    assert steps == 9  # the batch runs until its longest request is done
    for (tokens, reason), prompt, limit in zip(results, prompts, limits, strict=True):
        assert reason == "length"
        assert tokens == reference_greedy(model, prompt, limit)


def test_ignore_eos_runs_past_an_eos_token(
    model: LlamaForCausalLM, monkeypatch: pytest.MonkeyPatch
) -> None:
    prompt = prompt_of(6, 4)
    sequence = reference_greedy(model, prompt, 8)
    eos = next(sequence[i] for i in range(2, 8) if sequence.index(sequence[i]) == i)
    monkeypatch.setattr(model.generation_config, "eos_token_id", eos)

    _, results = asyncio.run(serve(HFGenerator(model), [(prompt, 8, True)]))
    assert results[0] == (sequence, "length")


def test_eos_stops_requests_that_honor_it(
    model: LlamaForCausalLM, monkeypatch: pytest.MonkeyPatch
) -> None:
    prompt = prompt_of(6, 4)
    sequence = reference_greedy(model, prompt, 8)
    eos = next(sequence[i] for i in range(2, 8) if sequence.index(sequence[i]) == i)
    stop_at = sequence.index(eos)
    monkeypatch.setattr(model.generation_config, "eos_token_id", eos)
    generator = HFGenerator(model)

    # Mixed batch: one request honors EOS, one ignores it. The batch still runs to its longest.
    steps, results = asyncio.run(serve(generator, [(prompt, 8, False), (prompt, 8, True)]))
    assert results[0] == (sequence[:stop_at], "stop")  # the EOS token itself is not delivered
    assert results[1] == (sequence, "length")
    assert steps == 8

    # Every request honors EOS: Hugging Face ends the batch as soon as all have stopped.
    steps, results = asyncio.run(serve(generator, [(prompt, 8, False), (prompt, 8, False)]))
    assert [r[1] for r in results] == ["stop", "stop"]
    assert steps == stop_at + 1 < 8


def test_generator_reads_multiple_eos_ids(
    model: LlamaForCausalLM, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(model.generation_config, "eos_token_id", [5, 6])
    generator = HFGenerator(model)
    assert generator.eos_ids == [5, 6] and generator.pad_id == 5


def make_client(model: LlamaForCausalLM) -> TestClient:
    app = build_app(model, model_id="tiny", max_batch_size=4, batch_window_s=0.01)
    return TestClient(app)


def sse_events(text: str) -> list[Any]:
    events: list[Any] = []
    for line in text.splitlines():
        if line.startswith("data:"):
            payload = line[len("data:") :].strip()
            events.append(payload if payload == "[DONE]" else json.loads(payload))
    return events


def test_streaming_completion_contract(model: LlamaForCausalLM) -> None:
    prompt = prompt_of(5, 7)
    with make_client(model) as client:
        response = client.post(
            "/v1/completions",
            json={
                "model": "m",
                "prompt": prompt,
                "max_tokens": 4,
                "stream": True,
                "stream_options": {"include_usage": True},
                "ignore_eos": True,
                "temperature": 0.0,
            },
        )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    events = sse_events(response.text)
    assert events[-1] == "[DONE]"

    token_events = [e for e in events if e != "[DONE]" and e["choices"] and e["choices"][0]["text"]]
    assert [e["choices"][0]["token_ids"][0] for e in token_events] == reference_greedy(
        model, prompt, 4
    )
    closing = events[len(token_events)]
    assert closing["choices"][0]["finish_reason"] == "length"
    assert closing["choices"][0]["text"] == ""
    usage = events[-2]
    assert usage["choices"] == []
    assert usage["usage"] == {"prompt_tokens": 5, "completion_tokens": 4, "total_tokens": 9}


def test_stream_without_usage_option_omits_usage(model: LlamaForCausalLM) -> None:
    with make_client(model) as client:
        response = client.post(
            "/v1/completions",
            json={"prompt": prompt_of(4, 8), "max_tokens": 2, "stream": True, "ignore_eos": True},
        )
    events = sse_events(response.text)
    assert not any(e != "[DONE]" and "usage" in e for e in events)


def test_non_streaming_completion(model: LlamaForCausalLM) -> None:
    prompt = prompt_of(5, 9)
    with make_client(model) as client:
        response = client.post(
            "/v1/completions", json={"prompt": prompt, "max_tokens": 3, "ignore_eos": True}
        )
    body = response.json()
    assert body["choices"][0]["token_ids"] == reference_greedy(model, prompt, 3)
    assert body["choices"][0]["finish_reason"] == "length"
    assert body["usage"]["completion_tokens"] == 3


@pytest.mark.parametrize(
    ("payload", "status"),
    [
        ({"prompt": [1, 2], "max_tokens": 2, "temperature": 0.7}, 400),
        ({"prompt": [1, CFG.vocab_size], "max_tokens": 2}, 400),
        ({"prompt": [-1, 2], "max_tokens": 2}, 400),
        ({"prompt": [1, 2], "max_tokens": CFG.max_position_embeddings}, 400),
        ({"prompt": "hello", "max_tokens": 2}, 422),  # token-id prompts only
        ({"prompt": [], "max_tokens": 2}, 422),
        ({"prompt": [1, 2], "max_tokens": 0}, 422),
        ({"prompt": [1, 2], "max_tokens": 2, "n": 2}, 422),
    ],
)
def test_invalid_requests_are_rejected(
    model: LlamaForCausalLM, payload: dict[str, Any], status: int
) -> None:
    with make_client(model) as client:
        assert client.post("/v1/completions", json=payload).status_code == status


def test_health_and_info(model: LlamaForCausalLM) -> None:
    with make_client(model) as client:
        assert client.get("/health").json() == {"status": "ok"}
        client.post("/v1/completions", json={"prompt": [1, 2, 3], "max_tokens": 2})
        info = client.get("/info").json()
    assert info["server"] == "hf-static-batching"
    assert info["model_id"] == "tiny"
    assert info["max_batch_size"] == 4 and info["batch_window_ms"] == pytest.approx(10)
    assert info["dtype"] == "float32" and info["device"] == "cpu"
    assert info["stats"]["requests"] == 1
    assert {"torch", "transformers"} <= info.keys()


def test_end_to_end_with_the_load_generator(model: LlamaForCausalLM, tmp_path: Path) -> None:
    """Real HTTP server, real sockets, the real load generator, a tiny model on CPU."""
    app = build_app(model, model_id="tiny", max_batch_size=4, batch_window_s=0.05)
    out = tmp_path / "e2e.jsonl"
    argv = [
        "run", "--model", "tiny", "--rate", "100", "--num-requests", "16", "--warmup", "1",
        "--vocab-size", "128", "--min-token-id", "3",
        "--prompt-median", "6", "--prompt-sigma", "0.5", "--prompt-max", "12",
        "--output-median", "5", "--output-sigma", "0.4", "--output-max", "8",
        "--fetch-server-info", "--out", str(out),
    ]  # fmt: skip
    with UvicornServer(app) as url:
        assert main([*argv, "--url", url]) == 0

    meta, results, summary = read_results(out)
    assert len(results) == 16 and all(r.status == "ok" for r in results)
    # Output lengths are forced by ignore_eos, so every request got exactly what it asked for.
    assert all(r.output_tokens == r.requested_output_tokens for r in results)
    assert all(r.tokens_from_usage for r in results)
    assert summary is not None and summary["failure_counts"] == {}

    before = meta["server_info"]["before"]["stats"]
    after = meta["server_info"]["after"]["stats"]
    assert after["requests"] - before["requests"] == 17  # 16 measured plus 1 warm-up
    assert 1 <= after["batches"] - before["batches"] <= 17
    assert meta["server_info"]["after"]["server"] == "hf-static-batching"
