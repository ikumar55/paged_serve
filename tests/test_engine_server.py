"""The engine's HTTP server on a tiny random-weight model (CPU), over real sockets."""

import json
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import httpx
import pytest
import torch
from fastapi import FastAPI

from bench.loadgen import main
from bench.results_io import read_results
from engine.config import LlamaConfig
from engine.generate import greedy_generate
from engine.paged_cache import KVPool
from engine.scheduler import Scheduler
from engine.server import create_app
from tests.fake_server import UvicornServer
from tests.test_paged_cache import make_model

CFG = LlamaConfig.tiny()


def make_app(
    eos_ids: Sequence[int] = (), num_blocks: int = 200, max_batch_size: int = 4
) -> tuple[FastAPI, Scheduler]:
    model = make_model(CFG)
    pool = KVPool(CFG, num_blocks, 4, torch.float32, "cpu")
    scheduler = Scheduler(model, pool, max_batch_size=max_batch_size)
    return create_app(scheduler, eos_ids, model_id="tiny"), scheduler


def prompt_of(length: int, seed: int) -> list[int]:
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(3, CFG.vocab_size, (length,), generator=generator).tolist()


def reference(prompt: list[int], count: int, eos: int | None = None) -> list[int]:
    return greedy_generate(make_model(CFG), prompt, count, eos)


def stream_events(client: httpx.Client, body: dict[str, Any]) -> list[Any]:
    """Return the decoded data events of a streaming request, '[DONE]' included as a string."""
    events: list[Any] = []
    with client.stream("POST", "/v1/completions", json={**body, "stream": True}) as response:
        assert response.status_code == 200
        for line in response.iter_lines():
            if line.startswith("data: "):
                data = line[6:]
                events.append(data if data == "[DONE]" else json.loads(data))
    return events


def wait_until(condition: Any, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.02)
    return False


def test_non_streaming_completion_matches_reference() -> None:
    app, _ = make_app()
    prompt = prompt_of(7, 1)
    with UvicornServer(app) as url, httpx.Client(base_url=url) as client:
        body = client.post("/v1/completions", json={"prompt": prompt, "max_tokens": 9}).json()
    choice = body["choices"][0]
    assert choice["token_ids"] == reference(prompt, 9)
    assert choice["finish_reason"] == "length"
    assert body["usage"] == {"prompt_tokens": 7, "completion_tokens": 9, "total_tokens": 16}


def test_streaming_events_usage_and_done() -> None:
    app, _ = make_app()
    prompt = prompt_of(5, 2)
    with UvicornServer(app) as url, httpx.Client(base_url=url) as client:
        events = stream_events(
            client,
            {"prompt": prompt, "max_tokens": 6, "stream_options": {"include_usage": True}},
        )
    assert events[-1] == "[DONE]"
    usage = events[-2]
    assert usage["choices"] == [] and usage["usage"]["completion_tokens"] == 6
    final = events[-3]["choices"][0]
    assert final["finish_reason"] == "length" and final["text"] == ""
    token_events = [e["choices"][0] for e in events[:-3]]
    assert [c["token_ids"][0] for c in token_events] == reference(prompt, 6)
    assert all(c["finish_reason"] is None for c in token_events)
    assert [c["text"] for c in token_events] == [f"<{t}>" for t in reference(prompt, 6)]


def test_concurrent_requests_are_batched_and_each_matches_reference() -> None:
    app, scheduler = make_app(max_batch_size=4)
    specs = [(prompt_of(3 + 2 * i, 10 + i), 8 + i) for i in range(8)]
    with UvicornServer(app) as url:

        def call(spec: tuple[list[int], int]) -> list[int]:
            prompt, count = spec
            with httpx.Client(base_url=url, timeout=30) as client:
                response = client.post(
                    "/v1/completions", json={"prompt": prompt, "max_tokens": count}
                )
                return list(response.json()["choices"][0]["token_ids"])

        with ThreadPoolExecutor(max_workers=8) as pool:
            outputs = list(pool.map(call, specs))
        info = httpx.get(f"{url}/info").json()

    for output, (prompt, count) in zip(outputs, specs, strict=True):
        assert output == reference(prompt, count)
    assert info["stats"]["requests_finished"] == 8
    assert info["stats"]["mean_decode_batch"] > 1.0, "requests were never decoded together"
    assert scheduler.pool.allocator.num_used == 0


def test_eos_ends_the_stream_without_emitting_it_unless_ignored() -> None:
    prompt = prompt_of(6, 3)
    full = reference(prompt, 12)
    eos = full[4]
    first = full.index(eos)  # the stream must stop at its first occurrence
    app, _ = make_app(eos_ids=[eos])
    with UvicornServer(app) as url, httpx.Client(base_url=url) as client:
        stopped = client.post("/v1/completions", json={"prompt": prompt, "max_tokens": 12}).json()
        ignored = client.post(
            "/v1/completions", json={"prompt": prompt, "max_tokens": 12, "ignore_eos": True}
        ).json()
    assert stopped["choices"][0]["token_ids"] == full[:first]
    assert stopped["choices"][0]["finish_reason"] == "stop"
    assert stopped["usage"]["completion_tokens"] == first
    assert ignored["choices"][0]["token_ids"] == full
    assert ignored["choices"][0]["finish_reason"] == "length"


@pytest.mark.parametrize(
    ("body", "fragment"),
    [
        ({"prompt": [1, 2], "max_tokens": 3, "temperature": 0.7}, "greedy"),
        ({"prompt": [1, 2, CFG.vocab_size], "max_tokens": 3}, "token ids"),
        ({"prompt": [-1], "max_tokens": 3}, "token ids"),
        ({"prompt": [1] * 250, "max_tokens": 20}, "exceed"),
        ({"prompt": [1, 2], "max_tokens": 0}, ""),
        ({"prompt": [], "max_tokens": 3}, ""),
    ],
)
def test_bad_requests_are_rejected_with_a_400_or_422(body: dict[str, Any], fragment: str) -> None:
    app, _ = make_app()
    with UvicornServer(app) as url:
        response = httpx.post(f"{url}/v1/completions", json=body)
    assert response.status_code in (400, 422)
    assert fragment in response.text


def test_a_request_too_big_for_the_pool_is_rejected_not_queued_forever() -> None:
    app, _ = make_app(num_blocks=4)  # 16 token slots
    with UvicornServer(app) as url:
        response = httpx.post(f"{url}/v1/completions", json={"prompt": [1] * 10, "max_tokens": 10})
    assert response.status_code == 400 and "pool" in response.text


def test_a_client_that_disconnects_mid_stream_frees_its_blocks() -> None:
    app, scheduler = make_app()
    real_step = scheduler.step

    def slow_step() -> Any:
        time.sleep(0.02)  # 200 tokens would take about 4 s, so the request cannot finish alone
        return real_step()

    scheduler.step = slow_step  # type: ignore[method-assign]
    with UvicornServer(app) as url, httpx.Client(base_url=url, timeout=30) as client:
        body = {"prompt": prompt_of(8, 5), "max_tokens": 200, "stream": True, "ignore_eos": True}
        with client.stream("POST", "/v1/completions", json=body) as response:
            lines = response.iter_lines()
            for _ in range(3):  # read a few tokens, then walk away
                next(line for line in lines if line.startswith("data: "))
        assert wait_until(lambda: scheduler.pool.allocator.num_used == 0, timeout=2.0)
        assert scheduler.aborted == 1
        assert scheduler.steps < 100, "the request ran to completion instead of being aborted"
        assert not scheduler.running and not scheduler.waiting
        # The engine is still healthy and serves the next request.
        again = client.post("/v1/completions", json={"prompt": [1, 2, 3], "max_tokens": 2})
        assert again.status_code == 200


def test_info_reports_the_server_and_scheduler_state() -> None:
    app, _ = make_app(max_batch_size=3)
    with UvicornServer(app) as url, httpx.Client(base_url=url) as client:
        assert client.get("/health").json() == {"status": "ok"}
        client.post("/v1/completions", json={"prompt": [1, 2, 3], "max_tokens": 2})
        info = client.get("/info").json()
    assert info["server"] == "pagedserve" and info["model_id"] == "tiny"
    assert info["max_batch_size"] == 3 and info["block_size"] == 4
    assert info["stats"]["requests_finished"] == 1 and info["stats"]["tokens_generated"] == 2
    assert info["stats"]["blocks_total"] == 200 and info["stats"]["blocks_used"] == 0


def test_info_accounts_for_the_engine_threads_time() -> None:
    """D-021: how much of the engine thread's time is the step, routing, the inbox, or idling."""
    app, _ = make_app(max_batch_size=3)
    with UvicornServer(app) as url, httpx.Client(base_url=url) as client:
        before = client.get("/info").json()["stats"]["engine_thread_seconds"]
        client.post("/v1/completions", json={"prompt": [1, 2, 3], "max_tokens": 8})
        after = client.get("/info").json()["stats"]["engine_thread_seconds"]
    assert set(after) == {"step", "route", "inbox", "idle"}
    assert after["step"] > before["step"] and after["route"] > before["route"]
    assert all(value >= 0 for value in after.values())
    assert after["idle"] >= before["idle"]


def test_end_to_end_with_the_load_generator(tmp_path: Path) -> None:
    """The same load generator that drives the Hugging Face baseline drives this engine."""
    app, _ = make_app(max_batch_size=4)
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
    assert all(r.output_tokens == r.requested_output_tokens for r in results)
    assert all(r.tokens_from_usage for r in results)
    assert summary is not None and summary["failure_counts"] == {}
    assert meta["server_info"]["after"]["server"] == "pagedserve"
    before = meta["server_info"]["before"]["stats"]
    after = meta["server_info"]["after"]["stats"]
    assert after["requests_finished"] - before["requests_finished"] == 17  # 16 plus the warm-up


def test_info_reports_prefix_cache_hits_for_a_shared_prompt_prefix() -> None:
    app, _ = make_app()
    shared = prompt_of(16, 6)  # four full blocks of four tokens
    with UvicornServer(app) as url, httpx.Client(base_url=url) as client:
        first = client.post("/v1/completions", json={"prompt": [*shared, 5, 6], "max_tokens": 4})
        second = client.post("/v1/completions", json={"prompt": [*shared, 7, 8], "max_tokens": 4})
        stats = client.get("/info").json()["stats"]
    assert first.json()["choices"][0]["token_ids"] == reference([*shared, 5, 6], 4)
    assert second.json()["choices"][0]["token_ids"] == reference([*shared, 7, 8], 4)
    assert stats["prefix_caching"] == 1 and stats["prefix_hit_tokens"] == 16
    assert stats["prefix_prompt_tokens"] == 36 and stats["prefix_hit_rate"] == pytest.approx(
        16 / 36
    )
    assert stats["blocks_cached"] >= 4 and stats["blocks_used"] == 0
