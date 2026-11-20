"""The mock backend: the real scheduler and block pool driven by a timing-only model (D-008)."""

import time
from typing import Any

import httpx
import pytest
import torch
from fastapi import FastAPI

from engine import server
from engine.mock import MockModel, TimingModel, mock_config, mock_next_token
from engine.paged_cache import KVPool
from engine.prefix_cache import block_hashes
from engine.scheduler import Scheduler
from tests.fake_server import UvicornServer

INSTANT = TimingModel(time_scale=0)
BLOCK = 4


def prompt_of(length: int, seed: int) -> list[int]:
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(1000, 100000, (length,), generator=generator).tolist()


def expected_tokens(prompt: list[int], count: int) -> list[int]:
    """What the mock must produce, computed without any model or scheduler."""
    tokens = list(prompt)
    for _ in range(count):
        tokens.append(mock_next_token(tokens[-1], len(tokens) - 1))
    return tokens[len(prompt) :]


def make_scheduler(
    num_blocks: int = 200, prefix_caching: bool = True, max_batch_size: int = 8
) -> tuple[Scheduler, MockModel]:
    model = MockModel(timing=INSTANT)
    pool = KVPool(model.cfg, num_blocks, BLOCK, torch.float32, "cpu")
    scheduler = Scheduler(
        model, pool, max_batch_size=max_batch_size, enable_prefix_caching=prefix_caching
    )
    return scheduler, model


def test_the_mock_config_is_tiny_but_accepts_real_token_ids() -> None:
    cfg = mock_config()
    assert cfg.vocab_size == 128256 and cfg.head_dim == 2  # a pool entry is a few bytes
    assert cfg.kv_bytes_per_token(4) < 100


def test_generation_is_a_pure_function_of_the_last_token_and_position() -> None:
    scheduler, _ = make_scheduler()
    prompts = [prompt_of(n, n) for n in (3, 9, 14, 6)]
    requests = [scheduler.add_request(i, p, 12) for i, p in enumerate(prompts)]
    scheduler.run_to_completion()
    for request, prompt in zip(requests, prompts, strict=True):
        assert request.generated == expected_tokens(prompt, 12)
    assert scheduler.pool.allocator.num_used == 0


def test_a_prefill_claims_blocks_exactly_as_a_real_model_would() -> None:
    """The mock has no data, so only the pool's accounting shows it behaves like a real model.

    If prefill claimed nothing, the pool would look emptier than it is and requests would be
    admitted, queued and preempted at different moments than on a real GPU.
    """
    scheduler, _ = make_scheduler(num_blocks=50)
    allocator = scheduler.pool.allocator
    scheduler.add_request(0, prompt_of(10, 1), 8)
    scheduler.step()  # admits and prefills: 10 tokens occupy ceil(10 / 4) = 3 blocks
    assert allocator.num_used == allocator.blocks_for(10) == 3
    scheduler.step()  # one decode step: 11 tokens still fit in 3 blocks
    assert allocator.num_used == 3
    scheduler.step()
    scheduler.step()  # 13 tokens now need a fourth block
    assert allocator.num_used == 4


def test_output_is_the_same_under_preemption() -> None:
    scheduler, _ = make_scheduler(num_blocks=14, max_batch_size=8)
    prompts = [prompt_of(11 + i, 40 + i) for i in range(6)]
    requests = [scheduler.add_request(i, p, 18) for i, p in enumerate(prompts)]
    scheduler.run_to_completion()
    assert scheduler.total_preemptions > 0
    assert [r.generated for r in requests] == [expected_tokens(p, 18) for p in prompts]


def test_cached_prefix_tokens_are_not_run_through_prefill() -> None:
    shared = prompt_of(16, 7)
    prompts = [[*shared, *prompt_of(5, 100 + i)] for i in range(3)]

    def run(prefix_caching: bool) -> tuple[MockModel, list[list[int]]]:
        scheduler, model = make_scheduler(prefix_caching=prefix_caching)
        outputs = []
        for index, prompt in enumerate(prompts):  # one at a time, so each is cached for the next
            request = scheduler.add_request(index, prompt, 4)
            scheduler.run_to_completion()
            outputs.append(request.generated)
        return model, outputs

    cached_model, cached_outputs = run(True)
    plain_model, plain_outputs = run(False)
    assert cached_outputs == plain_outputs == [expected_tokens(p, 4) for p in prompts]
    assert plain_model.prefill_tokens == 3 * 21
    assert cached_model.prefill_tokens == 21 + 2 * 5  # the two later prompts skip 16 tokens each


def test_durations_follow_the_timing_model() -> None:
    timing = TimingModel(
        prefill_base_ms=10, prefill_ms_per_token=1, decode_base_ms=5, decode_ms_per_sequence=0
    )
    assert timing.prefill_s(40) == pytest.approx(0.050)
    assert timing.decode_s(7) == pytest.approx(0.005)
    assert TimingModel(time_scale=0).prefill_s(1000) == 0
    assert TimingModel(time_scale=2).decode_s(1) == pytest.approx(2 * 0.0091)

    model = MockModel(timing=timing)
    pool = KVPool(model.cfg, 50, BLOCK, torch.float32, "cpu")
    scheduler = Scheduler(model, pool)
    scheduler.add_request(0, prompt_of(30, 1), 6)
    start = time.perf_counter()
    scheduler.run_to_completion()
    # One prefill (40 ms) and five decode steps (5 ms each) is at least 65 ms; sleep never returns
    # early, so this lower bound is safe even on a slow machine.
    assert time.perf_counter() - start >= 0.065 - 0.002


def serve(**scheduler_kwargs: Any) -> tuple[FastAPI, Scheduler]:
    scheduler, _ = make_scheduler(**scheduler_kwargs)
    return server.create_app(scheduler, [], model_id="mock"), scheduler


def test_the_mock_server_streams_the_expected_tokens() -> None:
    app, _ = serve()
    prompt = prompt_of(10, 3)
    with UvicornServer(app) as url, httpx.Client(base_url=url, timeout=30) as client:
        body = client.post("/v1/completions", json={"prompt": prompt, "max_tokens": 7}).json()
    assert body["choices"][0]["token_ids"] == expected_tokens(prompt, 7)
    assert body["usage"]["completion_tokens"] == 7


def test_cache_endpoint_lists_the_chain_hashes_of_cached_prompt_blocks() -> None:
    app, _ = serve()
    prompt = prompt_of(18, 4)  # four full blocks plus two tokens
    with UvicornServer(app) as url, httpx.Client(base_url=url, timeout=30) as client:
        assert client.get("/cache").json()["hashes"] == []
        client.post("/v1/completions", json={"prompt": prompt, "max_tokens": 2})
        body = client.get("/cache").json()
    assert body["block_size"] == BLOCK and body["prefix_caching"] is True
    held = {int(h, 16) for h in body["hashes"]}
    assert set(block_hashes(prompt, BLOCK)) <= held  # the router can recompute these itself
    assert all(len(h) == 32 for h in body["hashes"])


def test_cache_endpoint_is_empty_when_prefix_caching_is_off() -> None:
    app, _ = serve(prefix_caching=False)
    with UvicornServer(app) as url, httpx.Client(base_url=url, timeout=30) as client:
        client.post("/v1/completions", json={"prompt": prompt_of(12, 5), "max_tokens": 2})
        body = client.get("/cache").json()
    assert body["prefix_caching"] is False and body["hashes"] == []


def test_main_builds_a_mock_server_sized_like_the_real_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}
    monkeypatch.setattr(server.uvicorn, "run", lambda app, **kw: captured.update(app=app, **kw))
    code = server.main(
        ["--backend", "mock", "--port", "9", "--kv-cache-gb", "2", "--mock-time-scale", "0"]
    )
    assert code == 0 and captured["port"] == 9
    app = captured["app"]
    with UvicornServer(app) as url, httpx.Client(base_url=url, timeout=30) as client:
        info = client.get("/info").json()
        done = client.post("/v1/completions", json={"prompt": [5, 6, 7], "max_tokens": 3}).json()
    assert info["backend"] == "mock" and info["model_id"] == "mock"
    assert info["uncalibrated_timing"]["time_scale"] == 0
    # 2 GiB at 32 KiB per token (Llama-3.2-1B in bf16) and 16-token blocks: 4096 blocks.
    assert info["kv_blocks"] == 4096
    assert done["choices"][0]["token_ids"] == expected_tokens([5, 6, 7], 3)


def test_the_llama_backend_still_needs_a_model_directory() -> None:
    with pytest.raises(SystemExit, match="--model-dir"):
        server.main(["--backend", "llama"])
