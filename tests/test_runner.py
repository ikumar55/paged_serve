import asyncio
import time

import httpx
import pytest

from bench.runner import run_workload, warm_up
from bench.workload import LengthDist, WorkloadConfig, generate_workload
from tests.fake_server import echo_handler, mock_client, streaming_response, well_formed_stream


def small_workload(rate: float, count: int) -> WorkloadConfig:
    return WorkloadConfig(
        rate=rate,
        num_requests=count,
        prompt=LengthDist(8, 0.0, 1, 16),
        output=LengthDist(3, 0.0, 1, 8),
        seed=3,
    )


def test_every_request_completes_in_order_at_its_scheduled_time() -> None:
    specs = generate_workload(small_workload(rate=300, count=30))

    async def go() -> list:  # type: ignore[type-arg]
        async with mock_client(echo_handler()) as client:
            return await run_workload(specs, client, "m", timeout_s=5)

    results = asyncio.run(go())
    assert [r.id for r in results] == list(range(30))
    assert all(r.status == "ok" and r.output_tokens == 3 for r in results)
    # Requests are never sent before their scheduled time.
    assert all(r.sent_s >= r.scheduled_s - 1e-3 for r in results)


def test_load_is_open_loop() -> None:
    """A slow server must not slow arrivals: 10 requests at once, each taking 0.3 s."""
    specs = generate_workload(small_workload(rate=400, count=10))

    async def go() -> float:
        async with mock_client(echo_handler(delay=0.1)) as client:
            start = time.perf_counter()
            results = await run_workload(specs, client, "m", timeout_s=5)
            assert all(r.status == "ok" for r in results)
            return time.perf_counter() - start

    elapsed = asyncio.run(go())
    # Closed-loop sending would take about 10 x 0.3 s = 3 s.
    assert elapsed < 1.2


def test_failures_do_not_stop_the_run() -> None:
    specs = generate_workload(small_workload(rate=300, count=10))
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] % 2 == 0:
            return httpx.Response(500)
        return streaming_response(well_formed_stream(3))

    async def go() -> list:  # type: ignore[type-arg]
        async with mock_client(handler) as client:
            return await run_workload(specs, client, "m", timeout_s=5)

    results = asyncio.run(go())
    assert len(results) == 10
    assert sum(r.status == "ok" for r in results) == 5
    assert sum(r.status == "http_error" for r in results) == 5


def test_warm_up_sends_the_requested_number_of_requests() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return streaming_response(well_formed_stream(8))

    async def go() -> None:
        async with mock_client(handler) as client:
            await warm_up(client, "m", 3, vocab_size=5000, timeout_s=5)

    asyncio.run(go())
    assert calls["n"] == 3


def test_warm_up_failure_raises() -> None:
    async def go() -> None:
        async with mock_client(lambda request: httpx.Response(503)) as client:
            await warm_up(client, "m", 2, vocab_size=5000, timeout_s=5)

    with pytest.raises(RuntimeError, match="warm-up request 0 failed"):
        asyncio.run(go())
