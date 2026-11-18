"""Open-loop execution of a workload.

Requests are sent at their scheduled arrival times whether or not earlier ones have finished, so a
slow server accumulates queueing delay instead of silently throttling the load (which would hide
tail latency). Each result records how late it was sent, so a saturated load generator is visible.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Sequence

import httpx

from bench.client import RequestResult, execute_request
from bench.workload import RequestSpec


async def run_workload(
    specs: Sequence[RequestSpec],
    client: httpx.AsyncClient,
    model: str,
    *,
    timeout_s: float,
    ignore_eos: bool = True,
) -> list[RequestResult]:
    """Send every request at its arrival time and return results ordered by request id."""
    start = time.perf_counter()
    tasks: list[asyncio.Task[RequestResult]] = []
    for spec in specs:
        delay = start + spec.arrival_s - time.perf_counter()
        if delay > 0:
            await asyncio.sleep(delay)
        tasks.append(
            asyncio.create_task(
                execute_request(
                    client, model, spec, start, timeout_s=timeout_s, ignore_eos=ignore_eos
                )
            )
        )
    results = await asyncio.gather(*tasks)
    return sorted(results, key=lambda r: r.id)


async def warm_up(
    client: httpx.AsyncClient,
    model: str,
    count: int,
    *,
    vocab_size: int,
    timeout_s: float,
    min_token_id: int = 1000,
    prompt_tokens: int = 32,
    output_tokens: int = 8,
) -> None:
    """Send ``count`` sequential throwaway requests (GPU warm-up, graph capture, caches).

    Raises ``RuntimeError`` if any fails: measuring against a broken server is pointless.
    """
    for index in range(count):
        spec = RequestSpec(
            id=index,
            arrival_s=0.0,
            prompt_token_ids=[
                min_token_id + (index + i) % (vocab_size - min_token_id)
                for i in range(prompt_tokens)
            ],
            max_tokens=output_tokens,
        )
        result = await execute_request(
            client, model, spec, time.perf_counter(), timeout_s=timeout_s
        )
        if result.status != "ok":
            raise RuntimeError(f"warm-up request {index} failed: {result.error}")
