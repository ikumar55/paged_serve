"""One load-generator run split over several processes (D-030).

A single asyncio process reading thousands of streamed responses becomes the bottleneck of the
measurement: vLLM's 64-sequence run was flagged generator-saturated at 96 requests per second
(D-019). Here ``K`` processes share one run. Each generates the whole workload from its seed (so
every process holds exactly the requests a single process would), keeps the requests whose id
modulo ``K`` is its own and sends each at the request's own arrival time, measured from one start
instant chosen by the parent. The arrival process, the prompts and the output lengths are therefore
exactly the single process's; only the sending and reading are spread over ``K`` event loops. Each
process also measures its own event-loop lateness, so a generator that is still too busy shows up
in the result.
"""

from __future__ import annotations

import asyncio
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Any

import httpx

from bench.client import RequestResult
from bench.metrics import distribution
from bench.runner import run_workload, with_loop_lag
from bench.workload import WorkloadConfig, generate_workload

DEFAULT_LEAD_S = 5.0  # time given to every process to start up and build its workload


@dataclass(frozen=True)
class ShardJob:
    url: str
    model: str
    workload: dict[str, Any]
    shard: int
    shards: int
    start_wall: float  # time.time() at which arrival time 0 is
    timeout_s: float
    ignore_eos: bool


@dataclass(frozen=True)
class ShardOutcome:
    shard: int
    results: list[RequestResult]
    loop_lag: dict[str, float] | None  # distribution of the event loop's lateness (seconds)


def shard_of(request_id: int, shards: int) -> int:
    return request_id % shards


def monotonic_start(start_wall: float, now_wall: float, now_monotonic: float) -> float:
    """The ``time.perf_counter()`` value of the wall-clock instant ``start_wall``.

    The processes of one run share a wall-clock start but each has its own monotonic clock.
    """
    return now_monotonic + (start_wall - now_wall)


async def _run_shard(job: ShardJob) -> ShardOutcome:
    specs = [
        spec
        for spec in generate_workload(WorkloadConfig.from_dict(job.workload))
        if shard_of(spec.id, job.shards) == job.shard
    ]
    # Convert the shared wall-clock instant to this process's monotonic clock.
    start = monotonic_start(job.start_wall, time.time(), time.perf_counter())
    async with httpx.AsyncClient(
        base_url=job.url,
        timeout=None,
        limits=httpx.Limits(max_connections=None, max_keepalive_connections=None),
    ) as client:
        results, lag = await with_loop_lag(
            run_workload(
                specs,
                client,
                job.model,
                timeout_s=job.timeout_s,
                ignore_eos=job.ignore_eos,
                start=start,
            )
        )
    return ShardOutcome(job.shard, results, distribution(lag))


def run_shard(job: ShardJob) -> ShardOutcome:
    """Entry point of one generator process."""
    return asyncio.run(_run_shard(job))


def merge(outcomes: list[ShardOutcome], num_requests: int) -> list[RequestResult]:
    """Every shard's results as one list ordered by request id; every request exactly once."""
    merged = sorted((r for o in outcomes for r in o.results), key=lambda r: r.id)
    if [r.id for r in merged] != list(range(num_requests)):
        raise RuntimeError(
            f"the shards returned {len(merged)} results that are not requests 0 to "
            f"{num_requests - 1} exactly once"
        )
    return merged


def shard_stats(outcomes: list[ShardOutcome]) -> list[dict[str, Any]]:
    """Per process: requests, 99th percentile of the send lag and of the event-loop lag."""
    stats = []
    for outcome in sorted(outcomes, key=lambda o: o.shard):
        send_lag = distribution([max(0.0, r.sent_s - r.scheduled_s) for r in outcome.results])
        stats.append(
            {
                "shard": outcome.shard,
                "requests": len(outcome.results),
                "send_lag_p99_s": None if send_lag is None else send_lag["p99"],
                "loop_lag_p99_s": None if outcome.loop_lag is None else outcome.loop_lag["p99"],
            }
        )
    return stats


def run_sharded(
    url: str,
    model: str,
    workload: WorkloadConfig,
    shards: int,
    *,
    timeout_s: float,
    ignore_eos: bool,
    lead_s: float = DEFAULT_LEAD_S,
) -> list[ShardOutcome]:
    """Run ``workload`` against ``url`` from ``shards`` processes; returns each one's outcome."""
    if shards < 1:
        raise ValueError("shards must be at least 1")
    start_wall = time.time() + lead_s
    jobs = [
        ShardJob(url, model, workload.to_dict(), shard, shards, start_wall, timeout_s, ignore_eos)
        for shard in range(shards)
    ]
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=shards, mp_context=context) as pool:
        return list(pool.map(run_shard, jobs))
