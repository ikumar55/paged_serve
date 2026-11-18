"""Summary statistics for a load-generator run.

Latency statistics use successful requests only; failures are counted and broken down by cause.
Percentiles use linear interpolation between order statistics (``numpy.percentile`` default), so
tail percentiles of small samples are not meaningful: every distribution reports its ``count``.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from itertools import pairwise
from typing import Any

import numpy as np

from bench.client import RequestResult

# If the median-to-tail lag between scheduled and actual send time exceeds this, the load
# generator itself (not the server) was the bottleneck and latencies are understated.
GENERATOR_LAG_WARN_S = 0.05


def distribution(values: Sequence[float]) -> dict[str, float] | None:
    if not values:
        return None
    array = np.asarray(values, dtype=float)
    p50, p90, p99 = np.percentile(array, [50, 90, 99])
    return {
        "count": float(array.size),
        "mean": float(array.mean()),
        "p50": float(p50),
        "p90": float(p90),
        "p99": float(p99),
        "max": float(array.max()),
    }


def inter_token_latencies(result: RequestResult) -> list[float]:
    times = result.token_times_s
    return [later - earlier for earlier, later in pairwise(times)]


def time_per_output_token(result: RequestResult) -> float | None:
    """Mean decode time per token after the first, using the token count the server reported."""
    if result.status != "ok" or result.output_tokens < 2:
        return None
    if result.ttft_s is None or result.e2e_s is None:
        return None
    return (result.e2e_s - result.ttft_s) / (result.output_tokens - 1)


def error_label(result: RequestResult) -> str:
    return result.error or result.status


def summarize(results: Sequence[RequestResult]) -> dict[str, Any]:
    ok = [r for r in results if r.status == "ok"]
    failures = Counter(error_label(r) for r in results if r.status != "ok")

    sent = [r.sent_s for r in results]
    completions = [r.sent_s + r.e2e_s for r in results if r.e2e_s is not None]
    duration = (max(completions) - min(sent)) if results and completions else 0.0
    output_tokens = sum(r.output_tokens for r in ok)

    arrival_rate = None
    if len(results) > 1 and max(sent) > min(sent):
        arrival_rate = (len(results) - 1) / (max(sent) - min(sent))

    lags = [max(0.0, r.sent_s - r.scheduled_s) for r in results]
    lag = distribution(lags)

    itl = [gap for r in ok for gap in inter_token_latencies(r)]
    tpot = [t for r in ok if (t := time_per_output_token(r)) is not None]

    return {
        "num_requests": len(results),
        "num_ok": len(ok),
        "num_failed": len(results) - len(ok),
        "failure_counts": dict(failures),
        "duration_s": duration,
        "achieved_arrival_rate_rps": arrival_rate,
        "request_throughput_rps": len(ok) / duration if duration > 0 else None,
        "output_tokens_total": output_tokens,
        "output_tokens_per_s": output_tokens / duration if duration > 0 else None,
        "tokens_counted_from_usage": all(r.tokens_from_usage for r in ok) if ok else None,
        "ttft_s": distribution([r.ttft_s for r in ok if r.ttft_s is not None]),
        "itl_s": distribution(itl),
        "tpot_s": distribution(tpot),
        "e2e_s": distribution([r.e2e_s for r in ok if r.e2e_s is not None]),
        "client_lag_s": lag,
        "generator_saturated": bool(lag is not None and lag["p99"] > GENERATOR_LAG_WARN_S),
    }
