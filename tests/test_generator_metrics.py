"""Open-file limits, the steady-state window, the in-flight peak and event-loop lag (D-030)."""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

import httpx
import pytest

from bench.client import RequestResult
from bench.fdlimits import UNLIMITED, own_limits, parse_proc_limits, process_limits, soft_limit
from bench.metrics import (
    LOOP_LAG_WARN_S,
    STEADY_WINDOW_S,
    peak_inflight,
    steady_window,
    summarize,
)
from bench.runner import run_workload, with_loop_lag
from bench.workload import RequestSpec
from tests.fake_server import echo_handler, mock_client

PROC_LIMITS = """Limit                     Soft Limit           Hard Limit           Units
Max cpu time              unlimited            unlimited            seconds
Max open files            1024                 1048576              files
Max locked memory         8388608              8388608              bytes
"""


def result(
    id: int,
    *,
    sent: float = 0.0,
    e2e: float = 1.0,
    scheduled: float | None = None,
    status: str = "ok",
    error: str | None = None,
    tokens: int = 4,
) -> RequestResult:
    return RequestResult(
        id=id,
        scheduled_s=sent if scheduled is None else scheduled,
        sent_s=sent,
        prompt_tokens=8,
        requested_output_tokens=tokens,
        status=status,
        error=error,
        output_tokens=tokens if status == "ok" else 0,
        ttft_s=0.1 if status == "ok" else None,
        e2e_s=e2e,
    )


# -- open-file limits ------------------------------------------------------------------------------


def test_the_proc_limits_text_is_parsed() -> None:
    assert parse_proc_limits(PROC_LIMITS) == {"soft": 1024, "hard": 1048576}
    unlimited = PROC_LIMITS.replace("1024  ", "unlimited")
    assert parse_proc_limits(unlimited) == {"soft": UNLIMITED, "hard": 1048576}


@pytest.mark.parametrize(
    "text", ["", "Max cpu time unlimited unlimited seconds\n", "Max open files\n"]
)
def test_a_proc_limits_text_without_the_line_is_none_not_a_guess(text: str) -> None:
    assert parse_proc_limits(text) is None


def test_another_process_limits_are_read_from_its_proc_entry(tmp_path: Path) -> None:
    (tmp_path / "4242").mkdir()
    (tmp_path / "4242" / "limits").write_text(PROC_LIMITS)
    assert process_limits(4242, tmp_path) == {"soft": 1024, "hard": 1048576}
    assert process_limits(9999, tmp_path) is None  # no such process


def test_a_soft_limit_is_a_number_infinity_or_unknown() -> None:
    assert soft_limit({"soft": 1024, "hard": 4096}) == 1024.0
    assert soft_limit({"soft": UNLIMITED, "hard": UNLIMITED}) == float("inf")
    assert soft_limit(None) is None
    assert soft_limit({}) is None


def test_this_process_reports_its_own_limits_where_the_platform_has_them() -> None:
    limits = own_limits()
    if sys.platform == "win32":
        assert limits is None
    else:
        assert limits is not None and set(limits) == {"soft", "hard"}
        assert soft_limit(limits) is not None and soft_limit(limits) >= 3  # type: ignore[operator]


# -- the steady-state window -----------------------------------------------------------------------


def long_run(extra: list[RequestResult]) -> list[RequestResult]:
    """Results whose last scheduled arrival is at 60 s, so the 15 to 50 s window exists."""
    return [*extra, result(10_000, sent=60.0, e2e=0.1)]


def test_the_steady_window_counts_completions_between_15_and_50_seconds() -> None:
    assert STEADY_WINDOW_S == (15.0, 50.0)
    results = long_run(
        [
            result(0, sent=14.0, e2e=0.5),  # done at 14.5: before the window
            result(1, sent=14.0, e2e=1.0),  # done at 15.0: the window's first instant, inside
            result(2, sent=20.0, e2e=1.0),  # inside
            result(3, sent=48.0, e2e=1.9, tokens=10),  # done at 49.9: inside
            result(4, sent=49.0, e2e=1.0),  # done at 50.0: the end is exclusive
            result(5, sent=30.0, e2e=1.0, status="timeout", error="timeout"),  # failed
        ]
    )
    window = steady_window(results)
    assert window is not None
    assert window["completed_rps"] == pytest.approx(3 / 35)
    assert window["output_tokens_per_s"] == pytest.approx((4 + 4 + 10) / 35)
    assert (window["start_s"], window["end_s"]) == (15.0, 50.0)


def test_a_run_whose_arrivals_stop_before_the_window_ends_has_no_steady_window() -> None:
    assert steady_window([result(0, sent=10.0, e2e=1.0), result(1, sent=49.9)]) is None
    assert steady_window([]) is None


# -- the in-flight peak ----------------------------------------------------------------------------


def test_the_in_flight_peak_counts_overlap() -> None:
    results = [
        result(0, sent=0.0, e2e=3.0),
        result(1, sent=1.0, e2e=3.0),
        result(2, sent=2.0, e2e=3.0),
    ]
    assert peak_inflight(results) == 3
    assert peak_inflight([result(0, sent=0.0, e2e=1.0), result(1, sent=2.0, e2e=1.0)]) == 1


def test_a_completion_and_a_send_at_the_same_instant_do_not_overlap() -> None:
    assert peak_inflight([result(0, sent=0.0, e2e=1.0), result(1, sent=1.0, e2e=1.0)]) == 1


def test_timed_out_requests_held_their_connection_but_failed_connections_did_not() -> None:
    timed_out = result(0, sent=0.0, e2e=5.0, status="timeout", error="timeout")
    refused = result(1, sent=1.0, e2e=0.001, status="error", error="ConnectError")
    refused_slowly = result(2, sent=1.5, e2e=4.0, status="error", error="ConnectTimeout")
    served = result(3, sent=2.0, e2e=1.0)
    assert peak_inflight([timed_out, refused, refused_slowly, served]) == 2
    assert peak_inflight([]) == 0


# -- the summary -----------------------------------------------------------------------------------


def dist(p99: float) -> dict[str, float]:
    return {"count": 100.0, "mean": 0.0, "p50": 0.0, "p90": 0.0, "p99": p99, "max": p99}


def test_the_summary_carries_the_new_fields() -> None:
    results = long_run([result(i, sent=float(i), e2e=2.0) for i in range(60)])
    summary = summarize(results, [dist(0.001), dist(0.004)])
    assert summary["steady_window"]["completed_rps"] == pytest.approx(1.0)
    assert summary["peak_inflight"] == 2
    assert summary["event_loop_lag_p99_max_s"] == 0.004
    assert len(summary["event_loop_lag_s"]) == 2 and not summary["generator_saturated"]


def test_a_slow_event_loop_marks_the_generator_saturated_even_if_sends_were_on_time() -> None:
    results = [result(i, sent=float(i), e2e=1.0) for i in range(10)]  # sent exactly on schedule
    assert not summarize(results, [dist(LOOP_LAG_WARN_S)])["generator_saturated"]
    assert summarize(results, [dist(0.001), dist(LOOP_LAG_WARN_S + 0.005)])["generator_saturated"]


def test_without_loop_lag_samples_the_flag_depends_on_the_send_lag_alone() -> None:
    results = [result(i, sent=float(i), e2e=1.0) for i in range(10)]
    summary = summarize(results)
    assert summary["event_loop_lag_p99_max_s"] is None and not summary["generator_saturated"]
    late = [result(i, sent=float(i) + 0.5, scheduled=float(i), e2e=1.0) for i in range(10)]
    assert summarize(late)["generator_saturated"]


# -- the event-loop lag probe and the shared start -------------------------------------------------


def test_the_loop_lag_probe_sees_a_blocked_event_loop() -> None:
    async def blocked() -> str:
        await asyncio.sleep(0.05)
        time.sleep(0.2)  # the loop cannot run its timer
        await asyncio.sleep(0.05)
        return "done"

    value, samples = asyncio.run(with_loop_lag(blocked()))
    assert value == "done" and samples
    assert max(samples) > 0.15  # it was late by about the time it was blocked


def test_the_loop_lag_probe_is_small_when_nothing_blocks_and_stops_with_the_work() -> None:
    async def idle() -> None:
        await asyncio.sleep(0.15)

    _, samples = asyncio.run(with_loop_lag(idle()))
    assert samples and all(s >= 0.0 for s in samples)
    assert sorted(samples)[len(samples) // 2] < 0.009  # lateness only, not the interval itself
    count = len(samples)
    time.sleep(0.05)
    assert len(samples) == count  # the probe is no longer running


def test_a_run_started_in_the_past_reports_its_lateness_instead_of_hiding_it() -> None:
    async def go() -> list[RequestResult]:
        async with mock_client(echo_handler()) as client:
            specs = [RequestSpec(id=0, arrival_s=0.5, prompt_token_ids=[1, 2], max_tokens=2)]
            return await run_workload(
                specs, client, "m", timeout_s=5, start=time.perf_counter() - 2.0
            )

    (only,) = asyncio.run(go())
    assert only.scheduled_s == 0.5
    assert only.sent_s >= 2.0  # the arrival was 1.5 s overdue when it was sent, and that shows


def test_a_run_given_a_future_start_waits_for_it() -> None:
    async def go() -> list[RequestResult]:
        transport = httpx.MockTransport(echo_handler())
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
            specs = [RequestSpec(id=0, arrival_s=0.0, prompt_token_ids=[1], max_tokens=1)]
            return await run_workload(
                specs, client, "m", timeout_s=5, start=time.perf_counter() + 0.3
            )

    begun = time.perf_counter()
    (only,) = asyncio.run(go())
    assert time.perf_counter() - begun >= 0.25  # it waited for the shared start
    assert only.sent_s == pytest.approx(0.0, abs=0.1)  # and was sent at the run's time 0
