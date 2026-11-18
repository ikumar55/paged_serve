import pytest

from bench.client import RequestResult
from bench.metrics import distribution, inter_token_latencies, summarize, time_per_output_token


def ok_result(
    id: int,
    *,
    sent_s: float = 0.0,
    scheduled_s: float | None = None,
    ttft_s: float = 0.1,
    itl_s: float = 0.01,
    tokens: int = 5,
) -> RequestResult:
    times = [ttft_s + i * itl_s for i in range(tokens)]
    return RequestResult(
        id=id,
        scheduled_s=sent_s if scheduled_s is None else scheduled_s,
        sent_s=sent_s,
        prompt_tokens=10,
        requested_output_tokens=tokens,
        status="ok",
        http_status=200,
        output_tokens=tokens,
        tokens_from_usage=True,
        ttft_s=ttft_s,
        e2e_s=times[-1],
        token_times_s=times,
    )


def failed_result(
    id: int, status: str, error: str, http_status: int | None = None
) -> RequestResult:
    return RequestResult(
        id=id,
        scheduled_s=0.0,
        sent_s=0.0,
        prompt_tokens=10,
        requested_output_tokens=5,
        status=status,
        http_status=http_status,
        error=error,
        e2e_s=0.5,
    )


def test_distribution_uses_linear_interpolation() -> None:
    dist = distribution([0.1, 0.2, 0.3, 0.4])
    assert dist is not None
    assert dist["count"] == 4
    assert dist["mean"] == pytest.approx(0.25)
    assert dist["p50"] == pytest.approx(0.25)
    assert dist["p99"] == pytest.approx(0.397)
    assert dist["max"] == pytest.approx(0.4)


def test_distribution_of_nothing_is_none() -> None:
    assert distribution([]) is None


def test_latency_helpers() -> None:
    result = ok_result(0, ttft_s=0.1, itl_s=0.02, tokens=4)
    assert inter_token_latencies(result) == pytest.approx([0.02, 0.02, 0.02])
    assert time_per_output_token(result) == pytest.approx(0.02)
    assert time_per_output_token(ok_result(1, tokens=1)) is None  # needs two tokens
    assert time_per_output_token(failed_result(2, "timeout", "timeout")) is None


def test_summary_of_successful_requests() -> None:
    results = [
        ok_result(i, sent_s=i * 0.5, ttft_s=ttft) for i, ttft in enumerate([0.1, 0.2, 0.3, 0.4])
    ]
    summary = summarize(results)
    assert summary["num_requests"] == 4 and summary["num_ok"] == 4 and summary["num_failed"] == 0
    assert summary["ttft_s"]["p50"] == pytest.approx(0.25)
    assert summary["itl_s"]["p50"] == pytest.approx(0.01)
    assert summary["tpot_s"]["p50"] == pytest.approx(0.01)
    # First send at 0 s; the last request (sent at 1.5 s, e2e 0.44 s) finishes at 1.94 s.
    assert summary["duration_s"] == pytest.approx(1.94)
    assert summary["output_tokens_total"] == 20
    assert summary["output_tokens_per_s"] == pytest.approx(20 / 1.94)
    assert summary["request_throughput_rps"] == pytest.approx(4 / 1.94)
    assert summary["achieved_arrival_rate_rps"] == pytest.approx(3 / 1.5)
    assert summary["tokens_counted_from_usage"] is True
    assert summary["generator_saturated"] is False


def test_failures_are_counted_but_excluded_from_latency() -> None:
    results = [
        ok_result(0, ttft_s=0.1),
        ok_result(1, ttft_s=0.3),
        failed_result(2, "http_error", "http_429", 429),
        failed_result(3, "timeout", "timeout"),
        failed_result(4, "error", "ConnectError"),
        failed_result(5, "http_error", "http_429", 429),
    ]
    summary = summarize(results)
    assert summary["num_ok"] == 2 and summary["num_failed"] == 4
    assert summary["failure_counts"] == {"http_429": 2, "timeout": 1, "ConnectError": 1}
    assert summary["ttft_s"]["count"] == 2
    assert summary["ttft_s"]["p50"] == pytest.approx(0.2)
    assert summary["output_tokens_total"] == 10  # only successful requests count


def test_late_sending_flags_a_saturated_generator() -> None:
    results = [
        ok_result(i, sent_s=i * 0.1 + (0.2 if i % 5 == 0 else 0.0), scheduled_s=i * 0.1)
        for i in range(50)
    ]
    summary = summarize(results)
    assert summary["client_lag_s"]["max"] == pytest.approx(0.2)
    assert summary["generator_saturated"] is True


def test_summary_of_no_requests() -> None:
    summary = summarize([])
    assert summary["num_requests"] == 0
    assert summary["duration_s"] == 0.0
    assert summary["ttft_s"] is None and summary["output_tokens_per_s"] is None
    assert summary["generator_saturated"] is False


def test_summary_when_everything_failed() -> None:
    summary = summarize(
        [failed_result(0, "timeout", "timeout"), failed_result(1, "timeout", "timeout")]
    )
    assert summary["num_ok"] == 0
    assert summary["ttft_s"] is None
    assert summary["tokens_counted_from_usage"] is None
    assert summary["failure_counts"] == {"timeout": 2}
