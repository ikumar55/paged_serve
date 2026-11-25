"""The in-flight analysis of load-generator results (the ConnectError ceiling, D-025)."""

from __future__ import annotations

import json
from pathlib import Path

from bench import inflight as inf


def req(sent: float, e2e: float, error: str | None = None) -> dict[str, object]:
    return {"type": "request", "sent_s": sent, "e2e_s": e2e, "error": error}


def test_the_profile_counts_overlapping_requests() -> None:
    # three requests: [0, 4), [1, 3), [2, 6): in flight 1, 2, 3, then 2 (at 3), 1 (at 4), 0 (at 6)
    profile = inf.inflight_profile([req(0, 4), req(1, 2), req(2, 4)])
    assert max(c for _, c in profile) == 3
    assert [inf.inflight_at(profile, t) for t in (0, 1, 2, 2.5, 3, 4, 5, 6, 7)] == [
        1, 2, 3, 3, 2, 1, 1, 0, 0,
    ]  # fmt: skip
    assert inf.inflight_at(profile, -1) == 0  # before anything was sent


def test_a_request_ending_exactly_when_another_starts_is_not_counted_as_overlapping() -> None:
    profile = inf.inflight_profile([req(0, 1), req(1, 1)])
    assert max(c for _, c in profile) == 1


def test_failed_requests_never_held_a_connection() -> None:
    # a ceiling of 3: the fourth request fails at once; the failure must not raise the peak
    requests = [
        req(0, 10),
        req(1, 10),
        req(2, 10),
        req(3, 0.006, "ConnectError"),
        req(4, 0.006, "ConnectError"),
    ]
    report = inf.analyse(requests)
    assert report.requests == 5 and report.failed == 2
    assert report.first_failure_s == 3
    assert report.inflight_at_first_failure == 3
    assert report.peak_inflight == 3
    assert report.failure_kinds == ("ConnectError",)


def test_a_run_without_failures_reports_only_the_peak() -> None:
    report = inf.analyse([req(0, 2), req(1, 2)])
    assert (report.failed, report.first_failure_s, report.inflight_at_first_failure) == (
        0,
        None,
        None,
    )
    assert report.peak_inflight == 2 and report.failure_kinds == ()


def test_the_earliest_failure_is_the_first_whatever_the_order_in_the_file() -> None:
    report = inf.analyse([req(9, 0.1, "ConnectError"), req(0, 20), req(5, 0.1, "ReadTimeout")])
    assert report.first_failure_s == 5
    assert report.failure_kinds == ("ConnectError", "ReadTimeout")


def test_reading_a_result_file_and_rendering_it(tmp_path: Path) -> None:
    path = tmp_path / "run.jsonl"
    lines = [{"type": "meta-ish", "client_env": {}}, req(0, 5), req(1, 0.01, "ConnectError")]
    path.write_text("\n".join(json.dumps(line) for line in lines) + "\n", encoding="utf-8")
    assert len(inf.load_requests(path)) == 2  # the header line is not a request
    text = inf.render({"run": inf.analyse(inf.load_requests(path))})
    assert "| run | 2 | 1 | 1.0 | 1 | 1 | ConnectError |" in text
    assert inf.main([str(path)]) == 0


def test_it_reproduces_the_gather_path_finding_on_the_recorded_results() -> None:
    """The recorded gather runs peak at exactly 997 and fail at it; the kernel runs do not."""
    root = Path(__file__).resolve().parents[1] / "results"
    gather = [root / f"d025-sat-gather-rtx5090-seed{s}" / "b64-w0-r32.jsonl" for s in range(4)]
    kernel = [root / f"d025-sat-kernel-rtx5090-seed{s}" / "b64-w0-r32.jsonl" for s in range(4)]
    if not all(p.exists() for p in [*gather, *kernel]):
        return  # the recorded results are not in this checkout
    for path in gather:
        report = inf.analyse(inf.load_requests(path))
        assert report.failed > 0 and report.failure_kinds == ("ConnectError",)
        assert report.peak_inflight == 997 and report.inflight_at_first_failure == 997
    for path in kernel:
        report = inf.analyse(inf.load_requests(path))
        assert report.failed == 0 and report.peak_inflight < 997
