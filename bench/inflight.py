"""How many requests were in flight, and did the failures start at a ceiling?

At saturation a server that cannot keep up accumulates unfinished requests. If that number reaches a
process limit (the default of 1,024 open files per process on Linux allows about a thousand
connections), new connections fail at once and the load generator records a ``ConnectError``. Those
failures are then a property of the setup, not of the engine's speed, and a comparison between
systems that hit the ceiling and systems that do not is not like for like.

This reads the result files of ``bench.loadgen`` and reports, per run: how many requests failed,
when the first failure happened, how many connected requests were in flight at that moment, and the
peak in flight. In the four gather-path runs of D-025 all four peaks are exactly 997 and the first
failure happens at that value; the kernel and graph runs peak lower and have no failures.

Usage:
    python -m bench.inflight results/d025-sat-gather-rtx5090-seed0/b64-w0-r32.jsonl ...
"""

from __future__ import annotations

import argparse
import bisect
import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class InflightReport:
    requests: int
    failed: int
    first_failure_s: float | None  # send time of the earliest failed request
    inflight_at_first_failure: int | None  # connected requests in flight at that moment
    peak_inflight: int  # most connected requests in flight at once
    failure_kinds: tuple[str, ...]


def load_requests(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    return [row for row in rows if row.get("type") == "request"]


def inflight_profile(requests: Iterable[dict[str, Any]]) -> list[tuple[float, int]]:
    """``(time, requests in flight just after it)`` at every send and every completion."""
    events: list[tuple[float, int]] = []
    for request in requests:
        events.append((request["sent_s"], 1))
        events.append((request["sent_s"] + request["e2e_s"], -1))
    events.sort()
    current = 0
    profile = []
    for time_s, delta in events:
        current += delta
        profile.append((time_s, current))
    return profile


def inflight_at(profile: Sequence[tuple[float, int]], time_s: float) -> int:
    index = bisect.bisect_right([t for t, _ in profile], time_s) - 1
    return profile[index][1] if index >= 0 else 0


def analyse(requests: Sequence[dict[str, Any]]) -> InflightReport:
    """Failed requests are left out of the in-flight count: they never held a connection."""
    failed = sorted((r for r in requests if r.get("error")), key=lambda r: r["sent_s"])
    profile = inflight_profile(r for r in requests if not r.get("error"))
    first = failed[0]["sent_s"] if failed else None
    return InflightReport(
        requests=len(requests),
        failed=len(failed),
        first_failure_s=first,
        inflight_at_first_failure=inflight_at(profile, first) if first is not None else None,
        peak_inflight=max((count for _, count in profile), default=0),
        failure_kinds=tuple(sorted({str(r["error"]) for r in failed})),
    )


def render(reports: dict[str, InflightReport]) -> str:
    lines = [
        "| run | requests | failed | first failure (s) | in flight then | peak in flight | kinds |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for name, r in reports.items():
        first = "-" if r.first_failure_s is None else f"{r.first_failure_s:.1f}"
        then = "-" if r.inflight_at_first_failure is None else str(r.inflight_at_first_failure)
        kinds = ", ".join(r.failure_kinds) or "-"
        cells = [name, r.requests, r.failed, first, then, r.peak_inflight, kinds]
        lines.append("| " + " | ".join(str(cell) for cell in cells) + " |")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0] if __doc__ else None)
    parser.add_argument(
        "files", nargs="+", type=Path, help="result files (.jsonl) of bench.loadgen"
    )
    args = parser.parse_args(argv)
    reports = {str(path): analyse(load_requests(path)) for path in args.files}
    print(render(reports))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
