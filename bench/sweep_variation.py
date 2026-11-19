"""How much do repeated runs of the same configuration differ?

Takes several sweep directories (different workload seeds, or repeats) and, for every
configuration that appears in more than one, reports the number of valid runs and the mean and
range of throughput and latency. A single run says nothing about its own noise, so a comparison
between two configurations (or between systems) is only meaningful if it is larger than this
spread.

Usage:
    python -m bench.sweep_variation results/hf-static results/hf-static-ext \\
        results/hf-static-seed1 results/hf-static-seed2 --out results/variation.md
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from bench.sweep_report import Row, load_rows


@dataclass(frozen=True)
class Spread:
    n: int
    mean: float
    low: float
    high: float

    @property
    def relative_range(self) -> float:
        return (self.high - self.low) / self.mean if self.mean else float("inf")


def spread(values: Sequence[float]) -> Spread | None:
    if not values:
        return None
    return Spread(len(values), sum(values) / len(values), min(values), max(values))


def group_runs(directories: Sequence[Path]) -> dict[tuple[int, float, float], list[Row]]:
    """Valid runs, grouped by (batch size, window, rate), across every directory."""
    groups: dict[tuple[int, float, float], list[Row]] = defaultdict(list)
    for directory in directories:
        for row in load_rows(directory):
            if row.valid:
                groups[(row.batch_size, row.window_ms, row.rate)].append(row)
    return groups


def _fmt(s: Spread | None, scale: float, digits: int) -> str:
    if s is None:
        return "-"
    return (
        f"{s.mean * scale:.{digits}f} ({s.low * scale:.{digits}f} to {s.high * scale:.{digits}f})"
    )


def render(groups: dict[tuple[int, float, float], list[Row]], minimum: int = 2) -> str:
    lines = [
        "# Run-to-run variation",
        "",
        "Configurations with at least "
        f"{minimum} valid runs. Each cell is the mean with the minimum to maximum in brackets. "
        "Runs differ in workload seed (and in time), so this includes both system noise and "
        "which requests were drawn.",
        "",
        "| batch | window ms | rate req/s | runs | req/s | tok/s | TTFT p50 ms | TTFT p99 ms "
        "| e2e p99 s | req/s range / mean |",
        "| ---: | ---: | ---: | ---: | --- | --- | --- | --- | --- | ---: |",
    ]
    for (batch, window, rate), rows in sorted(groups.items(), key=lambda kv: (kv[0][2], kv[0])):
        if len(rows) < minimum:
            continue
        throughput = spread([r.throughput_rps for r in rows if r.throughput_rps is not None])
        tokens = spread([r.tokens_per_s for r in rows if r.tokens_per_s is not None])
        ttft50 = spread([r.ttft_p50_s for r in rows if r.ttft_p50_s is not None])
        ttft99 = spread([r.ttft_p99_s for r in rows if r.ttft_p99_s is not None])
        e2e99 = spread([r.e2e_p99_s for r in rows if r.e2e_p99_s is not None])
        relative = f"{throughput.relative_range:.0%}" if throughput else "-"
        lines.append(
            f"| {batch} | {window:g} | {rate:g} | {len(rows)} | {_fmt(throughput, 1, 2)} "
            f"| {_fmt(tokens, 1, 0)} | {_fmt(ttft50, 1000, 0)} | {_fmt(ttft99, 1000, 0)} "
            f"| {_fmt(e2e99, 1, 1)} | {relative} |"
        )
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0] if __doc__ else None)
    parser.add_argument("dirs", type=Path, nargs="+", help="sweep directories to combine")
    parser.add_argument("--out", type=Path, help="write the markdown here (never overwrites)")
    parser.add_argument("--min-runs", type=int, default=2)
    args = parser.parse_args(argv)
    text = render(group_runs(args.dirs), args.min_runs)
    if args.out is None:
        print(text)
    else:
        with args.out.open("x") as handle:
            handle.write(text)
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
