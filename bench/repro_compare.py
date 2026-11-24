"""Does a repeat of a sweep reproduce the original?

Pairs each run of a repeat sweep with the run of the same name in an original sweep (same seed,
same workload, same configuration; only the time differs) and reports the difference in delivered
throughput. Pairs in which either run was flagged contended are listed but kept out of the
summary. Below saturation delivered throughput equals the offered rate by construction, so the
informative pairs are the saturated ones; the table shows the tail latency too.

Usage:
    python -m bench.repro_compare \\
        --pair engine results/engine-rtx5090-seed1 results/repro-engine-rtx5090-seed1 \\
        --pair vllm results/vllm-rtx5090-seed1 results/repro-vllm-rtx5090-seed1 \\
        --out results/repro-vs-original-rtx5090.md
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from bench.sweep_report import Row, load_rows


@dataclass(frozen=True)
class Pair:
    label: str
    sweep: str
    original: Row
    repeat: Row

    @property
    def usable(self) -> bool:
        return (
            not self.original.contended
            and not self.repeat.contended
            and self.original.throughput_rps is not None
            and self.repeat.throughput_rps is not None
        )

    @property
    def diff_percent(self) -> float | None:
        a, b = self.original.throughput_rps, self.repeat.throughput_rps
        if not a or b is None:
            return None
        return (b - a) / a * 100


def pair_sweeps(label: str, original: Path, repeat: Path) -> list[Pair]:
    """Runs present in both sweeps, matched by run name, in the repeat sweep's order."""
    originals = {row.name: row for row in load_rows(original)}
    return [
        Pair(label, repeat.name, originals[row.name], row)
        for row in load_rows(repeat)
        if row.name in originals
    ]


def summarise(pairs: Sequence[Pair]) -> dict[str, tuple[int, float, float, float]]:
    """Per label: (pairs, mean absolute %, max absolute %, mean signed %) over usable pairs."""
    out: dict[str, tuple[int, float, float, float]] = {}
    for label in dict.fromkeys(p.label for p in pairs):
        diffs = [
            d for p in pairs if p.label == label and p.usable and (d := p.diff_percent) is not None
        ]
        if diffs:
            out[label] = (
                len(diffs),
                sum(abs(d) for d in diffs) / len(diffs),
                max(abs(d) for d in diffs),
                sum(diffs) / len(diffs),
            )
    return out


def render(pairs: Sequence[Pair]) -> str:
    def ms(value: float | None) -> str:
        return "-" if value is None else f"{value * 1000:.0f}"

    lines = [
        "# Repeat against original",
        "",
        "| system | sweep | run | original req/s | repeat req/s | diff % | original TTFT p99 ms "
        "| repeat TTFT p99 ms | original contended | repeat contended |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for p in pairs:
        diff = p.diff_percent
        lines.append(
            f"| {p.label} | {p.sweep} | {p.original.name} | {p.original.throughput_rps or 0:.2f} | "
            f"{p.repeat.throughput_rps or 0:.2f} | {'-' if diff is None else f'{diff:+.1f}'} | "
            f"{ms(p.original.ttft_p99_s)} | {ms(p.repeat.ttft_p99_s)} | "
            f"{p.original.contended} | {p.repeat.contended} |"
        )
    lines += ["", "Usable pairs (neither run contended):", ""]
    for label, (n, mean_abs, max_abs, mean_signed) in summarise(pairs).items():
        lines.append(
            f"- {label}: {n} pairs, mean |diff| {mean_abs:.1f}%, max |diff| {max_abs:.1f}%, "
            f"mean signed {mean_signed:+.1f}%"
        )
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--pair",
        nargs=3,
        action="append",
        metavar=("LABEL", "ORIGINAL_DIR", "REPEAT_DIR"),
        required=True,
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    pairs = [p for label, o, r in args.pair for p in pair_sweeps(label, Path(o), Path(r))]
    args.out.write_text(render(pairs), encoding="utf-8")
    print(f"wrote {args.out}: {len(pairs)} pairs, {sum(p.usable for p in pairs)} usable")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
