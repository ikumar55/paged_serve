"""Turn a batching sweep (``bench.sweep``'s manifest and result files) into a table and a plot.

The report lists every run, including failed, contended and overloaded ones, and for each arrival
rate it shows the best configuration *and* the worst, so the effect of the tuning knobs is
visible instead of one flattering setting being picked. Nothing is dropped from the table: a run
is only excluded from the best/worst comparison if it is invalid (and the reason is shown).

A run is *valid* if it finished, was not contended by other GPU users, the load generator kept
up, and at least 99% of requests succeeded. It is *sustained* if, in addition, the server kept up
with the offered load (throughput of at least 90% of the arrival rate).

Usage:
    python -m bench.sweep_report --dir results/hf-static --out results/hf-static/REPORT.md \\
        --plot results/hf-static/ttft.png
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bench.results_io import read_results

MIN_SUCCESS = 0.99
MIN_THROUGHPUT_FRACTION = 0.9


@dataclass
class Row:
    name: str
    batch_size: int
    window_ms: float
    rate: float
    status: str
    contended: bool
    num_ok: int = 0
    num_requests: int = 0
    throughput_rps: float | None = None
    tokens_per_s: float | None = None
    ttft_p50_s: float | None = None
    ttft_p99_s: float | None = None
    itl_p50_s: float | None = None
    e2e_p50_s: float | None = None
    e2e_p99_s: float | None = None
    generator_saturated: bool = False
    mean_batch_size: float | None = None
    wasted_slot_fraction: float | None = None

    @property
    def success_rate(self) -> float:
        return self.num_ok / self.num_requests if self.num_requests else 0.0

    @property
    def invalid_reason(self) -> str | None:
        if self.status != "ok":
            return self.status
        if self.contended:
            return "contended"
        if self.generator_saturated:
            return "load generator saturated"
        if self.success_rate < MIN_SUCCESS:
            return f"{self.success_rate:.0%} succeeded"
        return None

    @property
    def valid(self) -> bool:
        return self.invalid_reason is None

    @property
    def sustained(self) -> bool:
        return (
            self.valid
            and self.throughput_rps is not None
            and self.throughput_rps >= MIN_THROUGHPUT_FRACTION * self.rate
        )


def _p(dist: dict[str, float] | None, key: str) -> float | None:
    return None if not dist else dist.get(key)


def load_rows(directory: Path) -> list[Row]:
    rows: list[Row] = []
    for line in (directory / "manifest.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        row = Row(
            name=entry["name"],
            batch_size=int(entry["batch_size"]),
            window_ms=float(entry["window_ms"]),
            rate=float(entry["rate"]),
            status=entry["status"],
            contended=bool(entry["contended"]),
        )
        result = directory / entry["result_file"]
        if entry["status"] == "ok" and result.exists():
            meta, _, summary = read_results(result)
            if summary is not None:
                row.num_ok = summary["num_ok"]
                row.num_requests = summary["num_requests"]
                row.throughput_rps = summary["request_throughput_rps"]
                row.tokens_per_s = summary["output_tokens_per_s"]
                row.ttft_p50_s = _p(summary.get("ttft_s"), "p50")
                row.ttft_p99_s = _p(summary.get("ttft_s"), "p99")
                row.itl_p50_s = _p(summary.get("itl_s"), "p50")
                row.e2e_p50_s = _p(summary.get("e2e_s"), "p50")
                row.e2e_p99_s = _p(summary.get("e2e_s"), "p99")
                row.generator_saturated = bool(summary["generator_saturated"])
            after = ((meta.get("server_info") or {}).get("after") or {}).get("stats") or {}
            row.mean_batch_size = after.get("mean_batch_size")
            row.wasted_slot_fraction = after.get("wasted_slot_fraction")
        rows.append(row)
    return sorted(rows, key=lambda r: (r.rate, r.batch_size, r.window_ms))


def best_and_worst_per_rate(
    rows: Sequence[Row],
) -> dict[float, tuple[Row, Row, bool]]:
    """At each rate: the lowest and highest p99 time to first token, and whether any run kept up.

    Runs that kept up with the offered load (sustained) are compared when there are any. At a rate
    where every configuration is overloaded, the queue grows for the whole run and "best" only
    means least bad, so the valid runs are compared instead and the flag is False.
    """
    by_rate: dict[float, list[Row]] = defaultdict(list)
    for row in rows:
        if row.valid and row.ttft_p99_s is not None:
            by_rate[row.rate].append(row)
    result: dict[float, tuple[Row, Row, bool]] = {}
    for rate, group in sorted(by_rate.items()):
        sustained = [r for r in group if r.sustained]
        pool = sustained or group
        result[rate] = (
            min(pool, key=lambda r: r.ttft_p99_s or 0),
            max(pool, key=lambda r: r.ttft_p99_s or 0),
            bool(sustained),
        )
    return result


def _ms(value: float | None) -> str:
    return "-" if value is None else f"{value * 1000:.0f}"


def _s(value: float | None) -> str:
    return "-" if value is None else f"{value:.1f}"


def _num(value: float | None, fmt: str) -> str:
    return "-" if value is None else format(value, fmt)


def format_row(row: Row) -> str:
    note = row.invalid_reason or ("sustained" if row.sustained else "overloaded")
    return (
        f"| {row.batch_size} | {row.window_ms:g} | {row.rate:g} | {row.num_ok}/{row.num_requests} "
        f"| {_num(row.throughput_rps, '.2f')} | {_num(row.tokens_per_s, '.0f')} "
        f"| {_ms(row.ttft_p50_s)} | {_ms(row.ttft_p99_s)} | {_ms(row.itl_p50_s)} "
        f"| {_s(row.e2e_p50_s)} | {_s(row.e2e_p99_s)} "
        f"| {_num(row.mean_batch_size, '.1f')} | {_num(row.wasted_slot_fraction, '.0%')} | {note} |"
    )


TABLE_HEADER = (
    "| batch | window ms | rate req/s | ok | req/s | tok/s | TTFT p50 ms | TTFT p99 ms "
    "| ITL p50 ms | e2e p50 s | e2e p99 s | mean batch | wasted slots | status |\n"
    "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: "
    "| --- |"
)


def render_report(rows: Sequence[Row], heading: str = "Static batching sweep") -> str:
    lines = [f"# {heading}", ""]
    total = len(rows)
    valid = sum(r.valid for r in rows)
    lines.append(
        f"{total} runs, {valid} valid, {sum(r.sustained for r in rows)} sustained "
        f"(valid and throughput >= {MIN_THROUGHPUT_FRACTION:.0%} of the arrival rate)."
    )
    invalid = [r for r in rows if not r.valid]
    if invalid:
        lines.append("")
        lines.append(
            "Excluded from the best/worst comparison (still in the table): "
            + ", ".join(f"{r.name} ({r.invalid_reason})" for r in invalid)
        )
    lines += ["", "## Best and worst configuration at each arrival rate", ""]
    lines.append(
        "Ranked by p99 time to first token among the runs that kept up with the offered load. "
        "Where none did, every configuration is overloaded (its queue grows for the whole run), "
        "so the ranking is among valid runs and means only 'least bad'. The 90% cutoff for "
        "keeping up is a convention, so runs just either side of it are not meaningfully "
        "different."
    )
    lines += [
        "",
        "| rate req/s | ranked among | best (batch, window ms) | TTFT p99 ms "
        "| worst (batch, window ms) | TTFT p99 ms |",
        "| ---: | --- | --- | ---: | --- | ---: |",
    ]
    for rate, (best, worst, kept_up) in best_and_worst_per_rate(rows).items():
        pool = "sustained runs" if kept_up else "none sustained (least bad)"
        lines.append(
            f"| {rate:g} | {pool} | {best.batch_size}, {best.window_ms:g} | {_ms(best.ttft_p99_s)} "
            f"| {worst.batch_size}, {worst.window_ms:g} | {_ms(worst.ttft_p99_s)} |"
        )
    lines += ["", "## All runs", "", TABLE_HEADER, *map(format_row, rows), ""]
    lines.append(
        "Mean batch size and wasted slots come from the server's own counters and include the "
        "three unmeasured warm-up requests sent before each run (a fresh server per run)."
    )
    return "\n".join(lines) + "\n"


def plot_ttft(rows: Sequence[Row], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullFormatter, ScalarFormatter

    batches = sorted({r.batch_size for r in rows})
    windows = sorted({r.window_ms for r in rows})
    rates = sorted({r.rate for r in rows})
    fig, axes = plt.subplots(1, len(batches), figsize=(4 * len(batches), 3.6), sharey=True)
    axes_list: list[Any] = [axes] if len(batches) == 1 else list(axes)
    for axis, batch in zip(axes_list, batches, strict=True):
        for window in windows:
            group = sorted(
                (r for r in rows if r.batch_size == batch and r.window_ms == window and r.valid),
                key=lambda r: r.rate,
            )
            points = [(r.rate, r.ttft_p50_s * 1000) for r in group if r.ttft_p50_s is not None]
            if points:
                axis.plot(*zip(*points, strict=True), marker="o", label=f"window {window:g} ms")
        axis.set_title(f"max batch size {batch}")
        axis.set_xlabel("arrival rate (req/s)")
        axis.set_xscale("log")
        axis.set_yscale("log")
        # Label the rates that were actually run; matplotlib's default log ticks overlap here.
        axis.set_xticks(rates)
        axis.xaxis.set_major_formatter(ScalarFormatter())
        axis.xaxis.set_minor_formatter(NullFormatter())
        axis.grid(True, alpha=0.3)
    axes_list[0].set_ylabel("TTFT p50 (ms)")
    fig.suptitle(
        "Lines stop at the rate where a run stopped being valid (timeouts, or the load "
        "generator fell behind); the cliff is in the table",
        fontsize=9,
    )
    # The legend goes on the last panel that has any lines (a panel with no valid runs has none).
    for axis in reversed(axes_list):
        if axis.get_legend_handles_labels()[0]:
            axis.legend(fontsize=8)
            break
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0] if __doc__ else None)
    parser.add_argument("--dir", type=Path, required=True, help="sweep output directory")
    parser.add_argument(
        "--out", type=Path, help="write the markdown report here (never overwrites)"
    )
    parser.add_argument("--plot", type=Path, help="write the TTFT figure here")
    args = parser.parse_args(argv)
    rows = load_rows(args.dir)
    report = render_report(rows)
    if args.out is None:
        print(report)
    else:
        with args.out.open("x") as handle:
            handle.write(report)
        print(f"wrote {args.out}")
    if args.plot is not None:
        plot_ttft(rows, args.plot)
        print(f"wrote {args.plot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
