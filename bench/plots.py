"""Plots for single runs and for arrival-rate sweeps."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from matplotlib.axes import Axes
from matplotlib.axis import Axis
from matplotlib.figure import Figure
from matplotlib.ticker import NullFormatter, ScalarFormatter

from bench.client import RequestResult
from bench.metrics import inter_token_latencies


def _log_if_wide(ax: Axes, values: Sequence[float], axis: str) -> None:
    """Switch an axis to log scale only when the data spans more than a decade.

    Narrow ranges stay linear, and log axes get plain tick labels instead of scientific notation.
    """
    positive = [v for v in values if v > 0]
    if len(positive) < 2 or max(positive) / min(positive) < 10:
        return
    target: Axis
    if axis == "x":
        ax.set_xscale("log")
        target = ax.xaxis
    else:
        ax.set_yscale("log")
        target = ax.yaxis
    target.set_major_formatter(ScalarFormatter())
    target.set_minor_formatter(NullFormatter())


def _cdf(ax: Axes, values: Sequence[float], xlabel: str, *, log_x: bool = False) -> None:
    ax.set_xlabel(xlabel)
    ax.set_ylabel("fraction of requests")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    if not values:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        return
    xs: npt.NDArray[np.float64] = np.sort(np.asarray(values, dtype=float))
    ys = np.arange(1, xs.size + 1) / xs.size
    ax.step(xs, ys, where="post")
    if log_x:
        _log_if_wide(ax, values, "x")


def plot_run(meta: dict[str, Any], results: Sequence[RequestResult], out_path: Path) -> Path:
    """Four panels: TTFT, inter-token latency and end-to-end latency CDFs, TTFT over time."""
    ok = [r for r in results if r.status == "ok"]
    failed = [r for r in results if r.status != "ok"]

    figure = Figure(figsize=(11, 8), layout="constrained")
    ttft_ax, itl_ax, e2e_ax, time_ax = figure.subplots(2, 2).flatten()

    _cdf(ttft_ax, [r.ttft_s * 1000 for r in ok if r.ttft_s is not None], "TTFT (ms)")
    ttft_ax.set_title("Time to first token")
    _cdf(
        itl_ax,
        [gap * 1000 for r in ok for gap in inter_token_latencies(r)],
        "inter-token latency (ms)",
        log_x=True,
    )
    itl_ax.set_title("Inter-token latency (all gaps)")
    _cdf(e2e_ax, [r.e2e_s for r in ok if r.e2e_s is not None], "end-to-end latency (s)")
    e2e_ax.set_title("End-to-end latency")

    time_ax.set_title("TTFT over the run")
    time_ax.set_xlabel("scheduled arrival (s)")
    time_ax.set_ylabel("TTFT (ms)")
    time_ax.grid(alpha=0.3)
    time_ax.scatter(
        [r.scheduled_s for r in ok if r.ttft_s is not None],
        [r.ttft_s * 1000 for r in ok if r.ttft_s is not None],
        s=8,
        alpha=0.6,
        label="ok",
    )
    if failed:
        time_ax.scatter(
            [r.scheduled_s for r in failed],
            [0.0] * len(failed),
            marker="x",
            color="tab:red",
            label=f"failed ({len(failed)})",
        )
    time_ax.legend()

    rate = meta.get("workload", {}).get("rate")
    title = f"{meta.get('label') or 'run'}: {len(ok)}/{len(results)} ok"
    if rate is not None:
        title += f" at {rate:g} req/s offered"
    figure.suptitle(title)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out_path, dpi=130)
    return out_path


def plot_sweep(runs: Sequence[tuple[dict[str, Any], dict[str, Any]]], out_path: Path) -> Path:
    """Latency, throughput and failures against offered rate, one line per run label.

    ``runs`` holds ``(meta, summary)`` pairs.
    """
    grouped: dict[str, list[tuple[float, dict[str, Any]]]] = defaultdict(list)
    for meta, summary in runs:
        rate = float(meta["workload"]["rate"])
        grouped[str(meta.get("label") or "run")].append((rate, summary))

    figure = Figure(figsize=(11, 8), layout="constrained")
    p99_ax, p50_ax, tput_ax, fail_ax = figure.subplots(2, 2).flatten()
    for ax, title, ylabel in (
        (p99_ax, "p99 time to first token", "ms"),
        (p50_ax, "p50 time to first token", "ms"),
        (tput_ax, "Output throughput", "tokens / s"),
        (fail_ax, "Failed requests", "% of requests"),
    ):
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("offered rate (req/s)")
        ax.grid(alpha=0.3)

    plotted: dict[str, list[float]] = {"p99": [], "p50": []}
    for label, points in sorted(grouped.items()):
        points.sort(key=lambda p: p[0])
        rates = [rate for rate, _ in points]

        def series(
            getter: Any, rates: list[float] = rates, points: list[Any] = points
        ) -> tuple[list[float], list[float]]:
            pairs = [(r, getter(s)) for r, (_, s) in zip(rates, points, strict=True)]
            kept = [(r, v) for r, v in pairs if v is not None]
            return [r for r, _ in kept], [v for _, v in kept]

        for ax, stat in ((p99_ax, "p99"), (p50_ax, "p50")):
            xs, ys = series(
                lambda s, stat=stat: s["ttft_s"][stat] * 1000 if s.get("ttft_s") else None
            )
            ax.plot(xs, ys, marker="o", label=label)
            plotted[stat].extend(ys)
        xs, ys = series(lambda s: s.get("output_tokens_per_s"))
        tput_ax.plot(xs, ys, marker="o", label=label)
        xs, ys = series(
            lambda s: 100 * s["num_failed"] / s["num_requests"] if s["num_requests"] else None
        )
        fail_ax.plot(xs, ys, marker="o", label=label)

    _log_if_wide(p99_ax, plotted["p99"] + plotted["p50"], "y")  # shared range keeps panels alike
    _log_if_wide(p50_ax, plotted["p99"] + plotted["p50"], "y")
    tput_ax.set_ylim(bottom=0)
    fail_ax.set_ylim(0, max(1.0, fail_ax.get_ylim()[1]))
    p99_ax.legend()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out_path, dpi=130)
    return out_path
