"""Capacity of the three systems from saturation sweeps, as paired per-seed ratios (D-019, D-021).

Delivered throughput at a rate the server can keep up with is set by the arrivals, not by the
server (D-019), so capacity has to be read at an offered rate above it. The sweeps of pass 5 offer
each system more than it can serve (the engine and Hugging Face 32 req/s, vLLM 64 req/s), four
workload seeds, the systems interleaved within each seed. This report puts the systems side by
side per seed and gives the ratios between them seed by seed, since the seeds are the same
workload draws for every system, and, for the engine, where the engine thread's time went
(``engine_thread_seconds`` in ``/info``, D-021).

Usage:
    python -m bench.saturation_report \\
        --engine "results/sat-engine-rtx5090-seed{seed}" \\
        --hf "results/sat-hf-static-rtx5090-seed{seed}" \\
        --vllm "results/sat-vllm-rtx5090-seed{seed}" --seeds 0,1,2,3 \\
        --out results/saturation-rtx5090.md
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from bench.results_io import read_results
from bench.sweep import parse_numbers
from bench.sweep_report import Row, load_rows

THREAD_KEYS = ("step", "route", "inbox", "idle")


def find_run(rows: Sequence[Row], batch_size: int, rate: float) -> Row | None:
    for row in rows:
        if row.batch_size == batch_size and row.rate == rate:
            return row
    return None


def paired_ratios(
    numerator: Mapping[int, float], denominator: Mapping[int, float]
) -> dict[int, float]:
    """``numerator / denominator`` for every seed present in both (and a nonzero denominator)."""
    return {
        seed: numerator[seed] / denominator[seed]
        for seed in sorted(numerator.keys() & denominator.keys())
        if denominator[seed]
    }


def ratio_summary(ratios: Mapping[int, float]) -> tuple[float, float, float] | None:
    """(mean, minimum, maximum) of the per-seed ratios, or ``None`` when there are none."""
    if not ratios:
        return None
    values = list(ratios.values())
    return sum(values) / len(values), min(values), max(values)


def thread_accounting(
    before: Mapping[str, Any], after: Mapping[str, Any]
) -> dict[str, float] | None:
    """What the engine thread did during one run, from two ``/info`` stats snapshots.

    Returns the seconds in each activity, each one's share of the total, the milliseconds per
    scheduler step and per generated token, and the mean decode batch; ``None`` if the server did
    not report the thread timers (an older engine).
    """
    start, end = before.get("engine_thread_seconds"), after.get("engine_thread_seconds")
    if not start or not end:
        return None
    seconds = {key: end[key] - start[key] for key in THREAD_KEYS}
    total = sum(seconds.values())
    steps = after["decode_steps"] - before.get("decode_steps", 0)
    tokens = after["tokens_generated"] - before.get("tokens_generated", 0)
    out: dict[str, float] = {f"{key}_s": value for key, value in seconds.items()}
    out.update(
        {f"{key}_share": (value / total if total else 0.0) for key, value in seconds.items()}
    )
    out["ms_per_step"] = 1000 * seconds["step"] / steps if steps else float("nan")
    out["ms_per_token"] = 1000 * seconds["step"] / tokens if tokens else float("nan")
    out["mean_decode_batch"] = float(after.get("mean_decode_batch", float("nan")))
    return out


def run_stats(directory: Path, row: Row) -> dict[str, Any] | None:
    """The server stats snapshots of one run, or ``None`` if the run has no result file."""
    result = directory / f"{row.name}.jsonl"
    if not result.exists():
        return None
    meta, _, _ = read_results(result)
    info = meta.get("server_info") or {}
    before, after = (info.get("before") or {}).get("stats"), (info.get("after") or {}).get("stats")
    return {"before": before or {}, "after": after or {}}


def load_system(
    pattern: str, seeds: Sequence[int], batch_size: int, rate: float
) -> dict[int, tuple[Row, Path]]:
    found: dict[int, tuple[Row, Path]] = {}
    for seed in seeds:
        directory = Path(pattern.format(seed=seed))
        if not (directory / "manifest.jsonl").exists():
            continue
        row = find_run(load_rows(directory), batch_size, rate)
        if row is not None:
            found[seed] = (row, directory)
    return found


def requests_per_s(row: Row) -> float:
    return row.throughput_rps or 0.0


def tokens_per_s(row: Row) -> float:
    return row.tokens_per_s or 0.0


def ms(value: float | None) -> str:
    return "-" if value is None else f"{value * 1000:.0f}"


def render(
    systems: Mapping[str, dict[int, tuple[Row, Path]]],
    pairs: Sequence[tuple[str, str, str]],
    seeds: Sequence[int],
) -> str:
    lines = [
        "# Saturation (capacity) report",
        "",
        "Offered rates are above each system's capacity, so delivered throughput is the capacity "
        "(not capped by the arrivals). Seeds are the same workload draws for every system.",
        "",
        "## Per run",
        "",
        "| system | seed | req/s | tokens/s | TTFT p50 ms | ITL p50 ms | ok / sent | flags |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for name, runs in systems.items():
        for seed in seeds:
            if seed not in runs:
                continue
            row = runs[seed][0]
            flags = [
                f
                for f, on in (
                    ("contended", row.contended),
                    ("generator-saturated", row.generator_saturated),
                )
                if on
            ]
            lines.append(
                f"| {name} | {seed} | {requests_per_s(row):.2f} | {tokens_per_s(row):.0f} | "
                f"{ms(row.ttft_p50_s)} | {ms(row.itl_p50_s)} | {row.num_ok}/{row.num_requests} | "
                f"{', '.join(flags) or '-'} |"
            )
    lines += [
        "",
        "## Paired per-seed ratios",
        "",
        "| ratio | metric | per seed | mean (min to max) |",
        "| --- | --- | --- | --- |",
    ]
    for label, top, bottom in pairs:
        for metric, getter in (("req/s", requests_per_s), ("tokens/s", tokens_per_s)):
            numerator = {s: getter(r[0]) for s, r in systems[top].items()}
            denominator = {s: getter(r[0]) for s, r in systems[bottom].items()}
            ratios = paired_ratios(numerator, denominator)
            summary = ratio_summary(ratios)
            if summary is None:
                continue
            per_seed = ", ".join(f"{ratios[s]:.2f}" for s in sorted(ratios))
            lines.append(
                f"| {label} | {metric} | {per_seed} | "
                f"{summary[0]:.2f} ({summary[1]:.2f} to {summary[2]:.2f}) |"
            )
    lines += ["", "## Engine thread time (D-021)", ""]
    header_written = False
    for name, runs in systems.items():
        for seed in seeds:
            if seed not in runs:
                continue
            row, directory = runs[seed]
            stats = run_stats(directory, row)
            accounting = thread_accounting(stats["before"], stats["after"]) if stats else None
            if accounting is None:
                continue
            if not header_written:
                lines += [
                    "| run | seed | step share | route share | idle share | ms per step "
                    "| ms per token | mean decode batch |",
                    "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
                ]
                header_written = True
            lines.append(
                f"| {name} | {seed} | {accounting['step_share']:.0%} | "
                f"{accounting['route_share']:.1%} | {accounting['idle_share']:.1%} | "
                f"{accounting['ms_per_step']:.1f} | {accounting['ms_per_token']:.3f} | "
                f"{accounting['mean_decode_batch']:.1f} |"
            )
    if not header_written:
        lines.append("The engine runs carry no thread timers.")
    lines.append("")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0] if __doc__ else None)
    parser.add_argument("--engine", required=True, help="directory pattern with {seed}")
    parser.add_argument("--hf", required=True)
    parser.add_argument("--vllm", required=True)
    parser.add_argument("--seeds", default="0,1,2,3")
    parser.add_argument("--engine-rate", type=float, default=32.0)
    parser.add_argument("--vllm-rate", type=float, default=64.0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    seeds = parse_numbers(args.seeds, int)
    systems = {
        "engine b32": load_system(args.engine, seeds, 32, args.engine_rate),
        "engine b64": load_system(args.engine, seeds, 64, args.engine_rate),
        "hf-static b64": load_system(args.hf, seeds, 64, args.engine_rate),
        "vllm 32 seqs": load_system(args.vllm, seeds, 32, args.vllm_rate),
    }
    pairs = [
        ("engine b64 / hf b64", "engine b64", "hf-static b64"),
        ("engine b32 / hf b64", "engine b32", "hf-static b64"),
        ("vllm / engine b64", "vllm 32 seqs", "engine b64"),
        ("vllm / engine b32", "vllm 32 seqs", "engine b32"),
    ]
    if args.out.exists():
        raise SystemExit(f"{args.out} exists; results are never overwritten")
    args.out.write_text(render(systems, pairs, seeds), encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
