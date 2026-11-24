"""Where does the saturated engine thread's time go? (D-028)

Reads the start and end ``/info`` records that the load generator stores with every run of a
saturation sweep (written by ``bench.sweep``) and reports, per seed, how the engine thread's
scheduler-step time splits into making room for the decode, the batched decode and admission (the
prefills), what one admitted prompt costs, how many prompts are admitted per decode step, and the
shares of routing and the inbox. D-028 fixes the predicted ranges in advance; the table prints them
next to the measured values.

Usage:
    python -m bench.phase_report --engine "results/d028-sat-rtx5090-seed{seed}" --seeds 0,1,2,3 \\
        --out results/phase-report-rtx5090.md
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from bench.saturation_report import find_run, run_stats
from bench.sweep import parse_numbers
from bench.sweep_report import load_rows

# D-028's predictions: (name, low, high) on the quantity computed below.
PREDICTIONS = (
    ("admission share of step time", 0.20, 0.35),
    ("decode share of step time", 0.50, 0.70),
    ("room and other share of step time", 0.0, 0.05),
    ("route plus inbox share of thread time", 0.0, 0.05),
    ("ms per admitted prompt", 5.0, 12.0),
)


def phase_accounting(
    before: Mapping[str, Any], after: Mapping[str, Any]
) -> dict[str, float] | None:
    """The split of one run's engine-thread time, from two ``/info`` stats snapshots.

    ``None`` if the server did not report the phase timers (an engine from before D-028).
    """
    needed = ("phase_room_s", "phase_decode_s", "phase_admit_s", "prefills", "prefill_tokens")
    if any(key not in after for key in needed) or "engine_thread_seconds" not in after:
        return None

    def delta(key: str) -> float:
        return float(after[key]) - float(before.get(key, 0.0))

    room, decode, admit = delta("phase_room_s"), delta("phase_decode_s"), delta("phase_admit_s")
    thread = {
        key: float(after["engine_thread_seconds"][key])
        - float(before.get("engine_thread_seconds", {}).get(key, 0.0))
        for key in after["engine_thread_seconds"]
    }
    step = thread["step"]
    phases = room + decode + admit
    prefills, decode_steps = delta("prefills"), delta("decode_steps")
    total_thread = sum(thread.values())
    return {
        "step_s": step,
        "room_s": room,
        "decode_s": decode,
        "admit_s": admit,
        "admission_share": admit / step if step else float("nan"),
        "decode_share": decode / step if step else float("nan"),
        "other_share": (step - phases) / step if step else float("nan"),
        "route_inbox_share": (thread["route"] + thread["inbox"]) / total_thread
        if total_thread
        else float("nan"),
        "idle_share": thread["idle"] / total_thread if total_thread else float("nan"),
        "ms_per_prefill": 1000 * admit / prefills if prefills else float("nan"),
        "prefills_per_decode_step": prefills / decode_steps if decode_steps else float("nan"),
        "prefill_tokens_per_prefill": delta("prefill_tokens") / prefills
        if prefills
        else float("nan"),
        "ms_per_decode_step": 1000 * decode / decode_steps if decode_steps else float("nan"),
    }


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def check_predictions(rows: Mapping[int, Mapping[str, float]]) -> list[tuple[str, float, str]]:
    """For each D-028 prediction: the mean over seeds and whether it is inside the stated range."""
    keys = {
        "admission share of step time": "admission_share",
        "decode share of step time": "decode_share",
        "room and other share of step time": "other_share",
        "route plus inbox share of thread time": "route_inbox_share",
        "ms per admitted prompt": "ms_per_prefill",
    }
    out = []
    for name, low, high in PREDICTIONS:
        value = mean([row[keys[name]] for row in rows.values()])
        verdict = "inside" if low <= value <= high else "OUTSIDE"
        out.append((name, value, f"{low:g} to {high:g}: {verdict}"))
    return out


def render(rows: Mapping[int, Mapping[str, float]], throughput: Mapping[int, float]) -> str:
    lines = [
        "# Engine-thread phase report (D-028)",
        "",
        "Shares of the scheduler-step time unless stated. `other` is the step time not inside the "
        "three timed phases.",
        "",
        "| seed | req/s | decode | admission | other | ms per decode step | ms per prompt "
        "| prompts per decode step | tokens per prompt | route+inbox of thread | idle of thread |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for seed, row in sorted(rows.items()):
        lines.append(
            f"| {seed} | {throughput.get(seed, float('nan')):.2f} | {row['decode_share']:.0%} | "
            f"{row['admission_share']:.0%} | {row['other_share']:.1%} | "
            f"{row['ms_per_decode_step']:.2f} | {row['ms_per_prefill']:.1f} | "
            f"{row['prefills_per_decode_step']:.2f} | {row['prefill_tokens_per_prefill']:.0f} | "
            f"{row['route_inbox_share']:.1%} | {row['idle_share']:.1%} |"
        )
    lines += ["", "## Against the predictions fixed in D-028", "", "| quantity | mean | range |"]
    lines.append("| --- | ---: | --- |")
    for name, value, verdict in check_predictions(rows):
        lines.append(f"| {name} | {value:.3f} | {verdict} |")
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0] if __doc__ else None)
    parser.add_argument("--engine", required=True, help='sweep directory pattern with "{seed}"')
    parser.add_argument("--seeds", default="0,1,2,3")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--rate", type=float, default=32.0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise SystemExit(f"{args.out} exists; result files are never overwritten")
    rows: dict[int, dict[str, float]] = {}
    throughput: dict[int, float] = {}
    for seed in parse_numbers(args.seeds, int):
        directory = Path(args.engine.format(seed=seed))
        if not (directory / "manifest.jsonl").exists():
            continue
        row = find_run(load_rows(directory), args.batch_size, args.rate)
        stats = run_stats(directory, row) if row is not None else None
        accounting = phase_accounting(stats["before"], stats["after"]) if stats else None
        if row is None or accounting is None:
            raise SystemExit(f"seed {seed}: no run with phase timers in {directory}")
        rows[seed] = accounting
        throughput[seed] = row.throughput_rps or 0.0
    if not rows:
        raise SystemExit("no runs found")
    args.out.write_text(render(rows, throughput), encoding="utf-8")
    print(args.out.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
