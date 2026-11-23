"""Was the machine quiet while each run was measured?

Joins the samples written by ``bench.machine_monitor`` to the start and finish times in each
sweep manifest and reports, per run, what other users were doing and whether the GPU slowed
itself down. Nothing is discarded: a run is flagged, and a reader decides.

Flags (thresholds are judgments, set here so they are visible; change them with the evidence):

* ``no-coverage``: fewer than ``MIN_SAMPLES`` monitor samples fall inside the run, so nothing can
  be said about it. A run the monitor did not watch is **not** clean.
* ``other-cpu``: other ordinary users' processes used more than ``OTHER_CPU_MEAN_CORES`` cores on
  average or ``OTHER_CPU_PEAK_CORES`` at a sample.
* ``other-gpu-process``: a compute process that is not ours was on the GPU at some sample.
* ``slowdown``: the driver reported a hardware slowdown, software thermal, hardware thermal or
  power-brake throttle reason at some sample. The ordinary software power cap (bit 0x4) is
  reported in its own column and is not a flag: a fast GPU under load reaches it by design.

Usage:
    python -m bench.machine_audit --monitor results/machine-monitor-p3.jsonl \\
        --sweep results/prefix-engine-on-rtx5090-seed0 results/prefix-vllm-on-rtx5090-seed0 \\
        --out results/machine-audit-p3.md
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

MIN_SAMPLES = 3
OTHER_CPU_MEAN_CORES = 0.5
OTHER_CPU_PEAK_CORES = 2.0
SLOWDOWN_MASK = 0x8 | 0x20 | 0x40 | 0x80
POWER_CAP_BIT = 0x4


@dataclass(frozen=True)
class RunAudit:
    sweep: str
    name: str
    samples: int
    other_cpu_mean: float
    other_cpu_peak: float
    gpu_apps_other_peak: int
    power_capped_samples: int
    slowdown_samples: int
    temp_peak_c: float | None
    clock_ratio_min: float | None  # SM clock / max clock, only at samples where the GPU was busy
    flags: tuple[str, ...]


def parse_time(text: str) -> datetime:
    return datetime.fromisoformat(text)


def load_samples(paths: Iterable[Path]) -> list[dict[str, Any]]:
    """Monitor samples from one or more files, each with a parsed ``t`` and sorted by it."""
    samples: list[dict[str, Any]] = []
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                row["t"] = parse_time(row["t_utc"])
                samples.append(row)
    samples.sort(key=lambda row: row["t"])
    return samples


def load_manifest(sweep_dir: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in (sweep_dir / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def audit_run(sweep: str, entry: dict[str, Any], samples: Sequence[dict[str, Any]]) -> RunAudit:
    start = parse_time(entry["started_at_utc"])
    end = parse_time(entry["finished_at_utc"])
    inside = [row for row in samples if start <= row["t"] <= end]
    others = [row.get("cpu_cores_other_users", 0.0) for row in inside]
    bits = [row["gpu_throttle_bits"] for row in inside if row.get("gpu_throttle_bits") is not None]
    busy = [
        row
        for row in inside
        if (row.get("gpu_util_percent") or 0) > 50 and row.get("gpu_sm_clock_max_mhz")
    ]
    temps = [row["gpu_temp_c"] for row in inside if row.get("gpu_temp_c") is not None]
    mean = sum(others) / len(others) if others else 0.0
    peak = max(others, default=0.0)
    apps = max((row.get("gpu_apps_other", 0) for row in inside), default=0)
    slowdown = sum(1 for b in bits if b & SLOWDOWN_MASK)
    flags: list[str] = []
    if len(inside) < MIN_SAMPLES:
        flags.append("no-coverage")
    if mean > OTHER_CPU_MEAN_CORES or peak > OTHER_CPU_PEAK_CORES:
        flags.append("other-cpu")
    if apps > 0:
        flags.append("other-gpu-process")
    if slowdown:
        flags.append("slowdown")
    return RunAudit(
        sweep=sweep,
        name=entry["name"],
        samples=len(inside),
        other_cpu_mean=mean,
        other_cpu_peak=peak,
        gpu_apps_other_peak=apps,
        power_capped_samples=sum(1 for b in bits if b & POWER_CAP_BIT),
        slowdown_samples=slowdown,
        temp_peak_c=max(temps, default=None),
        clock_ratio_min=min(
            (row["gpu_sm_clock_mhz"] / row["gpu_sm_clock_max_mhz"] for row in busy), default=None
        ),
        flags=tuple(flags),
    )


def audit_sweeps(sweep_dirs: Sequence[Path], samples: Sequence[dict[str, Any]]) -> list[RunAudit]:
    return [
        audit_run(sweep_dir.name, entry, samples)
        for sweep_dir in sweep_dirs
        for entry in load_manifest(sweep_dir)
    ]


def render(audits: Sequence[RunAudit]) -> str:
    def fmt(value: float | None, spec: str) -> str:
        return "-" if value is None else format(value, spec)

    lines = [
        "# Machine audit",
        "",
        "Other users' CPU is in cores (1.0 = one core busy). Samples are those the monitor took "
        "between the run's start and finish (server start-up included). A run with no coverage is "
        "not known to be clean.",
        "",
        "| sweep | run | samples | other CPU mean | other CPU peak | other GPU procs | "
        "power-capped samples | slowdown samples | peak temp C | min clock ratio | flags |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for a in audits:
        lines.append(
            f"| {a.sweep} | {a.name} | {a.samples} | {a.other_cpu_mean:.2f} | "
            f"{a.other_cpu_peak:.2f} | {a.gpu_apps_other_peak} | {a.power_capped_samples} | "
            f"{a.slowdown_samples} | {fmt(a.temp_peak_c, '.0f')} | "
            f"{fmt(a.clock_ratio_min, '.2f')} | "
            f"{', '.join(a.flags) or 'clean'} |"
        )
    flagged = [a for a in audits if a.flags]
    lines += [
        "",
        f"{len(audits)} runs; {len(audits) - len(flagged)} clean; {len(flagged)} flagged"
        + (": " + ", ".join(f"{a.sweep}/{a.name}" for a in flagged) if flagged else "."),
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--monitor", type=Path, nargs="+", required=True)
    parser.add_argument("--sweep", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    audits = audit_sweeps(args.sweep, load_samples(args.monitor))
    args.out.write_text(render(audits), encoding="utf-8")
    flagged = sum(1 for a in audits if a.flags)
    print(f"wrote {args.out}: {len(audits)} runs, {flagged} flagged")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
