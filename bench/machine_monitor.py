"""Record what else is happening on a shared benchmark machine while the benchmarks run.

The sweep's own contention check looks at GPU utilisation for a few seconds before a run starts.
It cannot see another user's CPU work (this engine is Python and CPU bound, so that matters), a
foreign process that starts mid-run, or the GPU slowing itself down for heat or power. This
sampler runs beside the sweep and writes one JSON line every few seconds:

* CPU time, in cores (1.0 = one core fully busy), split into this user's processes, other
  ordinary users' processes and root's (system daemons);
* the GPU: utilisation, memory, SM clock against its maximum, power, temperature, the driver's
  throttle-reason bits and how many compute processes belong to someone else;
* the machine: one-minute load average and available memory.

``bench.machine_audit`` joins these lines to each run's start and finish times. A run is not
discarded for any of this; it is flagged so a reader can decide.

Usage (on the benchmark machine, started once before a pass and stopped after it):
    python -m bench.machine_monitor --out results/machine-monitor-<label>.jsonl --interval 2

Linux only (it reads ``/proc``). The output file is never overwritten.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ORDINARY_USER_UID = 1000  # uids from here up are people; below are system accounts (root, daemons)

GPU_QUERY = (
    "utilization.gpu,memory.used,clocks.sm,clocks.max.sm,power.draw,temperature.gpu,"
    "clocks_throttle_reasons.active"
)


def parse_proc_stat(text: str) -> tuple[int, int]:
    """(busy, total) jiffies over all cores from the first line of ``/proc/stat``."""
    first = text.splitlines()[0].split()
    if first[0] != "cpu":
        raise ValueError(f"not the aggregate cpu line: {first[0]!r}")
    values = [int(v) for v in first[1:9]]  # user nice system idle iowait irq softirq steal
    idle = values[3] + values[4]
    total = sum(values)
    return total - idle, total


def parse_pid_stat(text: str) -> int:
    """utime + stime (jiffies) from the text of ``/proc/<pid>/stat``.

    The command name sits in parentheses and may contain spaces and parentheses, so fields are
    counted from the last closing parenthesis.
    """
    rest = text[text.rindex(")") + 1 :].split()
    return int(rest[11]) + int(rest[12])  # fields 14 and 15 of the file


def read_process_jiffies(proc_root: Path = Path("/proc")) -> dict[int, tuple[int, int]]:
    """pid -> (uid, utime + stime) for every readable process."""
    found: dict[int, tuple[int, int]] = {}
    for entry in proc_root.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            uid = entry.stat().st_uid
            jiffies = parse_pid_stat((entry / "stat").read_text())
        except (OSError, ValueError, IndexError):
            continue  # the process ended while we were reading, or is not ours to read
        found[int(entry.name)] = (uid, jiffies)
    return found


def cpu_cores_by_group(
    before: Mapping[int, tuple[int, int]],
    after: Mapping[int, tuple[int, int]],
    *,
    own_uid: int,
    seconds: float,
    hz: int,
) -> dict[str, float]:
    """CPU use between two snapshots, in cores, for our user, other users and root.

    Only processes present in both snapshots count, so a process that started and ended between
    two samples is missed (a small undercount, the same for every run).
    """
    cores = {"ours": 0.0, "other_users": 0.0, "root": 0.0}
    for pid, (uid, jiffies_after) in after.items():
        if pid not in before:
            continue
        used = max(0, jiffies_after - before[pid][1]) / hz / seconds
        if uid == own_uid:
            cores["ours"] += used
        elif uid == 0:
            cores["root"] += used
        elif uid >= ORDINARY_USER_UID:
            cores["other_users"] += used
        # other system accounts (daemons) are neither ours nor a person's work: ignored
    return cores


def parse_gpu_line(line: str) -> dict[str, Any]:
    """One CSV line of ``GPU_QUERY`` (``--format=csv,noheader,nounits``)."""
    fields = [f.strip() for f in line.split(",")]
    if len(fields) != 7:
        raise ValueError(f"expected 7 fields, got {len(fields)}: {line!r}")

    def number(text: str) -> float | None:
        try:
            return float(text)
        except ValueError:  # "[N/A]" and similar
            return None

    return {
        "gpu_util_percent": number(fields[0]),
        "gpu_mem_used_mib": number(fields[1]),
        "gpu_sm_clock_mhz": number(fields[2]),
        "gpu_sm_clock_max_mhz": number(fields[3]),
        "gpu_power_w": number(fields[4]),
        "gpu_temp_c": number(fields[5]),
        "gpu_throttle_bits": int(fields[6], 16) if fields[6].startswith("0x") else None,
    }


def parse_compute_apps(text: str) -> list[int]:
    """Process ids from ``nvidia-smi --query-compute-apps=pid --format=csv,noheader``."""
    return [int(line.split(",")[0]) for line in text.splitlines() if line.strip()[:1].isdigit()]


def gpu_apps_of_others(pids: list[int], own_uid: int, uid_of: Callable[[int], int | None]) -> int:
    """How many GPU compute processes are not ours (one we cannot look up counts as not ours)."""
    return sum(1 for pid in pids if uid_of(pid) != own_uid)


def parse_loadavg(text: str) -> float:
    return float(text.split()[0])


def parse_mem_available_mib(text: str) -> float:
    for line in text.splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 1024
    raise ValueError("no MemAvailable line")


def _run(command: list[str]) -> str:
    return subprocess.run(command, capture_output=True, text=True, timeout=10, check=True).stdout


def _uid_of_pid(pid: int) -> int | None:
    try:
        return Path(f"/proc/{pid}").stat().st_uid
    except OSError:
        return None


def sample_gpu(own_uid: int) -> dict[str, Any]:
    """GPU fields plus the number of compute processes that are not ours; empty on failure."""
    try:
        first_line = _run(
            ["nvidia-smi", f"--query-gpu={GPU_QUERY}", "--format=csv,noheader,nounits"]
        ).splitlines()[0]
        apps = _run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"])
        out = parse_gpu_line(first_line)
        out["gpu_apps_other"] = gpu_apps_of_others(parse_compute_apps(apps), own_uid, _uid_of_pid)
        return out
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return {"gpu_error": True}


def monitor(
    out_path: Path, interval: float, duration: float | None, stop: Callable[[], bool]
) -> int:
    """Write samples until ``duration`` seconds pass or ``stop()`` is true; returns the count."""
    if sys.platform != "linux":
        raise RuntimeError("the machine monitor reads /proc and runs on Linux only")
    own_uid = os.getuid()
    hz = os.sysconf("SC_CLK_TCK")
    count = 0
    with out_path.open("x", encoding="utf-8") as handle:  # "x": refuse to overwrite
        previous = read_process_jiffies()
        previous_time = time.monotonic()
        previous_stat = parse_proc_stat(Path("/proc/stat").read_text())
        started = previous_time
        while not stop() and (duration is None or time.monotonic() - started < duration):
            time.sleep(interval)
            now = time.monotonic()
            current = read_process_jiffies()
            current_stat = parse_proc_stat(Path("/proc/stat").read_text())
            seconds = now - previous_time
            cores = cpu_cores_by_group(previous, current, own_uid=own_uid, seconds=seconds, hz=hz)
            busy_delta = current_stat[0] - previous_stat[0]
            sample: dict[str, Any] = {
                "t_utc": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
                "interval_s": round(seconds, 3),
                "cpu_cores_total": round(busy_delta / hz / seconds, 3),
                "cpu_cores_ours": round(cores["ours"], 3),
                "cpu_cores_other_users": round(cores["other_users"], 3),
                "cpu_cores_root": round(cores["root"], 3),
                "load1": parse_loadavg(Path("/proc/loadavg").read_text()),
                "mem_available_mib": round(
                    parse_mem_available_mib(Path("/proc/meminfo").read_text()), 1
                ),
                **sample_gpu(own_uid),
            }
            handle.write(json.dumps(sample, sort_keys=True) + "\n")
            handle.flush()
            count += 1
            previous, previous_time, previous_stat = current, now, current_stat
    return count


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--out", type=Path, required=True, help="JSONL file to create (not overwritten)"
    )
    parser.add_argument("--interval", type=float, default=2.0, help="seconds between samples")
    parser.add_argument("--duration", type=float, default=None, help="stop after this many seconds")
    args = parser.parse_args(argv)
    stopping = False

    def request_stop(_signum: int, _frame: object) -> None:
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    count = monitor(args.out, args.interval, args.duration, lambda: stopping)
    print(f"wrote {count} samples to {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
