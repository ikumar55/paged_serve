"""The machine audit: joining monitor samples to runs and flagging the ones that were not clean."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from bench import machine_audit as ma


def sample(second: int, **fields: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "t_utc": f"2026-11-23T10:00:{second:02d}+00:00",
        "cpu_cores_other_users": 0.0,
        "gpu_apps_other": 0,
        "gpu_throttle_bits": 0,
        "gpu_util_percent": 90.0,
        "gpu_sm_clock_mhz": 3000.0,
        "gpu_sm_clock_max_mhz": 3100.0,
        "gpu_temp_c": 60.0,
        **fields,
    }
    row["t"] = ma.parse_time(row["t_utc"])
    return row


ENTRY = {
    "name": "b32-w0-r4",
    "started_at_utc": "2026-11-23T10:00:10+00:00",
    "finished_at_utc": "2026-11-23T10:00:20+00:00",
}


def quiet(seconds: range) -> list[dict[str, Any]]:
    return [sample(s) for s in seconds]


def test_a_quiet_run_is_clean_and_only_samples_inside_its_window_count() -> None:
    noisy_outside = [sample(5, cpu_cores_other_users=9.0), sample(30, gpu_apps_other=2)]
    audit = ma.audit_run("s", ENTRY, noisy_outside + quiet(range(10, 21, 2)))
    assert audit.flags == ()
    assert audit.samples == 6  # seconds 10, 12, ..., 20: both ends are inside the window
    assert audit.other_cpu_peak == 0.0


def test_no_samples_means_no_coverage_not_clean() -> None:
    assert ma.audit_run("s", ENTRY, []).flags == ("no-coverage",)
    assert ma.audit_run("s", ENTRY, quiet(range(10, 12))).flags == ("no-coverage",)  # two samples
    assert ma.audit_run("s", ENTRY, quiet(range(10, 13))).flags == ()  # exactly three


def test_other_cpu_is_flagged_on_the_mean_or_on_the_peak() -> None:
    sustained = [sample(s, cpu_cores_other_users=0.6) for s in range(10, 20)]
    assert ma.audit_run("s", ENTRY, sustained).flags == ("other-cpu",)
    at_the_mean_limit = [sample(s, cpu_cores_other_users=0.5) for s in range(10, 20)]
    assert ma.audit_run("s", ENTRY, at_the_mean_limit).flags == ()
    spike = [*quiet(range(10, 20)), sample(15, cpu_cores_other_users=2.5)]
    assert ma.audit_run("s", ENTRY, spike).flags == ("other-cpu",)
    at_the_peak_limit = [*quiet(range(10, 20)), sample(15, cpu_cores_other_users=2.0)]
    assert ma.audit_run("s", ENTRY, at_the_peak_limit).flags == ()


def test_a_foreign_gpu_process_at_any_sample_is_flagged() -> None:
    samples = [*quiet(range(10, 20)), sample(14, gpu_apps_other=1)]
    audit = ma.audit_run("s", ENTRY, samples)
    assert audit.flags == ("other-gpu-process",)
    assert audit.gpu_apps_other_peak == 1


def test_power_cap_is_reported_but_thermal_and_slowdown_bits_are_flags() -> None:
    capped = [*quiet(range(10, 20)), sample(15, gpu_throttle_bits=4)]
    audit = ma.audit_run("s", ENTRY, capped)
    assert audit.flags == () and audit.power_capped_samples == 1 and audit.slowdown_samples == 0
    for bit in (0x8, 0x20, 0x40, 0x80):
        hot = [*quiet(range(10, 20)), sample(15, gpu_throttle_bits=bit | 4)]
        assert ma.audit_run("s", ENTRY, hot).flags == ("slowdown",), hex(bit)
    idle_bit = [*quiet(range(10, 20)), sample(15, gpu_throttle_bits=0x1)]  # GPU idle: normal
    assert ma.audit_run("s", ENTRY, idle_bit).flags == ()


def test_clock_ratio_uses_only_busy_samples() -> None:
    samples = [
        *quiet(range(10, 20)),
        sample(12, gpu_util_percent=0.0, gpu_sm_clock_mhz=210.0),  # idle: low clock is normal
        sample(14, gpu_sm_clock_mhz=2790.0),
    ]
    audit = ma.audit_run("s", ENTRY, samples)
    assert audit.clock_ratio_min == 2790.0 / 3100.0
    assert audit.temp_peak_c == 60.0


def test_missing_gpu_fields_do_not_break_the_audit() -> None:
    bare = [{"t_utc": f"2026-11-23T10:00:{s:02d}+00:00", "gpu_error": True} for s in range(10, 20)]
    for row in bare:
        row["t"] = ma.parse_time(row["t_utc"])
    audit = ma.audit_run("s", ENTRY, bare)
    assert audit.flags == () and audit.temp_peak_c is None and audit.clock_ratio_min is None


def test_cli_audits_every_run_of_every_sweep_and_lists_the_flagged(tmp_path: Path) -> None:
    sweep = tmp_path / "engine-seed1"
    sweep.mkdir()
    second = dict(
        ENTRY,
        name="b32-w0-r8",
        started_at_utc="2026-11-23T10:01:10+00:00",
        finished_at_utc="2026-11-23T10:01:20+00:00",
    )
    (sweep / "manifest.jsonl").write_text(json.dumps(ENTRY) + "\n" + json.dumps(second) + "\n")
    rows = [{k: v for k, v in sample(s).items() if k != "t"} for s in range(10, 21)]
    for s in range(10, 21):  # the second run has someone else's GPU process in it
        row = {k: v for k, v in sample(s).items() if k != "t"}
        row["t_utc"] = f"2026-11-23T10:01:{s:02d}+00:00"
        row["gpu_apps_other"] = 1
        rows.append(row)
    monitor = tmp_path / "m.jsonl"
    monitor.write_text("".join(json.dumps(r) + "\n" for r in rows))
    out = tmp_path / "audit.md"
    assert ma.main(["--monitor", str(monitor), "--sweep", str(sweep), "--out", str(out)]) == 0
    text = out.read_text()
    assert "2 runs; 1 clean; 1 flagged: engine-seed1/b32-w0-r8" in text
    assert "| engine-seed1 | b32-w0-r4 | 11 |" in text
    assert "other-gpu-process" in text
