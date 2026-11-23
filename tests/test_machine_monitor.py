"""The machine monitor's parsing and arithmetic, on fixed inputs (no GPU, no real /proc)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from bench import machine_monitor as mm

PROC_STAT = "cpu  100 10 50 800 40 5 5 0 0 0\ncpu0 1 1 1 1 1 1 1 1 1 1\n"


def test_proc_stat_busy_excludes_idle_and_iowait() -> None:
    busy, total = mm.parse_proc_stat(PROC_STAT)
    assert total == 100 + 10 + 50 + 800 + 40 + 5 + 5 + 0
    assert busy == total - 800 - 40


def test_proc_stat_rejects_a_per_core_line() -> None:
    with pytest.raises(ValueError):
        mm.parse_proc_stat("cpu0 1 2 3 4 5 6 7 8")


def test_pid_stat_counts_user_and_system_time() -> None:
    fields_after_comm = "S 1 2 3 4 5 6 7 8 9 10 111 222 0 0 20 0 1 0 5"
    assert mm.parse_pid_stat(f"42 (python) {fields_after_comm}") == 111 + 222


def test_pid_stat_survives_spaces_and_brackets_in_the_command_name() -> None:
    fields_after_comm = "R 1 2 3 4 5 6 7 8 9 10 30 40 0 0"
    assert mm.parse_pid_stat(f"7 (a) b (c) {fields_after_comm}") == 70


def test_read_process_jiffies_skips_non_pids_and_unreadable_entries(tmp_path: Path) -> None:
    (tmp_path / "self").mkdir()
    (tmp_path / "1234").mkdir()
    (tmp_path / "1234" / "stat").write_text("1234 (x) S 1 2 3 4 5 6 7 8 9 10 5 6 0 0")
    (tmp_path / "99").mkdir()  # no stat file: the process "ended"
    found = mm.read_process_jiffies(tmp_path)
    assert set(found) == {1234}
    assert found[1234][1] == 11


def test_cpu_cores_split_by_user_group() -> None:
    before = {1: (1018, 0), 2: (1019, 0), 3: (0, 0), 4: (100, 0), 5: (1019, 0)}
    after = {1: (1018, 300), 2: (1019, 100), 3: (0, 50), 4: (100, 999), 6: (1019, 999)}
    cores = mm.cpu_cores_by_group(before, after, own_uid=1018, seconds=2.0, hz=100)
    assert cores["ours"] == pytest.approx(300 / 100 / 2)
    assert cores["other_users"] == pytest.approx(100 / 100 / 2)  # pid 6 is new, pid 5 gone
    assert cores["root"] == pytest.approx(50 / 100 / 2)  # uid 100 (a daemon) is ignored


def test_a_counter_that_goes_backwards_is_not_negative_use() -> None:
    cores = mm.cpu_cores_by_group(
        {1: (1019, 500)}, {1: (1019, 100)}, own_uid=1018, seconds=1.0, hz=100
    )
    assert cores["other_users"] == 0.0


def test_gpu_line_parses_all_fields_and_the_throttle_bits() -> None:
    row = mm.parse_gpu_line("34, 11716, 2947, 3105, 148.77, 43, 0x0000000000000004")
    assert row == {
        "gpu_util_percent": 34.0,
        "gpu_mem_used_mib": 11716.0,
        "gpu_sm_clock_mhz": 2947.0,
        "gpu_sm_clock_max_mhz": 3105.0,
        "gpu_power_w": 148.77,
        "gpu_temp_c": 43.0,
        "gpu_throttle_bits": 4,
    }


def test_gpu_line_marks_unsupported_fields_as_missing() -> None:
    row = mm.parse_gpu_line("0, 2, 210, 3105, [N/A], 30, [N/A]")
    assert row["gpu_power_w"] is None
    assert row["gpu_throttle_bits"] is None


def test_gpu_line_with_the_wrong_number_of_fields_is_an_error() -> None:
    with pytest.raises(ValueError):
        mm.parse_gpu_line("1, 2, 3")


def test_compute_apps_and_ownership() -> None:
    pids = mm.parse_compute_apps("417582\n  12\n\n")
    assert pids == [417582, 12]
    owners = {417582: 1018, 12: 1019}
    assert mm.gpu_apps_of_others(pids, 1018, owners.get) == 1
    assert mm.gpu_apps_of_others([5], 1018, lambda _pid: None) == 1  # unknown counts as not ours
    assert mm.gpu_apps_of_others([], 1018, owners.get) == 0


def test_loadavg_and_memory() -> None:
    assert mm.parse_loadavg("1.25 1.48 1.37 2/910 417850") == 1.25
    meminfo = "MemTotal:       65523360 kB\nMemAvailable:   60898024 kB\n"
    assert mm.parse_mem_available_mib(meminfo) == pytest.approx(60898024 / 1024)
    with pytest.raises(ValueError):
        mm.parse_mem_available_mib("MemTotal: 1 kB\n")


@pytest.mark.skipif(sys.platform != "linux", reason="reads /proc")
def test_monitor_writes_samples_and_refuses_to_overwrite(tmp_path: Path) -> None:
    out = tmp_path / "m.jsonl"
    count = mm.monitor(out, interval=0.05, duration=0.3, stop=lambda: False)
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert count == len(rows) >= 2
    assert {"t_utc", "cpu_cores_ours", "cpu_cores_other_users", "load1"} <= set(rows[0])
    with pytest.raises(FileExistsError):
        mm.monitor(out, interval=0.05, duration=0.1, stop=lambda: False)
