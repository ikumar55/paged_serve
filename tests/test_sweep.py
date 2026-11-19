"""The sweep orchestrator's logic, with a fake backend (no servers, no GPU)."""

from __future__ import annotations

import socket
from pathlib import Path
from typing import Any

import pytest

from bench.sweep import RunConfig, execute_run, expand_grid, parse_numbers, port_in_use, run_sweep


class FakeBackend:
    def __init__(
        self,
        *,
        utilization: float = 0.0,
        ready: bool = True,
        exit_codes: dict[str, int] | None = None,
        crash_on: str | None = None,
    ) -> None:
        self.utilization = utilization
        self.ready = ready
        self.exit_codes = exit_codes or {}
        self.crash_on = crash_on
        self.events: list[str] = []

    def gpu_utilization(self) -> list[float]:
        return [self.utilization] * 3

    def start_server(self, config: RunConfig) -> str:
        self.events.append(f"start {config.name}")
        return config.name

    def wait_ready(self, handle: Any) -> bool:
        return self.ready

    def run_load(self, config: RunConfig, result_path: Path) -> int:
        self.events.append(f"load {config.name}")
        if self.crash_on == config.name:
            raise RuntimeError("load generator crashed")
        code = self.exit_codes.get(config.name, 0)
        if code == 0:
            result_path.write_text("{}\n")
        return code

    def stop_server(self, handle: Any) -> None:
        self.events.append(f"stop {handle}")


def test_port_in_use_detects_a_listening_socket() -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.bind(("127.0.0.1", 0))
        server.listen()
        port = server.getsockname()[1]
        assert port_in_use(port)
    assert not port_in_use(port)  # closed again


def test_parse_numbers() -> None:
    assert parse_numbers("1,4,8", int) == [1, 4, 8]
    assert parse_numbers("0, 20.5") == [0.0, 20.5]
    with pytest.raises(ValueError):
        parse_numbers("")
    with pytest.raises(ValueError):
        parse_numbers("1,1")


def test_grid_is_the_full_product_in_a_stable_order() -> None:
    grid = expand_grid([1, 8], [0, 20], [2, 4])
    assert len(grid) == 8
    assert [c.name for c in grid[:3]] == ["b1-w0-r2", "b1-w0-r4", "b1-w20-r2"]
    assert grid[-1] == RunConfig(8, 20, 4)


def test_every_run_gets_a_fresh_server_that_is_always_stopped(tmp_path: Path) -> None:
    backend = FakeBackend()
    entries = run_sweep(expand_grid([4], [0], [1, 2]), backend, tmp_path)
    assert [e["status"] for e in entries] == ["ok", "ok"]
    assert backend.events == [
        "start b4-w0-r1",
        "load b4-w0-r1",
        "stop b4-w0-r1",
        "start b4-w0-r2",
        "load b4-w0-r2",
        "stop b4-w0-r2",
    ]
    assert (tmp_path / "b4-w0-r1.jsonl").exists()


def test_a_crashing_load_generator_still_stops_the_server(tmp_path: Path) -> None:
    backend = FakeBackend(crash_on="b4-w0-r1")
    with pytest.raises(RuntimeError):
        execute_run(RunConfig(4, 0, 1), backend, tmp_path)
    assert backend.events[-1] == "stop b4-w0-r1"


def test_a_failed_run_is_recorded_and_the_sweep_continues(tmp_path: Path) -> None:
    backend = FakeBackend(exit_codes={"b1-w0-r1": 3})
    entries = run_sweep(expand_grid([1, 2], [0], [1]), backend, tmp_path)
    assert [(e["name"], e["status"]) for e in entries] == [
        ("b1-w0-r1", "loadgen_failed"),
        ("b2-w0-r1", "ok"),
    ]
    assert entries[0]["exit_code"] == 3


def test_a_server_that_never_starts_is_recorded_without_running_load(tmp_path: Path) -> None:
    backend = FakeBackend(ready=False)
    (entry,) = run_sweep(expand_grid([1], [0], [1]), backend, tmp_path)
    assert entry["status"] == "server_failed_to_start"
    assert not any(e.startswith("load") for e in backend.events)
    assert backend.events[-1].startswith("stop")


def test_other_users_gpu_load_marks_the_run_contended(tmp_path: Path) -> None:
    (quiet,) = run_sweep(expand_grid([1], [0], [1]), FakeBackend(utilization=2), tmp_path / "q")
    (busy,) = run_sweep(expand_grid([1], [0], [1]), FakeBackend(utilization=25), tmp_path / "b")
    assert quiet["contended"] is False
    assert busy["contended"] is True
    assert busy["gpu_before_load"]["max_percent"] == 25  # the evidence is kept in the manifest


def test_existing_results_are_never_overwritten(tmp_path: Path) -> None:
    grid = expand_grid([1, 2], [0], [1])
    run_sweep(grid[:1], FakeBackend(), tmp_path)
    with pytest.raises(FileExistsError):
        run_sweep(grid, FakeBackend(), tmp_path)
    backend = FakeBackend()
    entries = run_sweep(grid, backend, tmp_path, resume=True)
    assert [e["name"] for e in entries] == ["b2-w0-r1"]  # only the missing run was executed
    assert not any("b1-w0-r1" in event for event in backend.events)
