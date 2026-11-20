"""The sweep orchestrator's logic, with a fake backend (no servers, no GPU)."""

from __future__ import annotations

import json
import socket
import sys
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


# -- several systems: engine and vLLM as well as the Hugging Face baseline ----------------------


def sweep_args(*extra: str) -> Any:
    from bench.sweep import parse_args

    return parse_args(["--model", "models/m", "--out", "out", "--server-env", "env.json", *extra])


def vllm_env(tmp_path: Path, version: str | None = "0.34.0") -> Path:
    path = tmp_path / "env-vllm.json"
    path.write_text(json.dumps({"packages": {"vllm": version, "torch": "2.14.0"}}))
    return path


def test_hf_static_commands_are_exactly_what_the_recorded_sweeps_used() -> None:
    """The earlier results were produced by these commands; refactoring must not change them."""
    from bench.sweep import loadgen_command, server_command

    args = sweep_args("--port", "18437")
    config = RunConfig(batch_size=16, window_ms=20, rate=4)
    assert server_command(args, config) == [
        sys.executable, "-m", "bench.baselines.hf_server",
        "--model", str(Path("models/m")), "--model-id", "llama-3.2-1b-instruct",
        "--dtype", "bfloat16", "--max-batch-size", "16", "--batch-window-ms", "20",
        "--port", "18437",
    ]  # fmt: skip
    assert loadgen_command(args, config, Path("out/r.jsonl")) == [
        sys.executable, "-m", "bench.loadgen", "run",
        "--url", "http://127.0.0.1:18437", "--model", "llama-3.2-1b-instruct",
        "--rate", "4", "--duration", "60", "--vocab-size", "100000",
        "--label", "hf-static-b16-w20", "--fetch-server-info",
        "--server-env", "env.json", "--seed", "0", "--out", str(Path("out/r.jsonl")),
    ]  # fmt: skip


def test_defaults_depend_on_the_system() -> None:
    hf = sweep_args()
    assert (hf.system, hf.windows_ms, hf.startup_timeout) == ("hf-static", "0,20,100", 180.0)
    engine = sweep_args("--system", "engine")
    assert (engine.windows_ms, engine.startup_timeout) == ("0", 180.0)
    vllm = sweep_args("--system", "vllm")
    assert (vllm.windows_ms, vllm.startup_timeout) == ("0", 600.0)  # vLLM warms up for a while


def test_engine_command_and_label() -> None:
    from bench.sweep import loadgen_command, ready_path, server_command

    args = sweep_args("--system", "engine", "--kv-cache-gb", "6", "--block-size", "32")
    config = RunConfig(batch_size=32, window_ms=0, rate=8)
    command = server_command(args, config)
    assert command[1:3] == ["-m", "engine.server"]
    for flag, value in [
        ("--model-dir", str(Path("models/m"))),
        ("--max-batch-size", "32"),
        ("--block-size", "32"),
        ("--kv-cache-gb", "6"),
    ]:
        assert command[command.index(flag) + 1] == value
    assert "--batch-window-ms" not in command
    load = loadgen_command(args, config, Path("r.jsonl"))
    assert load[load.index("--label") + 1] == "engine-b32"
    assert "--fetch-server-info" in load
    assert ready_path("engine") == "/info"


def test_vllm_command_uses_its_own_python_and_has_no_info_endpoint(tmp_path: Path) -> None:
    from bench.sweep import loadgen_command, ready_path, server_command

    args = sweep_args(
        "--system", "vllm", "--vllm-python", "/venvs/vllm/bin/python",
        "--server-arg=--no-enable-prefix-caching",
    )  # fmt: skip
    config = RunConfig(batch_size=32, window_ms=0, rate=4)
    command = server_command(args, config)
    assert command[:3] == [
        str(Path("/venvs/vllm/bin/python")),
        "-m",
        "vllm.entrypoints.openai.api_server",
    ]
    for flag, value in [
        ("--max-num-seqs", "32"),
        ("--max-model-len", "4096"),
        ("--gpu-memory-utilization", "0.85"),
        ("--host", "127.0.0.1"),
        ("--served-model-name", "llama-3.2-1b-instruct"),
    ]:
        assert command[command.index(flag) + 1] == value
    assert command[-1] == "--no-enable-prefix-caching"  # extra flags are appended, in order
    load = loadgen_command(args, config, Path("r.jsonl"))
    assert "--fetch-server-info" not in load  # vLLM would only produce a warning
    assert ready_path("vllm") == "/health"


def test_validation_rejects_results_that_would_be_mislabelled(tmp_path: Path) -> None:
    from bench.sweep import validate_args

    assert validate_args(sweep_args()) == []
    assert validate_args(sweep_args("--system", "engine")) == []

    engine_with_windows = sweep_args("--system", "engine", "--windows-ms", "0,20")
    assert any("--windows-ms 0" in p for p in validate_args(engine_with_windows))

    no_python = sweep_args("--system", "vllm", "--server-env", str(vllm_env(tmp_path)))
    assert any("--vllm-python" in p for p in validate_args(no_python))

    # The environment file must describe vLLM's environment, or the result would not say what ran.
    engine_env = sweep_args(
        "--system", "vllm", "--vllm-python", "py", "--server-env", str(vllm_env(tmp_path, None))
    )
    assert any("does not record a vllm version" in p for p in validate_args(engine_env))
    missing = sweep_args("--system", "vllm", "--vllm-python", "py", "--server-env", "nope.json")
    assert any("cannot read" in p for p in validate_args(missing))

    good = sweep_args(
        "--system", "vllm", "--vllm-python", "py", "--server-env", str(vllm_env(tmp_path))
    )
    assert validate_args(good) == []


def test_main_refuses_an_invalid_sweep_before_starting_anything(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from bench import sweep

    out = tmp_path / "out"
    code = sweep.main(["--system", "vllm", "--model", "m", "--out", str(out), "--server-env", "x"])
    assert code == 2
    assert "--vllm-python is required" in capsys.readouterr().err
    assert not out.exists()  # nothing was created


def test_main_records_the_system_and_extra_server_args_in_the_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from bench import sweep

    class Backend(FakeBackend):
        def __init__(self, args: Any) -> None:
            super().__init__()

    monkeypatch.setattr(sweep, "SubprocessBackend", Backend)
    out = tmp_path / "engine"
    code = sweep.main([
        "--system", "engine", "--model", "m", "--out", str(out), "--server-env", "e.json",
        "--batch-sizes", "8,32", "--rates", "2,4", "--server-arg=--flag-x",
    ])  # fmt: skip
    assert code == 0
    entries = [json.loads(line) for line in (out / "manifest.jsonl").read_text().splitlines()]
    assert [e["name"] for e in entries] == ["b8-w0-r2", "b8-w0-r4", "b32-w0-r2", "b32-w0-r4"]
    assert {e["system"] for e in entries} == {"engine"}
    assert all(e["server_args"] == ["--flag-x"] for e in entries)


def test_loadgen_args_are_passed_to_every_run_and_recorded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from bench import sweep

    args = sweep_args(
        "--system", "engine",
        "--loadgen-arg=--shared-prefix-groups=8", "--loadgen-arg=--shared-prefix-len=512",
    )  # fmt: skip
    command = sweep.loadgen_command(args, RunConfig(32, 0, 4), Path("r.jsonl"))
    assert command[-2:] == ["--shared-prefix-groups=8", "--shared-prefix-len=512"]

    class Backend(FakeBackend):
        def __init__(self, args: Any) -> None:
            super().__init__()

    monkeypatch.setattr(sweep, "SubprocessBackend", Backend)
    out = tmp_path / "shared"
    code = sweep.main([
        "--system", "engine", "--model", "m", "--out", str(out), "--server-env", "e.json",
        "--batch-sizes", "8", "--rates", "2", "--loadgen-arg=--shared-prefix-groups=8",
    ])  # fmt: skip
    assert code == 0
    entry = json.loads((out / "manifest.jsonl").read_text().splitlines()[0])
    assert entry["loadgen_args"] == ["--shared-prefix-groups=8"]
