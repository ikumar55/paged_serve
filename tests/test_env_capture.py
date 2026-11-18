import json
from collections.abc import Sequence
from pathlib import Path

from bench.env_capture import (
    capture_environment,
    main,
    output_filename,
    parse_nvcc_release,
    parse_nvidia_smi,
)

FAKE_COMMIT = "0123456789abcdef0123456789abcdef01234567"


def fake_run(cmd: Sequence[str]) -> str | None:
    if cmd[0] == "nvidia-smi":
        return "NVIDIA A10, 550.54.15, 23028\nNVIDIA A10, 550.54.15, 23028\n"
    if cmd[:2] == ["git", "rev-parse"]:
        return FAKE_COMMIT + "\n"
    if cmd[:2] == ["git", "status"]:
        return ""
    if cmd[0] == "nvcc":
        return "Cuda compilation tools, release 12.4, V12.4.131\n"
    return None


def nothing_available(cmd: Sequence[str]) -> str | None:
    return None


def test_parse_nvidia_smi_rows() -> None:
    gpus = parse_nvidia_smi("NVIDIA GeForce RTX 4090, 560.35.03, 24564\n")
    assert gpus == [
        {
            "name": "NVIDIA GeForce RTX 4090",
            "driver_version": "560.35.03",
            "memory_total_mib": 24564,
        }
    ]


def test_parse_nvidia_smi_skips_garbage() -> None:
    assert parse_nvidia_smi("No devices were found\n") == []


def test_parse_nvcc_release() -> None:
    assert parse_nvcc_release("Cuda compilation tools, release 12.4, V12.4.131") == "12.4"
    assert parse_nvcc_release("not nvcc output") is None


def test_capture_with_gpu_tools() -> None:
    env = capture_environment(run=fake_run, args=["--out", "results"])
    assert len(env["gpus"]) == 2
    assert env["nvcc_release"] == "12.4"
    assert env["git"] == {"commit": FAKE_COMMIT, "dirty": False}
    assert env["command"] == "python -m bench.env_capture --out results"
    json.dumps(env)  # must be serializable


def test_capture_without_any_tools_degrades_gracefully() -> None:
    env = capture_environment(run=nothing_available)
    assert env["gpus"] == []
    assert env["nvcc_release"] is None
    assert env["git"] == {"commit": None, "dirty": None}


def test_capture_records_no_machine_identifiers() -> None:
    env = capture_environment(run=fake_run)
    forbidden = {"hostname", "node", "user", "username", "ip", "uuid", "serial"}
    assert forbidden.isdisjoint(env.keys())
    assert forbidden.isdisjoint(env["os"].keys())


def test_output_filename_marks_dirty_trees() -> None:
    clean = {"git": {"commit": FAKE_COMMIT, "dirty": False}}
    dirty = {"git": {"commit": FAKE_COMMIT, "dirty": True}}
    unknown = {"git": {"commit": None, "dirty": None}}
    assert output_filename(clean) == "env-0123456789ab.json"
    assert output_filename(dirty) == "env-0123456789ab-dirty.json"
    assert output_filename(unknown) == "env-nocommit.json"


def test_main_writes_file(tmp_path: Path, capsys: object) -> None:
    out = tmp_path / "results"
    assert main(["--out", str(out)]) == 0
    written = list(out.glob("env-*.json"))
    assert len(written) == 1
    data = json.loads(written[0].read_text())
    assert "packages" in data
    assert data["command"] == f"python -m bench.env_capture --out {out}"
