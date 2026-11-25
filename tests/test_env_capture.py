import json
from collections.abc import Sequence
from pathlib import Path

import pytest

from bench.env_capture import (
    capture_environment,
    git_info,
    main,
    output_filename,
    parse_nvcc_release,
    parse_nvidia_smi,
    redact_home,
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
    assert env["git"] == {"commit": FAKE_COMMIT, "dirty": False, "source": "git"}
    assert env["command"] == "python -m bench.env_capture --out results"
    json.dumps(env)  # must be serializable


def test_capture_without_any_tools_degrades_gracefully(tmp_path: Path) -> None:
    env = capture_environment(run=nothing_available, root=tmp_path, environ={})
    assert env["gpus"] == []
    assert env["nvcc_release"] is None
    assert env["git"] == {"commit": None, "dirty": None, "source": None}


def test_command_does_not_reveal_the_home_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression: an absolute --out path put the account name into published result files."""
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    out = tmp_path / "work" / "out"
    env = capture_environment(run=fake_run, args=["--out", str(out)])
    assert env["command"] == f"python -m bench.env_capture --out ~{str(out)[len(str(tmp_path)) :]}"
    assert str(tmp_path) not in json.dumps(env)


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
    assert data["command"] == redact_home(f"python -m bench.env_capture --out {out}")


# -- which code produced a result, when git cannot say (a shipped archive, a Docker image) ------

OTHER_COMMIT = "fedcba9876543210fedcba9876543210fedcba98"


def test_git_answers_first_and_says_so(tmp_path: Path) -> None:
    (tmp_path / "COMMIT").write_text(OTHER_COMMIT)  # present but not used: git is asked first
    info = git_info(fake_run, tmp_path, {"PAGEDSERVE_COMMIT": OTHER_COMMIT})
    assert info == {"commit": FAKE_COMMIT, "dirty": False, "source": "git"}


def test_the_environment_variable_answers_when_git_cannot(tmp_path: Path) -> None:
    (tmp_path / "COMMIT").write_text(FAKE_COMMIT)
    info = git_info(
        nothing_available, tmp_path, {"PAGEDSERVE_COMMIT": " " + OTHER_COMMIT.upper() + "\n"}
    )
    assert info == {"commit": OTHER_COMMIT, "dirty": None, "source": "env"}  # trimmed, lower case


def test_the_commit_file_answers_when_neither_git_nor_the_variable_can(tmp_path: Path) -> None:
    (tmp_path / "COMMIT").write_text(FAKE_COMMIT + "\n")
    info = git_info(nothing_available, tmp_path, {})
    assert info == {"commit": FAKE_COMMIT, "dirty": None, "source": "file"}


@pytest.mark.parametrize(
    "junk", ["unknown", "", "abc123", FAKE_COMMIT[:39], FAKE_COMMIT + "0", "g" * 40]
)
def test_only_a_full_hexadecimal_commit_counts(tmp_path: Path, junk: str) -> None:
    (tmp_path / "COMMIT").write_text(junk)
    assert git_info(nothing_available, tmp_path, {"PAGEDSERVE_COMMIT": junk})["source"] is None


def test_a_missing_commit_file_is_not_an_error(tmp_path: Path) -> None:
    assert git_info(nothing_available, tmp_path / "does-not-exist", {}) == {
        "commit": None,
        "dirty": None,
        "source": None,
    }


def test_the_real_environment_and_the_shipped_commit_file_reach_capture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PAGEDSERVE_COMMIT", OTHER_COMMIT)
    assert capture_environment(run=nothing_available, root=tmp_path)["git"]["source"] == "env"
    monkeypatch.delenv("PAGEDSERVE_COMMIT")
    (tmp_path / "COMMIT").write_text(FAKE_COMMIT)
    env = capture_environment(run=nothing_available, root=tmp_path)
    assert env["git"] == {"commit": FAKE_COMMIT, "dirty": None, "source": "file"}
    assert output_filename(env) == "env-0123456789ab.json"  # a file named after the commit
