"""Capture hardware and library versions so every result file is reproducible.

Deliberately records no hostnames, usernames, IP addresses, file paths or GPU UUIDs and serials:
result files are meant to be published.

Usage:
    python -m bench.env_capture              # print JSON to stdout
    python -m bench.env_capture --out results   # write results/env-<commit>.json
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import re
import subprocess
import sys
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Runs a command and returns its stdout, or None if it failed or is unavailable.
Runner = Callable[[Sequence[str]], str | None]

TRACKED_PACKAGES = (
    "torch",
    "transformers",
    "numpy",
    "triton",
    "vllm",
    "fastapi",
    "uvicorn",
    "prometheus-client",
    "pytest",
)

NVIDIA_SMI_QUERY = (
    "nvidia-smi",
    "--query-gpu=name,driver_version,memory.total",
    "--format=csv,noheader,nounits",
)


def run_command(cmd: Sequence[str]) -> str | None:
    try:
        proc = subprocess.run(list(cmd), capture_output=True, text=True, timeout=30, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    return proc.stdout if proc.returncode == 0 else None


def package_versions(names: Sequence[str] = TRACKED_PACKAGES) -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def parse_nvidia_smi(csv_text: str) -> list[dict[str, Any]]:
    """Parse ``name, driver_version, memory.total`` rows (memory in MiB)."""
    gpus: list[dict[str, Any]] = []
    for line in csv_text.strip().splitlines():
        parts = [p.strip() for p in line.rsplit(",", 2)]
        if len(parts) != 3 or not parts[2].isdigit():
            continue
        gpus.append(
            {"name": parts[0], "driver_version": parts[1], "memory_total_mib": int(parts[2])}
        )
    return gpus


def parse_nvcc_release(text: str) -> str | None:
    match = re.search(r"release (\d+\.\d+)", text)
    return match.group(1) if match else None


def git_info(run: Runner) -> dict[str, Any]:
    commit = run(["git", "rev-parse", "HEAD"])
    status = run(["git", "status", "--porcelain"])
    return {
        "commit": commit.strip() if commit else None,
        "dirty": bool(status.strip()) if status is not None else None,
    }


def torch_info() -> dict[str, Any]:
    try:
        import torch
    except ImportError:
        return {"installed": False}
    info: dict[str, Any] = {
        "installed": True,
        "version": torch.__version__,
        "cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "mps_available": bool(torch.backends.mps.is_available()),
        "devices": [],
    }
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            info["devices"].append(
                {
                    "name": props.name,
                    "compute_capability": f"{props.major}.{props.minor}",
                    "total_memory_bytes": props.total_memory,
                }
            )
    return info


def redact_home(text: str) -> str:
    """Replace the user's home directory with ``~``: an absolute path argument would otherwise put
    the account name into a file that is meant to be published."""
    home = str(Path.home())
    return text.replace(home, "~") if len(home) > 1 else text


def capture_environment(run: Runner = run_command, args: Sequence[str] = ()) -> dict[str, Any]:
    nvidia = run(NVIDIA_SMI_QUERY)
    nvcc = run(["nvcc", "--version"])
    return {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "command": redact_home(" ".join(["python", "-m", "bench.env_capture", *args])),
        "git": git_info(run),
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
        },
        "os": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "gpus": parse_nvidia_smi(nvidia) if nvidia else [],
        "nvcc_release": parse_nvcc_release(nvcc) if nvcc else None,
        "torch": torch_info(),
        "packages": package_versions(),
    }


def output_filename(env: dict[str, Any]) -> str:
    commit = env["git"]["commit"]
    stem = commit[:12] if commit else "nocommit"
    if env["git"]["dirty"]:
        stem += "-dirty"
    return f"env-{stem}.json"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0] if __doc__ else None)
    parser.add_argument("--out", type=Path, help="directory to write env-<commit>.json into")
    args = parser.parse_args(argv)

    cli_args = list(sys.argv[1:] if argv is None else argv)
    env = capture_environment(args=cli_args)
    text = json.dumps(env, indent=2, sort_keys=True) + "\n"
    if args.out is None:
        sys.stdout.write(text)
    else:
        args.out.mkdir(parents=True, exist_ok=True)
        path = args.out / output_filename(env)
        path.write_text(text)
        sys.stdout.write(f"wrote {path}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
