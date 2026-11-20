"""Sweep a serving system over batch size, (for the baseline) batch window and arrival rate.

The systems are the Hugging Face static-batching baseline (``hf-static``, the default), this
engine (``engine``) and vLLM (``vllm``). For the last two the batch window does not exist, so the
sweep varies the maximum number of concurrent sequences and the arrival rate only.

Every run gets a fresh server process. A request that times out client-side can leave the
baseline's worker still generating for it, which would slow the next run and quietly flatter or
punish whichever configuration happened to come second, and the server's statistics are
cumulative. Before each run the GPU is sampled for other users' load; a run that still sees any
is recorded as ``contended`` (never silently kept as clean).

Results are one write-once JSONL file per run (``bench.loadgen``'s format) plus a manifest listing
every run: configuration, result file, exit status, GPU load before and after, timing. A run
that fails or is contended stays in the manifest, so the sweep reports everything it tried.

Usage (on the GPU machine, so the load generator reaches the server over localhost):
    python -m bench.sweep --model models/Llama-3.2-1B-Instruct --out results/hf-static \\
        --server-env results/env-<commit>.json --batch-sizes 1,4,8,16 --windows-ms 0,20,100 \\
        --rates 1,2,4,8

    python -m bench.sweep --system engine --model models/Llama-3.2-1B-Instruct \\
        --out results/engine --server-env results/env-engine.json --batch-sizes 32 --rates 2,4,8

    python -m bench.sweep --system vllm --vllm-python ~/venvs/vllm/bin/python \\
        --model models/Llama-3.2-1B-Instruct --out results/vllm \\
        --server-env results/env-vllm.json --batch-sizes 32 --rates 2,4,8
"""

from __future__ import annotations

import argparse
import itertools
import json
import socket
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

SYSTEMS = ("hf-static", "engine", "vllm")
GPU_BUSY_PERCENT = 10  # other users' load above this makes a run "contended"
GPU_SAMPLES = 5


@dataclass(frozen=True)
class RunConfig:
    batch_size: int
    window_ms: float
    rate: float

    @property
    def name(self) -> str:
        return f"b{self.batch_size}-w{self.window_ms:g}-r{self.rate:g}"


def parse_numbers(text: str, kind: type[float] | type[int] = float) -> list[Any]:
    values = [kind(part) for part in text.split(",") if part.strip()]
    if not values or len(set(values)) != len(values):
        raise ValueError(f"expected a non-empty list without duplicates, got {text!r}")
    return values


def expand_grid(
    batch_sizes: Sequence[int], windows_ms: Sequence[float], rates: Sequence[float]
) -> list[RunConfig]:
    """Every combination, ordered by (batch size, window, rate)."""
    return [
        RunConfig(batch, window, rate)
        for batch, window, rate in itertools.product(batch_sizes, windows_ms, rates)
    ]


class Backend(Protocol):
    """The parts of a run that touch processes and hardware, so the logic above can be tested."""

    def gpu_utilization(self) -> list[float]: ...
    def start_server(self, config: RunConfig) -> Any: ...
    def wait_ready(self, handle: Any) -> bool: ...
    def run_load(self, config: RunConfig, result_path: Path) -> int: ...
    def stop_server(self, handle: Any) -> None: ...


def sample_gpu(backend: Backend, samples: int = GPU_SAMPLES) -> dict[str, Any]:
    values = backend.gpu_utilization()[:samples]
    return {"max_percent": max(values, default=None), "samples": values}


def execute_run(
    config: RunConfig,
    backend: Backend,
    out_dir: Path,
    *,
    settle: Callable[[], None] = lambda: None,
) -> dict[str, Any]:
    """One run on a fresh server; returns its manifest entry."""
    result_path = out_dir / f"{config.name}.jsonl"
    entry: dict[str, Any] = {
        **asdict(config),
        "name": config.name,
        "result_file": result_path.name,
        "started_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    entry["gpu_before"] = sample_gpu(backend)
    handle = backend.start_server(config)
    try:
        if not backend.wait_ready(handle):
            entry.update(status="server_failed_to_start", exit_code=None)
        else:
            settle()
            entry["gpu_before_load"] = sample_gpu(backend)
            entry["exit_code"] = backend.run_load(config, result_path)
            entry["status"] = "ok" if entry["exit_code"] == 0 else "loadgen_failed"
    finally:
        backend.stop_server(handle)
    entry["gpu_after"] = sample_gpu(backend)
    # Another user's job is visible as load before our own server has done anything.
    busy = entry.get("gpu_before_load", entry["gpu_before"])["max_percent"]
    entry["contended"] = bool(busy is not None and busy > GPU_BUSY_PERCENT)
    entry["finished_at_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return entry


def run_sweep(
    configs: Sequence[RunConfig],
    backend: Backend,
    out_dir: Path,
    *,
    resume: bool = False,
    on_entry: Callable[[dict[str, Any]], None] = lambda entry: None,
    settle: Callable[[], None] = lambda: None,
) -> list[dict[str, Any]]:
    """Run each configuration in order. With ``resume``, configurations whose result file
    already exists are skipped (never overwritten); otherwise an existing file is an error."""
    out_dir.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, Any]] = []
    for config in configs:
        if (out_dir / f"{config.name}.jsonl").exists():
            if not resume:
                raise FileExistsError(f"{config.name}.jsonl exists; pass --resume to skip it")
            continue
        entry = execute_run(config, backend, out_dir, settle=settle)
        entries.append(entry)
        on_entry(entry)
    return entries


def port_in_use(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1)
        return sock.connect_ex((host, port)) == 0


def server_command(args: argparse.Namespace, config: RunConfig) -> list[str]:
    """The command that starts the server for one run of ``args.system``."""
    port = str(args.port)
    if args.system == "hf-static":
        command = [
            sys.executable, "-m", "bench.baselines.hf_server",
            "--model", str(args.model),
            "--model-id", args.model_name,
            "--dtype", args.dtype,
            "--max-batch-size", str(config.batch_size),
            "--batch-window-ms", f"{config.window_ms:g}",
            "--port", port,
        ]  # fmt: skip
    elif args.system == "engine":
        command = [
            sys.executable, "-m", "engine.server",
            "--model-dir", str(args.model),
            "--model-id", args.model_name,
            "--dtype", args.dtype,
            "--max-batch-size", str(config.batch_size),
            "--block-size", str(args.block_size),
            "--kv-cache-gb", f"{args.kv_cache_gb:g}",
            "--port", port,
        ]  # fmt: skip
    else:  # vllm, from its own environment (it pins a different torch)
        command = [
            str(args.vllm_python), "-m", "vllm.entrypoints.openai.api_server",
            "--model", str(args.model),
            "--served-model-name", args.model_name,
            "--dtype", args.dtype,
            "--max-num-seqs", str(config.batch_size),
            "--max-model-len", str(args.max_model_len),
            "--gpu-memory-utilization", f"{args.gpu_memory_utilization:g}",
            "--host", "127.0.0.1",
            "--port", port,
        ]  # fmt: skip
    return [*command, *args.server_arg]


def ready_path(system: str) -> str:
    """vLLM has no ``/info``; its ``/health`` answers once the model is loaded."""
    return "/health" if system == "vllm" else "/info"


def loadgen_command(args: argparse.Namespace, config: RunConfig, result_path: Path) -> list[str]:
    label = (
        f"hf-static-b{config.batch_size}-w{config.window_ms:g}"
        if args.system == "hf-static"
        else f"{args.system}-b{config.batch_size}"
    )
    command = [
        sys.executable, "-m", "bench.loadgen", "run",
        "--url", f"http://127.0.0.1:{args.port}",
        "--model", args.model_name,
        "--rate", f"{config.rate:g}",
        "--duration", f"{args.duration:g}",
        "--vocab-size", str(args.vocab_size),
        "--label", label,
    ]  # fmt: skip
    if args.system != "vllm":  # only our own servers have /info
        command.append("--fetch-server-info")
    command += ["--server-env", str(args.server_env), "--seed", str(args.seed)]
    command += ["--out", str(result_path)]
    return command


def validate_args(args: argparse.Namespace) -> list[str]:
    """Problems that would make the sweep's results wrong or mislabelled (empty if none)."""
    problems: list[str] = []
    if args.system != "hf-static" and parse_numbers(args.windows_ms) != [0.0]:
        problems.append(
            f"the batch window only applies to hf-static; use --windows-ms 0 for {args.system}"
        )
    if args.system == "vllm":
        if args.vllm_python is None:
            problems.append(
                "--vllm-python is required for --system vllm (vLLM has its own environment)"
            )
        try:
            versions = json.loads(args.server_env.read_text())["packages"]
        except (OSError, ValueError, KeyError):
            problems.append(f"cannot read the package versions in {args.server_env}")
        else:
            if not versions.get("vllm"):
                problems.append(
                    "--server-env does not record a vllm version, so the result would not say "
                    "what it measured. Capture it with the vLLM environment's Python: "
                    "<vllm-python> -m bench.env_capture > results/env-vllm.json"
                )
    return problems


class SubprocessBackend:
    """Starts the system's server and ``bench.loadgen`` as separate processes."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args

    def gpu_utilization(self) -> list[float]:
        readings: list[float] = []
        for _ in range(GPU_SAMPLES):
            proc = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                check=False,
            )
            lines = proc.stdout.split()
            if proc.returncode == 0 and lines:
                readings.append(float(lines[0]))
            time.sleep(1)
        return readings

    def start_server(self, config: RunConfig) -> subprocess.Popen[bytes]:
        # On a shared machine another service may own the port; waiting for "ready" would then
        # find that service and the load generator would send our load to it.
        if port_in_use(self.args.port):
            raise RuntimeError(f"port {self.args.port} is already in use; pick another with --port")
        command = server_command(self.args, config)
        return subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def wait_ready(self, handle: subprocess.Popen[bytes]) -> bool:
        import httpx

        deadline = time.monotonic() + self.args.startup_timeout
        while time.monotonic() < deadline:
            if handle.poll() is not None:
                return False
            try:
                url = f"http://127.0.0.1:{self.args.port}{ready_path(self.args.system)}"
                if httpx.get(url, timeout=2).is_success:
                    return True
            except httpx.HTTPError:
                pass
            time.sleep(1)
        return False

    def run_load(self, config: RunConfig, result_path: Path) -> int:
        command = loadgen_command(self.args, config, result_path)
        return subprocess.run(command, check=False).returncode

    def stop_server(self, handle: subprocess.Popen[bytes]) -> None:
        handle.terminate()
        try:
            handle.wait(timeout=30)
        except subprocess.TimeoutExpired:
            handle.kill()
            handle.wait()


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0] if __doc__ else None)
    parser.add_argument("--system", choices=SYSTEMS, default="hf-static")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-name", default="llama-3.2-1b-instruct")
    parser.add_argument("--out", type=Path, required=True, help="directory for results + manifest")
    parser.add_argument("--server-env", type=Path, required=True, help="env_capture JSON")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument(
        "--batch-sizes",
        default="1,4,8,16",
        help="max batch size (hf-static, engine) or max concurrent sequences (vllm)",
    )
    parser.add_argument(
        "--windows-ms", help="hf-static only (default 0,20,100); other systems use 0"
    )
    parser.add_argument("--rates", default="1,2,4,8")
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--port", type=int, default=18437, help="must be free; checked per run")
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=100000,
        help="random prompt ids stay below this (Llama 3's special tokens start at 128000)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--startup-timeout", type=float, help="seconds (default 180; 600 for vllm, which warms up)"
    )
    parser.add_argument("--vllm-python", type=Path, help="Python of the separate vLLM environment")
    parser.add_argument("--max-model-len", type=int, default=4096, help="vllm only")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85, help="vllm only")
    parser.add_argument("--kv-cache-gb", type=float, default=8.0, help="engine only")
    parser.add_argument("--block-size", type=int, default=16, help="engine only")
    parser.add_argument(
        "--server-arg",
        action="append",
        default=[],
        help=(
            "extra flag for the server command, repeatable; write it with an equals sign, "
            "e.g. --server-arg=--no-enable-prefix-caching"
        ),
    )
    parser.add_argument("--resume", action="store_true", help="skip runs whose result exists")
    args = parser.parse_args(argv)
    if args.windows_ms is None:
        args.windows_ms = "0,20,100" if args.system == "hf-static" else "0"
    if args.startup_timeout is None:
        args.startup_timeout = 600.0 if args.system == "vllm" else 180.0
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    problems = validate_args(args)
    if problems:
        for problem in problems:
            print(f"error: {problem}", file=sys.stderr)
        return 2
    configs = expand_grid(
        parse_numbers(args.batch_sizes, int),
        parse_numbers(args.windows_ms),
        parse_numbers(args.rates),
    )
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = args.out / "manifest.jsonl"
    print(f"{len(configs)} runs, {args.duration:g} s of arrivals each", file=sys.stderr)
    with manifest.open("a" if args.resume else "x") as handle:

        def record(entry: dict[str, Any]) -> None:
            entry = {**entry, "system": args.system, "server_args": args.server_arg}
            handle.write(json.dumps(entry, sort_keys=True) + "\n")
            handle.flush()
            flag = " CONTENDED" if entry["contended"] else ""
            print(f"{entry['name']}: {entry['status']}{flag}", file=sys.stderr)

        entries = run_sweep(
            configs, SubprocessBackend(args), args.out, resume=args.resume, on_entry=record
        )
    bad = [e["name"] for e in entries if e["status"] != "ok"]
    contended = [e["name"] for e in entries if e["contended"]]
    print(f"done: {len(entries)} runs; not ok: {bad or 'none'}; contended: {contended or 'none'}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
