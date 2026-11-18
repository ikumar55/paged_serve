"""Load generator command line.

    python -m bench.loadgen run --url http://localhost:8000 --model NAME --rate 4 --duration 60 \\
        --out results/run-rate4.jsonl --server-env results/env-<commit>.json
    python -m bench.loadgen summarize results/run-rate4.jsonl
    python -m bench.loadgen plot results/run-rate*.jsonl --out-dir results/plots

``--server-env`` is the JSON written by ``python -m bench.env_capture`` on the machine that runs
the server. Without it the result does not record the server's hardware and must not be published.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

from bench.client import RequestResult
from bench.env_capture import capture_environment
from bench.metrics import summarize
from bench.plots import plot_run, plot_sweep
from bench.results_io import read_results, write_results
from bench.runner import run_workload, warm_up
from bench.workload import LengthDist, WorkloadConfig, generate_workload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m bench.loadgen", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="send a Poisson workload to a server and record results")
    run.add_argument("--url", required=True, help="server base URL, e.g. http://localhost:8000")
    run.add_argument("--model", required=True, help="model name sent in each request")
    run.add_argument("--rate", type=float, required=True, help="mean arrivals per second")
    size = run.add_mutually_exclusive_group(required=True)
    size.add_argument("--duration", type=float, help="seconds of arrivals (rate x duration reqs)")
    size.add_argument("--num-requests", type=int, help="exact number of requests")
    run.add_argument("--prompt-median", type=float, default=256)
    run.add_argument("--prompt-sigma", type=float, default=0.7)
    run.add_argument("--prompt-max", type=int, default=2048)
    run.add_argument("--output-median", type=float, default=128)
    run.add_argument("--output-sigma", type=float, default=0.5)
    run.add_argument("--output-max", type=int, default=512)
    run.add_argument("--seed", type=int, default=0)
    run.add_argument(
        "--vocab-size", type=int, default=32000, help="prompt token ids stay below this"
    )
    run.add_argument(
        "--min-token-id", type=int, default=1000, help="prompt token ids start at this value"
    )
    run.add_argument("--timeout", type=float, default=120.0, help="per-request deadline in seconds")
    run.add_argument("--no-ignore-eos", action="store_true", help="let the model stop at EOS")
    run.add_argument("--warmup", type=int, default=3, help="unmeasured sequential requests first")
    run.add_argument("--label", default="", help="name of the system under test, used in plots")
    run.add_argument("--server-env", type=Path, help="env_capture JSON from the server machine")
    run.add_argument(
        "--fetch-server-info",
        action="store_true",
        help="record GET /info from the server before and after the run (our own servers)",
    )
    run.add_argument("--out", type=Path, required=True, help="result file to create (JSONL)")

    summary = sub.add_parser("summarize", help="print the summary of a result file")
    summary.add_argument("path", type=Path)

    plot = sub.add_parser("plot", help="plot result files (adds a sweep plot for two or more)")
    plot.add_argument("paths", type=Path, nargs="+")
    plot.add_argument("--out-dir", type=Path, required=True)
    return parser


def _ms(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 1000:.1f}"


def format_summary(summary: dict[str, Any]) -> str:
    lines = [
        f"requests: {summary['num_ok']}/{summary['num_requests']} ok"
        + (f", failures: {summary['failure_counts']}" if summary["failure_counts"] else ""),
        f"duration: {summary['duration_s']:.2f} s",
    ]
    if summary["output_tokens_per_s"] is not None:
        lines.append(
            f"output: {summary['output_tokens_total']} tokens, "
            f"{summary['output_tokens_per_s']:.1f} tokens/s, "
            f"{summary['request_throughput_rps']:.2f} req/s"
        )
    for key, name in (("ttft_s", "TTFT"), ("itl_s", "inter-token"), ("e2e_s", "end-to-end")):
        dist = summary.get(key)
        if dist:
            in_seconds = key == "e2e_s"
            unit = "s" if in_seconds else "ms"

            def show(value: float, in_seconds: bool = in_seconds) -> str:
                return f"{value:.2f}" if in_seconds else _ms(value)

            lines.append(
                f"{name} ({unit}): p50 {show(dist['p50'])}, p90 {show(dist['p90'])}, "
                f"p99 {show(dist['p99'])}, max {show(dist['max'])} (n={int(dist['count'])})"
            )
    if summary["generator_saturated"]:
        lines.append(
            "WARNING: the load generator was sending late (p99 lag above 50 ms); "
            "latencies are understated."
        )
    return "\n".join(lines)


def _workload_from_args(args: argparse.Namespace) -> WorkloadConfig:
    count = args.num_requests or max(1, round(args.rate * args.duration))
    return WorkloadConfig(
        rate=args.rate,
        num_requests=count,
        prompt=LengthDist(args.prompt_median, args.prompt_sigma, 1, args.prompt_max),
        output=LengthDist(args.output_median, args.output_sigma, 1, args.output_max),
        seed=args.seed,
        vocab_size=args.vocab_size,
        min_token_id=args.min_token_id,
    )


def sanitize_argv(argv: Sequence[str]) -> list[str]:
    """Make a command line safe to publish: hide the server URL and the home directory.

    Server addresses can be internal hostnames or IPs, and absolute paths contain the username.
    """
    home = str(Path.home())
    clean: list[str] = []
    redact_next = False
    for arg in argv:
        if redact_next:
            clean.append("<redacted>")
            redact_next = False
        elif arg == "--url":
            clean.append(arg)
            redact_next = True
        elif arg.startswith("--url="):
            clean.append("--url=<redacted>")
        else:
            clean.append(arg.replace(home, "~"))
    return clean


def _build_meta(
    args: argparse.Namespace, workload: WorkloadConfig, argv: Sequence[str]
) -> dict[str, Any]:
    argv = sanitize_argv(argv)
    client_env = capture_environment(args=argv)
    client_env.pop("command")  # the env_capture command is not the command that produced this run
    server_env: dict[str, Any] | None = None
    if args.server_env is not None:
        server_env = json.loads(args.server_env.read_text())
    return {
        "label": args.label,
        "command": " ".join(["python", "-m", "bench.loadgen", *argv]),
        "started_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": args.model,
        "workload": workload.to_dict(),
        "request_params": {
            "timeout_s": args.timeout,
            "ignore_eos": not args.no_ignore_eos,
            "warmup_requests": args.warmup,
        },
        "client_env": client_env,
        "server_env": server_env,
    }


async def _execute(
    args: argparse.Namespace,
    workload: WorkloadConfig,
    transport: httpx.AsyncBaseTransport | None,
) -> list[RequestResult]:
    async with httpx.AsyncClient(
        base_url=args.url,
        transport=transport,
        timeout=None,  # per-request deadlines are enforced by the client, not per network phase
        limits=httpx.Limits(max_connections=None, max_keepalive_connections=None),
    ) as client:
        if args.warmup:
            await warm_up(
                client,
                args.model,
                args.warmup,
                vocab_size=workload.vocab_size,
                min_token_id=workload.min_token_id,
                timeout_s=args.timeout,
            )
        return await run_workload(
            generate_workload(workload),
            client,
            args.model,
            timeout_s=args.timeout,
            ignore_eos=not args.no_ignore_eos,
        )


async def _fetch_info(
    url: str, transport: httpx.AsyncBaseTransport | None
) -> dict[str, Any] | None:
    """The server's ``/info`` (configuration and cumulative stats), or None if unavailable."""
    try:
        async with httpx.AsyncClient(base_url=url, transport=transport, timeout=10) as client:
            response = await client.get("/info")
            response.raise_for_status()
            data = response.json()
    except (httpx.HTTPError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _command_run(
    args: argparse.Namespace, argv: Sequence[str], transport: httpx.AsyncBaseTransport | None
) -> int:
    if args.out.exists():
        print(f"error: {args.out} already exists; results are never overwritten", file=sys.stderr)
        return 2
    if args.server_env is None:
        print(
            "warning: no --server-env given; this result does not record the server's hardware "
            "and must not be published.",
            file=sys.stderr,
        )
    workload = _workload_from_args(args)
    meta = _build_meta(args, workload, argv)
    info_before = asyncio.run(_fetch_info(args.url, transport)) if args.fetch_server_info else None
    if args.fetch_server_info and info_before is None:
        print("warning: could not fetch /info from the server; recording null", file=sys.stderr)
    try:
        results = asyncio.run(_execute(args, workload, transport))
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.fetch_server_info:
        # Server stats are cumulative since it started, so keep both ends to take the difference.
        meta["server_info"] = {
            "before": info_before,
            "after": asyncio.run(_fetch_info(args.url, transport)),
        }
    summary = summarize(results)
    write_results(args.out, meta, results, summary)
    print(format_summary(summary))
    print(f"wrote {args.out}")
    return 0 if summary["num_ok"] > 0 else 1


def main(
    argv: Sequence[str] | None = None, *, transport: httpx.AsyncBaseTransport | None = None
) -> int:
    args_list = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(args_list)

    if args.command == "run":
        return _command_run(args, args_list, transport)

    if args.command == "summarize":
        _, results, _ = read_results(args.path)
        print(format_summary(summarize(results)))
        return 0

    runs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for path in args.paths:
        meta, results, _ = read_results(path)
        plot_run(meta, results, args.out_dir / f"{path.stem}.png")
        runs.append((meta, summarize(results)))
    if len(runs) > 1:
        plot_sweep(runs, args.out_dir / "sweep.png")
    print(f"wrote plots to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
