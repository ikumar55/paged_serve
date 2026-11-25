"""Compare the router policies (D-007) on a fleet of replicas under a shared-prefix workload.

    python -m bench.router_compare --out results/router-compare-mock-a.json \\
        --replicas 4 --rate 30 --num-requests 300 --seeds 0,1,2 \\
        --shared-prefix-groups 8 --shared-prefix-len 512 --shared-prefix-skew 1 \\
        --approx-max-entries 128

**Mock backend, uncalibrated timing.** The replicas are ``engine.server --backend mock`` processes
(D-008): the scheduler, block pool and prefix cache are the real ones, but a forward pass is a sleep
whose length is a placeholder. The results therefore describe the routing policies and the plumbing
(cache hit rates, balance, queueing under these made-up step times), and say nothing about a GPU.
Every output file says so.

**Real replicas (D-022).** ``--backend llama --model-dir <weights>`` starts real ``engine.server``
replicas instead (bf16 by default). They run as separate processes on the one GPU and CPU of the
machine, together with the router and the load generator: that sharing is a confound, and the result
files say so instead of the mock banner. Each fresh replica first receives a few tiny requests
directly (``--warmup-requests``, shorter than a block, so nothing enters its prefix cache) to
initialise CUDA before the run.

Method. For each workload seed, every variant gets a fresh fleet (new replica processes, so no
cache carries over, and a new router) and the *same* requests (a digest of the workload is stored
per run, and a test checks that variants of one seed share it). Variants are run seed by seed,
interleaved, so a slow minute on the machine does not land on one policy only. There is no
warm-up, so every policy pays for its first sight of each prefix. A variant is a policy, plus for
the approximate policy optionally a memory limit (``--approx-max-entries``), which shows how far
its guesses drift once it must forget.

Output: one write-once JSON file (every run, with its configuration) and a Markdown table next to
it. Neither contains addresses, ports, hostnames or user names.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import socket
import subprocess
import sys
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import AbstractContextManager, contextmanager
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import httpx

from bench.client import RequestResult
from bench.env_capture import capture_environment, redact_home
from bench.metrics import summarize
from bench.runner import run_workload
from bench.sweep import parse_numbers
from bench.workload import LengthDist, RequestSpec, WorkloadConfig, generate_workload
from engine.mock import TimingModel
from router.policies import POLICIES

SCHEMA_VERSION = 1
PREFIX_POLICIES = ("approx-prefix", "precise-prefix")  # the policies with a load cap
REPO_ROOT = Path(__file__).resolve().parents[1]

BANNER = (
    "MOCK BACKEND, UNCALIBRATED TIMING. These numbers describe the routing policies and the "
    "plumbing on a CPU, with placeholder step times (D-008). They are not GPU performance and "
    "must not be quoted as such."
)

WARMUP_PROMPT = tuple(range(1000, 1008))  # 8 tokens: shorter than a block of 16, so no cache entry

BANNER_REAL = (
    "REAL ENGINE REPLICAS SHARING ONE GPU AND ONE CPU. The replicas, the router and the load "
    "generator run on the same machine, so latencies include contention between the replicas "
    "(D-022). One GPU, one model, a synthetic shared-prefix workload."
)


@dataclass(frozen=True)
class Variant:
    """A router configuration.

    ``max_entries`` (the memory limit) only applies to the approximate policy. ``load_cap``
    overrides the fleet's cap for one variant and only applies to the two prefix policies, the
    only ones that have a cap; without it the fleet's setting is used.
    """

    policy: str
    max_entries: int | None = None
    load_cap: int | None = None

    def __post_init__(self) -> None:
        if self.policy not in POLICIES:
            raise ValueError(f"unknown policy {self.policy!r}; choose from {', '.join(POLICIES)}")
        if self.max_entries is not None:
            if self.policy != "approx-prefix":
                raise ValueError("a memory limit (max_entries) only applies to approx-prefix")
            if self.max_entries <= 0:
                raise ValueError("max_entries must be positive")
        if self.load_cap is not None:
            if self.policy not in PREFIX_POLICIES:
                raise ValueError("a load cap only applies to the prefix policies")
            if self.load_cap <= 0:
                raise ValueError("load_cap must be positive")

    @property
    def label(self) -> str:
        settings = []
        if self.max_entries is not None:
            settings.append(f"max-entries={self.max_entries}")
        if self.load_cap is not None:
            settings.append(f"cap={self.load_cap}")
        return f"{self.policy}[{','.join(settings)}]" if settings else self.policy

    def effective_load_cap(self, fleet: FleetConfig) -> int | None:
        """The cap this variant's router runs with, or None if its policy has no cap."""
        if self.policy not in PREFIX_POLICIES:
            return None
        return self.load_cap if self.load_cap is not None else fleet.load_cap


@dataclass(frozen=True)
class FleetConfig:
    """The replicas and the router's settings; the same for every variant in a file."""

    replicas: int = 4
    max_batch_size: int = 32
    block_size: int = 16
    kv_cache_gb: float = 4.0  # the mock sizes its pool as Llama-3.2-1B in bf16 would be
    load_cap: int = 16
    poll_interval_s: float = 0.5  # precise-prefix: how often the router reads each /cache
    time_scale: float = 1.0  # multiplies every mock step time; 0 makes the mock instantaneous
    backend: str = "mock"  # "mock" (D-008) or "llama" (real replicas, D-022)
    model_dir: str | None = None  # the weights directory, for the llama backend only
    dtype: str = "bfloat16"  # the llama backend's weight and cache dtype
    warmup_requests: int = 0  # tiny direct requests per fresh replica before the run

    def __post_init__(self) -> None:
        if self.replicas < 1:
            raise ValueError("replicas must be at least 1")
        if self.backend not in ("mock", "llama"):
            raise ValueError("backend must be 'mock' or 'llama'")
        if self.backend == "llama" and not self.model_dir:
            raise ValueError("the llama backend needs --model-dir")
        if self.warmup_requests < 0:
            raise ValueError("warmup_requests must not be negative")

    @property
    def model_name(self) -> str:
        """The name sent in each request: the weights directory's name, or ``mock``."""
        return Path(self.model_dir).name if self.backend == "llama" and self.model_dir else "mock"


@dataclass(frozen=True)
class FleetEndpoints:
    router_url: str
    replica_urls: list[str]


class FleetError(RuntimeError):
    """A fleet could not be started (or died while starting)."""


StartFleet = Callable[[Variant, FleetConfig], AbstractContextManager[FleetEndpoints]]


# -- starting a fleet as real processes ----------------------------------------------------------


def _free_ports(count: int) -> list[int]:
    sockets = [socket.socket() for _ in range(count)]
    try:
        for sock in sockets:
            sock.bind(("127.0.0.1", 0))
        return [int(sock.getsockname()[1]) for sock in sockets]
    finally:
        for sock in sockets:
            sock.close()


def replica_command(port: int, fleet: FleetConfig) -> list[str]:
    if fleet.backend == "llama":
        return [
            sys.executable, "-m", "engine.server",
            "--model-dir", str(fleet.model_dir), "--dtype", fleet.dtype,
            "--port", str(port),
            "--max-batch-size", str(fleet.max_batch_size),
            "--block-size", str(fleet.block_size),
            "--kv-cache-gb", f"{fleet.kv_cache_gb:g}",
        ]  # fmt: skip
    return [
        sys.executable, "-m", "engine.server", "--backend", "mock",
        "--port", str(port),
        "--max-batch-size", str(fleet.max_batch_size),
        "--block-size", str(fleet.block_size),
        "--kv-cache-gb", f"{fleet.kv_cache_gb:g}",
        "--mock-time-scale", f"{fleet.time_scale:g}",
    ]  # fmt: skip


def router_command(
    port: int, replica_urls: Sequence[str], variant: Variant, fleet: FleetConfig
) -> list[str]:
    command = [
        sys.executable, "-m", "router.server",
        "--port", str(port),
        "--policy", variant.policy,
        "--chunk-tokens", str(fleet.block_size),  # chunks match the replicas' blocks
        "--load-cap", str(variant.load_cap if variant.load_cap is not None else fleet.load_cap),
        "--poll-interval", f"{fleet.poll_interval_s:g}",
    ]  # fmt: skip
    for url in replica_urls:
        command += ["--replica", url]
    if variant.max_entries is not None:
        command += ["--max-entries", str(variant.max_entries)]
    return command


def _wait_ready(
    processes: Sequence[subprocess.Popen[bytes]], urls: Sequence[str], timeout_s: float
) -> None:
    deadline = time.monotonic() + timeout_s
    pending = list(urls)
    while pending:
        if any(p.poll() is not None for p in processes):
            raise FleetError("a server process exited while starting")
        if time.monotonic() > deadline:
            raise FleetError(f"servers not ready after {timeout_s:g} s")
        try:
            if httpx.get(pending[0], timeout=2).is_success:
                pending.pop(0)
                continue
        except httpx.HTTPError:
            pass
        time.sleep(0.2)


def warm_up_replicas(urls: Sequence[str], count: int, model: str, timeout_s: float) -> None:
    """Send ``count`` tiny requests straight to every replica (not through the router).

    The prompt is 8 tokens, shorter than a block, so nothing enters a replica's prefix cache; the
    point is to pay for CUDA initialisation and kernel selection before the measured run, not
    during its first requests. Raises ``FleetError`` if a replica does not answer.
    """
    body = {"model": model, "prompt": list(WARMUP_PROMPT), "max_tokens": 4, "temperature": 0}
    for url in urls:
        for _ in range(count):
            try:
                response = httpx.post(f"{url}/v1/completions", json=body, timeout=timeout_s)
            except httpx.HTTPError as error:
                raise FleetError(
                    f"a replica did not answer its warm-up: {type(error).__name__}"
                ) from error
            if not response.is_success:
                raise FleetError(f"a replica refused its warm-up (HTTP {response.status_code})")


@contextmanager
def subprocess_fleet(
    variant: Variant, fleet: FleetConfig, *, startup_timeout_s: float = 90.0
) -> Iterator[FleetEndpoints]:
    """Replica and router processes on free local ports, stopped on exit."""
    ports = _free_ports(fleet.replicas + 1)
    replica_urls = [f"http://127.0.0.1:{p}" for p in ports[:-1]]
    router_url = f"http://127.0.0.1:{ports[-1]}"
    processes: list[subprocess.Popen[bytes]] = []
    try:
        for port in ports[:-1]:
            processes.append(
                subprocess.Popen(
                    replica_command(port, fleet),
                    cwd=REPO_ROOT,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            )
        processes.append(
            subprocess.Popen(
                router_command(ports[-1], replica_urls, variant, fleet),
                cwd=REPO_ROOT,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        )
        _wait_ready(
            processes,
            [f"{url}/info" for url in replica_urls] + [f"{router_url}/health"],
            startup_timeout_s,
        )
        if fleet.warmup_requests:
            warm_up_replicas(
                replica_urls, fleet.warmup_requests, fleet.model_name, startup_timeout_s
            )
        yield FleetEndpoints(router_url, replica_urls)
    finally:
        for process in reversed(processes):  # router first, so it stops before its replicas
            process.terminate()
        for process in processes:
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


# -- one run -------------------------------------------------------------------------------------


def workload_digest(specs: Sequence[RequestSpec]) -> str:
    """A short hash of exactly what will be sent, so two runs can be shown to have had the same."""
    digest = hashlib.sha256()
    for spec in specs:
        digest.update(
            json.dumps(
                [spec.id, round(spec.arrival_s, 9), spec.max_tokens, spec.prompt_token_ids]
            ).encode()
        )
    return digest.hexdigest()[:16]


def imbalance(values: Sequence[float]) -> float | None:
    """Largest share relative to an even split: 1.0 is perfectly even, 2.0 means the busiest
    replica did twice its fair share. None if there was no work to compare."""
    if not values or sum(values) == 0:
        return None
    return max(values) / (sum(values) / len(values))


def summarize_run(
    results: Sequence[RequestResult], info: dict[str, Any], variant: Variant, fleet: FleetConfig
) -> dict[str, Any]:
    """Reduce one run's client results and the router's ``/info?fleet=true`` to what is compared.

    Raises ``ValueError`` if the fleet report is incomplete (a replica that did not answer would
    otherwise count as zero hits and quietly flatter or punish a policy), or if the router is not
    running the policy this run is labelled with.
    """
    if info.get("policy") != variant.policy:
        raise ValueError(
            f"router reports policy {info.get('policy')!r}, expected {variant.policy!r}"
        )
    replica_rows = info["replicas"]
    if len(replica_rows) != fleet.replicas:
        raise ValueError(f"router reports {len(replica_rows)} replicas, expected {fleet.replicas}")
    own_info = info["fleet"]["replicas"]
    per_replica: list[dict[str, int]] = []
    for row in replica_rows:
        stats = own_info.get(row["url"], {}).get("stats")
        if stats is None:
            raise ValueError("a replica's own /info is missing from the fleet report")
        prompt, hit = int(stats["prefix_prompt_tokens"]), int(stats["prefix_hit_tokens"])
        per_replica.append(
            {
                "served": int(row["served"]),
                "prompt_tokens": prompt,
                "prefix_hit_tokens": hit,
                "computed_prompt_tokens": prompt - hit,
                "preemptions": int(stats["preemptions"]),
                "block_evictions": int(stats["block_evictions"]),
            }
        )

    summary = summarize(results)
    prompt_total = sum(r["prompt_tokens"] for r in per_replica)
    hit_total = sum(r["prefix_hit_tokens"] for r in per_replica)
    return {
        "requests": summary["num_requests"],
        "ok": summary["num_ok"],
        "failure_counts": summary["failure_counts"],
        "prefix_hit_rate": hit_total / prompt_total if prompt_total else 0.0,
        "prefix_hit_tokens": hit_total,
        "prefix_prompt_tokens": prompt_total,
        "ttft_s": summary["ttft_s"],
        "e2e_s": summary["e2e_s"],
        "request_throughput_rps": summary["request_throughput_rps"],
        "generator_saturated": summary["generator_saturated"],
        # How evenly the work spread: requests, and the prompt tokens actually computed (the cost
        # that a cache hit avoids), which is what a hot replica would be short of.
        "served_imbalance": imbalance([r["served"] for r in per_replica]),
        "computed_tokens_imbalance": imbalance([r["computed_prompt_tokens"] for r in per_replica]),
        "replicas": per_replica,
        "policy_stats": info["policy_stats"],
        "router_counters": {
            key: info[key] for key in ("failovers", "upstream_errors", "no_replica", "bad_requests")
        },
    }


async def _drive(
    url: str, specs: Sequence[RequestSpec], timeout_s: float, model: str = "mock"
) -> list[RequestResult]:
    async with httpx.AsyncClient(
        base_url=url,
        timeout=None,
        limits=httpx.Limits(max_connections=None, max_keepalive_connections=None),
    ) as client:
        return await run_workload(specs, client, model, timeout_s=timeout_s)


def run_variant(
    start_fleet: StartFleet,
    variant: Variant,
    fleet: FleetConfig,
    specs: Sequence[RequestSpec],
    *,
    timeout_s: float,
) -> dict[str, Any]:
    """One variant on a fresh fleet; returns the run's entry (``status`` says whether it counts)."""
    entry: dict[str, Any] = {
        "variant": variant.label,
        "policy": variant.policy,
        "max_entries": variant.max_entries,
        "load_cap": variant.effective_load_cap(fleet),  # None for policies without a cap
        "workload_digest": workload_digest(specs),  # of what is sent here, not of what was meant
    }
    try:
        with start_fleet(variant, fleet) as endpoints:
            results = asyncio.run(_drive(endpoints.router_url, specs, timeout_s, fleet.model_name))
            response = httpx.get(
                f"{endpoints.router_url}/info", params={"fleet": "true"}, timeout=30
            )
            response.raise_for_status()
            info = response.json()
    except (FleetError, httpx.HTTPError) as error:
        return {**entry, "status": "fleet_error", "error": type(error).__name__}
    try:
        entry["summary"] = summarize_run(results, info, variant, fleet)
    except (ValueError, KeyError) as error:
        return {**entry, "status": "invalid_report", "error": str(error)}
    entry["status"] = "ok"
    return entry


# -- many runs, one file -------------------------------------------------------------------------


def build_variants(
    policies: Sequence[str],
    approx_max_entries: Sequence[int],
    load_caps: Sequence[int] = (),
) -> list[Variant]:
    """One variant per policy, plus approximate variants with a memory limit. With ``load_caps``
    each prefix policy gets one variant per cap instead of a single one at the fleet's cap."""
    variants: list[Variant] = []
    for policy in policies:
        if load_caps and policy in PREFIX_POLICIES:
            variants += [Variant(policy, load_cap=cap) for cap in load_caps]
        else:
            variants.append(Variant(policy))
    if load_caps and not any(p in PREFIX_POLICIES for p in policies):
        raise ValueError("--load-caps needs a prefix policy among the policies")
    if approx_max_entries and "approx-prefix" not in policies:
        raise ValueError("--approx-max-entries needs approx-prefix among the policies")
    variants += [Variant("approx-prefix", limit) for limit in approx_max_entries]
    labels = [v.label for v in variants]
    if len(set(labels)) != len(labels):
        raise ValueError("duplicate variants")
    return variants


def run_comparison(
    start_fleet: StartFleet,
    variants: Sequence[Variant],
    fleet: FleetConfig,
    workload: WorkloadConfig,
    seeds: Sequence[int],
    *,
    timeout_s: float,
    on_run: Callable[[dict[str, Any]], None] = lambda entry: None,
) -> list[dict[str, Any]]:
    """Every variant for every seed, seed by seed. Variants of a seed share one workload."""
    runs: list[dict[str, Any]] = []
    for seed in seeds:
        specs = generate_workload(replace(workload, seed=seed))
        for variant in variants:
            entry = run_variant(start_fleet, variant, fleet, specs, timeout_s=timeout_s)
            entry = {"seed": seed, **entry}
            runs.append(entry)
            on_run(entry)
    return runs


# -- reporting -----------------------------------------------------------------------------------


def spread(values: Sequence[float], scale: float = 1.0, digits: int = 1) -> str:
    """``mean (min to max)`` of the values, each multiplied by ``scale``."""
    scaled = [v * scale for v in values]
    mean = sum(scaled) / len(scaled)
    return f"{mean:.{digits}f} ({min(scaled):.{digits}f} to {max(scaled):.{digits}f})"


def _dist(summary: dict[str, Any], key: str, stat: str) -> float | None:
    dist = summary[key]
    return None if dist is None else float(dist[stat])


# (column title, value from a run summary, scale, digits)
Column = tuple[str, Callable[[dict[str, Any]], float | None], float, int]
COLUMNS: list[Column] = [
    ("prefix hit rate %", lambda s: float(s["prefix_hit_rate"]), 100.0, 1),
    ("TTFT p50 ms", lambda s: _dist(s, "ttft_s", "p50"), 1000.0, 0),
    ("TTFT p99 ms", lambda s: _dist(s, "ttft_s", "p99"), 1000.0, 0),
    ("req/s", lambda s: s["request_throughput_rps"], 1.0, 1),
    ("served imbalance", lambda s: s["served_imbalance"], 1.0, 2),
    ("computed-token imbalance", lambda s: s["computed_tokens_imbalance"], 1.0, 2),
]


def _cell(runs: Sequence[dict[str, Any]], column: Column) -> str:
    _, pick, scale, digits = column
    values = [v for run in runs if (v := pick(run["summary"])) is not None]
    return "n/a" if not values else spread(values, scale, digits)


def _fleet_kind(doc: dict[str, Any]) -> str:
    return "real-engine fleet on one shared GPU" if doc.get("backend") == "llama" else "mock fleet"


def format_table(doc: dict[str, Any]) -> str:
    """The Markdown report for a result document."""
    runs = doc["runs"]
    seeds = doc["seeds"]
    lines = [
        f"# Router policy comparison, {_fleet_kind(doc)} ({doc['label'] or 'unlabelled'})",
        "",
        f"**{doc.get('banner', BANNER)}**",
        "",
        f"{doc['fleet']['replicas']} {'real engine' if doc.get('backend') == 'llama' else 'mock'} "
        f"replicas, workload seeds {seeds}, "
        f"{doc['workload']['num_requests']} requests per run at {doc['workload']['rate']:g} req/s. "
        "Each cell is the mean over seeds with the minimum to maximum in brackets; with this few "
        "seeds the brackets are a spread, not a confidence interval.",
        "",
        "| variant | runs | "
        + " | ".join(title for title, *_ in COLUMNS)
        + " | failed requests | load fallbacks |",
        "| --- | ---: | " + " | ".join("---" for _ in COLUMNS) + " | ---: | ---: |",
    ]
    for label in doc["variant_labels"]:
        mine = [r for r in runs if r["variant"] == label]
        ok = [r for r in mine if r["status"] == "ok"]
        bad = len(mine) - len(ok)
        runs_cell = f"{len(ok)}" + (f" (+{bad} not ok)" if bad else "")
        if not ok:
            lines.append(f"| {label} | {runs_cell} |" + " |" * (len(COLUMNS) + 2))
            continue
        failed = sum(r["summary"]["requests"] - r["summary"]["ok"] for r in ok)
        fallbacks: int | str = "n/a"  # only the prefix-aware policies have a load cap to hit
        if ok[0]["policy"] in PREFIX_POLICIES:
            fallbacks = sum(r["summary"]["policy_stats"]["load_fallbacks"] for r in ok)
        cells = " | ".join(_cell(ok, column) for column in COLUMNS)
        lines.append(f"| {label} | {runs_cell} | {cells} | {failed} | {fallbacks} |")
    lines += [
        "",
        "Imbalance is the busiest replica's share relative to an even split (1.00 is perfectly "
        "even). Computed tokens are prompt tokens a replica really ran through the model, i.e. "
        "not served from its prefix cache.",
    ]
    if any(r.get("summary", {}).get("generator_saturated") for r in runs):
        lines += [
            "",
            "WARNING: in some runs the load generator sent late; latencies are understated.",
        ]
    preempted = sum(
        rep["preemptions"] for r in runs if r["status"] == "ok" for rep in r["summary"]["replicas"]
    )
    if preempted:
        lines += [
            "",
            f"{preempted} preemptions happened across the runs; prompt tokens re-admitted after a "
            "preemption are counted again in the hit rate.",
        ]
    lines += ["", *[f"- {note}" for note in doc["notes"]], ""]
    return "\n".join(lines)


def sanitize_argv(argv: Sequence[str]) -> list[str]:
    """The command line without the output path (an absolute path carries the account name, and
    where the file went says nothing about the result), with the weights directory reduced to its
    name (a path to it can carry the account name too) and with the home directory hidden."""
    clean: list[str] = []
    redact_next = False
    basename_next = False
    for arg in argv:
        if redact_next:
            clean.append("<redacted>")
            redact_next = False
        elif basename_next:
            clean.append(Path(arg.replace("\\", "/")).name)
            basename_next = False
        elif arg == "--out":
            clean.append(arg)
            redact_next = True
        elif arg == "--model-dir":
            clean.append(arg)
            basename_next = True
        elif arg.startswith("--out="):
            clean.append("--out=<redacted>")
        elif arg.startswith("--model-dir="):
            clean.append("--model-dir=" + Path(arg.split("=", 1)[1].replace("\\", "/")).name)
        else:
            clean.append(redact_home(arg))
    return clean


def build_document(
    *,
    label: str,
    argv: Sequence[str],
    variants: Sequence[Variant],
    fleet: FleetConfig,
    workload: WorkloadConfig,
    seeds: Sequence[int],
    runs: list[dict[str, Any]],
) -> dict[str, Any]:
    argv = sanitize_argv(argv)
    env = capture_environment(args=argv)
    env.pop("command")  # that is the env_capture command, not this one
    workload_dict = workload.to_dict()
    workload_dict.pop("seed")  # one per run
    real = fleet.backend == "llama"
    fleet_dict = asdict(fleet)
    fleet_dict.pop("model_dir")  # a path can carry an account name: record the model's name only
    if real:
        fleet_dict["model"] = fleet.model_name
    cold = (
        "No warm-up of the prefix caches: every policy starts with cold caches and pays for its "
        "first sight of each prefix."
    )
    same_workload = (
        "Fresh replica and router processes for every run; variants of a seed receive the "
        "identical workload (see workload_digest)."
    )
    shape = (
        "The shared-prefix workload shape (groups, length, skew) is a modelling choice, not a "
        "measurement of real traffic."
    )
    document: dict[str, Any] = {
        "type": "router-policy-comparison",
        "schema_version": SCHEMA_VERSION,
        "backend": fleet.backend,
        "banner": BANNER_REAL if real else BANNER,
        "label": label,
        "command": " ".join(["python", "-m", "bench.router_compare", *argv]),
        "env": env,
        "fleet": fleet_dict,
        "workload": workload_dict,
        "seeds": list(seeds),
        "variant_labels": [v.label for v in variants],
        "runs": runs,
    }
    if real:
        document["shared_gpu_confound"] = True
        document["warmup_requests"] = fleet.warmup_requests
        document["notes"] = [
            "Real engine replicas started as separate processes on one GPU: the replicas, the "
            "router and the load generator share the GPU and the CPU, so latencies include "
            "contention between the replicas.",
            cold,
            f"Each fresh replica first received {fleet.warmup_requests} tiny direct requests of 8 "
            "tokens (shorter than a block, so nothing entered its prefix cache) to initialise "
            "CUDA; their prompt tokens are counted in the replicas' totals and are not subtracted.",
            same_workload,
            shape,
        ]
    else:
        document["calibrated"] = False
        document["mock_timing"] = TimingModel(time_scale=fleet.time_scale).to_dict()
        document["notes"] = [
            "Placeholder mock timing (see mock_timing), not calibrated against any real run.",
            cold,
            same_workload,
            f"Run on {env['os']['system']}: sleeps and scheduling here are not those of a "
            "Linux box, and the replicas, router and load generator share one machine.",
            shape,
        ]
    return document


def write_outputs(json_path: Path, markdown_path: Path, doc: dict[str, Any]) -> None:
    """Write both files, refusing to overwrite either (results are write-once). Both are checked
    before anything is written, so a refusal never leaves one half of a result behind."""
    for path in (json_path, markdown_path):
        if path.exists():
            raise FileExistsError(f"{path} already exists; results are never overwritten")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("x") as handle:
        handle.write(json.dumps(doc, indent=2, sort_keys=True) + "\n")
    with markdown_path.open("x") as handle:
        handle.write(format_table(doc))


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0] if __doc__ else None)
    parser.add_argument(
        "--out", type=Path, required=True, help="JSON to create; .md goes beside it"
    )
    parser.add_argument("--label", default="", help="a name for this experiment")
    parser.add_argument(
        "--policies", default=",".join(POLICIES), help=f"comma separated, from {POLICIES}"
    )
    parser.add_argument(
        "--approx-max-entries",
        default="",
        help="extra approx-prefix variants with this memory limit (chunks); comma separated",
    )
    parser.add_argument(
        "--load-caps",
        default="",
        help="comma separated; each prefix policy then runs once per cap instead of once at "
        "--load-cap (round-robin and least-outstanding have no cap)",
    )
    parser.add_argument("--seeds", default="0,1,2", help="workload seeds, comma separated")
    fleet = parser.add_argument_group("fleet")
    defaults = FleetConfig()
    fleet.add_argument("--replicas", type=int, default=defaults.replicas)
    fleet.add_argument("--max-batch-size", type=int, default=defaults.max_batch_size)
    fleet.add_argument("--block-size", type=int, default=defaults.block_size)
    fleet.add_argument("--kv-cache-gb", type=float, default=defaults.kv_cache_gb)
    fleet.add_argument("--load-cap", type=int, default=defaults.load_cap)
    fleet.add_argument("--poll-interval", type=float, default=defaults.poll_interval_s)
    fleet.add_argument("--mock-time-scale", type=float, default=defaults.time_scale)
    fleet.add_argument("--startup-timeout", type=float, default=90.0)
    fleet.add_argument(
        "--backend",
        choices=("mock", "llama"),
        default=defaults.backend,
        help="mock: placeholder timings (D-008); llama: real engine replicas on this machine's "
        "GPU (D-022; needs --model-dir)",
    )
    fleet.add_argument("--model-dir", help="weights directory, for --backend llama")
    fleet.add_argument("--dtype", default=defaults.dtype, help="--backend llama only")
    fleet.add_argument(
        "--warmup-requests",
        type=int,
        default=defaults.warmup_requests,
        help="tiny direct requests per fresh replica before the run (D-022 uses 2)",
    )
    load = parser.add_argument_group("workload (as bench.loadgen)")
    load.add_argument("--rate", type=float, default=30.0, help="fleet-wide arrivals per second")
    load.add_argument("--num-requests", type=int, default=300)
    load.add_argument("--shared-prefix-groups", type=int, default=8)
    load.add_argument("--shared-prefix-len", type=int, default=512)
    load.add_argument("--shared-prefix-skew", type=float, default=1.0)
    load.add_argument("--prompt-median", type=float, default=64, help="unique suffix, tokens")
    load.add_argument("--prompt-sigma", type=float, default=0.5)
    load.add_argument("--prompt-max", type=int, default=256)
    load.add_argument("--output-median", type=float, default=32)
    load.add_argument("--output-sigma", type=float, default=0.5)
    load.add_argument("--output-max", type=int, default=128)
    load.add_argument("--vocab-size", type=int, default=100000)
    load.add_argument("--timeout", type=float, default=120.0, help="per-request deadline, s")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None, *, start_fleet: StartFleet | None = None) -> int:
    args_list = list(sys.argv[1:] if argv is None else argv)
    args = parse_args(args_list)
    markdown_path = args.out.with_suffix(".md")
    for path in (args.out, markdown_path):
        if path.exists():
            print(f"error: {path} already exists; results are never overwritten", file=sys.stderr)
            return 2

    try:
        variants = build_variants(
            [p for p in args.policies.split(",") if p],
            parse_numbers(args.approx_max_entries, int) if args.approx_max_entries else [],
            parse_numbers(args.load_caps, int) if args.load_caps else [],
        )
        seeds = parse_numbers(args.seeds, int)
        fleet = FleetConfig(
            args.replicas,
            args.max_batch_size,
            args.block_size,
            args.kv_cache_gb,
            args.load_cap,
            args.poll_interval,
            args.mock_time_scale,
            backend=args.backend,
            model_dir=args.model_dir,
            dtype=args.dtype,
            warmup_requests=args.warmup_requests,
        )
        workload = WorkloadConfig(
            rate=args.rate,
            num_requests=args.num_requests,
            prompt=LengthDist(args.prompt_median, args.prompt_sigma, 1, args.prompt_max),
            output=LengthDist(args.output_median, args.output_sigma, 1, args.output_max),
            vocab_size=args.vocab_size,
            shared_prefix_groups=args.shared_prefix_groups,
            shared_prefix_len=args.shared_prefix_len,
            shared_prefix_skew=args.shared_prefix_skew,
        )
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    if start_fleet is None:

        def start_fleet(
            variant: Variant, config: FleetConfig
        ) -> AbstractContextManager[FleetEndpoints]:
            return subprocess_fleet(variant, config, startup_timeout_s=args.startup_timeout)

    total = len(variants) * len(seeds)
    print(f"{total} runs: {len(variants)} variants x {len(seeds)} seeds", file=sys.stderr)

    def progress(entry: dict[str, Any]) -> None:
        detail = ""
        if entry["status"] == "ok":
            detail = f" hit rate {entry['summary']['prefix_hit_rate']:.3f}"
        print(
            f"seed {entry['seed']} {entry['variant']}: {entry['status']}{detail}", file=sys.stderr
        )

    runs = run_comparison(
        start_fleet, variants, fleet, workload, seeds, timeout_s=args.timeout, on_run=progress
    )
    doc = build_document(
        label=args.label,
        argv=args_list,
        variants=variants,
        fleet=fleet,
        workload=workload,
        seeds=seeds,
        runs=runs,
    )
    write_outputs(args.out, markdown_path, doc)
    print(format_table(doc))
    print(f"wrote {args.out} and {markdown_path}")
    not_ok = [r for r in runs if r["status"] != "ok"]
    return 1 if not_ok else 0


if __name__ == "__main__":
    raise SystemExit(main())
