"""Where does a decode step of the engine spend its time?

Drives the scheduler directly (no HTTP, no load generator) with a full batch of running sequences
and measures one decode step three ways:

* **wall time** per step, without the profiler (the inter-token latency the engine can reach);
* **GPU busy time** per step, the sum of the kernels' durations from ``torch.profiler``;
* the **number of kernels** per step and which kernels take the time, grouped by what they do.

If the GPU is busy for a small part of the wall time, the step is bound by Python and kernel
launches (the thing CUDA graphs or fewer, larger kernels would fix). If the GPU is busy for most
of it and gather or copy kernels dominate, the cost is memory traffic from rebuilding contiguous
keys and values every step (the thing a fused paged-attention kernel would fix). Nothing here is
a throughput benchmark: it isolates one step so a speed change can be pointed at its cause.

Usage (on a GPU machine):
    python -m bench.profile_decode --model-dir models/Llama-3.2-1B-Instruct \\
        --batch-sizes 1,8,32,64 --prompt-len 256 --out results/profile-decode-<hw>.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from torch.profiler import ProfilerActivity, profile

from bench.workload import LengthDist, WorkloadConfig, generate_workload
from engine.paged_cache import KVPool
from engine.scheduler import LanguageModel, Scheduler

# (substring of the lowercase kernel name, category), checked in order: the first match wins.
CATEGORIES: tuple[tuple[tuple[str, ...], str], ...] = (
    # Attention first: PyTorch's memory-efficient attention kernel is named "fmha_cutlassF_...".
    (("fmha", "flash", "attention", "sdpa", "softmax"), "attention"),
    (("gemm", "cutlass", "cublas", "nvjet", "gemv", "sgemm", "xmma"), "matmul"),
    (
        ("index", "gather", "scatter", "copy", "memcpy", "cat", "permute", "transpose"),
        "gather/copy",
    ),
)


def categorize(kernel_name: str) -> str:
    lowered = kernel_name.lower()
    for needles, category in CATEGORIES:
        if any(needle in lowered for needle in needles):
            return category
    return "elementwise/other"


def summarise_kernels(
    kernels: Sequence[tuple[str, int, float]], steps: int, top: int = 10
) -> dict[str, Any]:
    """Per-step kernel statistics from ``(name, launches, total device microseconds)`` rows."""
    if steps <= 0:
        raise ValueError("steps must be positive")
    total_us = sum(us for _, _, us in kernels)
    by_category: dict[str, float] = {}
    for name, _, us in kernels:
        category = categorize(name)
        by_category[category] = by_category.get(category, 0.0) + us
    ranked = sorted(kernels, key=lambda row: row[2], reverse=True)[:top]
    return {
        "gpu_busy_ms_per_step": total_us / steps / 1000,
        "kernels_per_step": sum(count for _, count, _ in kernels) / steps,
        "busy_share_by_category": {
            category: (us / total_us if total_us else 0.0)
            for category, us in sorted(by_category.items(), key=lambda item: -item[1])
        },
        "top_kernels": [
            {
                "name": name[:90],
                "launches_per_step": count / steps,
                "ms_per_step": us / steps / 1000,
                "share": us / total_us if total_us else 0.0,
            }
            for name, count, us in ranked
        ],
    }


def kernel_rows(prof: Any) -> list[tuple[str, int, float]]:
    """``(name, launches, device microseconds)`` for every GPU kernel the profiler recorded."""
    rows: list[tuple[str, int, float]] = []
    for event in prof.key_averages():
        device_us = float(
            getattr(event, "self_device_time_total", getattr(event, "self_cuda_time_total", 0.0))
        )
        is_device = str(getattr(event, "device_type", "")).endswith("CUDA")
        if is_device and device_us > 0:
            rows.append((event.key, int(event.count), device_us))
    return rows


def fill_batch(
    model: LanguageModel,
    pool: KVPool,
    batch_size: int,
    prompt_len: int,
    new_tokens: int,
    seed: int = 0,
) -> Scheduler:
    """A scheduler with ``batch_size`` sequences prefilled and running, none finished."""
    scheduler = Scheduler(model, pool, max_batch_size=batch_size, enable_prefix_caching=False)
    generator = torch.Generator().manual_seed(seed)
    vocab = model.cfg.vocab_size
    for request_id in range(batch_size):
        prompt = torch.randint(0, vocab, (prompt_len,), generator=generator).tolist()
        scheduler.add_request(request_id, prompt, new_tokens)
    while scheduler.waiting:
        scheduler.step()
    if len(scheduler.running) != batch_size:
        raise RuntimeError(f"only {len(scheduler.running)} of {batch_size} sequences are running")
    return scheduler


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def profile_batch(
    model: LanguageModel,
    pool: KVPool,
    *,
    batch_size: int,
    prompt_len: int,
    steps: int,
    warmup: int = 5,
) -> dict[str, Any]:
    """One batch size: wall time of decode steps, then kernel statistics of the same steps."""
    new_tokens = warmup + 2 * steps + 8  # nobody finishes while we measure
    scheduler = fill_batch(model, pool, batch_size, prompt_len, new_tokens)
    device = model.device
    for _ in range(warmup):
        scheduler.step()
    walls: list[float] = []
    for _ in range(steps):
        sync(device)
        started = time.perf_counter()
        scheduler.step()  # a decode step: it ends in .tolist(), which waits for the GPU
        sync(device)
        walls.append((time.perf_counter() - started) * 1000)
    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)
    with profile(activities=activities) as prof:
        for _ in range(steps):
            scheduler.step()
        sync(device)
    summary = summarise_kernels(kernel_rows(prof), steps)
    wall_median = statistics.median(walls)
    busy = summary["gpu_busy_ms_per_step"]
    return {
        "batch_size": batch_size,
        "prompt_len": prompt_len,
        "context_len_at_end": prompt_len + warmup + 2 * steps,
        "steps": steps,
        "wall_ms_per_step_median": wall_median,
        "wall_ms_per_step_min": min(walls),
        "wall_ms_per_step_max": max(walls),
        "gpu_idle_fraction": (1 - busy / wall_median) if busy and wall_median else None,
        **summary,
    }


def standard_requests(
    seed: int, rate: float = 32.0, duration_s: float = 60.0, vocab_size: int = 100000
) -> list[tuple[list[int], int]]:
    """``(prompt, new tokens)`` of the standard workload exactly as the load generator builds it.

    The saturation sweeps offered ``rate`` x ``duration_s`` requests with prompt lengths lognormal
    around 256 tokens (clipped to 2048) and output lengths around 128 (clipped to 512). Using the
    same generator and seed gives the same requests, so an offline run can be compared with a
    server run request for request (D-021).
    """
    config = WorkloadConfig(
        rate=rate,
        num_requests=max(1, round(rate * duration_s)),
        prompt=LengthDist(256, 0.7, 1, 2048),
        output=LengthDist(128, 0.5, 1, 512),
        seed=seed,
        vocab_size=vocab_size,
    )
    return [(spec.prompt_token_ids, spec.max_tokens) for spec in generate_workload(config)]


def offline_throughput(
    model: LanguageModel,
    pool: KVPool,
    *,
    requests: int,
    prompt_len: int,
    new_tokens: int,
    max_batch_size: int,
    seed: int = 0,
) -> dict[str, Any]:
    """The scheduler offline on random prompts of one fixed length, prefix cache off."""
    generator = torch.Generator().manual_seed(seed)
    specs = [
        (
            torch.randint(0, model.cfg.vocab_size, (prompt_len,), generator=generator).tolist(),
            new_tokens,
        )
        for _ in range(requests)
    ]
    result = offline_run(model, pool, specs, max_batch_size=max_batch_size, prefix_caching=False)
    return {**result, "prompt_len": prompt_len, "new_tokens": new_tokens}


def offline_run(
    model: LanguageModel,
    pool: KVPool,
    specs: Sequence[tuple[list[int], int]],
    *,
    max_batch_size: int,
    prefix_caching: bool,
) -> dict[str, Any]:
    """Run the scheduler to completion with no server or load generator and split its time.

    Every request is waiting from the start, so the batch is as full as the scheduler can keep
    it. The time of each step is divided into ``decode`` (the batched decode), ``admit`` (the
    prefills of newly admitted requests and their bookkeeping) and ``other``; the cost per
    generated token is defined as the server's counters define it (time inside the scheduler step
    over tokens generated), so the two can be compared (D-021).
    """
    requests = len(specs)
    new_tokens_total = sum(new_tokens for _, new_tokens in specs)
    prompt_tokens_total = sum(len(prompt) for prompt, _ in specs)
    scheduler = Scheduler(
        model, pool, max_batch_size=max_batch_size, enable_prefix_caching=prefix_caching
    )
    for request_id, (prompt, new_tokens) in enumerate(specs):
        scheduler.add_request(request_id, list(prompt), new_tokens)
    device = model.device
    spent = {"decode": 0.0, "admit": 0.0}

    def timed(label: str, method: Any) -> Any:
        def run(result: Any) -> None:
            sync(device)
            started = time.perf_counter()
            method(result)
            sync(device)
            spent[label] += time.perf_counter() - started

        return run

    scheduler._decode = timed("decode", scheduler._decode)  # type: ignore[method-assign]
    scheduler._admit = timed("admit", scheduler._admit)  # type: ignore[method-assign]
    sync(device)
    started = time.perf_counter()
    while scheduler.has_work:
        scheduler.step()
    sync(device)
    elapsed = time.perf_counter() - started
    stats = scheduler.stats()
    return {
        "max_batch_size": max_batch_size,
        "prefix_caching": prefix_caching,
        "requests": requests,
        "prompt_tokens_total": prompt_tokens_total,
        "new_tokens_total": new_tokens_total,
        "elapsed_s": elapsed,
        "requests_per_s": requests / elapsed,
        "tokens_per_s": new_tokens_total / elapsed,
        "ms_per_token": 1000 * elapsed / new_tokens_total,
        "decode_share": spent["decode"] / elapsed,
        "admit_share": spent["admit"] / elapsed,
        "other_share": max(0.0, 1 - (spent["decode"] + spent["admit"]) / elapsed),
        "decode_steps": scheduler.decode_steps,
        "mean_decode_batch": stats.get("mean_decode_batch"),
        "preemptions": scheduler.total_preemptions,
        "prefix_hit_tokens": scheduler.prefix_hit_tokens,
        "block_evictions": stats.get("block_evictions"),
    }


def render_offline(results: Sequence[dict[str, Any]]) -> str:
    lines = [
        "| max batch | workload seed | prefix cache | req/s | tokens/s | ms per token "
        "| decode share | admit (prefill) share | other | mean decode batch | preemptions "
        "| evictions |",
        "| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in results:
        caching = r.get("prefix_caching")
        cache = "-" if caching is None else ("on" if caching else "off")
        lines.append(
            f"| {r['max_batch_size']} | {r.get('workload_seed', 'fixed')} | {cache} | "
            f"{r['requests_per_s']:.1f} | {r['tokens_per_s']:.0f} | "
            f"{r.get('ms_per_token', float('nan')):.3f} | {r['decode_share']:.0%} | "
            f"{r['admit_share']:.0%} | {r['other_share']:.0%} | {r['mean_decode_batch']:.1f} | "
            f"{r['preemptions']} | {r.get('block_evictions', '-')} |"
        )
    return "\n".join(lines)


def render(results: Sequence[dict[str, Any]]) -> str:
    lines = [
        "| batch | wall ms/step | GPU busy ms/step | GPU idle | kernels/step | matmul | attention "
        "| gather/copy | elementwise/other |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in results:
        share = r["busy_share_by_category"]
        idle = r["gpu_idle_fraction"]
        lines.append(
            f"| {r['batch_size']} | {r['wall_ms_per_step_median']:.2f} | "
            f"{r['gpu_busy_ms_per_step']:.2f} | {'-' if idle is None else f'{idle:.0%}'} | "
            f"{r['kernels_per_step']:.0f} | {share.get('matmul', 0):.0%} | "
            f"{share.get('attention', 0):.0%} | {share.get('gather/copy', 0):.0%} | "
            f"{share.get('elementwise/other', 0):.0%} |"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    from engine.weights import load_model

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--batch-sizes", default="1,8,32,64")
    parser.add_argument("--prompt-len", type=int, default=256)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--kv-cache-gb", type=float, default=8.0)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--out", type=Path, required=True, help="JSON file to create (not overwritten)"
    )
    parser.add_argument(
        "--offline-requests",
        type=int,
        default=0,
        help="also run this many requests through the scheduler with no server and report "
        "where the time goes (0 = skip)",
    )
    parser.add_argument("--offline-new-tokens", type=int, default=128)
    parser.add_argument(
        "--offline-workload-seeds",
        default="",
        help="comma separated: also run the scheduler offline on the standard workload's own "
        "requests (lognormal lengths, as the saturation sweeps sent them) for these seeds, "
        "at each batch size (D-021)",
    )
    parser.add_argument(
        "--offline-prefix-cache",
        choices=("on", "off", "both"),
        default="both",
        help="prefix cache setting of the --offline-workload-seeds runs (both: off then on, "
        "interleaved within each seed)",
    )
    parser.add_argument(
        "--skip-profile",
        action="store_true",
        help="only the offline runs (needs --offline-requests or --offline-workload-seeds)",
    )
    args = parser.parse_args(argv)
    if args.skip_profile and not (args.offline_requests or args.offline_workload_seeds):
        parser.error("--skip-profile needs --offline-requests or --offline-workload-seeds")

    dtype = getattr(torch, args.dtype)
    model, _ = load_model(args.model_dir, dtype, args.device)
    per_token = model.cfg.kv_bytes_per_token(torch.empty((), dtype=dtype).element_size())
    num_blocks = int(args.kv_cache_gb * 2**30 // (per_token * args.block_size))
    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]
    results = []
    offline = []
    with torch.inference_mode():
        for batch_size in [] if args.skip_profile else batch_sizes:
            pool = KVPool(model.cfg, num_blocks, args.block_size, dtype, args.device)
            results.append(
                profile_batch(
                    model,
                    pool,
                    batch_size=batch_size,
                    prompt_len=args.prompt_len,
                    steps=args.steps,
                )
            )
            del pool
            print(f"batch {batch_size}: {results[-1]['wall_ms_per_step_median']:.2f} ms/step")
        for batch_size in batch_sizes if args.offline_requests else []:
            pool = KVPool(model.cfg, num_blocks, args.block_size, dtype, args.device)
            offline.append(
                offline_throughput(
                    model,
                    pool,
                    requests=args.offline_requests,
                    prompt_len=args.prompt_len,
                    new_tokens=args.offline_new_tokens,
                    max_batch_size=batch_size,
                )
            )
            del pool
            print(f"offline, max batch {batch_size}: {offline[-1]['tokens_per_s']:.0f} tokens/s")
        caches = {"on": [True], "off": [False], "both": [False, True]}[args.offline_prefix_cache]
        for seed in [int(x) for x in args.offline_workload_seeds.split(",") if x]:
            specs = standard_requests(seed)
            for batch_size in batch_sizes:
                for caching in caches:  # interleaved within a seed
                    pool = KVPool(model.cfg, num_blocks, args.block_size, dtype, args.device)
                    run = offline_run(
                        model, pool, specs, max_batch_size=batch_size, prefix_caching=caching
                    )
                    offline.append({**run, "workload_seed": seed})
                    del pool
                    print(
                        f"offline workload seed {seed}, batch {batch_size}, cache "
                        f"{'on' if caching else 'off'}: {run['ms_per_token']:.3f} ms/token, "
                        f"{run['tokens_per_s']:.0f} tokens/s"
                    )
    record = {
        "device": torch.cuda.get_device_name(model.device)
        if model.device.type == "cuda"
        else "cpu",
        "torch": torch.__version__,
        "dtype": args.dtype,
        "block_size": args.block_size,
        "results": results,
        "offline": offline,
    }
    with args.out.open("x", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2, sort_keys=True)
    if results:
        print(render(results))
    if offline:
        print(render_offline(offline))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
