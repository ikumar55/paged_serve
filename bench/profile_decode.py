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
    args = parser.parse_args(argv)

    dtype = getattr(torch, args.dtype)
    model, _ = load_model(args.model_dir, dtype, args.device)
    per_token = model.cfg.kv_bytes_per_token(torch.empty((), dtype=dtype).element_size())
    num_blocks = int(args.kv_cache_gb * 2**30 // (per_token * args.block_size))
    results = []
    with torch.inference_mode():
        for batch_size in (int(x) for x in args.batch_sizes.split(",")):
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
    record = {
        "device": torch.cuda.get_device_name(model.device)
        if model.device.type == "cuda"
        else "cpu",
        "torch": torch.__version__,
        "dtype": args.dtype,
        "block_size": args.block_size,
        "results": results,
    }
    with args.out.open("x", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2, sort_keys=True)
    print(render(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
