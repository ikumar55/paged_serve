"""Does the scheduler (batching, paging, prefix cache, preemption) produce the reference's text?

``bench.greedy_eval`` checks the plain single-request engine against Hugging Face. The server does
not use that path: it uses the scheduler, which batches sequences of different lengths over a
paged cache, may reuse cached prefix blocks and may preempt and recompute. Every throughput number
assumes that path serves correct text, so this checks it directly: the same prompts as
``greedy_eval``, greedy decoding with end-of-sequence ignored, free-running, compared with
Hugging Face ``generate`` on the same device and dtype, under three scheduler configurations:

* ``served``: the settings the server uses (prefix cache on, a pool that holds everything);
* ``no-prefix-cache``: the same with the prefix cache off;
* ``preempting``: a pool too small for the batch, so sequences are preempted and recomputed.

In bf16 a different batch shape can change rounding, so a flip at a near-tie is possible without a
bug; every divergence is reported with its first position and the config, and nothing is hidden.
Free-running comparison exaggerates one flip into a cascade, so the count of identical outputs is a
lower bound on agreement.

Usage (on the GPU machine):
    python -m bench.scheduler_eval --model-dir models/Llama-3.2-1B-Instruct --out results \\
        --label rtx5090-bf16 --device cuda --dtype bfloat16
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from engine.paged_cache import KVPool
from engine.scheduler import LanguageModel, Scheduler

SCHEMA_VERSION = 1
BLOCK_SIZE = 16


@dataclass(frozen=True)
class SchedulerConfig:
    name: str
    num_blocks: int
    max_batch_size: int
    prefix_caching: bool


def run_scheduler(
    model: LanguageModel,
    prompts: Sequence[Sequence[int]],
    n_tokens: int,
    config: SchedulerConfig,
    dtype: torch.dtype,
) -> tuple[list[list[int]], dict[str, Any]]:
    """Serve every prompt to ``n_tokens`` new tokens; returns the outputs and scheduler counters."""
    pool = KVPool(model.cfg, config.num_blocks, BLOCK_SIZE, dtype, model.device)
    scheduler = Scheduler(
        model,
        pool,
        max_batch_size=config.max_batch_size,
        enable_prefix_caching=config.prefix_caching,
    )
    for index, ids in enumerate(prompts):
        scheduler.add_request(index, list(ids), n_tokens)
    outputs = scheduler.run_to_completion()
    if pool.allocator.num_used != 0 and not config.prefix_caching:
        raise RuntimeError("blocks leaked")
    stats = {
        "preemptions": scheduler.total_preemptions,
        "steps": scheduler.steps,
        "mean_decode_batch": scheduler.stats().get("mean_decode_batch"),
        "prefix_hit_tokens": scheduler.prefix_hit_tokens,
    }
    return [outputs[i] for i in range(len(prompts))], stats


def first_divergence(a: Sequence[int], b: Sequence[int]) -> int | None:
    """Index of the first differing token, ``None`` if equal (a length difference counts)."""
    for index, (x, y) in enumerate(zip(a, b, strict=False)):
        if x != y:
            return index
    return None if len(a) == len(b) else min(len(a), len(b))


def compare(
    reference: Sequence[Sequence[int]], produced: Sequence[Sequence[int]]
) -> dict[str, Any]:
    """Counts of identical outputs and the first divergence of each differing one."""
    if len(reference) != len(produced):
        raise ValueError("different numbers of outputs")
    diverged = []
    for index, (ref, got) in enumerate(zip(reference, produced, strict=True)):
        position = first_divergence(ref, got)
        if position is not None:
            diverged.append({"prompt": index, "first_divergence": position})
    return {
        "prompts": len(reference),
        "identical": len(reference) - len(diverged),
        "diverged": diverged,
    }


def configs_for(
    num_prompts: int, tokens: int, max_prompt: int, max_batch: int
) -> list[SchedulerConfig]:
    """The three configurations. The small pool holds about a third of the batch at its longest."""
    per_sequence = -(-(max_prompt + tokens) // BLOCK_SIZE) + 1
    ample = per_sequence * num_prompts + 8
    small = max(per_sequence + 2, per_sequence * max_batch // 3)
    return [
        SchedulerConfig("served", ample, max_batch, True),
        SchedulerConfig("no-prefix-cache", ample, max_batch, False),
        SchedulerConfig("preempting", small, max_batch, True),
    ]


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0] if __doc__ else None)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="directory for the result file")
    parser.add_argument("--label", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="bfloat16")
    parser.add_argument("--prompts", type=int, default=50)
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-batch", type=int, default=32)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from bench.greedy_eval import (
        build_prompt_set,
        counts_for,
        hf_generate,
        load_engine,
        prompt_set_hash,
    )

    args = parse_args(argv)
    path = args.out / f"scheduler-eval-{args.label}.json"
    if path.exists():
        raise SystemExit(f"{path} exists; result files are never overwritten")
    dtype = torch.float32 if args.dtype == "float32" else torch.bfloat16
    if dtype == torch.float32:  # keep "float32" meaning float32 on GPUs that would use TF32
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    reference_model = AutoModelForCausalLM.from_pretrained(args.model_dir, dtype=dtype)
    reference_model = reference_model.to(args.device).eval()
    engine = load_engine(args.model_dir, dtype, args.device)
    pad = reference_model.generation_config.pad_token_id
    eos = reference_model.generation_config.eos_token_id
    pad_id = int(pad) if pad is not None else int(eos[0] if isinstance(eos, list) else eos or 0)

    repo_root = Path(__file__).resolve().parents[1]
    prompts = build_prompt_set(
        lambda text: tokenizer(text, add_special_tokens=False)["input_ids"],
        int(tokenizer.bos_token_id),
        repo_root,
        args.prompts,
        args.seed,
        engine.cfg.vocab_size,
    )
    ids = [p.ids for p in prompts]
    started = time.perf_counter()
    reference = [hf_generate(reference_model, p, args.tokens, args.device, pad_id) for p in ids]
    print(f"reference done in {time.perf_counter() - started:.0f}s", file=sys.stderr)
    del reference_model

    results: dict[str, Any] = {}
    with torch.inference_mode():
        for config in configs_for(len(ids), args.tokens, max(len(p) for p in ids), args.max_batch):
            produced, stats = run_scheduler(engine, ids, args.tokens, config, dtype)
            results[config.name] = {
                "config": {
                    "num_blocks": config.num_blocks,
                    "max_batch_size": config.max_batch_size,
                    "prefix_caching": config.prefix_caching,
                },
                "scheduler": stats,
                **compare(reference, produced),
            }
            r = results[config.name]
            print(
                f"{config.name}: {r['identical']}/{r['prompts']} identical, "
                f"{stats['preemptions']} preemptions",
                file=sys.stderr,
            )
    record = {
        "schema_version": SCHEMA_VERSION,
        "meta": {
            "captured_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "label": args.label,
            "model": args.model_dir.name,
            "device": torch.cuda.get_device_name(0) if args.device.startswith("cuda") else "cpu",
            "dtype": args.dtype,
            "torch": torch.__version__,
            "tokens": args.tokens,
            "block_size": BLOCK_SIZE,
            "prompt_set": {
                "seed": args.seed,
                "counts": counts_for(args.prompts),
                "sha256": prompt_set_hash(prompts),
            },
            "reference": "transformers generate, greedy, EOS disabled, same device and dtype",
        },
        "results": results,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2, sort_keys=True)
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
