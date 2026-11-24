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
import hashlib
import json
import random
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from engine.cuda_graph import CudaGraphDecoder
from engine.paged_attention import DecodeAttention, resolve_kernel
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
    decode_attention: DecodeAttention | None = None,
    cuda_graphs: bool = False,
    graph_capture: bool = True,
) -> tuple[list[list[int]], dict[str, Any]]:
    """Serve every prompt to ``n_tokens`` new tokens; returns the outputs and scheduler counters."""
    # the graph decoder keeps one block as scratch; one more keeps the usable capacity equal
    pool = KVPool(
        model.cfg, config.num_blocks + (1 if cuda_graphs else 0), BLOCK_SIZE, dtype, model.device
    )
    decoder = None
    if cuda_graphs:
        if decode_attention is None:
            raise ValueError("CUDA graphs need the decode-attention kernel")
        decoder = CudaGraphDecoder(
            model, pool, decode_attention, config.max_batch_size, capture=graph_capture
        )
    scheduler = Scheduler(
        model,
        pool,
        max_batch_size=config.max_batch_size,
        enable_prefix_caching=config.prefix_caching,
        decode_attention=decode_attention,
        graph_decoder=decoder,
    )
    for index, ids in enumerate(prompts):
        scheduler.add_request(index, list(ids), n_tokens)
    outputs = scheduler.run_to_completion()
    held = 1 if decoder is not None else 0  # the graph decoder keeps one block as scratch
    if pool.allocator.num_used != held and not config.prefix_caching:
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


def shared_prefix_prompts(
    count: int, groups: int, prefix_len: int, suffix_len: int, vocab_size: int, seed: int
) -> list[list[int]]:
    """Prompts that share long prefixes, so the prefix cache is actually hit.

    ``groups`` random prefixes of ``prefix_len`` tokens; prompt ``i`` is prefix ``i % groups``
    followed by its own random suffix of ``suffix_len`` tokens. Token ids come from the same range
    as the random prompts of ``bench.greedy_eval`` (no special or very rare tokens).
    """
    rng = random.Random(seed)
    low, high = min(1000, max(1, vocab_size // 10)), min(100000, vocab_size)
    prefixes = [[rng.randrange(low, high) for _ in range(prefix_len)] for _ in range(groups)]
    return [
        prefixes[i % groups] + [rng.randrange(low, high) for _ in range(suffix_len)]
        for i in range(count)
    ]


def load_recorded_prompts(path: Path) -> tuple[list[list[int]], dict[str, Any]]:
    """The exact prompts of a recorded ``bench.greedy_eval`` result, checked against its hash.

    ``greedy_eval`` builds its prompts from this repository's own source files, so regenerating
    them after the repository changes gives a different set (a recorded hash shows it). Reading
    the recorded ids freezes the set, so two evaluations can be compared prompt for prompt.
    """
    from bench.greedy_eval import Prompt, prompt_set_hash

    record = json.loads(path.read_text(encoding="utf-8"))
    prompts = [Prompt(p["kind"], list(p["prompt_ids"])) for p in record["prompts"]]
    expected = record["meta"]["prompt_set"]["sha256"]
    actual = prompt_set_hash(prompts)
    if actual != expected:
        raise ValueError(
            f"{path.name}: prompt hash {actual[:12]} does not match recorded {expected[:12]}"
        )
    return [p.ids for p in prompts], {
        "source": path.name,
        "sha256": expected,
        **record["meta"]["prompt_set"],
    }


def logit_gap_record(
    logits: torch.Tensor, reference_token: int, scheduler_token: int
) -> dict[str, Any]:
    """How far apart the reference model's own logits put its choice and the scheduler's.

    ``logits`` are the reference's logits at the step where the two first differ. If the two
    tokens are within one unit of the logits' numeric resolution (0.125 for bf16 logits of size 16
    to 32), the difference is a tie broken differently by different rounding, not a wrong token.
    """
    values = logits.float()
    top = values.topk(2).values
    return {
        "reference_token_minus_scheduler_token_logit": float(
            values[reference_token] - values[scheduler_token]
        ),
        "reference_top1_minus_top2": float(top[0] - top[1]),
        "reference_token_is_top1": bool(int(values.argmax()) == reference_token),
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
    parser.add_argument(
        "--shared-prefix-groups",
        type=int,
        default=0,
        help="use synthetic prompts sharing this many long prefixes instead (so the prefix cache "
        "is hit); see --shared-prefix-len and --shared-prefix-suffix",
    )
    parser.add_argument("--shared-prefix-len", type=int, default=1024)
    parser.add_argument("--shared-prefix-suffix", type=int, default=64)
    parser.add_argument(
        "--prompts-from",
        type=Path,
        default=None,
        help="a recorded greedy-eval result: use its exact prompts (--prompts and --seed are "
        "then ignored); without it the prompts are regenerated from this repository's files",
    )
    parser.add_argument(
        "--paged-attention-kernel",
        action="store_true",
        help="decode attention reads the KV blocks in place with the Triton kernel (D-025); "
        "the default is the gather path",
    )
    parser.add_argument(
        "--cuda-graphs",
        action="store_true",
        help="decode steps run as CUDA graphs (D-020; needs --paged-attention-kernel)",
    )
    args = parser.parse_args(argv)
    if args.cuda_graphs and not args.paged_attention_kernel:
        parser.error("--cuda-graphs needs --paged-attention-kernel")
    return args


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
    decode_attention = resolve_kernel("triton") if args.paged_attention_kernel else None
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

    if args.prompts_from is not None and args.shared_prefix_groups:
        raise SystemExit("--prompts-from and --shared-prefix-groups are alternatives")
    if args.prompts_from is not None:
        ids, prompt_meta = load_recorded_prompts(args.prompts_from)
    elif args.shared_prefix_groups:
        ids = shared_prefix_prompts(
            args.prompts,
            args.shared_prefix_groups,
            args.shared_prefix_len,
            args.shared_prefix_suffix,
            engine.cfg.vocab_size,
            args.seed,
        )
        prompt_meta = {
            "source": "synthetic shared prefixes",
            "seed": args.seed,
            "count": args.prompts,
            "groups": args.shared_prefix_groups,
            "prefix_len": args.shared_prefix_len,
            "suffix_len": args.shared_prefix_suffix,
            "sha256": hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest(),
        }
    else:
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
        prompt_meta = {
            "source": "regenerated from this repository's files (changes when they change)",
            "seed": args.seed,
            "counts": counts_for(args.prompts),
            "sha256": prompt_set_hash(prompts),
        }
    started = time.perf_counter()
    reference = [hf_generate(reference_model, p, args.tokens, args.device, pad_id) for p in ids]
    print(f"reference done in {time.perf_counter() - started:.0f}s", file=sys.stderr)

    results: dict[str, Any] = {}
    outputs: dict[str, list[list[int]]] = {}
    with torch.inference_mode():
        for config in configs_for(len(ids), args.tokens, max(len(p) for p in ids), args.max_batch):
            produced, stats = run_scheduler(
                engine, ids, args.tokens, config, dtype, decode_attention, args.cuda_graphs
            )
            outputs[config.name] = produced
            comparison = compare(reference, produced)
            # a fingerprint of the outputs, so two runs can be compared without storing them
            comparison["outputs_sha256"] = hashlib.sha256(
                json.dumps(produced, separators=(",", ":")).encode()
            ).hexdigest()
            for entry in comparison["diverged"]:  # why did it differ? (see logit_gap_record)
                index, position = entry["prompt"], entry["first_divergence"]
                if position < min(len(reference[index]), len(produced[index])):
                    context = [*ids[index], *reference[index][:position]]
                    logits = reference_model(torch.tensor([context], device=args.device)).logits
                    entry.update(
                        logit_gap_record(
                            logits[0, -1], reference[index][position], produced[index][position]
                        )
                    )
            results[config.name] = {
                "config": {
                    "num_blocks": config.num_blocks,
                    "max_batch_size": config.max_batch_size,
                    "prefix_caching": config.prefix_caching,
                },
                "scheduler": stats,
                **comparison,
            }
            r = results[config.name]
            print(
                f"{config.name}: {r['identical']}/{r['prompts']} identical, "
                f"{stats['preemptions']} preemptions",
                file=sys.stderr,
            )
    # The prefix cache's own effect on the text: the same scheduler with and without it.
    cache_effect = compare(outputs["no-prefix-cache"], outputs["served"])
    print(
        f"prefix cache on vs off: {cache_effect['identical']}/{cache_effect['prompts']} identical, "
        f"{results['served']['scheduler']['prefix_hit_tokens']} prompt tokens served from cache",
        file=sys.stderr,
    )
    record = {
        "schema_version": SCHEMA_VERSION,
        "prefix_cache_on_vs_off": cache_effect,
        "meta": {
            "captured_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "label": args.label,
            "model": args.model_dir.name,
            "device": torch.cuda.get_device_name(0) if args.device.startswith("cuda") else "cpu",
            "dtype": args.dtype,
            "torch": torch.__version__,
            "tokens": args.tokens,
            "block_size": BLOCK_SIZE,
            "paged_attention_kernel": args.paged_attention_kernel,
            "cuda_graphs": args.cuda_graphs,
            "prompt_set": prompt_meta,
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
