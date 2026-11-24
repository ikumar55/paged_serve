"""How far is each decode path from an fp32 reference, position by position? (D-026)

Four variants run in lockstep over the same prompts, and at every decode step every variant is
fed the same token (the fp32 gather path's own greedy token), so all of them see the same context:

* ``truth``: fp32 weights, the gather path (50 of 50 identical to Hugging Face on every prompt set
  tried, D-019, so it is the reference here);
* ``fp32-kernel``: fp32 weights, the decode-attention function under test;
* ``bf16-gather``: bf16 weights, the gather path;
* ``bf16-kernel``: bf16 weights, the decode-attention function under test.

For each non-truth variant and each position it records the maximum absolute logit difference from
``truth``, whether its argmax differs (a "flip") and ``truth``'s top-1 minus top-2 margin. A flip at
a small margin is a near-tie; a flip at a large margin is a real error. D-025's gate compared bf16
text with Hugging Face's bf16 text, which cannot say which of two noisy computations is closer to
the right answer; this does.

Usage (on a GPU machine):
    python -m bench.bf16_logit_error --model-dir models/Llama-3.2-1B-Instruct --out results \\
        --label rtx5090-frozen --prompts-from results/greedy-eval-rtx5090-bf16.json
    python -m bench.bf16_logit_error ... --label rtx5090-prefix-s0 --shared-prefix-groups 8 --seed 0
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from engine.cuda_graph import CudaGraphDecoder
from engine.paged_attention import DecodeAttention, resolve_kernel
from engine.paged_cache import KVPool, PagedBatchCache, PagedSequenceCache
from engine.scheduler import LanguageModel

SCHEMA_VERSION = 1
NEAR_TIE_MARGIN = 0.125  # truth margins below this are "near-ties" (one bf16 logit step)
VARIANTS = ("fp32-kernel", "bf16-gather", "bf16-kernel")
GRAPH_VARIANTS = ("fp32-graphs", "bf16-graphs")  # added by --graphs (D-027)


@dataclass
class Variant:
    name: str
    model: LanguageModel
    pool: KVPool
    attention: DecodeAttention | None
    caches: list[PagedSequenceCache] = field(default_factory=list)
    decoder: CudaGraphDecoder | None = None  # when set, decode steps replay CUDA graphs (D-027)


@dataclass
class Records:
    """Per-position measurements of one variant against truth, in step-major order."""

    max_error: list[float] = field(default_factory=list)
    flip: list[bool] = field(default_factory=list)
    margin: list[float] = field(default_factory=list)  # truth's top-1 minus top-2
    prompt: list[int] = field(default_factory=list)
    step: list[int] = field(default_factory=list)  # index of the generated token (first is 0)


def prefill(variant: Variant, prompts: Sequence[Sequence[int]]) -> torch.Tensor:
    """Prefill every prompt; returns the logits of each prompt's last token, ``[n, vocab]``."""
    device = variant.model.device
    rows = []
    for ids in prompts:
        cache = PagedSequenceCache(variant.pool)
        tensor = torch.tensor([list(ids)], dtype=torch.long, device=device)
        rows.append(variant.model(tensor, cache=cache, last_only=True)[0, -1].float())
        variant.caches.append(cache)
    return torch.stack(rows)


def decode_step(variant: Variant, tokens: torch.Tensor) -> torch.Tensor:
    """One batched decode step with the given next tokens; logits ``[n, vocab]``."""
    step = PagedBatchCache(variant.caches, attention=variant.attention)
    if variant.decoder is not None:
        logits = variant.decoder.run(step, tokens.unsqueeze(1))
    else:
        logits = variant.model(tokens.unsqueeze(1), positions=step.positions(), cache=step)
    return logits[:, -1].float()


def compare(
    truth: torch.Tensor, other: torch.Tensor
) -> tuple[list[float], list[bool], list[float]]:
    """Per row: max absolute logit error, flip, truth's top-1 minus top-2 margin."""
    error = (other - truth).abs().amax(dim=-1)
    flip = other.argmax(dim=-1) != truth.argmax(dim=-1)
    top = truth.topk(2, dim=-1).values
    margin = top[:, 0] - top[:, 1]
    return error.tolist(), flip.tolist(), margin.tolist()


def run_lockstep(
    truth: Variant,
    others: Sequence[Variant],
    prompts: Sequence[Sequence[int]],
    steps: int,
    log: Callable[[str], None] | None = None,
) -> dict[str, Records]:
    """Prefill, then ``steps`` batched decode steps; compare every variant with ``truth``.

    Generated token 0 comes from the prefill (the same computation for every variant of one
    dtype, the kernel is not involved), tokens 1 to ``steps`` from decode steps. Only decode
    steps are recorded, since the decode-attention function is used there.
    """
    records = {variant.name: Records() for variant in others}
    truth_logits = prefill(truth, prompts)
    for variant in others:
        prefill(variant, prompts)
    tokens = truth_logits.argmax(dim=-1)
    for step in range(1, steps + 1):
        truth_logits = decode_step(truth, tokens)
        for variant in others:
            error, flip, margin = compare(truth_logits, decode_step(variant, tokens))
            record = records[variant.name]
            record.max_error += error
            record.flip += flip
            record.margin += margin
            record.prompt += list(range(len(prompts)))
            record.step += [step] * len(prompts)
        tokens = truth_logits.argmax(dim=-1)
        if log is not None and step % 16 == 0:
            log(f"step {step}/{steps}")
    return records


def percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(fraction * len(ordered)))]


def summarise(records: Records) -> dict[str, Any]:
    flips = [i for i, f in enumerate(records.flip) if f]
    near = [i for i in flips if records.margin[i] < NEAR_TIE_MARGIN]
    return {
        "positions": len(records.max_error),
        "error_mean": statistics.fmean(records.max_error),
        "error_p50": percentile(records.max_error, 0.50),
        "error_p99": percentile(records.max_error, 0.99),
        "error_max": max(records.max_error),
        "flips": len(flips),
        "flips_near_tie": len(near),
        "flips_not_near_tie": len(flips) - len(near),
        "flip_list": [
            {
                "prompt": records.prompt[i],
                "step": records.step[i],
                "truth_margin": records.margin[i],
                "max_error": records.max_error[i],
            }
            for i in flips
        ],
    }


def combine(parts: Sequence[dict[str, Records]]) -> dict[str, Records]:
    """Pool the records of several prompt sets; prompt ids become ``set index * 1000 + id``."""
    merged: dict[str, Records] = {}
    for index, part in enumerate(parts):
        for name, record in part.items():
            target = merged.setdefault(name, Records())
            target.max_error += record.max_error
            target.flip += record.flip
            target.margin += record.margin
            target.prompt += [index * 1000 + p for p in record.prompt]
            target.step += record.step
    return merged


def load_records(path: Path) -> dict[str, Records]:
    document = json.loads(path.read_text(encoding="utf-8"))
    return {
        name: Records(
            max_error=list(raw["max_error"]),
            flip=[bool(f) for f in raw["flip"]],
            margin=list(raw["margin"]),
            prompt=list(raw["prompt"]),
            step=list(raw["step"]),
        )
        for name, raw in document["records"].items()
    }


def report(paths: Sequence[Path]) -> dict[str, Any]:
    """Pool result files; evaluate D-026's criteria (and D-027's when there are graph variants)."""
    pooled = combine([load_records(path) for path in paths])
    summaries = {name: summarise(record) for name, record in pooled.items()}
    document: dict[str, Any] = {
        "files": [path.name for path in paths],
        "summaries": summaries,
        "criteria": criteria(summaries),
    }
    if all(name in summaries for name in GRAPH_VARIANTS):
        document["criteria_graphs"] = criteria(
            summaries, candidate="bf16-graphs", fp32_variant="fp32-graphs"
        )
    return document


def criteria(
    summaries: dict[str, dict[str, Any]],
    candidate: str = "bf16-kernel",
    fp32_variant: str = "fp32-kernel",
) -> dict[str, Any]:
    """D-026's four criteria, as written there, from the summaries of the pooled records.

    ``candidate`` and ``fp32_variant`` name the bf16 variant judged against the gather path and the
    fp32 variant used for the sanity check; D-027 applies the same criteria to the graph variants.
    """
    gather, kernel, fp32 = (
        summaries["bf16-gather"],
        summaries[candidate],
        summaries[fp32_variant],
    )
    result = {
        "i_mean_error_ratio": kernel["error_mean"] / gather["error_mean"],
        "i_pass": kernel["error_mean"] <= 1.10 * gather["error_mean"],
        "ii_p99_error_ratio": kernel["error_p99"] / gather["error_p99"],
        "ii_pass": kernel["error_p99"] <= 1.25 * gather["error_p99"],
        "iii_flips_not_near_tie": {
            "kernel": kernel["flips_not_near_tie"],
            "gather": gather["flips_not_near_tie"],
        },
        "iii_pass": kernel["flips_not_near_tie"] <= 1.25 * gather["flips_not_near_tie"] + 2,
        "iv_fp32_kernel_flips": fp32["flips"],
        "iv_fp32_kernel_error_mean": fp32["error_mean"],
        "iv_pass": fp32["flips"] == 0 and fp32["error_mean"] < 1e-4,
    }
    result["all_pass"] = all(result[k] for k in ("i_pass", "ii_pass", "iii_pass", "iv_pass"))
    return result


def build_variants(
    truth_model: LanguageModel,
    bf16_model: LanguageModel,
    attention: DecodeAttention,
    num_blocks: int,
    block_size: int,
    device: str,
    graphs: bool = False,
    graph_max_batch: int = 64,
    graph_capture: bool = True,
) -> tuple[Variant, list[Variant]]:
    def pool(model: LanguageModel, dtype: torch.dtype) -> KVPool:
        return KVPool(model.cfg, num_blocks, block_size, dtype, device)

    truth = Variant("truth", truth_model, pool(truth_model, torch.float32), None)
    others = [
        Variant("fp32-kernel", truth_model, pool(truth_model, torch.float32), attention),
        Variant("bf16-gather", bf16_model, pool(bf16_model, torch.bfloat16), None),
        Variant("bf16-kernel", bf16_model, pool(bf16_model, torch.bfloat16), attention),
    ]
    if graphs:
        # Built for a larger maximum batch than the prompt set, so a step runs in a bucket with
        # padding rows: the case D-027 is about.
        for name, model, dtype in (
            ("fp32-graphs", truth_model, torch.float32),
            ("bf16-graphs", bf16_model, torch.bfloat16),
        ):
            variant = Variant(name, model, pool(model, dtype), attention)
            assert attention is not None
            variant.decoder = CudaGraphDecoder(
                model, variant.pool, attention, graph_max_batch, capture=graph_capture
            )
            others.append(variant)
    return truth, others


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0] if __doc__ else None)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="directory for the result file")
    parser.add_argument("--label", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=63)
    parser.add_argument("--prompts", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--prompts-from", type=Path, default=None)
    parser.add_argument("--shared-prefix-groups", type=int, default=0)
    parser.add_argument("--shared-prefix-len", type=int, default=1024)
    parser.add_argument("--shared-prefix-suffix", type=int, default=64)
    parser.add_argument("--kv-cache-gb", type=float, default=2.0, help="per variant pool")
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument(
        "--graphs",
        action="store_true",
        help="also run the fp32 and bf16 models through CUDA graphs (D-027; needs a GPU)",
    )
    parser.add_argument(
        "--attention",
        choices=("triton", "reference"),
        default="triton",
        help="the decode-attention function under test",
    )
    args = parser.parse_args(argv)
    if (args.prompts_from is None) == (args.shared_prefix_groups == 0):
        parser.error("give exactly one of --prompts-from and --shared-prefix-groups")
    return args


def report_main(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(prog="bench.bf16_logit_error report")
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, required=True, help="JSON file to create")
    args = parser.parse_args(argv)
    document = report(args.files)
    with args.out.open("x", encoding="utf-8") as handle:
        json.dump(document, handle, indent=2, sort_keys=True)
    for name, summary in document["summaries"].items():
        print(
            f"{name}: positions {summary['positions']}, mean err {summary['error_mean']:.4f}, "
            f"p99 {summary['error_p99']:.4f}, flips {summary['flips']} "
            f"({summary['flips_not_near_tie']} not near-ties)"
        )
    print(json.dumps(document["criteria"], indent=2, sort_keys=True))
    if "criteria_graphs" in document:
        print("graph variants:")
        print(json.dumps(document["criteria_graphs"], indent=2, sort_keys=True))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    if argv and argv[0] == "report":
        return report_main(argv[1:])
    from bench.scheduler_eval import load_recorded_prompts, shared_prefix_prompts
    from engine.weights import load_model

    args = parse_args(argv)
    path = args.out / f"bf16-logit-error-{args.label}.json"
    if path.exists():
        raise SystemExit(f"{path} exists; result files are never overwritten")
    torch.set_float32_matmul_precision("highest")  # "fp32" must mean fp32, not TF32
    torch.backends.cuda.matmul.allow_tf32 = False

    truth_model, _ = load_model(args.model_dir, torch.float32, args.device)
    bf16_model, _ = load_model(args.model_dir, torch.bfloat16, args.device)
    cfg = truth_model.cfg
    if args.prompts_from is not None:
        prompts, prompt_meta = load_recorded_prompts(args.prompts_from)
    else:
        prompts = shared_prefix_prompts(
            args.prompts,
            args.shared_prefix_groups,
            args.shared_prefix_len,
            args.shared_prefix_suffix,
            cfg.vocab_size,
            args.seed,
        )
        prompt_meta = {
            "source": "synthetic shared prefixes",
            "seed": args.seed,
            "count": args.prompts,
            "groups": args.shared_prefix_groups,
            "prefix_len": args.shared_prefix_len,
            "suffix_len": args.shared_prefix_suffix,
        }
    per_token = cfg.kv_bytes_per_token(4)  # the fp32 pools are the larger
    num_blocks = int(args.kv_cache_gb * 2**30 // (per_token * args.block_size))
    longest = max(len(p) for p in prompts) + args.steps + 1
    needed = len(prompts) * (-(-longest // args.block_size) + 1) + 1  # +1: graph scratch block
    if num_blocks < needed:
        raise SystemExit(f"--kv-cache-gb too small: {num_blocks} blocks, {needed} needed")

    attention = resolve_kernel(args.attention)
    truth, others = build_variants(
        truth_model,
        bf16_model,
        attention,
        num_blocks,
        args.block_size,
        args.device,
        graphs=args.graphs,
    )
    started = time.perf_counter()
    with torch.inference_mode():
        records = run_lockstep(
            truth,
            others,
            prompts,
            args.steps,
            log=lambda message: print(message, file=sys.stderr),
        )
    elapsed = time.perf_counter() - started
    summaries = {name: summarise(record) for name, record in records.items()}
    args.out.mkdir(parents=True, exist_ok=True)
    document = {
        "schema_version": SCHEMA_VERSION,
        "meta": {
            "captured_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "label": args.label,
            "device": torch.cuda.get_device_name(0) if args.device.startswith("cuda") else "cpu",
            "torch": torch.__version__,
            "steps": args.steps,
            "attention": args.attention,
            "near_tie_margin": NEAR_TIE_MARGIN,
            "prompt_set": prompt_meta,
            "elapsed_s": elapsed,
            "reference": "fp32 gather path, forced with its own greedy tokens",
        },
        "summaries": summaries,
        "records": {
            name: {
                "max_error": record.max_error,
                "flip": [int(f) for f in record.flip],
                "margin": record.margin,
                "prompt": record.prompt,
                "step": record.step,
            }
            for name, record in records.items()
        },
    }
    with path.open("x", encoding="utf-8") as handle:
        json.dump(document, handle, sort_keys=True)
    for name, summary in summaries.items():
        print(
            f"{name}: mean err {summary['error_mean']:.4f}, p99 {summary['error_p99']:.4f}, "
            f"flips {summary['flips']} ({summary['flips_not_near_tie']} not near-ties)",
            file=sys.stderr,
        )
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
