"""bf16 accuracy on real text: KL divergence and perplexity against Hugging Face (D-033).

Earlier accuracy checks (D-026, D-027) compared the largest absolute logit difference, a number
dominated by bf16 rounding of single logits, against the engine's own fp32 path, on random or
generated tokens. This tool asks the question that matters for a language model: how far is the
whole next-token distribution of each path from an independent fp32 reference, and how likely does
each path find the real next token, on natural text?

Real text (the project's own documents and code) is cut into non-overlapping windows. Each window
is a prompt, prefilled normally, followed by teacher-forced decode steps that feed the real text one
token at a time, so every decode path (gather, Triton kernel, CUDA graphs) runs on real contexts.
For each variant and position it records the KL divergence KL(truth || variant) over the whole
vocabulary, the negative log-likelihood of the real next token (so perplexity) and top-1 agreement
with the truth. Hugging Face's full forward pass in fp32 is the truth; Hugging Face's own bf16 run
is the yardstick for ordinary bf16 noise. Confidence intervals come from a bootstrap over windows.

Usage (on a GPU machine):
    python -m bench.kl_perplexity --model-dir models/Llama-3.2-1B-Instruct --out results \\
        --label rtx5090
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt
import torch

from bench.bf16_logit_error import Variant, build_variants, decode_step, graph_statistics, prefill
from engine.paged_attention import DecodeAttention, resolve_kernel

SCHEMA_VERSION = 1
REPO_ROOT = Path(__file__).resolve().parents[1]
# The fixed text, in this order (D-033, amendment 1): prose, tables and code, about 270 KB.
DEFAULT_TEXT_FILES = (
    "docs/decisions.md",
    "docs/STATUS.md",
    "README.md",
    "deploy/README.md",
    "engine/scheduler.py",
    "engine/model.py",
    "router/app.py",
)
# The extended set, used only for the 400-window widening (amendment 2): 390 KB, 404 windows.
EXTENDED_TEXT_FILES = (
    *DEFAULT_TEXT_FILES,
    "docs/independent-review-2026-11-24.md",
    "docs/second-review-2026-11-24.md",
    "docs/gpu-runbook.md",
    "bench/loadgen.py",
    "bench/sweep.py",
    "bench/runner.py",
    "bench/scheduler_eval.py",
)
Array = npt.NDArray[np.float64]

# Variant names. "hf-fp32" is the truth; the engine's own paths are measured against it.
HF_TRUTH, HF_BF16 = "hf-fp32", "hf-bf16"
ENGINE_NAMES = {  # names used by bench.bf16_logit_error.build_variants -> names here
    "truth": "engine-fp32-gather",
    "fp32-kernel": "engine-fp32-kernel",
    "bf16-gather": "engine-bf16-gather",
    "bf16-kernel": "engine-bf16-kernel",
    "fp32-graphs": "engine-fp32-graphs",
    "bf16-graphs": "engine-bf16-graphs",
}
GATHER, KERNEL, GRAPHS = "engine-bf16-gather", "engine-bf16-kernel", "engine-bf16-graphs"
# D-033's limits L and confirmation bounds B (amendment 1).
LIMIT_KL_RATIO, BOUND_KL_RATIO = 1.10, 1.25  # (i)
LIMIT_PPL_DIFF, BOUND_PPL_DIFF = 0.002, 0.003  # (ii)
LIMIT_HF_RATIO, BOUND_HF_RATIO = 2.0, 2.5  # (iii)
LIMIT_FP32_KL, BOUND_FP32_KL = 1e-6, 2e-6  # (iv)
LIMIT_FP32_PPL, BOUND_FP32_PPL = 1e-4, 2e-4  # (iv)


# -- the text and its windows ------------------------------------------------------------------


def load_text(root: Path, files: Sequence[str]) -> tuple[str, str]:
    """The concatenated text (line endings normalised to LF) and the SHA-256 of its bytes."""
    parts = []
    for name in files:
        parts.append((root / name).read_bytes().replace(b"\r\n", b"\n").decode("utf-8"))
    text = "\n\n".join(parts)
    return text, hashlib.sha256(text.encode("utf-8")).hexdigest()


def make_windows(
    stream: Sequence[int], bos: int, count: int, window_len: int, seed: int
) -> list[list[int]]:
    """``count`` non-overlapping windows: the start token followed by an aligned slice of the text.

    The stream is cut into slices of ``window_len - 1`` tokens and ``count`` of them are sampled
    without replacement with a fixed seed, so no two windows share a token (the bootstrap over
    windows assumes independence).
    """
    slice_len = window_len - 1
    available = len(stream) // slice_len
    if available < count:
        raise ValueError(
            f"{len(stream)} tokens give {available} windows of {window_len}, not {count}"
        )
    picks = sorted(random.Random(seed).sample(range(available), count))
    return [[bos, *stream[i * slice_len : (i + 1) * slice_len]] for i in picks]


# -- per-position metrics --------------------------------------------------------------------------


@dataclass
class VariantMetrics:
    """Per-window means over the decode positions of one variant, plus every position's KL."""

    kl: list[Array] = field(default_factory=list)  # per chunk, [windows, steps]
    nll: list[Array] = field(default_factory=list)
    top1: list[Array] = field(default_factory=list)

    def add(self, kl: torch.Tensor, nll: torch.Tensor, top1: torch.Tensor) -> None:
        self.kl.append(kl.double().cpu().numpy())
        self.nll.append(nll.double().cpu().numpy())
        self.top1.append(top1.double().cpu().numpy())

    def arrays(self) -> tuple[Array, Array, Array]:
        return (
            np.concatenate(self.kl, axis=0),
            np.concatenate(self.nll, axis=0),
            np.concatenate(self.top1, axis=0),
        )


def position_metrics(
    truth_lp: torch.Tensor, variant_lp: torch.Tensor, target: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """KL(truth || variant), the variant NLL of ``target`` and its top-1 agreement with the truth.

    ``truth_lp`` and ``variant_lp`` are log-probabilities over the vocabulary ``[..., vocab]`` in
    float32; ``target`` is ``[...]``.
    """
    kl = (truth_lp.exp() * (truth_lp - variant_lp)).sum(dim=-1)
    nll = -variant_lp.gather(-1, target.unsqueeze(-1)).squeeze(-1)
    top1 = variant_lp.argmax(dim=-1) == truth_lp.argmax(dim=-1)
    return kl, nll, top1


def sequence_metrics(
    truth_lp: torch.Tensor, variant_lp: torch.Tensor, target: torch.Tensor, step_block: int = 16
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``position_metrics`` over ``[windows, steps, vocab]``, a few steps at a time (memory)."""
    parts = [
        position_metrics(
            truth_lp[:, i : i + step_block],
            variant_lp[:, i : i + step_block],
            target[:, i : i + step_block],
        )
        for i in range(0, truth_lp.shape[1], step_block)
    ]
    return tuple(torch.cat([p[k] for p in parts], dim=1) for k in range(3))  # type: ignore[return-value]


class Reference(Protocol):
    """Hugging Face's forward passes: log-probabilities where the decode steps predict."""

    def __call__(self, windows: torch.Tensor, prompt_len: int) -> dict[str, torch.Tensor]:
        """``{"truth": [B, steps, V], "hf-bf16": [B, steps, V]}`` float32 log-probabilities."""


class HuggingFaceReference:
    """The truth (fp32) and the yardstick (bf16), from two Hugging Face causal language models."""

    def __init__(self, fp32_model: Any, bf16_model: Any) -> None:
        self.models = {"truth": fp32_model, HF_BF16: bf16_model}

    def __call__(self, windows: torch.Tensor, prompt_len: int) -> dict[str, torch.Tensor]:
        steps = windows.shape[1] - 1 - prompt_len
        keep = windows.shape[1] - prompt_len  # the positions prompt_len .. end
        out = {}
        for name, model in self.models.items():
            logits = model(windows, logits_to_keep=keep).logits[:, :steps]
            out[name] = torch.log_softmax(logits.float(), dim=-1)
            del logits
        return out


# -- running ---------------------------------------------------------------------------------


def reset(variant: Variant) -> None:
    for cache in variant.caches:
        cache.release()
    variant.caches.clear()


def run_chunk(
    windows: torch.Tensor,
    reference: Reference,
    variants: Sequence[Variant],
    prompt_len: int,
    metrics: dict[str, VariantMetrics],
) -> None:
    """Add one chunk of windows ``[B, T]`` (token ids on the device) to ``metrics``."""
    steps = windows.shape[1] - 1 - prompt_len
    targets = windows[:, prompt_len + 1 : prompt_len + 1 + steps]
    refs = reference(windows, prompt_len)
    truth = refs["truth"]
    metrics.setdefault(HF_TRUTH, VariantMetrics()).add(*sequence_metrics(truth, truth, targets))
    metrics.setdefault(HF_BF16, VariantMetrics()).add(
        *sequence_metrics(truth, refs[HF_BF16], targets)
    )
    del refs

    prompts = windows[:, :prompt_len].tolist()
    for variant in variants:
        prefill(variant, prompts)  # the variants are empty: each chunk resets them when it ends
    per_variant: dict[str, list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]] = {
        v.name: [] for v in variants
    }
    for step in range(steps):
        tokens = windows[:, prompt_len + step]  # real text, fed one token at a time
        for variant in variants:
            logprobs = torch.log_softmax(decode_step(variant, tokens).float(), dim=-1)
            per_variant[variant.name].append(
                position_metrics(truth[:, step], logprobs, targets[:, step])
            )
    for variant in variants:
        rows = per_variant[variant.name]
        stacked = tuple(torch.stack([r[k] for r in rows], dim=1) for k in range(3))
        metrics.setdefault(ENGINE_NAMES[variant.name], VariantMetrics()).add(*stacked)
    for variant in variants:
        reset(variant)


# -- statistics ------------------------------------------------------------------------------


@dataclass(frozen=True)
class WindowStats:
    """One variant's per-window means (the unit of independence for the bootstrap)."""

    kl: Array  # [windows] mean KL over the window's decode positions
    nll: Array  # [windows] mean negative log-likelihood
    top1: Array  # [windows] fraction of positions agreeing with the truth's argmax
    kl_positions: Array  # every position's KL, flattened (for the quantiles)


def window_stats(metrics: VariantMetrics) -> WindowStats:
    kl, nll, top1 = metrics.arrays()
    return WindowStats(kl.mean(axis=1), nll.mean(axis=1), top1.mean(axis=1), kl.reshape(-1))


def summarise(stats: WindowStats) -> dict[str, float]:
    return {
        "windows": float(len(stats.kl)),
        "positions": float(len(stats.kl_positions)),
        "mean_kl": float(stats.kl.mean()),
        "kl_p50": float(np.percentile(stats.kl_positions, 50)),
        "kl_p99": float(np.percentile(stats.kl_positions, 99)),
        "mean_nll": float(stats.nll.mean()),
        "perplexity": float(np.exp(stats.nll.mean())),
        "top1_agreement": float(stats.top1.mean()),
    }


def bootstrap_indices(windows: int, resamples: int, seed: int) -> npt.NDArray[np.int64]:
    """The same resampled window indices for every variant, so that ratios are paired."""
    return np.random.default_rng(seed).integers(0, windows, size=(resamples, windows))


def interval(values: Array) -> tuple[float, float]:
    low, high = np.percentile(values, [2.5, 97.5])
    return float(low), float(high)


def verdict(point: float, upper: float, limit: float, bound: float) -> str:
    """D-033's three-state rule (amendment 1): met, inconclusive or failed."""
    if point > limit:
        return "failed"
    return "met" if upper <= bound else "inconclusive"


def ratio_criterion(
    numerator: np.ndarray[Any, Any],
    denominator: np.ndarray[Any, Any],
    index: npt.NDArray[np.int64],
    limit: float,
    bound: float,
) -> dict[str, Any]:
    point = float(numerator.mean() / denominator.mean())
    boot = numerator[index].mean(axis=1) / denominator[index].mean(axis=1)
    low, high = interval(boot)
    return {
        "value": point, "ci": [low, high], "limit": limit, "bound": bound,
        "verdict": verdict(point, high, limit, bound),
    }  # fmt: skip


def perplexity_criterion(
    nll: np.ndarray[Any, Any],
    reference_nll: np.ndarray[Any, Any],
    index: npt.NDArray[np.int64],
    limit: float,
    bound: float,
) -> dict[str, Any]:
    """Absolute relative perplexity difference |exp(mean nll - mean reference nll) - 1|."""
    signed = float(np.exp(nll.mean() - reference_nll.mean()) - 1.0)
    boot = np.exp(nll[index].mean(axis=1) - reference_nll[index].mean(axis=1)) - 1.0
    low, high = interval(boot)
    upper = max(abs(low), abs(high))
    return {
        "value": abs(signed), "signed": signed, "ci": [low, high], "limit": limit, "bound": bound,
        "verdict": verdict(abs(signed), upper, limit, bound),
    }  # fmt: skip


def mean_criterion(
    values: np.ndarray[Any, Any], index: npt.NDArray[np.int64], limit: float, bound: float
) -> dict[str, Any]:
    point = float(values.mean())
    low, high = interval(values[index].mean(axis=1))
    return {
        "value": point, "ci": [low, high], "limit": limit, "bound": bound,
        "verdict": verdict(point, high, limit, bound),
    }  # fmt: skip


def evaluate(stats: dict[str, WindowStats], resamples: int = 2000, seed: int = 0) -> dict[str, Any]:
    """D-033's criteria (i) to (iv) with bootstrap intervals; absent variants are skipped."""
    windows = len(stats[HF_TRUTH].kl)
    index = bootstrap_indices(windows, resamples, seed)
    result: dict[str, Any] = {"i": {}, "ii": {}, "iii": {}, "iv": {}}
    for name in (KERNEL, GRAPHS):
        if name in stats:
            result["i"][name] = ratio_criterion(
                stats[name].kl, stats[GATHER].kl, index, LIMIT_KL_RATIO, BOUND_KL_RATIO
            )
            result["ii"][name] = perplexity_criterion(
                stats[name].nll, stats[GATHER].nll, index, LIMIT_PPL_DIFF, BOUND_PPL_DIFF
            )
    for name in (GATHER, KERNEL, GRAPHS):
        if name in stats:
            result["iii"][name] = ratio_criterion(
                stats[name].kl, stats[HF_BF16].kl, index, LIMIT_HF_RATIO, BOUND_HF_RATIO
            )
    for name in ("engine-fp32-gather", "engine-fp32-kernel"):
        if name in stats:
            result["iv"][name] = {
                "mean_kl": mean_criterion(stats[name].kl, index, LIMIT_FP32_KL, BOUND_FP32_KL),
                "perplexity": perplexity_criterion(
                    stats[name].nll, stats[HF_TRUTH].nll, index, LIMIT_FP32_PPL, BOUND_FP32_PPL
                ),
            }
    verdicts = [c["verdict"] for group in ("i", "ii", "iii") for c in result[group].values()] + [
        part["verdict"] for c in result["iv"].values() for part in c.values()
    ]
    result["overall"] = (
        "failed"
        if "failed" in verdicts
        else "inconclusive"
        if "inconclusive" in verdicts
        else "met"
    )
    result["counts"] = {v: verdicts.count(v) for v in ("met", "inconclusive", "failed")}
    return result


# -- command line ----------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0] if __doc__ else None)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="directory for the result file")
    parser.add_argument("--label", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--text-set",
        choices=("default", "extended"),
        default="default",
        help="default: the 7 files of D-033 (200 windows); extended: 14 files (400 windows)",
    )
    parser.add_argument("--windows", type=int, default=200)
    parser.add_argument("--window-len", type=int, default=256)
    parser.add_argument("--prompt-len", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--chunk", type=int, default=25, help="windows run together")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--attention", choices=("triton", "reference"), default="triton")
    parser.add_argument("--no-graphs", action="store_true", help="skip the CUDA-graph variants")
    args = parser.parse_args(argv)
    if not 0 < args.prompt_len < args.window_len - 1:
        parser.error("--prompt-len must be positive and leave at least two tokens to predict")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from engine.weights import load_model

    args = parse_args(argv)
    path = args.out / f"kl-perplexity-{args.label}.json"
    if path.exists():
        raise SystemExit(f"{path} exists; result files are never overwritten")
    torch.set_float32_matmul_precision("highest")  # "fp32" must mean fp32, not TF32
    torch.backends.cuda.matmul.allow_tf32 = False

    text_files = EXTENDED_TEXT_FILES if args.text_set == "extended" else DEFAULT_TEXT_FILES
    text, text_hash = load_text(REPO_ROOT, text_files)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    stream = tokenizer(text, add_special_tokens=False)["input_ids"]
    windows = make_windows(
        stream, int(tokenizer.bos_token_id), args.windows, args.window_len, args.seed
    )
    all_windows = torch.tensor(windows, dtype=torch.long, device=args.device)

    hf_fp32 = AutoModelForCausalLM.from_pretrained(args.model_dir, dtype=torch.float32)
    hf_bf16 = AutoModelForCausalLM.from_pretrained(args.model_dir, dtype=torch.bfloat16)
    hf_fp32, hf_bf16 = hf_fp32.to(args.device).eval(), hf_bf16.to(args.device).eval()
    engine_fp32, _ = load_model(args.model_dir, torch.float32, args.device)
    engine_bf16, _ = load_model(args.model_dir, torch.bfloat16, args.device)
    attention: DecodeAttention = resolve_kernel(args.attention)
    per_sequence = -(-args.window_len // args.block_size) + 1
    num_blocks = args.chunk * per_sequence + 4
    gather_fp32, variants = build_variants(  # its first variant is the engine's fp32 gather path
        engine_fp32, engine_bf16, attention, num_blocks, args.block_size, args.device,
        graphs=not args.no_graphs, graph_max_batch=max(64, args.chunk),
    )  # fmt: skip
    measured = [gather_fp32, *variants]

    reference = HuggingFaceReference(hf_fp32, hf_bf16)
    metrics: dict[str, VariantMetrics] = {}
    started = time.perf_counter()
    with torch.inference_mode():
        for start in range(0, len(windows), args.chunk):
            run_chunk(
                all_windows[start : start + args.chunk],
                reference,
                measured,
                args.prompt_len,
                metrics,
            )
            print(
                f"windows {min(start + args.chunk, len(windows))}/{len(windows)}", file=sys.stderr
            )
    elapsed = time.perf_counter() - started

    stats = {name: window_stats(m) for name, m in metrics.items()}
    evaluation = evaluate(stats, args.bootstrap, args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    document: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "meta": {
            "captured_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "label": args.label,
            "device": torch.cuda.get_device_name(0) if args.device.startswith("cuda") else "cpu",
            "torch": torch.__version__,
            "text_set": args.text_set,
            "text_files": list(text_files),
            "text_sha256": text_hash,
            "text_tokens": len(stream),
            "windows": args.windows,
            "window_len": args.window_len,
            "prompt_len": args.prompt_len,
            "decode_steps_per_window": args.window_len - 1 - args.prompt_len,
            "seed": args.seed,
            "bootstrap_resamples": args.bootstrap,
            "attention": args.attention,
            "elapsed_s": elapsed,
            "truth": "Hugging Face, fp32, full forward pass, TF32 off",
        },
        "summaries": {name: summarise(s) for name, s in stats.items()},
        "criteria": evaluation,
        "graph_statistics": graph_statistics(measured),
        "per_window": {
            name: {"kl": s.kl.tolist(), "nll": s.nll.tolist(), "top1": s.top1.tolist()}
            for name, s in stats.items()
        },
    }
    with path.open("x", encoding="utf-8") as handle:
        json.dump(document, handle, sort_keys=True)
    for name, summary in document["summaries"].items():
        print(
            f"{name}: mean KL {summary['mean_kl']:.3e}, perplexity {summary['perplexity']:.4f}, "
            f"top-1 {summary['top1_agreement']:.4f}",
            file=sys.stderr,
        )
    print(f"overall: {evaluation['overall']} {evaluation['counts']}", file=sys.stderr)
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
