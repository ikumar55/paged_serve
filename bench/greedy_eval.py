"""Greedy-decoding agreement between this engine and a Hugging Face reference.

Both models decode the same prompts greedily for a fixed number of tokens (end-of-sequence is
ignored, so every output has the same length). Two views are reported, because one alone misleads:

* **Free-running**: does the engine produce exactly the reference's tokens, and where does it first
  differ? This is what a user sees, but one near-tie flip changes every later token, so it
  exaggerates disagreement.
* **Teacher-forced**: both models read the *reference's* output as input and we compare their
  top-1 prediction (forward pass against forward pass) at every step. A flip is then a single
  step, not a cascade, and we can measure the reference's logit gap between its choice and the
  engine's at that step. Real bugs flip tokens with a large gap; numerical noise flips only
  near-ties. The reference's own ``generate`` can disagree with its own forward pass at exact
  ties (seen in bf16); that is reported separately as ``reference_self_mismatches``.

For any disagreeing step the reference's gap ``g`` satisfies ``g <= 2 * step_max_logit_diff``
(each logit moves by at most the step's maximum difference), so the useful number is how large the
logit differences are and how many steps are close enough to flip at all (``steps_at_risk``).

Usage:
    python -m bench.greedy_eval --model-dir models/Llama-3.2-1B-Instruct --out results \\
        --label gpu-fp32 --device cuda --dtype float32
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import sys
import time
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel

from bench.env_capture import capture_environment, git_info, run_command
from bench.gpu_selftest import failed_checks, run_selftest
from engine.config import LlamaConfig
from engine.generate import greedy_generate
from engine.model import Llama
from engine.weights import load_hf_state_dict

SCHEMA_VERSION = 1
DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16}
PROSE_FILES = ("README.md", "docs/decisions.md")
CODE_GLOBS = ("engine/*.py", "bench/*.py")
NATURAL_LENGTHS = (16, 128)  # prompt window sizes in tokens, before the BOS token
RANDOM_LENGTHS = (8, 64)
RANDOM_TOKEN_RANGE = (1000, 100000)  # skips special and very rare ids


@dataclass(frozen=True)
class Prompt:
    kind: str
    ids: list[int]


@dataclass
class PromptResult:
    index: int
    kind: str
    prompt_ids: list[int]
    reference_tokens: list[int]
    engine_tokens: list[int]
    first_divergence: int | None  # free-running; None when all tokens match
    gap_at_first_divergence: float | None  # reference logit gap at that step (same context)
    teacher_forced_disagreements: list[dict[str, Any]]
    reference_self_mismatches: int  # steps where the reference's full forward disagrees with itself
    exact_tie_steps: int  # steps where the reference's two best logits are exactly equal
    max_logit_diff: float
    ref_logit_gaps: list[float]  # top-1 minus top-2 reference logit at every step


def counts_for(total: int) -> dict[str, int]:
    """Split ``total`` prompts 40% prose, 30% code, 30% random tokens."""
    prose = round(0.4 * total)
    code = round(0.3 * total)
    return {"prose": prose, "code": code, "random": total - prose - code}


def _window(ids: Sequence[int], rng: random.Random, lengths: tuple[int, int]) -> list[int]:
    length = rng.randint(*lengths)
    if len(ids) < length:
        raise ValueError(f"corpus has {len(ids)} tokens, need at least {length}")
    start = rng.randint(0, len(ids) - length)
    return list(ids[start : start + length])


def build_prompt_set(
    encode: Callable[[str], list[int]],
    bos_id: int,
    root: Path,
    total: int,
    seed: int,
    vocab_size: int,
) -> list[Prompt]:
    """Deterministic prompts: windows of this repo's prose and code, plus random token sequences.

    The token ids are stored in the result file, so later edits to the source files cannot change
    what a recorded result was measured on.
    """
    rng = random.Random(seed)
    prose = encode("\n\n".join((root / name).read_text() for name in PROSE_FILES))
    code_paths = sorted(p for pattern in CODE_GLOBS for p in root.glob(pattern))
    code = encode("\n\n".join(p.read_text() for p in code_paths if p.name != "greedy_eval.py"))
    high = min(RANDOM_TOKEN_RANGE[1], vocab_size)
    prompts: list[Prompt] = []
    for kind, count in counts_for(total).items():
        for _ in range(count):
            if kind == "random":
                length = rng.randint(*RANDOM_LENGTHS)
                ids = [rng.randrange(RANDOM_TOKEN_RANGE[0], high) for _ in range(length)]
            else:
                ids = [bos_id, *_window(prose if kind == "prose" else code, rng, NATURAL_LENGTHS)]
            prompts.append(Prompt(kind, ids))
    return prompts


def prompt_set_hash(prompts: Sequence[Prompt]) -> str:
    payload = json.dumps([[p.kind, p.ids] for p in prompts], separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def load_engine(model_dir: Path, dtype: torch.dtype, device: torch.device | str) -> Llama:
    raw = json.loads((model_dir / "config.json").read_text())
    model = Llama(LlamaConfig.from_hf_dict(raw), dtype=dtype, device=device)
    load_hf_state_dict(model, load_file(model_dir / "model.safetensors"))
    return model.eval()


@torch.inference_mode()
def hf_generate(
    model: PreTrainedModel, ids: list[int], n_tokens: int, device: torch.device | str, pad_id: int
) -> list[int]:
    """Greedy generation that cannot stop early (see D-015 on why ``eos_token_id=None`` is not
    enough)."""
    config = copy.deepcopy(model.generation_config)
    config.do_sample = False
    config.temperature = None  # sampling settings shipped with chat models are irrelevant
    config.top_p = None
    config.top_k = None
    config.max_new_tokens = n_tokens
    config.eos_token_id = [int(model.config.vocab_size)]  # an id that is never produced
    config.pad_token_id = pad_id
    batch = torch.tensor([ids], device=device)
    out = cast("Any", model).generate(
        input_ids=batch, attention_mask=torch.ones_like(batch), generation_config=config
    )
    new: list[int] = out[0, len(ids) :].tolist()
    if len(new) != n_tokens:
        raise RuntimeError(f"reference produced {len(new)} tokens, expected {n_tokens}")
    return new


@torch.inference_mode()
def evaluate_prompt(
    index: int,
    prompt: Prompt,
    reference: PreTrainedModel,
    engine: Llama,
    n_tokens: int,
    ref_device: torch.device | str,
    pad_id: int,
) -> PromptResult:
    ref_tokens = hf_generate(reference, prompt.ids, n_tokens, ref_device, pad_id)
    eng_tokens = greedy_generate(engine, prompt.ids, n_tokens)

    # Teacher forcing: both models read prompt + the reference's output. The logits at position
    # p - 1 + i predict reference token i, with an identical context for both models.
    p = len(prompt.ids)
    seq = prompt.ids + ref_tokens
    window = slice(p - 1, p - 1 + n_tokens)
    ref_logits = reference(torch.tensor([seq], device=ref_device)).logits[0, window].float()
    eng_logits = (
        engine(torch.tensor([seq], device=engine.device))[0, window].float().to(ref_logits.device)
    )

    ref_t = torch.tensor(ref_tokens, device=ref_logits.device)
    ref_arg = ref_logits.argmax(-1)
    eng_arg = eng_logits.argmax(-1)
    top2 = ref_logits.topk(2, dim=-1).values
    gaps = top2[:, 0] - top2[:, 1]
    step_diff = (eng_logits - ref_logits).abs().amax(-1)

    # Forward pass against forward pass, so only implementation differences are counted. Whether
    # the reference's own generate (KV cache) agrees with its own full forward is a separate
    # number (reference_self_mismatches); in bf16 it differs at exact ties.
    disagreements = []
    for step in (eng_arg != ref_arg).nonzero().flatten().tolist():
        chosen = int(eng_arg[step])
        preferred = int(ref_arg[step])
        disagreements.append(
            {
                "step": step,
                "reference_token": preferred,
                "engine_token": chosen,
                "reference_logit_gap": float(
                    ref_logits[step, preferred] - ref_logits[step, chosen]
                ),
                "step_max_logit_diff": float(step_diff[step]),
            }
        )

    divergence = next(
        (i for i, (a, b) in enumerate(zip(ref_tokens, eng_tokens, strict=True)) if a != b), None
    )
    gap_at_divergence = None
    if divergence is not None:
        gap_at_divergence = float(
            ref_logits[divergence, ref_tokens[divergence]]
            - ref_logits[divergence, eng_tokens[divergence]]
        )
    return PromptResult(
        index=index,
        kind=prompt.kind,
        prompt_ids=prompt.ids,
        reference_tokens=ref_tokens,
        engine_tokens=eng_tokens,
        first_divergence=divergence,
        gap_at_first_divergence=gap_at_divergence,
        teacher_forced_disagreements=disagreements,
        reference_self_mismatches=int((ref_arg != ref_t).sum()),
        exact_tie_steps=int((gaps == 0).sum()),
        max_logit_diff=float(step_diff.max()),
        ref_logit_gaps=[round(g, 4) for g in gaps.tolist()],
    )


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a proportion."""
    if total == 0:
        return (0.0, 1.0)
    p = successes / total
    denom = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denom
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def summarize(results: Sequence[PromptResult], n_tokens: int) -> dict[str, Any]:
    total = len(results)
    exact = sum(r.first_divergence is None for r in results)
    low, high = wilson_interval(exact, total)
    by_kind: dict[str, list[PromptResult]] = defaultdict(list)
    for r in results:
        by_kind[r.kind].append(r)
    prefix_fractions = [
        (n_tokens if r.first_divergence is None else r.first_divergence) / n_tokens for r in results
    ]
    steps = total * n_tokens
    disagreements = [d for r in results for d in r.teacher_forced_disagreements]
    max_diff = max((r.max_logit_diff for r in results), default=0.0)
    at_risk = sum(g <= 2 * max_diff for r in results for g in r.ref_logit_gaps)
    return {
        "prompts": total,
        "tokens_per_prompt": n_tokens,
        "free_running": {
            "exact_match": exact,
            "exact_match_rate": exact / total if total else None,
            "exact_match_rate_ci95": [round(low, 4), round(high, 4)],
            "mean_matched_prefix_fraction": sum(prefix_fractions) / total if total else None,
            "by_kind": {
                kind: {
                    "prompts": len(group),
                    "exact_match": sum(r.first_divergence is None for r in group),
                }
                for kind, group in sorted(by_kind.items())
            },
            "first_divergence_steps": sorted(
                r.first_divergence for r in results if r.first_divergence is not None
            ),
        },
        "teacher_forced": {
            "steps": steps,
            "top1_disagreements": len(disagreements),
            "top1_agreement": 1 - len(disagreements) / steps if steps else None,
            "largest_gap_among_disagreements": max(
                (d["reference_logit_gap"] for d in disagreements), default=None
            ),
            "steps_at_risk": at_risk,
            "at_risk_definition": "reference top-1/top-2 gap <= 2 * max_logit_diff",
            "exact_tie_steps": sum(r.exact_tie_steps for r in results),
        },
        "max_logit_diff": max_diff,
        "mean_per_prompt_max_logit_diff": (
            sum(r.max_logit_diff for r in results) / total if total else None
        ),
        "reference_self_mismatches": sum(r.reference_self_mismatches for r in results),
    }


def format_summary(summary: dict[str, Any]) -> str:
    free = summary["free_running"]
    forced = summary["teacher_forced"]
    low, high = free["exact_match_rate_ci95"]
    lines = [
        f"prompts: {summary['prompts']} x {summary['tokens_per_prompt']} tokens",
        f"free-running exact match: {free['exact_match']}/{summary['prompts']} "
        f"({free['exact_match_rate']:.1%}, 95% CI {low:.1%} to {high:.1%}); "
        f"mean matched prefix {free['mean_matched_prefix_fraction']:.1%}",
    ]
    for kind, stats in free["by_kind"].items():
        lines.append(f"  {kind}: {stats['exact_match']}/{stats['prompts']}")
    lines.append(f"first divergences at steps: {free['first_divergence_steps']}")
    lines.append(
        f"teacher-forced top-1: {forced['steps'] - forced['top1_disagreements']}/{forced['steps']} "
        f"agree ({forced['top1_agreement']:.4%}); disagreements: {forced['top1_disagreements']}; "
        f"steps at risk of flipping: {forced['steps_at_risk']}"
    )
    lines.append(
        f"max logit diff over all steps: {summary['max_logit_diff']:.3e}; "
        f"largest reference gap among disagreements: {forced['largest_gap_among_disagreements']}; "
        f"reference self-mismatches (its generate vs its own forward): "
        f"{summary['reference_self_mismatches']}; steps with an exact top-2 tie: "
        f"{forced['exact_tie_steps']}"
    )
    return "\n".join(lines)


def redact(text: str) -> str:
    return text.replace(str(Path.home()), "~")


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0] if __doc__ else None)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="directory for the result file")
    parser.add_argument("--label", required=True, help="names the result file")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--ref-device", default=None, help="defaults to --device")
    parser.add_argument("--dtype", choices=DTYPES, default="float32", help="engine dtype")
    parser.add_argument("--ref-dtype", choices=DTYPES, default=None, help="defaults to --dtype")
    parser.add_argument("--prompts", type=int, default=200)
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--skip-selftest",
        action="store_true",
        help="measure even if the GPU self-test fails (the result file records the skip)",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    cli_args = list(sys.argv[1:] if argv is None else argv)
    ref_device = args.ref_device or args.device
    dtype, ref_dtype = DTYPES[args.dtype], DTYPES[args.ref_dtype or args.dtype]
    path = args.out / f"greedy-eval-{args.label}.json"
    if path.exists():
        raise SystemExit(f"{path} exists; result files are never overwritten")

    # Keep "float32" meaning float32 on GPUs that would otherwise use TF32 for matmuls.
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    # A GPU that returns different results for identical inputs invalidates everything below, so
    # check before spending time on the run, and record the outcome in the result file.
    selftest: dict[str, Any] | None = None
    devices = {args.device, ref_device}
    if any(torch.device(d).type == "cuda" for d in devices):
        cuda_device = next(d for d in devices if torch.device(d).type == "cuda")
        if args.skip_selftest:
            selftest = {"skipped": True}
        else:
            selftest = run_selftest(cuda_device)
            if not selftest["passed"]:
                raise SystemExit(
                    f"GPU self-test failed ({', '.join(failed_checks(selftest))}); this device "
                    "returns inconsistent results, so nothing measured on it can be trusted. "
                    "Use another GPU (or --skip-selftest to record that you chose not to check)."
                )

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    reference = AutoModelForCausalLM.from_pretrained(args.model_dir, dtype=ref_dtype)
    reference = reference.to(ref_device).eval()
    engine = load_engine(args.model_dir, dtype, args.device)
    vocab_size = engine.cfg.vocab_size
    pad = reference.generation_config.pad_token_id
    eos = reference.generation_config.eos_token_id
    pad_id = int(pad) if pad is not None else int(eos[0] if isinstance(eos, list) else eos or 0)

    repo_root = Path(__file__).resolve().parents[1]
    prompts = build_prompt_set(
        lambda text: tokenizer(text, add_special_tokens=False)["input_ids"],
        int(tokenizer.bos_token_id),
        repo_root,
        args.prompts,
        args.seed,
        vocab_size,
    )

    results: list[PromptResult] = []
    started = time.perf_counter()
    for index, prompt in enumerate(prompts):
        results.append(
            evaluate_prompt(index, prompt, reference, engine, args.tokens, ref_device, pad_id)
        )
        if (index + 1) % 20 == 0 or index + 1 == len(prompts):
            print(
                f"[{index + 1}/{len(prompts)}] {time.perf_counter() - started:.0f}s",
                file=sys.stderr,
            )

    summary = summarize(results, args.tokens)
    record = {
        "schema_version": SCHEMA_VERSION,
        "meta": {
            "captured_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "command": redact(" ".join(["python", "-m", "bench.greedy_eval", *cli_args])),
            "label": args.label,
            "model": args.model_dir.name,
            "engine": {"device_type": torch.device(args.device).type, "dtype": args.dtype},
            "reference": {
                "implementation": "transformers generate, greedy, EOS disabled",
                "device_type": torch.device(ref_device).type,
                "dtype": args.ref_dtype or args.dtype,
            },
            "tf32_matmul_allowed": False,
            "gpu_selftest": selftest,
            "prompt_set": {
                "seed": args.seed,
                "counts": counts_for(args.prompts),
                "sha256": prompt_set_hash(prompts),
            },
            "git": git_info(run_command),
            "environment": capture_environment(args=cli_args),
        },
        "summary": summary,
        "prompts": [asdict(r) for r in results],
    }
    args.out.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:  # "x": fail rather than overwrite an existing result
        handle.write(json.dumps(record, sort_keys=True) + "\n")
    print(format_summary(summary))
    print(f"wrote {redact(str(path))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
