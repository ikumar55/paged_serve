"""Checks against the real Llama-3.2-1B-Instruct weights.

Skipped unless the files are in ``models/Llama-3.2-1B-Instruct`` (gitignored; downloaded by hand
after accepting the model license), so CI never needs them. Run with ``make test-real``.

Memory: the Hugging Face reference runs first and is freed before this engine's model loads, so
peak use is about one fp32 copy of the 1B model (5 GB) plus saved logits.
"""

from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Any

import pytest
import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from engine.config import LlamaConfig
from engine.generate import greedy_generate
from engine.model import Llama
from engine.weights import load_hf_state_dict

MODEL_DIR = Path(__file__).resolve().parents[1] / "models" / "Llama-3.2-1B-Instruct"

pytestmark = [
    pytest.mark.real_model,
    pytest.mark.skipif(
        not (MODEL_DIR / "model.safetensors").exists(),
        reason="real weights not downloaded (see tests/test_real_llama.py)",
    ),
]

# The project target for fp32 logits (docs/decisions.md, D-010).
FP32_LOGIT_TARGET = 1e-3
GENERATED_TOKENS = 24
LONG_PROMPT_TOKENS = 2048  # long enough for Llama-3 RoPE scaling to matter

TEXT_PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):\n    if n < 2:\n        return n\n    return",
    "Photosynthesis is the process by which plants",
    "Once upon a time, in a small village by the sea,",
    "The three laws of thermodynamics state that",
    "To make a good cup of coffee, you should first",
    "In 1969, the Apollo 11 mission",
    "SELECT name, COUNT(*) FROM orders GROUP BY",
    "A binary search tree is a data structure where",
    "Dear team, I am writing to let you know that",
]


@pytest.fixture(scope="module")
def prompts() -> list[list[int]]:
    """Fifty prompts: ten real sentences plus forty random token sequences of varying length."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    texts = [tokenizer(text)["input_ids"] for text in TEXT_PROMPTS]
    generator = torch.Generator().manual_seed(0)
    lengths = torch.randint(4, 33, (40,), generator=generator).tolist()
    random_ids = [torch.randint(1000, 100000, (n,), generator=generator).tolist() for n in lengths]
    result = texts + random_ids
    assert len(result) == 50 and max(len(p) for p in result) <= 32
    return result


@pytest.fixture(scope="module")
def reference(prompts: list[list[int]]) -> dict[str, Any]:
    """Hugging Face fp32 logits for every prompt and greedy continuations for the first eight."""
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.float32).eval()
    with torch.no_grad():
        logits = [model(torch.tensor([p])).logits[0].clone() for p in prompts]
        generated = []
        for prompt in prompts[:8]:
            ids = torch.tensor([prompt])
            out = model.generate(
                ids,
                attention_mask=torch.ones_like(ids),
                max_new_tokens=GENERATED_TOKENS,
                do_sample=False,
                temperature=None,
                top_p=None,
                eos_token_id=None,
            )
            generated.append(out[0, len(prompt) :].tolist())
        long_ids = torch.randint(
            1000, 100000, (1, LONG_PROMPT_TOKENS), generator=torch.Generator().manual_seed(1)
        )
        long_logits = model(long_ids, logits_to_keep=1).logits[0, -1].clone()
    del model
    gc.collect()
    return {
        "logits": logits,
        "generated": generated,
        "long_ids": long_ids,
        "long_logits": long_logits,
    }


def load_engine(dtype: torch.dtype) -> Llama:
    raw = json.loads((MODEL_DIR / "config.json").read_text())
    model = Llama(LlamaConfig.from_hf_dict(raw), dtype=dtype)
    load_hf_state_dict(model, load_file(MODEL_DIR / "model.safetensors"))
    return model.eval()


@pytest.fixture(scope="module")
def engine_fp32(reference: dict[str, Any]) -> Llama:
    return load_engine(torch.float32)


def test_preset_matches_the_published_config() -> None:
    raw = json.loads((MODEL_DIR / "config.json").read_text())
    assert LlamaConfig.from_hf_dict(raw) == LlamaConfig.llama_3_2_1b()


def test_fp32_logits_match_reference_on_fifty_prompts(
    engine_fp32: Llama,
    prompts: list[list[int]],
    reference: dict[str, Any],
    record_property: Any,
) -> None:
    worst = 0.0
    with torch.no_grad():
        for prompt, expected in zip(prompts, reference["logits"], strict=True):
            actual = engine_fp32(torch.tensor([prompt]))[0]
            worst = max(worst, (actual - expected).abs().max().item())
    record_property("max_abs_logit_diff_fp32", worst)
    print(f"\nmax abs logit difference over 50 prompts (fp32): {worst:.3e}")
    assert worst < FP32_LOGIT_TARGET


def test_fp32_logits_match_reference_on_a_long_prompt(
    engine_fp32: Llama, reference: dict[str, Any], record_property: Any
) -> None:
    """At 2048 tokens the scaled low-frequency rotations matter; short prompts barely see them."""
    with torch.no_grad():
        actual = engine_fp32(reference["long_ids"], last_only=True)[0, -1]
    worst = (actual - reference["long_logits"]).abs().max().item()
    record_property("max_abs_logit_diff_fp32_2048_tokens", worst)
    print(f"\nmax abs logit difference, {LONG_PROMPT_TOKENS}-token prompt (fp32): {worst:.3e}")
    assert worst < FP32_LOGIT_TARGET


def test_greedy_generation_matches_reference(
    engine_fp32: Llama, prompts: list[list[int]], reference: dict[str, Any]
) -> None:
    for index, expected in enumerate(reference["generated"]):
        actual = greedy_generate(engine_fp32, prompts[index], GENERATED_TOKENS)
        assert actual == expected, (
            f"prompt {index} diverges at token {_first_diff(actual, expected)}"
        )


def _first_diff(a: list[int], b: list[int]) -> int:
    return next(
        (i for i, (x, y) in enumerate(zip(a, b, strict=False)) if x != y), min(len(a), len(b))
    )


def test_bf16_agreement_with_fp32_reference(
    engine_fp32: Llama, prompts: list[list[int]], reference: dict[str, Any], record_property: Any
) -> None:
    """bf16 is reported, not held to a tight absolute bound (docs/decisions.md, D-010)."""
    engine = load_engine(torch.bfloat16)
    agree = total = 0
    diffs: list[float] = []
    with torch.no_grad():
        for prompt, expected in zip(prompts, reference["logits"], strict=True):
            actual = engine(torch.tensor([prompt]))[0].float()
            agree += int((actual.argmax(-1) == expected.argmax(-1)).sum())
            total += expected.shape[0]
            diffs.append((actual - expected).abs().mean().item())
    top1 = agree / total
    mean_diff = sum(diffs) / len(diffs)
    record_property("bf16_top1_agreement", top1)
    record_property("bf16_mean_abs_logit_diff", mean_diff)
    print(f"\nbf16 vs fp32 reference: top-1 agreement {top1:.4f}, mean |diff| {mean_diff:.4f}")
    assert top1 > 0.8  # a loose guard against gross bugs; the measured value is recorded
