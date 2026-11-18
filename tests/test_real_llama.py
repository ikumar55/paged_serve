"""Checks against the real Llama-3.2-1B-Instruct weights.

Skipped unless the files are in ``models/Llama-3.2-1B-Instruct`` (gitignored; downloaded by hand
after accepting the model license), so CI never needs them. Run with ``make test-real``.

Memory: the Hugging Face reference runs first and is freed before this engine's model loads, so
peak use is about one fp32 copy of the 1B model (5 GB) plus saved logits.
"""

from __future__ import annotations

import gc
import json
from typing import Any

import pytest
import torch
from transformers import AutoModelForCausalLM

from engine.config import LlamaConfig
from engine.generate import greedy_generate
from engine.model import Llama
from tests.real_model_support import (
    LONG_PROMPT_TOKENS,
    MODEL_DIR,
    WEIGHTS_PRESENT,
    build_prompts,
    first_diff,
    load_engine,
    long_prompt_ids,
)

pytestmark = [
    pytest.mark.real_model,
    pytest.mark.skipif(
        not WEIGHTS_PRESENT,
        reason="real weights not downloaded (see tests/test_real_llama.py)",
    ),
]

# The project target for fp32 logits (docs/decisions.md, D-010).
FP32_LOGIT_TARGET = 1e-3
GENERATED_TOKENS = 24


@pytest.fixture(scope="module")
def prompts() -> list[list[int]]:
    return build_prompts()


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
        long_ids = long_prompt_ids()
        long_logits = model(long_ids, logits_to_keep=1).logits[0, -1].clone()
    del model
    gc.collect()
    return {
        "logits": logits,
        "generated": generated,
        "long_ids": long_ids,
        "long_logits": long_logits,
    }


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
            f"prompt {index} diverges at token {first_diff(actual, expected)}"
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
