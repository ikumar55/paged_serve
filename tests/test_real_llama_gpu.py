"""GPU checks against the real Llama-3.2-1B-Instruct weights. Run by hand: ``make test-gpu``.

Skipped without the weights (see ``tests/real_model_support.py``) or without a CUDA device, so CI
and the CPU-only machines never run it.

The golden reference is Hugging Face in fp32 on CPU, the same one ``test_real_llama.py`` uses.
Because a GPU legitimately differs from the CPU in the last bits (different matmul kernels and
accumulation orders), every check also reports the *floor*: how far Hugging Face's own GPU output
is from its own CPU output on the same prompts. An engine difference is only meaningful relative
to that floor, so it is printed and recorded next to each result. The floor is ordinary fp32
error, not something specific to the GPU: measured against a float64 reference, each fp32
implementation (CPU or GPU) is off by about 4e-4 at worst, so two of them differ by up to about
twice that (see docs/decisions.md, D-002, and ``tests/fp64_truth_report.py``).

TF32 is switched off for the duration so that "fp32" really means fp32 on Ampere and newer GPUs.
"""

from __future__ import annotations

import gc
from collections.abc import Iterator
from typing import Any

import pytest
import torch
from transformers import AutoModelForCausalLM

from bench.gpu_selftest import failed_checks, run_selftest
from tests.real_model_support import (
    LONG_PROMPT_TOKENS,
    MODEL_DIR,
    WEIGHTS_PRESENT,
    build_prompts,
    load_engine,
    long_prompt_ids,
)

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.real_model,
    pytest.mark.skipif(not WEIGHTS_PRESENT, reason="real weights not downloaded"),
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU"),
]

# The fp32 contract on a GPU (docs/decisions.md, D-019). The earlier fixed target of 1e-3 against
# Hugging Face on the CPU (D-010) is kept for the CPU tests, but a GPU's own fp32 output differs
# from the CPU's by an amount that depends on the card (1.19e-3 for Hugging Face itself on an
# RTX 5090), so a fixed bound against the CPU tests the card and not the engine. Two checks that
# do test the engine: it must reproduce Hugging Face on the same GPU almost exactly, and it must
# be no further from the CPU reference than Hugging Face's own GPU run is.
SAME_GPU_TOLERANCE = 1e-4
FLOOR_MARGIN = 1.05
DEVICE = "cuda"


def assert_fp32_contract(vs_cpu: float, vs_hf_gpu: float, floor: float) -> None:
    assert vs_hf_gpu <= SAME_GPU_TOLERANCE, (
        f"engine on the GPU differs from Hugging Face on the same GPU by {vs_hf_gpu:.3e} "
        f"(limit {SAME_GPU_TOLERANCE:.0e})"
    )
    assert vs_cpu <= floor * FLOOR_MARGIN, (
        f"engine is {vs_cpu:.3e} from the CPU reference, further than Hugging Face's own GPU run "
        f"({floor:.3e} x {FLOOR_MARGIN})"
    )


@pytest.fixture(scope="module", autouse=True)
def gpu_passes_selftest() -> None:
    """Refuse to measure on a GPU that returns different results for identical inputs."""
    report = run_selftest(DEVICE)
    if not report["passed"]:
        pytest.fail(
            f"GPU self-test failed ({', '.join(failed_checks(report))}): this device returns "
            "inconsistent results, so nothing measured on it can be trusted",
            pytrace=False,
        )


@pytest.fixture(scope="module", autouse=True)
def strict_fp32() -> Iterator[None]:
    """Disable TF32 so fp32 matmuls are true fp32, restoring the previous settings afterwards."""
    previous = (
        torch.get_float32_matmul_precision(),
        torch.backends.cuda.matmul.allow_tf32,
        torch.backends.cudnn.allow_tf32,
    )
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    yield
    torch.set_float32_matmul_precision(previous[0])
    torch.backends.cuda.matmul.allow_tf32 = previous[1]
    torch.backends.cudnn.allow_tf32 = previous[2]


@pytest.fixture(scope="module")
def prompts() -> list[list[int]]:
    return build_prompts()


@pytest.fixture(scope="module")
def references(prompts: list[list[int]]) -> dict[str, Any]:
    """Hugging Face logits: fp32 on CPU (golden), fp32 on GPU (the floor), bf16 on GPU."""
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.float32).eval()
    long_ids = long_prompt_ids()
    with torch.no_grad():
        cpu = [model(torch.tensor([p])).logits[0].clone() for p in prompts]
        cpu_long = model(long_ids, logits_to_keep=1).logits[0, -1].clone()

        model.to(DEVICE)
        gpu = [model(torch.tensor([p], device=DEVICE)).logits[0].cpu() for p in prompts]
        gpu_long = model(long_ids.to(DEVICE), logits_to_keep=1).logits[0, -1].cpu()

        # The checkpoint is stored in bf16, so rounding the fp32 copy back is exact.
        model.to(torch.bfloat16)
        bf16 = [model(torch.tensor([p], device=DEVICE)).logits[0].float().cpu() for p in prompts]
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return {
        "cpu": cpu,
        "cpu_long": cpu_long,
        "gpu": gpu,
        "gpu_long": gpu_long,
        "bf16": bf16,
        "long_ids": long_ids,
    }


@pytest.fixture(scope="module")
def engine_fp32(references: dict[str, Any]) -> Any:
    return load_engine(torch.float32, DEVICE)


def max_abs_diff(a: list[torch.Tensor], b: list[torch.Tensor]) -> float:
    return max((x - y).abs().max().item() for x, y in zip(a, b, strict=True))


def top1_agreement(a: list[torch.Tensor], b: list[torch.Tensor]) -> float:
    agree = sum(int((x.argmax(-1) == y.argmax(-1)).sum()) for x, y in zip(a, b, strict=True))
    return agree / sum(x.shape[0] for x in b)


def test_gpu_fp32_logits_match_cpu_reference_on_fifty_prompts(
    engine_fp32: Any, prompts: list[list[int]], references: dict[str, Any], record_property: Any
) -> None:
    with torch.no_grad():
        engine = [engine_fp32(torch.tensor([p], device=DEVICE))[0].cpu() for p in prompts]
    vs_cpu = max_abs_diff(engine, references["cpu"])
    vs_hf_gpu = max_abs_diff(engine, references["gpu"])
    floor = max_abs_diff(references["gpu"], references["cpu"])
    record_property("engine_gpu_vs_hf_cpu_max_abs_diff", vs_cpu)
    record_property("engine_gpu_vs_hf_gpu_max_abs_diff", vs_hf_gpu)
    record_property("hf_gpu_vs_hf_cpu_noise_floor", floor)
    print(
        f"\nfp32, 50 prompts, max abs logit difference:"
        f"\n  engine(GPU) vs HF(CPU) : {vs_cpu:.3e}   <- at most the floor x {FLOOR_MARGIN}"
        f"\n  engine(GPU) vs HF(GPU) : {vs_hf_gpu:.3e}   <- at most {SAME_GPU_TOLERANCE:.0e}"
        f"\n  HF(GPU)     vs HF(CPU) : {floor:.3e}   <- floor: two fp32 implementations differ"
    )
    assert_fp32_contract(vs_cpu, vs_hf_gpu, floor)


def test_gpu_fp32_logits_match_cpu_reference_on_a_long_prompt(
    engine_fp32: Any, references: dict[str, Any], record_property: Any
) -> None:
    with torch.no_grad():
        actual = engine_fp32(references["long_ids"].to(DEVICE), last_only=True)[0, -1].cpu()
    vs_cpu = (actual - references["cpu_long"]).abs().max().item()
    vs_hf_gpu = (actual - references["gpu_long"]).abs().max().item()
    floor = (references["gpu_long"] - references["cpu_long"]).abs().max().item()
    record_property("engine_gpu_vs_hf_cpu_max_abs_diff_2048_tokens", vs_cpu)
    record_property("engine_gpu_vs_hf_gpu_max_abs_diff_2048_tokens", vs_hf_gpu)
    record_property("hf_gpu_vs_hf_cpu_noise_floor_2048_tokens", floor)
    print(
        f"\nfp32, {LONG_PROMPT_TOKENS}-token prompt, max abs logit difference:"
        f"\n  engine(GPU) vs HF(CPU) : {vs_cpu:.3e}   <- at most the floor x {FLOOR_MARGIN}"
        f"\n  engine(GPU) vs HF(GPU) : {vs_hf_gpu:.3e}   <- at most {SAME_GPU_TOLERANCE:.0e}"
        f"\n  HF(GPU)     vs HF(CPU) : {floor:.3e}   <- floor: two fp32 implementations differ"
    )
    assert_fp32_contract(vs_cpu, vs_hf_gpu, floor)


def test_gpu_bf16_agreement_with_fp32_reference(
    prompts: list[list[int]], references: dict[str, Any], record_property: Any
) -> None:
    """bf16 is reported next to Hugging Face's own bf16, not held to a tight absolute bound."""
    engine = load_engine(torch.bfloat16, DEVICE)
    with torch.no_grad():
        logits = [engine(torch.tensor([p], device=DEVICE))[0].float().cpu() for p in prompts]
    ours = top1_agreement(logits, references["cpu"])
    theirs = top1_agreement(references["bf16"], references["cpu"])
    ours_diff = sum(
        (x - y).abs().mean().item() for x, y in zip(logits, references["cpu"], strict=True)
    )
    theirs_diff = sum(
        (x - y).abs().mean().item()
        for x, y in zip(references["bf16"], references["cpu"], strict=True)
    )
    count = len(prompts)
    record_property("engine_bf16_top1_agreement", ours)
    record_property("hf_bf16_top1_agreement", theirs)
    record_property("engine_bf16_mean_abs_logit_diff", ours_diff / count)
    record_property("hf_bf16_mean_abs_logit_diff", theirs_diff / count)
    print(
        f"\nbf16 on GPU vs the fp32 CPU reference (50 prompts):"
        f"\n  engine : top-1 agreement {ours:.4f}, mean |diff| {ours_diff / count:.4f}"
        f"\n  HF     : top-1 agreement {theirs:.4f}, mean |diff| {theirs_diff / count:.4f}"
    )
    assert ours > 0.8  # a loose guard against gross bugs; the measured values are recorded
