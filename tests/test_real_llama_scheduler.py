"""The paged cache, batched decode and scheduler against Hugging Face, on the real weights (CPU).

Skipped unless the weights are in ``models/Llama-3.2-1B-Instruct`` (see ``test_real_llama.py``).
Run with ``make test-real``.

Ground truth is Hugging Face's own ``generate`` in fp32 on the CPU, not this engine's single-request
path. fp32 on the CPU is where the engine's logits were bit-identical to Hugging Face's, so token
equality is the expected result here; a mismatch is reported with its first position. This is a
sample (a subset of the 50 shared prompts, a few dozen tokens each), not the 200-prompt
evaluation of ``bench/greedy_eval.py``.

Memory: the Hugging Face model is freed before this engine's weights load, so the peak is about one
fp32 copy of the model (5 GB) plus the KV pool (64 KB per token in fp32).
"""

from __future__ import annotations

import gc

import pytest
import torch
from transformers import AutoModelForCausalLM

from bench.greedy_eval import hf_generate
from engine.model import Llama
from engine.paged_cache import KVPool
from engine.scheduler import Scheduler
from tests.real_model_support import (
    MODEL_DIR,
    WEIGHTS_PRESENT,
    build_prompts,
    first_diff,
    load_engine,
)

pytestmark = [
    pytest.mark.real_model,
    pytest.mark.skipif(
        not WEIGHTS_PRESENT,
        reason="real weights not downloaded (see tests/test_real_llama.py)",
    ),
]

NEW_TOKENS = 24
PAD_ID = 128001  # irrelevant for one unpadded prompt, but generate wants one
BLOCK_SIZE = 16


def chosen_prompts() -> list[list[int]]:
    prompts = build_prompts()
    return prompts[:6] + prompts[10:20]  # six sentences and ten random-token prompts


@pytest.fixture(scope="module")
def reference_tokens() -> list[list[int]]:
    """Hugging Face ``generate`` output for the chosen prompts, computed once."""
    reference = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.float32).eval()
    tokens = [hf_generate(reference, p, NEW_TOKENS, "cpu", PAD_ID) for p in chosen_prompts()]
    del reference
    gc.collect()
    return tokens


@pytest.fixture(scope="module")
def engine() -> Llama:
    return load_engine(torch.float32)


def run_scheduler(engine: Llama, num_blocks: int, max_batch_size: int) -> tuple[Scheduler, list]:  # type: ignore[type-arg]
    pool = KVPool(engine.cfg, num_blocks, BLOCK_SIZE, torch.float32, "cpu")
    scheduler = Scheduler(engine, pool, max_batch_size=max_batch_size)
    requests = [
        scheduler.add_request(i, prompt, NEW_TOKENS) for i, prompt in enumerate(chosen_prompts())
    ]
    scheduler.run_to_completion()
    assert pool.allocator.num_used == 0, "blocks leaked"
    return scheduler, requests


def assert_matches_reference(requests: list, reference_tokens: list[list[int]]) -> None:  # type: ignore[type-arg]
    for request, expected in zip(requests, reference_tokens, strict=True):
        assert request.generated == expected, (
            f"request {request.id} diverges from Hugging Face at token "
            f"{first_diff(request.generated, expected)} "
            f"(preempted {request.preemptions} times)"
        )


def test_batched_scheduler_matches_hugging_face(
    engine: Llama, reference_tokens: list[list[int]]
) -> None:
    scheduler, requests = run_scheduler(engine, num_blocks=200, max_batch_size=8)
    assert scheduler.total_preemptions == 0
    assert scheduler.stats()["mean_decode_batch"] > 2, "requests were not decoded together"
    assert_matches_reference(requests, reference_tokens)


def test_scheduler_with_preemption_by_recompute_matches_hugging_face(
    engine: Llama, reference_tokens: list[list[int]]
) -> None:
    # A request needs at most 4 blocks of 16 tokens; 14 blocks cannot hold eight at once.
    scheduler, requests = run_scheduler(engine, num_blocks=14, max_batch_size=8)
    assert scheduler.total_preemptions > 0, "the pool was meant to be too small"
    print(f"\npreemptions: {scheduler.total_preemptions}, steps: {scheduler.steps}")
    assert_matches_reference(requests, reference_tokens)
