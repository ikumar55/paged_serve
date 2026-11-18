"""Shared helpers for the tests that use the real Llama-3.2-1B-Instruct weights.

The weights live in ``models/Llama-3.2-1B-Instruct`` (gitignored; downloaded by hand after
accepting the model license). Both ``test_real_llama.py`` (CPU) and ``test_real_llama_gpu.py`` use
the same prompts and the same loader so their numbers are directly comparable.
"""

from __future__ import annotations

from pathlib import Path

import torch
from transformers import AutoTokenizer

from bench.greedy_eval import load_engine as load_engine_from_dir
from engine.model import Llama

MODEL_DIR = Path(__file__).resolve().parents[1] / "models" / "Llama-3.2-1B-Instruct"
WEIGHTS_PRESENT = (MODEL_DIR / "model.safetensors").exists()

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


def build_prompts() -> list[list[int]]:
    """Fifty prompts: ten real sentences plus forty random token sequences of varying length."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    texts = [tokenizer(text)["input_ids"] for text in TEXT_PROMPTS]
    generator = torch.Generator().manual_seed(0)
    lengths = torch.randint(4, 33, (40,), generator=generator).tolist()
    random_ids = [torch.randint(1000, 100000, (n,), generator=generator).tolist() for n in lengths]
    result = texts + random_ids
    assert len(result) == 50 and max(len(p) for p in result) <= 32
    return result


def long_prompt_ids() -> torch.Tensor:
    """One random 2048-token prompt, shape ``[1, 2048]``."""
    return torch.randint(
        1000, 100000, (1, LONG_PROMPT_TOKENS), generator=torch.Generator().manual_seed(1)
    )


def load_engine(dtype: torch.dtype, device: torch.device | str = "cpu") -> Llama:
    """This engine with the real weights, in eval mode, on ``device``."""
    return load_engine_from_dir(MODEL_DIR, dtype, device)


def first_diff(a: list[int], b: list[int]) -> int:
    """Index of the first position where ``a`` and ``b`` differ (or the shorter length)."""
    return next(
        (i for i, (x, y) in enumerate(zip(a, b, strict=False)) if x != y), min(len(a), len(b))
    )
