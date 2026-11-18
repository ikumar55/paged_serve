"""Single-request greedy generation with a contiguous KV cache."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from engine.kv_cache import ContiguousKVCache
from engine.model import Llama


@torch.inference_mode()
def greedy_generate(
    model: Llama,
    prompt_ids: Sequence[int],
    max_new_tokens: int,
    eos_token_id: int | None = None,
) -> list[int]:
    """Generate up to ``max_new_tokens`` tokens, stopping early after ``eos_token_id``.

    Returns only the newly generated tokens (including the EOS token if one was produced).
    """
    if not prompt_ids:
        raise ValueError("prompt must contain at least one token")
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    total = len(prompt_ids) + max_new_tokens
    if total > model.cfg.max_position_embeddings:
        raise ValueError(
            f"{total} tokens exceed the model's maximum of {model.cfg.max_position_embeddings}"
        )

    cache = ContiguousKVCache(model.cfg, 1, total, model.dtype, model.device)
    logits = model(
        torch.tensor([list(prompt_ids)], device=model.device), cache=cache, last_only=True
    )

    generated: list[int] = []
    while True:
        next_token = int(logits[0, -1].argmax())
        generated.append(next_token)
        if next_token == eos_token_id or len(generated) == max_new_tokens:
            return generated
        step = torch.tensor([[next_token]], device=model.device)
        logits = model(step, cache=cache, last_only=True)
