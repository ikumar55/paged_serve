"""Contiguous (non-paged) KV cache: one preallocated buffer per sequence slot.

This is the baseline layout the paged allocator will later replace, and the reference its outputs
are checked against.
"""

from __future__ import annotations

import torch
from torch import Tensor

from engine.config import LlamaConfig


class ContiguousKVCache:
    """Per-layer K and V buffers of shape ``[batch, kv_heads, max_len, head_dim]``."""

    def __init__(
        self,
        cfg: LlamaConfig,
        batch_size: int,
        max_len: int,
        dtype: torch.dtype,
        device: torch.device | str,
    ) -> None:
        if batch_size <= 0 or max_len <= 0:
            raise ValueError("batch_size and max_len must be positive")
        shape = (cfg.num_layers, batch_size, cfg.num_kv_heads, max_len, cfg.head_dim)
        self.k = torch.zeros(shape, dtype=dtype, device=device)
        self.v = torch.zeros(shape, dtype=dtype, device=device)
        self.max_len = max_len
        self.length = 0  # tokens already cached; advanced once per forward pass

    def update(self, layer: int, k_new: Tensor, v_new: Tensor) -> tuple[Tensor, Tensor]:
        """Write new keys and values at the current length and return all cached ones.

        ``k_new`` and ``v_new`` have shape ``[batch, kv_heads, new_tokens, head_dim]``. The caller
        calls :meth:`advance` once every layer has been updated.
        """
        end = self.length + k_new.shape[2]
        if end > self.max_len:
            raise ValueError(f"cache overflow: {end} tokens exceed capacity {self.max_len}")
        self.k[layer, :, :, self.length : end] = k_new
        self.v[layer, :, :, self.length : end] = v_new
        return self.k[layer, :, :, :end], self.v[layer, :, :, :end]

    def advance(self, num_tokens: int) -> None:
        self.length += num_tokens
