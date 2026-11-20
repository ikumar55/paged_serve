"""Paged KV cache: one shared pool of blocks, one block table per sequence.

``KVPool`` owns the memory and the allocator. ``PagedSequenceCache`` is the view one sequence has
of it: a list of block ids, in order, plus the number of tokens stored. It exposes the same
``length`` / ``update`` / ``advance`` interface as the contiguous cache, so the model forward pass
runs unchanged and the two layouts can be checked against each other.

This is the gather-based version of D-003: ``update`` copies a sequence's blocks into one
contiguous tensor before attention. That is correct and slow; a fused kernel that reads the blocks
in place would come later and be tested against this one.

Scope of this first version: one sequence per cache object (batch size 1). Batching several
sequences with different lengths needs per-sequence positions and masks in the forward pass, which
is the scheduler's job and not done here.
"""

from __future__ import annotations

import torch
from torch import Tensor

from engine.block_allocator import BlockAllocator, OutOfBlocksError
from engine.config import LlamaConfig


class KVPool:
    """Key and value storage for ``num_blocks`` blocks of ``block_size`` tokens, all layers."""

    def __init__(
        self,
        cfg: LlamaConfig,
        num_blocks: int,
        block_size: int,
        dtype: torch.dtype,
        device: torch.device | str,
    ) -> None:
        self.allocator = BlockAllocator(num_blocks, block_size)
        shape = (cfg.num_layers, num_blocks, cfg.num_kv_heads, block_size, cfg.head_dim)
        self.k = torch.zeros(shape, dtype=dtype, device=device)
        self.v = torch.zeros(shape, dtype=dtype, device=device)
        self.block_size = block_size

    @property
    def device(self) -> torch.device:
        return self.k.device

    def bytes_per_block(self) -> int:
        per_tensor = self.k[:, 0].numel() * self.k.element_size()
        return 2 * per_tensor


class PagedSequenceCache:
    """One sequence's block table over a shared :class:`KVPool`."""

    def __init__(self, pool: KVPool) -> None:
        self.pool = pool
        self.blocks: list[int] = []
        self.length = 0  # tokens already cached; advanced once per forward pass

    def update(self, layer: int, k_new: Tensor, v_new: Tensor) -> tuple[Tensor, Tensor]:
        """Write new keys and values after the cached ones and return all of them.

        ``k_new`` and ``v_new`` have shape ``[1, kv_heads, new_tokens, head_dim]``. Blocks are
        claimed here as needed (every layer asks for the same ones, so this is idempotent). A
        :class:`~engine.block_allocator.OutOfBlocksError` leaves the cache holding only the
        blocks it already had plus any claimed before the pool ran out; the caller should
        :meth:`release` it or retry after memory frees up.
        """
        if k_new.shape[0] != 1:
            raise ValueError("a paged sequence cache holds one sequence (batch size 1)")
        new = k_new.shape[2]
        end = self.length + new
        self._ensure_capacity(end)

        block_size = self.pool.block_size
        positions = torch.arange(self.length, end, device=self.pool.device)
        table = torch.tensor(self.blocks, dtype=torch.long, device=self.pool.device)
        slot_blocks = table[positions // block_size]
        slot_offsets = positions % block_size
        # Indexing with two tensors around a slice puts the token dimension first.
        self.pool.k[layer, slot_blocks, :, slot_offsets] = k_new[0].transpose(0, 1)
        self.pool.v[layer, slot_blocks, :, slot_offsets] = v_new[0].transpose(0, 1)

        used = table[: self.pool.allocator.blocks_for(end)]
        return self._gather(self.pool.k[layer], used, end), self._gather(
            self.pool.v[layer], used, end
        )

    def advance(self, num_tokens: int) -> None:
        self.length += num_tokens

    def release(self) -> None:
        """Give every block back. The cache is empty afterwards and can be reused."""
        for block in self.blocks:
            self.pool.allocator.release(block)
        self.blocks = []
        self.length = 0

    def _ensure_capacity(self, num_tokens: int) -> None:
        allocator = self.pool.allocator
        missing = allocator.blocks_for(num_tokens) - len(self.blocks)
        if missing <= 0:
            return
        if not allocator.can_allocate(missing):
            # Fail before taking anything, so a failed step does not leak half an allocation.
            raise OutOfBlocksError(
                f"need {missing} more blocks, only {allocator.num_free} are free"
            )
        for _ in range(missing):
            self.blocks.append(allocator.allocate())

    @staticmethod
    def _gather(layer_pool: Tensor, table: Tensor, num_tokens: int) -> Tensor:
        """Concatenate a sequence's blocks into ``[1, kv_heads, num_tokens, head_dim]``."""
        blocks = layer_pool[table]  # [num_blocks_used, kv_heads, block_size, head_dim]
        kv_heads, head_dim = blocks.shape[1], blocks.shape[3]
        flat = blocks.permute(1, 0, 2, 3).reshape(kv_heads, -1, head_dim)
        return flat[:, :num_tokens].unsqueeze(0)
