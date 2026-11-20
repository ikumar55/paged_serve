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

    def attention_mask(self, new_tokens: int) -> Tensor | None:
        return None

    def blocks_missing(self, num_tokens: int) -> int:
        """How many more blocks this sequence needs before it can hold ``num_tokens`` tokens."""
        return max(0, self.pool.allocator.blocks_for(num_tokens) - len(self.blocks))


class PagedBatchCache:
    """A decode step over several sequences of different lengths, one new token each.

    Each sequence keeps its own block table. Keys and values are gathered into one padded tensor
    and an attention mask hides the padding, so the padded batch computes what each sequence
    would alone. Callers must pass explicit ``positions`` (:meth:`positions`) to the model, since
    the sequences are at different places.

    Blocks for the new token are claimed when the object is built, all or nothing, so a pool that
    cannot hold the step fails before anything changes (the scheduler then preempts or waits).
    """

    def __init__(self, caches: list[PagedSequenceCache]) -> None:
        if not caches:
            raise ValueError("a batch needs at least one sequence")
        self.pool = caches[0].pool
        if any(cache.pool is not self.pool for cache in caches):
            raise ValueError("all sequences in a batch must share one pool")
        self.caches = caches
        self.length = 0  # unused: positions and masks are per sequence

        allocator = self.pool.allocator
        needed = sum(cache.blocks_missing(cache.length + 1) for cache in caches)
        if not allocator.can_allocate(needed):
            raise OutOfBlocksError(f"need {needed} more blocks, only {allocator.num_free} are free")
        for cache in caches:
            cache._ensure_capacity(cache.length + 1)

        width = max(len(cache.blocks) for cache in caches)
        table = [cache.blocks + [0] * (width - len(cache.blocks)) for cache in caches]
        device = self.pool.device
        self._tables = torch.tensor(table, dtype=torch.long, device=device)
        self._lengths = torch.tensor([cache.length for cache in caches], device=device)

    def positions(self) -> Tensor:
        """Position of each sequence's new token, shape ``[batch, 1]``."""
        return self._lengths.unsqueeze(1)

    def update(self, layer: int, k_new: Tensor, v_new: Tensor) -> tuple[Tensor, Tensor]:
        """Write each sequence's new key and value, return all keys and values padded.

        ``k_new`` and ``v_new`` have shape ``[batch, kv_heads, 1, head_dim]``; the result has
        shape ``[batch, kv_heads, padded_len, head_dim]``.
        """
        if k_new.shape[0] != len(self.caches) or k_new.shape[2] != 1:
            raise ValueError("a batch step takes exactly one new token per sequence")
        block_size = self.pool.block_size
        rows = torch.arange(len(self.caches), device=self.pool.device)
        slot_blocks = self._tables[rows, self._lengths // block_size]
        slot_offsets = self._lengths % block_size
        self.pool.k[layer, slot_blocks, :, slot_offsets] = k_new[:, :, 0, :]
        self.pool.v[layer, slot_blocks, :, slot_offsets] = v_new[:, :, 0, :]
        return self._gather(self.pool.k[layer]), self._gather(self.pool.v[layer])

    def attention_mask(self, new_tokens: int) -> Tensor:
        """``[batch, 1, 1, padded_len]``: each sequence sees its cached tokens and its new one."""
        padded = self._tables.shape[1] * self.pool.block_size
        columns = torch.arange(padded, device=self.pool.device)
        return (columns.unsqueeze(0) <= self._lengths.unsqueeze(1)).view(-1, 1, 1, padded)

    def advance(self, num_tokens: int) -> None:
        for cache in self.caches:
            cache.advance(num_tokens)

    def _gather(self, layer_pool: Tensor) -> Tensor:
        blocks = layer_pool[self._tables]  # [batch, width, kv_heads, block_size, head_dim]
        batch, width, kv_heads, block_size, head_dim = blocks.shape
        return blocks.permute(0, 2, 1, 3, 4).reshape(batch, kv_heads, width * block_size, head_dim)
