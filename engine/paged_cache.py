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
from engine.paged_attention import DecodeAttention
from engine.prefix_cache import hash_block


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
        self.hashes: list[int] = []  # content hash of each leading block already registered
        self.length = 0  # tokens already cached; advanced once per forward pass

    def adopt(self, blocks: list[int], hashes: list[int]) -> None:
        """Start from cached full blocks that already hold this sequence's first tokens.

        The caller has taken a holder on each block (see ``match_prefix``); this cache now owns
        those holders and gives them back in :meth:`release`.
        """
        if self.blocks or self.length:
            raise ValueError("only an empty cache can adopt blocks")
        if len(blocks) != len(hashes):
            raise ValueError("one hash per adopted block")
        self.blocks = list(blocks)
        self.hashes = list(hashes)
        self.length = len(blocks) * self.pool.block_size

    def register_full_blocks(self, tokens: list[int]) -> None:
        """Index every block that has become full so later prompts can reuse it.

        ``tokens`` are this sequence's tokens from the start; only those already cached
        (``self.length`` of them) are used. Safe to call after every step: it does work only when
        a block has just filled.
        """
        size = self.pool.block_size
        full = min(self.length // size, len(self.blocks))
        while len(self.hashes) < full:
            index = len(self.hashes)
            parent = self.hashes[-1] if self.hashes else None
            content_hash = hash_block(parent, tokens[index * size : (index + 1) * size])
            self.pool.allocator.register(self.blocks[index], content_hash)
            self.hashes.append(content_hash)

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
        self.hashes = []
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


def write_new_tokens(
    pool: KVPool, tables: Tensor, lengths: Tensor, layer: int, k_new: Tensor, v_new: Tensor
) -> None:
    """Write each sequence's new key and value at its next slot (``lengths`` tokens are cached).

    ``k_new`` and ``v_new`` are ``[batch, kv_heads, 1, head_dim]``. Shared by the ordinary batch
    cache and the static-buffer one a CUDA graph replays (``engine.cuda_graph``).
    """
    if k_new.shape[0] != tables.shape[0] or k_new.shape[2] != 1:
        raise ValueError("a batch step takes exactly one new token per sequence")
    rows = torch.arange(tables.shape[0], device=pool.device)
    slot_blocks = tables[rows, lengths // pool.block_size]
    slot_offsets = lengths % pool.block_size
    pool.k[layer, slot_blocks, :, slot_offsets] = k_new[:, :, 0, :]
    pool.v[layer, slot_blocks, :, slot_offsets] = v_new[:, :, 0, :]


class PagedBatchCache:
    """A decode step over several sequences of different lengths, one new token each.

    Each sequence keeps its own block table. Keys and values are gathered into one padded tensor
    and an attention mask hides the padding, so the padded batch computes what each sequence
    would alone. Callers must pass explicit ``positions`` (:meth:`positions`) to the model, since
    the sequences are at different places.

    Blocks for the new token are claimed when the object is built, all or nothing, so a pool that
    cannot hold the step fails before anything changes (the scheduler then preempts or waits).
    """

    def __init__(
        self,
        caches: list[PagedSequenceCache],
        attention: DecodeAttention | None = None,
    ) -> None:
        if not caches:
            raise ValueError("a batch needs at least one sequence")
        self.pool = caches[0].pool
        if any(cache.pool is not self.pool for cache in caches):
            raise ValueError("all sequences in a batch must share one pool")
        self.caches = caches
        self.length = 0  # unused: positions and masks are per sequence
        # With a decode-attention function (D-025) the model calls attend_in_place instead of
        # update plus its own attention, and nothing is gathered. None keeps the gather path.
        self.attention = attention

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
        self._write_new(layer, k_new, v_new)
        return self._gather(self.pool.k[layer]), self._gather(self.pool.v[layer])

    @property
    def reads_in_place(self) -> bool:
        """True when attention reads the pool directly (the model then calls attend_in_place)."""
        return self.attention is not None

    def attend_in_place(self, layer: int, q: Tensor, k_new: Tensor, v_new: Tensor) -> Tensor:
        """Write each sequence's new key and value, then attend over the pool without gathering.

        ``q`` is ``[batch, heads, 1, head_dim]``; the result has the same shape.
        """
        if self.attention is None:
            raise RuntimeError("no decode-attention function was given to this batch cache")
        self._write_new(layer, k_new, v_new)
        return self.attention(
            q, self.pool.k[layer], self.pool.v[layer], self._tables, self._lengths
        )

    def _write_new(self, layer: int, k_new: Tensor, v_new: Tensor) -> None:
        write_new_tokens(self.pool, self._tables, self._lengths, layer, k_new, v_new)

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
