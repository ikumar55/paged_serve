"""Shared helpers for the decode-attention tests (CPU tests and the GPU kernel tests).

``make_case`` builds a ragged decode batch over a random pool with scrambled block tables and
padded table entries pointing at block 0, and ``gather_attention`` is the engine's existing path
(gather into a padded tensor, mask, SDPA) written over the same arguments, so any implementation
can be compared with it and with the float64 reference.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from engine.model import attend


@dataclass
class Case:
    q: Tensor  # [batch, heads, 1, head_dim]
    k_pool: Tensor  # [num_blocks, kv_heads, block_size, head_dim]
    v_pool: Tensor
    tables: Tensor  # [batch, width] int64, padded with block 0
    lengths: Tensor  # [batch] int64: tokens cached before the new one
    valid: Tensor  # [num_blocks, block_size] bool: slots some sequence really uses


def make_case(
    lengths: list[int],
    *,
    heads: int = 32,
    kv_heads: int = 8,
    head_dim: int = 64,
    block_size: int = 16,
    dtype: torch.dtype = torch.float32,
    device: str = "cpu",
    seed: int = 0,
    spare_blocks: int = 5,
    poison: float | None = None,
    own_block_zero: bool = False,
) -> Case:
    """A decode batch whose sequences have ``lengths`` cached tokens each (plus the new one).

    Blocks are handed out in random order, so tables are scrambled. ``poison`` fills every slot no
    sequence uses (stale slots in a last block, whole unused blocks including block 0 when nobody
    owns it) with that value; ``None`` leaves them random. ``own_block_zero`` makes the first
    sequence own block 0, so the padded entries of the others point at live data of another
    sequence (the engine's real situation: the allocator hands out block 0 like any other).
    """
    generator = torch.Generator().manual_seed(seed)
    need = [math.ceil((n + 1) / block_size) for n in lengths]
    num_blocks = sum(need) + spare_blocks
    order = torch.randperm(num_blocks, generator=generator).tolist()
    if own_block_zero:  # the first sequence's first block is block 0, as the allocator would
        order.remove(0)
        order.insert(0, 0)
    width = max(need)
    tables = torch.zeros(len(lengths), width, dtype=torch.long)
    valid = torch.zeros(num_blocks, block_size, dtype=torch.bool)
    taken = 0
    for row, (n, count) in enumerate(zip(lengths, need, strict=True)):
        blocks = order[taken : taken + count]
        taken += count
        tables[row, :count] = torch.tensor(blocks)
        for position in range(n + 1):
            valid[blocks[position // block_size], position % block_size] = True

    def randn(*shape: int) -> Tensor:
        return torch.randn(*shape, generator=generator).to(dtype)

    k_pool = randn(num_blocks, kv_heads, block_size, head_dim)
    v_pool = randn(num_blocks, kv_heads, block_size, head_dim)
    if poison is not None:
        mask = valid[:, None, :, None]
        k_pool = torch.where(mask, k_pool, torch.tensor(poison, dtype=dtype))
        v_pool = torch.where(mask, v_pool, torch.tensor(poison, dtype=dtype))
    q = randn(len(lengths), heads, 1, head_dim)
    return Case(
        q.to(device),
        k_pool.to(device),
        v_pool.to(device),
        tables.to(device),
        torch.tensor(lengths, dtype=torch.long, device=device),
        valid.to(device),
    )


def gather_attention(
    q: Tensor, k_pool: Tensor, v_pool: Tensor, tables: Tensor, lengths: Tensor
) -> Tensor:
    """The existing path: gather the blocks into a padded tensor, mask, run SDPA."""
    batch, width = tables.shape
    block_size = k_pool.shape[2]

    def gather(pool: Tensor) -> Tensor:
        blocks = pool[tables]  # [batch, width, kv_heads, block_size, head_dim]
        _, _, kv_heads, _, head_dim = blocks.shape
        return blocks.permute(0, 2, 1, 3, 4).reshape(batch, kv_heads, width * block_size, head_dim)

    padded = width * block_size
    columns = torch.arange(padded, device=q.device)
    mask = (columns.unsqueeze(0) <= lengths.unsqueeze(1)).view(-1, 1, 1, padded)
    return attend(q, gather(k_pool), gather(v_pool), past=0, mask=mask)
