"""Split-KV variant of the Triton decode-attention kernel (D-031).

``engine.triton_paged_attention`` runs one program per (sequence, key-value head) that walks the
whole sequence one block at a time. On a batch of 60 sequences that is 480 programs of two warps on
a card with 170 multiprocessors: about three programs per multiprocessor, too few to hide memory
latency. This variant divides each sequence's blocks over ``SPLITS`` programs (the idea of
Flash-Decoding), each producing a partial online-softmax result (running maximum, running sum,
weighted values) for its own blocks, and a second small kernel merges the partial results by
rescaling to the largest maximum.

The blocks are dealt out by :func:`engine.paged_attention.split_blocks`, the rule the float64
reference :func:`engine.paged_attention.split_decode_attention_reference` uses (checked on a CPU);
a split with no blocks writes a maximum of minus infinity and zeros and is ignored by the merge.
The arithmetic is float32 with multiply-and-sum, as in the unsplit kernel.

Needs a CUDA GPU and Triton. **Written on a machine with neither and never compiled or run when
this was committed**: its first use, on the GPU machine, is the correctness gate of
``bench.kernel_microbench`` (a variant that fails the gate is reported as failed and is not timed).
"""

from __future__ import annotations

import math
from collections.abc import Callable

import torch
import triton
import triton.language as tl
from torch import Tensor

ALLOWED_SPLITS = (1, 2, 4, 8)  # the merge kernel needs a power of two


@triton.jit
def _partial_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    tables_ptr,
    lengths_ptr,
    max_ptr,
    sum_ptr,
    acc_ptr,
    stride_qb,
    stride_qh,
    stride_kb,
    stride_kh,
    stride_kt,
    stride_vb,
    stride_vh,
    stride_vt,
    stride_tb,
    num_kv_heads,
    scale,
    GROUP: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    SPLITS: tl.constexpr,
):
    seq = tl.program_id(0)
    kv_head = tl.program_id(1)
    split = tl.program_id(2)
    count = tl.load(lengths_ptr + seq) + 1

    group = tl.arange(0, GROUP)
    slot = tl.arange(0, BLOCK_SIZE)
    dim = tl.arange(0, HEAD_DIM)

    heads = kv_head * GROUP + group
    q = tl.load(q_ptr + seq * stride_qb + heads[:, None] * stride_qh + dim[None, :])
    q = q.to(tl.float32) * scale

    running_max = tl.full([GROUP], float("-inf"), dtype=tl.float32)
    running_sum = tl.zeros([GROUP], dtype=tl.float32)
    acc = tl.zeros([GROUP, HEAD_DIM], dtype=tl.float32)

    num_blocks = tl.cdiv(count, BLOCK_SIZE)
    per_split = tl.cdiv(num_blocks, SPLITS)
    first = tl.minimum(split * per_split, num_blocks)
    last = tl.minimum(num_blocks, first + per_split)
    for index in range(first, last):
        block = tl.load(tables_ptr + seq * stride_tb + index)
        position = index * BLOCK_SIZE + slot
        valid = position < count

        k_offset = block * stride_kb + kv_head * stride_kh
        keys = tl.load(
            k_ptr + k_offset + slot[:, None] * stride_kt + dim[None, :],
            mask=valid[:, None],
            other=0.0,
        ).to(tl.float32)
        scores = tl.sum(q[:, None, :] * keys[None, :, :], axis=2)
        scores = tl.where(valid[None, :], scores, float("-inf"))

        new_max = tl.maximum(running_max, tl.max(scores, axis=1))
        weights = tl.exp(scores - new_max[:, None])
        rescale = tl.exp(running_max - new_max)
        running_sum = running_sum * rescale + tl.sum(weights, axis=1)

        v_offset = block * stride_vb + kv_head * stride_vh
        values = tl.load(
            v_ptr + v_offset + slot[:, None] * stride_vt + dim[None, :],
            mask=valid[:, None],
            other=0.0,
        ).to(tl.float32)
        acc = acc * rescale[:, None] + tl.sum(weights[:, :, None] * values[None, :, :], axis=1)
        running_max = new_max

    slot_index = (seq * num_kv_heads + kv_head) * SPLITS + split  # one partial result
    tl.store(max_ptr + slot_index * GROUP + group, running_max)
    tl.store(sum_ptr + slot_index * GROUP + group, running_sum)
    tl.store(
        acc_ptr + (slot_index * GROUP + group[:, None]) * HEAD_DIM + dim[None, :],
        acc,
    )


@triton.jit
def _merge_kernel(
    max_ptr,
    sum_ptr,
    acc_ptr,
    out_ptr,
    stride_ob,
    stride_oh,
    num_kv_heads,
    GROUP: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    SPLITS: tl.constexpr,
):
    seq = tl.program_id(0)
    kv_head = tl.program_id(1)

    split = tl.arange(0, SPLITS)
    group = tl.arange(0, GROUP)
    dim = tl.arange(0, HEAD_DIM)

    slot_index = (seq * num_kv_heads + kv_head) * SPLITS + split  # [SPLITS]
    maxima = tl.load(max_ptr + slot_index[:, None] * GROUP + group[None, :])  # [SPLITS, GROUP]
    sums = tl.load(sum_ptr + slot_index[:, None] * GROUP + group[None, :])
    acc = tl.load(
        acc_ptr
        + (slot_index[:, None, None] * GROUP + group[None, :, None]) * HEAD_DIM
        + dim[None, None, :]
    )  # [SPLITS, GROUP, HEAD_DIM]

    overall = tl.max(maxima, axis=0)  # [GROUP]; finite: every sequence has at least one position
    scale = tl.exp(maxima - overall[None, :])  # zero for a split with no blocks
    total = tl.sum(scale * sums, axis=0)
    merged = tl.sum(scale[:, :, None] * acc, axis=0)
    result = merged / total[:, None]

    heads = kv_head * GROUP + group
    tl.store(
        out_ptr + seq * stride_ob + heads[:, None] * stride_oh + dim[None, :],
        result.to(out_ptr.dtype.element_ty),
    )


def _is_power_of_two(value: int) -> bool:
    return value > 0 and value & (value - 1) == 0


def paged_split_decode_attention(
    q: Tensor,
    k_pool: Tensor,
    v_pool: Tensor,
    tables: Tensor,
    lengths: Tensor,
    splits: int = 4,
    num_warps: int = 2,
) -> Tensor:
    """Decode attention over the paged pool with each sequence split over ``splits`` programs."""
    batch, heads, new, head_dim = q.shape
    kv_heads, block_size = k_pool.shape[1], k_pool.shape[2]
    if new != 1:
        raise ValueError("decode attention takes one new token per sequence")
    if splits not in ALLOWED_SPLITS:
        raise ValueError(f"splits must be one of {ALLOWED_SPLITS}, got {splits}")
    if k_pool.shape[3] != head_dim or heads % kv_heads:
        raise ValueError("pool and query shapes do not match")
    group = heads // kv_heads
    for name, size in (("group", group), ("block size", block_size), ("head size", head_dim)):
        if not _is_power_of_two(size):
            raise ValueError(f"the kernel needs a power-of-two {name}, got {size}")
    if not (q.is_cuda and k_pool.is_cuda and tables.is_cuda and lengths.is_cuda):
        raise ValueError("the Triton kernel needs every tensor on the GPU")
    if tables.dtype != torch.long or lengths.dtype != torch.long:
        raise ValueError("tables and lengths must be int64")
    if k_pool.stride(3) != 1 or v_pool.stride(3) != 1 or q.stride(3) != 1:
        raise ValueError("the last dimension of q and of the pools must be contiguous")

    partials = batch * kv_heads * splits
    maxima = torch.empty(partials * group, dtype=torch.float32, device=q.device)
    sums = torch.empty_like(maxima)
    acc = torch.empty(partials * group * head_dim, dtype=torch.float32, device=q.device)
    out = torch.empty_like(q)
    _partial_kernel[(batch, kv_heads, splits)](
        q,
        k_pool,
        v_pool,
        tables,
        lengths,
        maxima,
        sums,
        acc,
        q.stride(0),
        q.stride(1),
        k_pool.stride(0),
        k_pool.stride(1),
        k_pool.stride(2),
        v_pool.stride(0),
        v_pool.stride(1),
        v_pool.stride(2),
        tables.stride(0),
        kv_heads,
        1.0 / math.sqrt(head_dim),
        GROUP=group,
        BLOCK_SIZE=block_size,
        HEAD_DIM=head_dim,
        SPLITS=splits,
        num_warps=num_warps,
    )
    _merge_kernel[(batch, kv_heads)](
        maxima,
        sums,
        acc,
        out,
        out.stride(0),
        out.stride(1),
        kv_heads,
        GROUP=group,
        HEAD_DIM=head_dim,
        SPLITS=splits,
        num_warps=num_warps,
    )
    return out


def split_variant(splits: int, num_warps: int = 2) -> Callable[..., Tensor]:
    """``paged_split_decode_attention`` with ``splits`` and ``num_warps`` fixed."""

    def run(q: Tensor, k_pool: Tensor, v_pool: Tensor, tables: Tensor, lengths: Tensor) -> Tensor:
        return paged_split_decode_attention(q, k_pool, v_pool, tables, lengths, splits, num_warps)

    return run
