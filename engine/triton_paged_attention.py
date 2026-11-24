"""Triton decode-attention kernel that reads the paged KV pool in place (D-025).

One program per (sequence, key-value head). It loads the query heads that share that key-value
head (grouped-query attention), walks the sequence's own block table one block at a time and keeps
an online softmax (running max, running sum, running weighted values), so nothing is gathered into
a padded tensor and no work is done for positions a sequence does not have.

Arithmetic is float32 throughout, with plain multiply-and-sum rather than ``tl.dot``: ``tl.dot`` on
float32 inputs may use TF32, which would break the fp32 correctness gate, and decode attention is
bound by memory reads, not arithmetic, so the tile shape that ``tl.dot`` wants is not worth it.

Safety: the loop runs only over the blocks a sequence has, so padded table entries (block 0 in the
engine) are never read; inside the last block the slots past the sequence's length are masked on
load and on the score.

Needs a CUDA GPU; importing this module needs Triton. This file is checked only on the GPU machine
(``tests/test_paged_attention_gpu.py``); it cannot be run on a CPU.
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl
from torch import Tensor


@triton.jit
def _decode_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    out_ptr,
    tables_ptr,
    lengths_ptr,
    stride_qb,
    stride_qh,
    stride_kb,
    stride_kh,
    stride_kt,
    stride_vb,
    stride_vh,
    stride_vt,
    stride_tb,
    stride_ob,
    stride_oh,
    scale,
    GROUP: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    seq = tl.program_id(0)
    kv_head = tl.program_id(1)
    count = tl.load(lengths_ptr + seq) + 1  # cached tokens plus the new one

    group = tl.arange(0, GROUP)
    slot = tl.arange(0, BLOCK_SIZE)
    dim = tl.arange(0, HEAD_DIM)

    heads = kv_head * GROUP + group
    q = tl.load(q_ptr + seq * stride_qb + heads[:, None] * stride_qh + dim[None, :])
    q = q.to(tl.float32) * scale  # [GROUP, HEAD_DIM]

    running_max = tl.full([GROUP], float("-inf"), dtype=tl.float32)
    running_sum = tl.zeros([GROUP], dtype=tl.float32)
    acc = tl.zeros([GROUP, HEAD_DIM], dtype=tl.float32)

    num_blocks = tl.cdiv(count, BLOCK_SIZE)
    for index in range(0, num_blocks):
        block = tl.load(tables_ptr + seq * stride_tb + index)
        position = index * BLOCK_SIZE + slot
        valid = position < count  # [BLOCK_SIZE]; false only in the last block

        k_offset = block * stride_kb + kv_head * stride_kh
        keys = tl.load(
            k_ptr + k_offset + slot[:, None] * stride_kt + dim[None, :],
            mask=valid[:, None],
            other=0.0,
        ).to(tl.float32)  # [BLOCK_SIZE, HEAD_DIM]
        scores = tl.sum(q[:, None, :] * keys[None, :, :], axis=2)  # [GROUP, BLOCK_SIZE]
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

    result = acc / running_sum[:, None]
    tl.store(
        out_ptr + seq * stride_ob + heads[:, None] * stride_oh + dim[None, :],
        result.to(out_ptr.dtype.element_ty),
    )


def _is_power_of_two(value: int) -> bool:
    return value > 0 and value & (value - 1) == 0


def paged_decode_attention(
    q: Tensor, k_pool: Tensor, v_pool: Tensor, tables: Tensor, lengths: Tensor
) -> Tensor:
    """Decode attention over the paged pool without gathering. See ``engine.paged_attention``."""
    batch, heads, new, head_dim = q.shape
    num_blocks, kv_heads, block_size, pool_dim = k_pool.shape
    del num_blocks
    if new != 1:
        raise ValueError("decode attention takes one new token per sequence")
    if pool_dim != head_dim or heads % kv_heads:
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

    out = torch.empty_like(q)
    _decode_kernel[(batch, kv_heads)](
        q,
        k_pool,
        v_pool,
        out,
        tables,
        lengths,
        q.stride(0),
        q.stride(1),
        k_pool.stride(0),
        k_pool.stride(1),
        k_pool.stride(2),
        v_pool.stride(0),
        v_pool.stride(1),
        v_pool.stride(2),
        tables.stride(0),
        out.stride(0),
        out.stride(1),
        1.0 / math.sqrt(head_dim),
        GROUP=group,
        BLOCK_SIZE=block_size,
        HEAD_DIM=head_dim,
        num_warps=2,
    )
    return out
