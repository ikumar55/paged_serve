"""Decode attention that reads each sequence's KV blocks in place (D-025).

The gather path (``PagedBatchCache._gather``) copies every sequence's blocks into one padded
tensor and masks the padding, so the batch costs as much as its longest sequence. The functions
here take the pool tensors, the block tables and the lengths directly and touch only the positions
each sequence really has.

Two implementations share one signature (:data:`DecodeAttention`):

* :func:`decode_attention_reference` is plain PyTorch in float64. It is the yardstick the kernel
  is checked against, and it also runs on a CPU, so the plumbing around it can be tested without
  a GPU.
* ``engine.triton_paged_attention.paged_decode_attention`` is the Triton kernel (GPU only; Triton
  is imported only when it is asked for).

Conventions, the same as ``PagedBatchCache``: ``k_pool`` and ``v_pool`` are one layer's pool,
``[num_blocks, kv_heads, block_size, head_dim]``; ``tables`` is ``[batch, width]`` block ids, padded
with block 0 beyond a sequence's own blocks; ``lengths`` is the number of tokens each sequence had
cached **before** the new one, so the new token's key and value (already written to the pool) sit
at index ``lengths`` and each sequence attends over ``lengths + 1`` positions.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import torch
from torch import Tensor

# (q [batch, heads, 1, head_dim], k_pool, v_pool, tables, lengths) -> [batch, heads, 1, head_dim]
DecodeAttention = Callable[[Tensor, Tensor, Tensor, Tensor, Tensor], Tensor]


def decode_attention_reference(
    q: Tensor,
    k_pool: Tensor,
    v_pool: Tensor,
    tables: Tensor,
    lengths: Tensor,
    dtype: torch.dtype | None = None,
) -> Tensor:
    """Decode attention for each sequence alone, in float64, reading only its own positions.

    Slow (a Python loop over the batch) and meant as ground truth. The result has ``q``'s dtype
    unless ``dtype`` says otherwise (the correctness gate asks for float64, to measure another
    implementation's error without this one's final rounding). Table entries past a sequence's
    last block are never indexed, so whatever they point at (block 0 in the engine) cannot leak
    in.
    """
    batch, heads, new, head_dim = q.shape
    if new != 1:
        raise ValueError("decode attention takes one new token per sequence")
    kv_heads, block_size = k_pool.shape[1], k_pool.shape[2]
    group = heads // kv_heads
    out_dtype = q.dtype if dtype is None else dtype
    out = torch.empty(q.shape, dtype=out_dtype, device=q.device)
    for row in range(batch):
        count = int(lengths[row]) + 1
        blocks = tables[row, : math.ceil(count / block_size)]
        keys = k_pool[blocks].permute(1, 0, 2, 3).reshape(kv_heads, -1, head_dim)[:, :count]
        values = v_pool[blocks].permute(1, 0, 2, 3).reshape(kv_heads, -1, head_dim)[:, :count]
        keys = keys.double().repeat_interleave(group, dim=0)  # [heads, count, head_dim]
        values = values.double().repeat_interleave(group, dim=0)
        query = q[row, :, 0, :].double()  # [heads, head_dim]
        scores = torch.einsum("hd,htd->ht", query, keys) / math.sqrt(head_dim)
        weights = torch.softmax(scores, dim=-1)
        out[row, :, 0, :] = torch.einsum("ht,htd->hd", weights, values).to(out_dtype)
    return out


def resolve_kernel(name: str) -> DecodeAttention:
    """The decode-attention function called ``name``: ``"triton"`` or ``"reference"``.

    ``"triton"`` needs a CUDA GPU and Triton and says so if either is missing, rather than
    quietly falling back (a benchmark run with the flag on must really use the kernel).
    """
    if name == "reference":
        return decode_attention_reference
    if name == "triton":
        if not torch.cuda.is_available():
            raise RuntimeError("the Triton paged-attention kernel needs a CUDA GPU")
        try:
            from engine.triton_paged_attention import paged_decode_attention
        except ImportError as error:  # pragma: no cover - depends on the machine
            raise RuntimeError(
                "the Triton paged-attention kernel needs Triton installed"
            ) from error
        return paged_decode_attention
    raise ValueError(f"unknown paged-attention implementation {name!r}")
