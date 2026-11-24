"""CUDA graphs for the decode step (D-020, re-based on the in-place kernel path).

A decode step is about 950 small kernels. Launching them one by one from Python leaves the GPU
idle between them; capturing the whole step once and replaying it removes that. A graph replays
fixed memory addresses, so everything it reads is a **static buffer** that is filled before each
replay, and the batch is padded up to one of a few **bucket** sizes (one graph per bucket).

What makes padding safe:

* A padding row has length 0 and a block table whose first entry is a **scratch block** that the
  decoder reserved from the pool and never gives back. Its new key and value are written into the
  scratch block, and its attention reads one position of it. No live sequence ever owns it, so
  padding rows cannot corrupt a live sequence, and their outputs are dropped.
* Real rows copy their block table into the static one. Columns past a row's own blocks may hold
  stale ids from an earlier step; the in-place kernel never reads them (it loops only over the
  blocks a sequence really has), which is also why the kernel path needs no mask and a fixed
  table width.

This needs the in-place decode attention of D-025 (the gather path pads to the longest sequence,
a shape that changes every step). A step whose block tables are wider than the static width, or
with more sequences than the largest bucket, runs eagerly instead and is counted.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import torch
from torch import Tensor

from engine.paged_attention import DecodeAttention
from engine.paged_cache import KVPool, PagedBatchCache, write_new_tokens
from engine.scheduler import LanguageModel

DEFAULT_MAX_BLOCKS = 256  # table width: 4,096 tokens at block size 16


def default_buckets(max_batch: int) -> list[int]:
    """1, 2, 4, 8, then steps of 8 up to ``max_batch`` (which is always included)."""
    sizes = {1, 2, 4, 8, *range(16, max_batch + 1, 8), max_batch}
    return sorted(size for size in sizes if size <= max_batch)


class StaticBatchCache:
    """The batch cache a captured step uses: ``PagedBatchCache``'s interface over static tensors."""

    reads_in_place = True

    def __init__(
        self, pool: KVPool, tables: Tensor, lengths: Tensor, attention: DecodeAttention
    ) -> None:
        self.pool = pool
        self._tables = tables
        self._lengths = lengths
        self.attention = attention
        self.length = 0

    def positions(self) -> Tensor:
        return self._lengths.unsqueeze(1)

    def attend_in_place(self, layer: int, q: Tensor, k_new: Tensor, v_new: Tensor) -> Tensor:
        write_new_tokens(self.pool, self._tables, self._lengths, layer, k_new, v_new)
        return self.attention(
            q, self.pool.k[layer], self.pool.v[layer], self._tables, self._lengths
        )

    def advance(self, num_tokens: int) -> None:
        pass  # the real sequences' lengths are advanced by the caller, outside the graph

    # The rest of the cache interface, which the in-place path never reaches.
    def update(self, layer: int, k_new: Tensor, v_new: Tensor) -> tuple[Tensor, Tensor]:
        raise RuntimeError("a static batch cache only attends in place")

    def attention_mask(self, new_tokens: int) -> Tensor | None:
        raise RuntimeError("a static batch cache only attends in place")


@dataclass
class _Bucket:
    size: int
    tokens: Tensor
    tables: Tensor
    lengths: Tensor
    cache: StaticBatchCache
    logits: Tensor | None = None
    graph: torch.cuda.CUDAGraph | None = None


@dataclass
class GraphStats:
    replays: int = 0
    eager_fallbacks: int = 0
    padded_rows: int = 0
    by_bucket: dict[int, int] = field(default_factory=dict)


class CudaGraphDecoder:
    """Runs batched decode steps through captured graphs, one per batch-size bucket.

    ``capture=False`` runs the same static-buffer, padded path eagerly (no graph): that is what
    the CPU tests exercise, and what a GPU test compares a real graph with, bit for bit.
    """

    def __init__(
        self,
        model: LanguageModel,
        pool: KVPool,
        attention: DecodeAttention,
        max_batch: int,
        *,
        max_blocks: int = DEFAULT_MAX_BLOCKS,
        buckets: Sequence[int] | None = None,
        capture: bool = True,
    ) -> None:
        if max_batch <= 0 or max_blocks <= 0:
            raise ValueError("max_batch and max_blocks must be positive")
        self.model = model
        self.pool = pool
        self.attention = attention
        self.max_blocks = max_blocks
        self.capture = capture
        self.stats = GraphStats()
        self.scratch_block = pool.allocator.allocate()  # held for the decoder's lifetime
        sizes = sorted(set(buckets)) if buckets is not None else default_buckets(max_batch)
        if not sizes or sizes[0] < 1:
            raise ValueError("buckets must be positive sizes")
        self.buckets = {size: self._make_bucket(size) for size in sizes}
        self.max_bucket = sizes[-1]
        if capture:
            with torch.inference_mode():  # no autograd bookkeeping in the captured step
                for bucket in self.buckets.values():
                    self._capture(bucket)

    # -- running ---------------------------------------------------------------------------

    def bucket_for(self, count: int) -> int | None:
        """The smallest bucket that holds ``count`` sequences, or ``None`` if there is none."""
        for size in sorted(self.buckets):
            if size >= count:
                return size
        return None

    def can_run(self, step: PagedBatchCache) -> bool:
        return len(step.caches) <= self.max_bucket and step._tables.shape[1] <= self.max_blocks

    def run(self, step: PagedBatchCache, last_tokens: Tensor) -> Tensor:
        """The logits ``[n, 1, vocab]`` of one decode step over ``step``'s sequences.

        ``last_tokens`` is ``[n, 1]``. The sequences' cached lengths are advanced here (the model
        does it in the eager path), so this is a drop-in for ``model(...)`` on the step.
        """
        count = len(step.caches)
        size = self.bucket_for(count)
        if size is None or not self.can_run(step):
            raise ValueError("this step does not fit the graph; use the eager path")
        bucket = self.buckets[size]
        width = step._tables.shape[1]
        bucket.tokens.zero_()
        bucket.tokens[:count] = last_tokens
        bucket.lengths.zero_()
        bucket.lengths[:count] = step._lengths
        bucket.tables[:count, :width] = step._tables
        bucket.tables[count:, 0] = self.scratch_block
        if bucket.graph is not None:
            bucket.graph.replay()
        else:
            bucket.logits = self._forward(bucket)
        assert bucket.logits is not None
        step.advance(1)
        self.stats.replays += 1
        self.stats.padded_rows += size - count
        self.stats.by_bucket[size] = self.stats.by_bucket.get(size, 0) + 1
        return bucket.logits[:count]

    # -- building --------------------------------------------------------------------------

    def _make_bucket(self, size: int) -> _Bucket:
        device = self.pool.device
        tokens = torch.zeros(size, 1, dtype=torch.long, device=device)
        tables = torch.zeros(size, self.max_blocks, dtype=torch.long, device=device)
        tables[:, 0] = self.scratch_block
        lengths = torch.zeros(size, dtype=torch.long, device=device)
        cache = StaticBatchCache(self.pool, tables, lengths, self.attention)
        return _Bucket(size, tokens, tables, lengths, cache)

    def _forward(self, bucket: _Bucket) -> Tensor:
        return self.model(bucket.tokens, positions=bucket.cache.positions(), cache=bucket.cache)

    def _capture(self, bucket: _Bucket) -> None:
        """Warm up on a side stream, then capture one decode step of this bucket's static state."""
        if self.pool.device.type != "cuda":
            raise RuntimeError("CUDA graphs need a CUDA device")
        side = torch.cuda.Stream()  # type: ignore[no-untyped-call]
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):  # lets cuBLAS and Triton set up outside the capture
                self._forward(bucket)
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            bucket.logits = self._forward(bucket)
        bucket.graph = graph
