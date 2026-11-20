"""Fixed-size block allocator with reference counts.

The KV cache memory is carved into ``num_blocks`` blocks of ``block_size`` token slots. Sequences
hold lists of block ids instead of one contiguous buffer, so memory is claimed one block at a
time and returned the moment a sequence ends. Reference counts let several sequences share a
block (needed later for prefix caching, D-005); a block goes back to the free list only when its
last holder releases it.
"""

from __future__ import annotations

from collections import deque


class OutOfBlocksError(RuntimeError):
    """No free block is left. The scheduler reacts by waiting or preempting (D-004)."""


class BlockAllocator:
    def __init__(self, num_blocks: int, block_size: int) -> None:
        if num_blocks <= 0 or block_size <= 0:
            raise ValueError("num_blocks and block_size must be positive")
        self.num_blocks = num_blocks
        self.block_size = block_size
        # FIFO free list: a released block is reused last, which keeps recently freed contents
        # around longest. That matters once freed blocks can still hold a cacheable prefix.
        self._free: deque[int] = deque(range(num_blocks))
        self._refs = [0] * num_blocks

    @property
    def num_free(self) -> int:
        return len(self._free)

    @property
    def num_used(self) -> int:
        return self.num_blocks - len(self._free)

    def blocks_for(self, num_tokens: int) -> int:
        """Number of blocks needed to hold ``num_tokens`` tokens."""
        if num_tokens < 0:
            raise ValueError("num_tokens must be non-negative")
        return -(-num_tokens // self.block_size)

    def can_allocate(self, count: int) -> bool:
        return count <= len(self._free)

    def allocate(self) -> int:
        """Take one free block (reference count 1)."""
        if not self._free:
            raise OutOfBlocksError(f"all {self.num_blocks} blocks are in use")
        block = self._free.popleft()
        self._refs[block] = 1
        return block

    def share(self, block: int) -> None:
        """Add a holder to a block that is already in use."""
        self._check_in_use(block)
        self._refs[block] += 1

    def release(self, block: int) -> None:
        """Drop one holder; the block becomes free when the last holder lets go."""
        self._check_in_use(block)
        self._refs[block] -= 1
        if self._refs[block] == 0:
            self._free.append(block)

    def ref_count(self, block: int) -> int:
        self._check_range(block)
        return self._refs[block]

    def _check_range(self, block: int) -> None:
        if not 0 <= block < self.num_blocks:
            raise ValueError(f"block {block} is outside 0..{self.num_blocks - 1}")

    def _check_in_use(self, block: int) -> None:
        self._check_range(block)
        if self._refs[block] == 0:
            raise ValueError(f"block {block} is not allocated")
