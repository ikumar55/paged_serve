"""Fixed-size block allocator with reference counts and an optional content index.

The KV cache memory is carved into ``num_blocks`` blocks of ``block_size`` token slots. Sequences
hold lists of block ids instead of one contiguous buffer, so memory is claimed one block at a
time and returned the moment a sequence ends. Reference counts let several sequences share a
block; a block is released when its last holder lets go.

Prefix caching (D-005) adds a content index. A *full* block can be registered under a hash that
identifies its tokens and everything before them (see :mod:`engine.prefix_cache`). When such a
block's last holder releases it, it is not returned to the free list but kept, still holding its
keys and values, on an *evictable* list in least-recently-released order. A later request whose
prompt hashes to it can take it back (:meth:`lookup`) instead of recomputing it. When the free list
runs dry, :meth:`allocate` evicts the oldest evictable block. Blocks that are free or evictable
both count as available.
"""

from __future__ import annotations

from collections import OrderedDict, deque


class OutOfBlocksError(RuntimeError):
    """No free block is left. The scheduler reacts by waiting or preempting (D-004)."""


class BlockAllocator:
    def __init__(self, num_blocks: int, block_size: int) -> None:
        if num_blocks <= 0 or block_size <= 0:
            raise ValueError("num_blocks and block_size must be positive")
        self.num_blocks = num_blocks
        self.block_size = block_size
        self._free: deque[int] = deque(range(num_blocks))
        self._refs = [0] * num_blocks
        self._hash_of: dict[int, int] = {}  # block -> content hash, for registered blocks
        self._by_hash: dict[int, int] = {}  # content hash -> block
        self._evictable: OrderedDict[int, None] = OrderedDict()  # unreferenced, still cached
        self.evictions = 0

    @property
    def num_free(self) -> int:
        """Blocks that can be allocated now: free ones plus cached ones nobody holds."""
        return len(self._free) + len(self._evictable)

    @property
    def num_used(self) -> int:
        """Blocks held by at least one sequence."""
        return self.num_blocks - self.num_free

    @property
    def num_cached(self) -> int:
        """Blocks registered in the content index, held or not."""
        return len(self._by_hash)

    def blocks_for(self, num_tokens: int) -> int:
        """Number of blocks needed to hold ``num_tokens`` tokens."""
        if num_tokens < 0:
            raise ValueError("num_tokens must be non-negative")
        return -(-num_tokens // self.block_size)

    def can_allocate(self, count: int) -> bool:
        return count <= self.num_free

    def allocate(self) -> int:
        """Take one block (reference count 1): a free one, else evict the oldest cached one."""
        if self._free:
            block = self._free.popleft()
        elif self._evictable:
            block, _ = self._evictable.popitem(last=False)
            del self._by_hash[self._hash_of.pop(block)]
            self.evictions += 1
        else:
            raise OutOfBlocksError(f"all {self.num_blocks} blocks are in use")
        self._refs[block] = 1
        return block

    def share(self, block: int) -> None:
        """Add a holder to a block that is already in use."""
        self._check_in_use(block)
        self._refs[block] += 1

    def release(self, block: int) -> None:
        """Drop one holder. The last one frees the block, or keeps it cached if it is registered."""
        self._check_in_use(block)
        self._refs[block] -= 1
        if self._refs[block] > 0:
            return
        if block in self._hash_of:
            self._evictable[block] = None  # most recently released goes last, so it is evicted last
        else:
            self._free.append(block)

    def ref_count(self, block: int) -> int:
        self._check_range(block)
        return self._refs[block]

    # -- content index (prefix caching) ----------------------------------------------------

    def register(self, block: int, content_hash: int) -> bool:
        """Record that ``block`` (full and held) contains the content ``content_hash``.

        Returns False and changes nothing if the hash is already indexed to another block (the
        same prefix was computed twice) or the block is already registered.
        """
        self._check_in_use(block)
        if content_hash in self._by_hash or block in self._hash_of:
            return False
        self._hash_of[block] = content_hash
        self._by_hash[content_hash] = block
        return True

    def cached_hashes(self) -> list[int]:
        """The content hashes currently indexed, held or not.

        Meant for another thread to read (a router polls it): copying a dictionary's keys is a
        single step under CPython's global lock, so it never sees the dictionary mid-change.
        """
        return list(self._by_hash)

    def lookup(self, content_hash: int) -> int | None:
        """Take a holder on the block with this content, if one is cached; None otherwise."""
        block = self._by_hash.get(content_hash)
        if block is None:
            return None
        if self._refs[block] == 0:
            del self._evictable[block]  # revived: no longer a candidate for eviction
        self._refs[block] += 1
        return block

    def _check_range(self, block: int) -> None:
        if not 0 <= block < self.num_blocks:
            raise ValueError(f"block {block} is outside 0..{self.num_blocks - 1}")

    def _check_in_use(self, block: int) -> None:
        self._check_range(block)
        if self._refs[block] == 0:
            raise ValueError(f"block {block} is not allocated")
