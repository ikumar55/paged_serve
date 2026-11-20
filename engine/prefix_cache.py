"""Block hashes for prefix caching (D-005) and the matching of a prompt against cached blocks.

A block's hash covers its own tokens *and its parent block's hash*, so equal hashes mean equal
token sequences from the start of the prompt up to and including that block. Hashing the tokens
alone would not be safe: with ``A`` and ``B`` two blocks of tokens, the prompt ``[A, A]`` would
find the cached block for ``A`` at its second position, where its keys carry the rotary position
of the first. The hash is a 128-bit digest, so an accidental collision (which would silently reuse
the wrong keys and values) is not a practical concern.

The idea, hashing each block together with its parent's hash, is the one documented for vLLM's
automatic prefix caching; the code is this project's own.
"""

from __future__ import annotations

import hashlib
from array import array
from collections.abc import Sequence

from engine.block_allocator import BlockAllocator

_ROOT = b"\xff" * 16


def hash_block(parent: int | None, tokens: Sequence[int]) -> int:
    """Hash of one full block of ``tokens`` that follows the block hashed as ``parent``."""
    digest = hashlib.blake2b(digest_size=16)
    digest.update(_ROOT if parent is None else parent.to_bytes(16, "big"))
    digest.update(array("I", tokens).tobytes())
    return int.from_bytes(digest.digest(), "big")


def block_hashes(tokens: Sequence[int], block_size: int) -> list[int]:
    """The chain of hashes of every full block of ``tokens`` (a partial last block has none)."""
    hashes: list[int] = []
    parent: int | None = None
    for start in range(0, len(tokens) - block_size + 1, block_size):
        parent = hash_block(parent, tokens[start : start + block_size])
        hashes.append(parent)
    return hashes


def match_prefix(
    allocator: BlockAllocator, tokens: Sequence[int], max_tokens: int
) -> tuple[list[int], list[int]]:
    """Take holders on the longest run of cached blocks that starts ``tokens``.

    At most ``max_tokens`` tokens are matched (whole blocks only); the caller uses this to leave at
    least one token to compute, since the model needs one forward pass to produce the next token.
    Returns the blocks (each with a holder added, to be released by the caller) and their hashes.
    """
    size = allocator.block_size
    blocks: list[int] = []
    hashes: list[int] = []
    parent: int | None = None
    while (len(blocks) + 1) * size <= min(max_tokens, len(tokens)):
        start = len(blocks) * size
        content_hash = hash_block(parent, tokens[start : start + size])
        block = allocator.lookup(content_hash)
        if block is None:
            break
        blocks.append(block)
        hashes.append(content_hash)
        parent = content_hash
    return blocks, hashes
