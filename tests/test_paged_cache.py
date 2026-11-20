"""Block allocator and paged KV cache: bookkeeping, and equality with the contiguous cache."""

import pytest
import torch

from engine.block_allocator import BlockAllocator, OutOfBlocksError
from engine.config import LlamaConfig
from engine.kv_cache import ContiguousKVCache
from engine.model import Llama
from engine.paged_cache import KVPool, PagedSequenceCache


def test_blocks_for_rounds_up() -> None:
    allocator = BlockAllocator(num_blocks=4, block_size=16)
    assert [allocator.blocks_for(n) for n in (0, 1, 16, 17, 32, 33)] == [0, 1, 1, 2, 2, 3]
    with pytest.raises(ValueError):
        allocator.blocks_for(-1)


def test_allocate_until_empty_then_raise() -> None:
    allocator = BlockAllocator(num_blocks=3, block_size=4)
    got = {allocator.allocate() for _ in range(3)}
    assert got == {0, 1, 2}
    assert allocator.num_free == 0 and allocator.num_used == 3
    with pytest.raises(OutOfBlocksError):
        allocator.allocate()


def test_release_returns_block_to_the_pool() -> None:
    allocator = BlockAllocator(num_blocks=2, block_size=4)
    a, b = allocator.allocate(), allocator.allocate()
    allocator.release(a)
    assert allocator.num_free == 1
    assert allocator.allocate() == a
    assert allocator.ref_count(b) == 1


def test_shared_block_is_freed_only_by_its_last_holder() -> None:
    allocator = BlockAllocator(num_blocks=2, block_size=4)
    block = allocator.allocate()
    allocator.share(block)
    assert allocator.ref_count(block) == 2
    allocator.release(block)
    assert allocator.num_free == 1  # the other block only; this one is still held
    allocator.release(block)
    assert allocator.num_free == 2


def test_double_release_and_unknown_blocks_are_rejected() -> None:
    allocator = BlockAllocator(num_blocks=2, block_size=4)
    block = allocator.allocate()
    allocator.release(block)
    with pytest.raises(ValueError):
        allocator.release(block)
    with pytest.raises(ValueError):
        allocator.share(block)
    with pytest.raises(ValueError):
        allocator.release(99)


def test_allocator_rejects_nonpositive_sizes() -> None:
    with pytest.raises(ValueError):
        BlockAllocator(0, 4)
    with pytest.raises(ValueError):
        BlockAllocator(4, 0)


def make_model(cfg: LlamaConfig) -> Llama:
    torch.manual_seed(0)
    model = Llama(cfg)
    for parameter in model.parameters():
        torch.nn.init.normal_(parameter, std=0.2)  # the default init is too flat to test with
    return model.eval()


def scrambled_pool(cfg: LlamaConfig, num_blocks: int, block_size: int) -> KVPool:
    """A pool whose free list is out of order, so a sequence's blocks are not contiguous."""
    pool = KVPool(cfg, num_blocks, block_size, torch.float32, "cpu")
    held = [pool.allocator.allocate() for _ in range(num_blocks)]
    order = torch.randperm(num_blocks, generator=torch.manual_seed(7)).tolist()
    for index in order:
        pool.allocator.release(held[index])
    return pool


@pytest.mark.parametrize("block_size", [1, 3, 4, 16])
def test_paged_logits_equal_contiguous_logits_exactly(block_size: int) -> None:
    cfg = LlamaConfig.tiny()
    model = make_model(cfg)
    ids = torch.randint(0, cfg.vocab_size, (1, 23), generator=torch.manual_seed(1))
    prompt_len = 9

    pool = scrambled_pool(cfg, num_blocks=64, block_size=block_size)
    paged = PagedSequenceCache(pool)
    contiguous = ContiguousKVCache(cfg, 1, 32, model.dtype, model.device)

    with torch.no_grad():
        for start, stop in [(0, prompt_len), *[(i, i + 1) for i in range(prompt_len, 23)]]:
            expected = model(ids[:, start:stop], cache=contiguous)
            actual = model(ids[:, start:stop], cache=paged)
            assert torch.equal(expected, actual), f"logits differ at tokens {start}:{stop}"

    assert len(paged.blocks) == pool.allocator.blocks_for(23)
    # The test is only meaningful if the blocks really were out of order (checked where the
    # sequence spans enough blocks for a random order to be unlikely to come out sorted).
    if len(paged.blocks) >= 6:
        assert paged.blocks != sorted(paged.blocks)


def test_two_sequences_sharing_a_pool_do_not_corrupt_each_other() -> None:
    cfg = LlamaConfig.tiny()
    model = make_model(cfg)
    a_ids = torch.randint(0, cfg.vocab_size, (1, 14), generator=torch.manual_seed(2))
    b_ids = torch.randint(0, cfg.vocab_size, (1, 14), generator=torch.manual_seed(3))

    pool = KVPool(cfg, num_blocks=32, block_size=4, dtype=torch.float32, device="cpu")
    a, b = PagedSequenceCache(pool), PagedSequenceCache(pool)
    solo_a = ContiguousKVCache(cfg, 1, 16, model.dtype, model.device)
    solo_b = ContiguousKVCache(cfg, 1, 16, model.dtype, model.device)

    with torch.no_grad():
        # Interleave the two sequences token by token after a short prompt each.
        for cache, solo, ids in ((a, solo_a, a_ids), (b, solo_b, b_ids)):
            assert torch.equal(model(ids[:, :5], cache=cache), model(ids[:, :5], cache=solo))
        for i in range(5, 14):
            for cache, solo, ids in ((a, solo_a, a_ids), (b, solo_b, b_ids)):
                step = ids[:, i : i + 1]
                assert torch.equal(model(step, cache=cache), model(step, cache=solo))

    assert set(a.blocks).isdisjoint(b.blocks)


def test_release_frees_blocks_and_the_cache_can_be_reused() -> None:
    cfg = LlamaConfig.tiny()
    model = make_model(cfg)
    ids = torch.randint(0, cfg.vocab_size, (1, 10), generator=torch.manual_seed(4))
    pool = KVPool(cfg, num_blocks=8, block_size=4, dtype=torch.float32, device="cpu")
    cache = PagedSequenceCache(pool)
    with torch.no_grad():
        first = model(ids, cache=cache)
    assert pool.allocator.num_used == 3
    cache.release()
    assert pool.allocator.num_used == 0 and cache.length == 0 and cache.blocks == []
    with torch.no_grad():
        again = model(ids, cache=cache)
    assert torch.equal(first, again)  # stale data in recycled blocks must never leak through


def test_running_out_of_blocks_takes_nothing_and_can_be_retried() -> None:
    cfg = LlamaConfig.tiny()
    model = make_model(cfg)
    pool = KVPool(cfg, num_blocks=2, block_size=4, dtype=torch.float32, device="cpu")
    cache = PagedSequenceCache(pool)
    too_long = torch.randint(0, cfg.vocab_size, (1, 9), generator=torch.manual_seed(5))
    with torch.no_grad(), pytest.raises(OutOfBlocksError):
        model(too_long, cache=cache)
    assert pool.allocator.num_used == 0  # nothing half-allocated
    assert cache.length == 0

    fits = too_long[:, :8]
    with torch.no_grad():
        model(fits, cache=cache)
    assert pool.allocator.num_free == 0


def test_batch_larger_than_one_is_refused() -> None:
    cfg = LlamaConfig.tiny()
    model = make_model(cfg)
    pool = KVPool(cfg, num_blocks=8, block_size=4, dtype=torch.float32, device="cpu")
    with pytest.raises(ValueError, match="batch size 1"), torch.no_grad():
        model(torch.zeros(2, 3, dtype=torch.long), cache=PagedSequenceCache(pool))


def test_bytes_per_block_matches_the_tensor_layout() -> None:
    cfg = LlamaConfig.tiny()
    pool = KVPool(cfg, num_blocks=4, block_size=8, dtype=torch.float32, device="cpu")
    expected = 2 * cfg.num_layers * cfg.num_kv_heads * 8 * cfg.head_dim * 4
    assert pool.bytes_per_block() == expected
