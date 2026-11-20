"""Prefix caching: the content index, hash chain, eviction, and reuse inside the scheduler."""

import pytest
import torch

from engine.block_allocator import BlockAllocator, OutOfBlocksError
from engine.config import LlamaConfig
from engine.generate import greedy_generate
from engine.paged_cache import KVPool, PagedSequenceCache
from engine.prefix_cache import block_hashes, hash_block, match_prefix
from engine.scheduler import Scheduler
from tests.test_paged_cache import make_model

CFG = LlamaConfig.tiny()
BLOCK = 4


def tokens(seed: int, length: int) -> list[int]:
    generator = torch.manual_seed(seed)
    return torch.randint(3, CFG.vocab_size, (length,), generator=generator).tolist()


# -- hashes ---------------------------------------------------------------------------------


def test_hashes_are_deterministic_and_depend_on_content() -> None:
    assert hash_block(None, [1, 2, 3, 4]) == hash_block(None, [1, 2, 3, 4])
    assert hash_block(None, [1, 2, 3, 4]) != hash_block(None, [1, 2, 3, 5])
    assert hash_block(None, [1, 2, 3, 4]) != hash_block(None, [4, 3, 2, 1])


def test_a_block_hash_includes_everything_before_it() -> None:
    first, second = hash_block(None, [1, 1, 1, 1]), hash_block(None, [2, 2, 2, 2])
    same_tokens = [9, 9, 9, 9]
    assert hash_block(first, same_tokens) != hash_block(second, same_tokens)
    assert hash_block(first, same_tokens) != hash_block(None, same_tokens)


def test_block_hashes_covers_full_blocks_only() -> None:
    ids = list(range(10))
    chain = block_hashes(ids, 4)
    assert len(chain) == 2  # the last two tokens do not fill a block
    assert chain[0] == hash_block(None, ids[0:4])
    assert chain[1] == hash_block(chain[0], ids[4:8])
    assert block_hashes(ids[:3], 4) == []


# -- allocator index ---------------------------------------------------------------------------


def test_a_released_registered_block_stays_cached_and_can_be_taken_back() -> None:
    allocator = BlockAllocator(num_blocks=4, block_size=BLOCK)
    block = allocator.allocate()
    assert allocator.register(block, 111)
    allocator.release(block)
    assert allocator.num_used == 0 and allocator.num_free == 4  # cached blocks are available
    assert allocator.num_cached == 1
    assert allocator.lookup(111) == block
    assert allocator.ref_count(block) == 1 and allocator.num_used == 1
    assert allocator.lookup(999) is None


def test_lookup_of_a_held_block_adds_a_holder() -> None:
    allocator = BlockAllocator(num_blocks=2, block_size=BLOCK)
    block = allocator.allocate()
    allocator.register(block, 5)
    assert allocator.lookup(5) == block and allocator.ref_count(block) == 2
    allocator.release(block)
    allocator.release(block)
    assert allocator.num_used == 0 and allocator.lookup(5) == block  # still cached


def test_free_blocks_are_used_before_cached_ones_are_evicted() -> None:
    allocator = BlockAllocator(num_blocks=3, block_size=BLOCK)
    cached = allocator.allocate()
    allocator.register(cached, 7)
    allocator.release(cached)
    taken = [allocator.allocate(), allocator.allocate()]  # the two never-used blocks
    assert cached not in taken and allocator.evictions == 0
    assert allocator.lookup(7) == cached  # still there
    allocator.release(cached)
    allocator.allocate()  # now nothing is free, so the cached block is evicted
    assert allocator.evictions == 1 and allocator.lookup(7) is None


def test_eviction_takes_the_least_recently_released_block() -> None:
    allocator = BlockAllocator(num_blocks=3, block_size=BLOCK)
    blocks = [allocator.allocate() for _ in range(3)]
    for index, block in enumerate(blocks):
        allocator.register(block, 100 + index)
    for block in (blocks[1], blocks[0], blocks[2]):  # release order decides the eviction order
        allocator.release(block)
    allocator.allocate()
    assert allocator.lookup(101) is None  # the first one released went first
    assert allocator.lookup(100) == blocks[0] and allocator.lookup(102) == blocks[2]


def test_blocks_held_by_a_sequence_are_never_evicted() -> None:
    allocator = BlockAllocator(num_blocks=2, block_size=BLOCK)
    held, other = allocator.allocate(), allocator.allocate()
    allocator.register(held, 1)
    allocator.register(other, 2)
    allocator.release(other)
    assert allocator.allocate() == other  # the only evictable block
    with pytest.raises(OutOfBlocksError):
        allocator.allocate()  # `held` is in use and stays put
    assert allocator.lookup(1) == held


def test_register_refuses_duplicates_and_unheld_blocks() -> None:
    allocator = BlockAllocator(num_blocks=3, block_size=BLOCK)
    a, b = allocator.allocate(), allocator.allocate()
    assert allocator.register(a, 9)
    assert not allocator.register(b, 9)  # that content is already indexed to `a`
    assert not allocator.register(a, 10)  # `a` already has a hash
    allocator.release(b)
    with pytest.raises(ValueError):
        allocator.register(b, 11)  # not held


def test_match_prefix_stops_at_the_first_missing_block_and_honours_the_cap() -> None:
    allocator = BlockAllocator(num_blocks=8, block_size=BLOCK)
    ids = tokens(1, 12)
    chain = block_hashes(ids, BLOCK)
    for content_hash in chain:
        block = allocator.allocate()
        allocator.register(block, content_hash)
        allocator.release(block)
    blocks, hashes = match_prefix(allocator, ids, max_tokens=12)
    assert len(blocks) == 3 and hashes == chain
    for block in blocks:
        allocator.release(block)
    capped, _ = match_prefix(allocator, ids, max_tokens=11)  # one token must stay uncached
    assert len(capped) == 2
    for block in capped:
        allocator.release(block)
    changed = ids[:4] + [0] * 4 + ids[8:]
    partial, _ = match_prefix(allocator, changed, max_tokens=12)
    assert len(partial) == 1  # the second block differs, so matching stops there
    assert allocator.num_used == 1


def test_a_sequence_registers_blocks_as_they_fill_and_adopts_cached_ones() -> None:
    model = make_model(CFG)
    pool = KVPool(CFG, num_blocks=16, block_size=BLOCK, dtype=torch.float32, device="cpu")
    ids = tokens(2, 10)
    first = PagedSequenceCache(pool)
    with torch.no_grad():
        model(torch.tensor([ids]), cache=first)
    first.register_full_blocks(ids)
    assert first.hashes == block_hashes(ids, BLOCK) and pool.allocator.num_cached == 2
    first.release()

    blocks, hashes = match_prefix(pool.allocator, ids, max_tokens=len(ids) - 1)
    second = PagedSequenceCache(pool)
    second.adopt(blocks, hashes)
    assert second.length == 8 and second.blocks == blocks
    with pytest.raises(ValueError):
        second.adopt(blocks, hashes)  # not empty any more


# -- inside the scheduler ---------------------------------------------------------------------


def run(
    prompts: list[list[int]],
    new_tokens: int = 8,
    num_blocks: int = 100,
    max_batch_size: int = 4,
    prefix_caching: bool = True,
    one_at_a_time: bool = False,
) -> tuple[Scheduler, list[list[int]]]:
    model = make_model(CFG)
    pool = KVPool(CFG, num_blocks, BLOCK, torch.float32, "cpu")
    scheduler = Scheduler(
        model, pool, max_batch_size=max_batch_size, enable_prefix_caching=prefix_caching
    )
    requests = []
    for index, prompt in enumerate(prompts):
        requests.append(scheduler.add_request(index, prompt, new_tokens))
        if one_at_a_time:  # finish each before the next arrives, so its blocks are cached
            while scheduler.has_work:
                scheduler.step()
    scheduler.run_to_completion()
    return scheduler, [r.generated for r in requests]


def expected(prompts: list[list[int]], new_tokens: int = 8) -> list[list[int]]:
    model = make_model(CFG)
    return [greedy_generate(model, p, new_tokens) for p in prompts]


def test_a_shared_prefix_is_not_recomputed_and_outputs_are_unchanged() -> None:
    shared = tokens(3, 16)  # four full blocks
    prompts = [shared + tokens(4, 5), shared + tokens(5, 7), shared + tokens(6, 3)]
    scheduler, outputs = run(prompts, one_at_a_time=True)
    assert outputs == expected(prompts)
    stats = scheduler.stats()
    assert stats["prefix_hit_tokens"] == 32  # the two later requests each reuse 16 tokens
    assert stats["prefix_prompt_tokens"] == 21 + 23 + 19
    assert scheduler.pool.allocator.num_used == 0


def test_requests_in_the_same_batch_share_blocks_while_running() -> None:
    shared = tokens(7, 12)
    prompts = [shared + tokens(8, 2), shared + tokens(9, 2), shared + tokens(10, 2)]
    model = make_model(CFG)
    pool = KVPool(CFG, 100, BLOCK, torch.float32, "cpu")
    scheduler = Scheduler(model, pool, max_batch_size=4)
    requests = [scheduler.add_request(i, p, 6) for i, p in enumerate(prompts)]
    scheduler.step()  # admits and prefills all three; later ones find the first one's blocks
    held = [scheduler._caches[r.id].blocks[:3] for r in requests]
    assert held[0] == held[1] == held[2]
    assert all(pool.allocator.ref_count(b) == 3 for b in held[0])
    scheduler.run_to_completion()
    assert [r.generated for r in requests] == expected(prompts, 6)
    assert scheduler.stats()["prefix_hit_tokens"] == 24
    assert pool.allocator.num_used == 0


def test_a_prompt_that_is_entirely_cached_still_computes_its_last_token() -> None:
    prompt = tokens(11, 16)  # exactly four blocks
    scheduler, outputs = run([prompt, prompt], one_at_a_time=True)
    assert outputs == expected([prompt, prompt])
    # The second request may reuse at most 15 tokens, which is three whole blocks.
    assert scheduler.stats()["prefix_hit_tokens"] == 12


def test_the_hash_chain_stops_a_repeated_block_from_matching_the_wrong_position() -> None:
    """[A, A] must not reuse the cached block of [A, B]'s first A at its second position."""
    a, b = tokens(12, BLOCK), tokens(13, BLOCK)
    prompts = [a + b + [5], a + a + [5]]
    scheduler, outputs = run(prompts, one_at_a_time=True)
    assert outputs == expected(prompts)
    assert scheduler.stats()["prefix_hit_tokens"] == BLOCK  # only the genuinely shared first block


def test_correct_under_memory_pressure_with_eviction_and_preemption() -> None:
    shared = tokens(14, 8)
    prompts = [shared + tokens(20 + i, 3 + i) for i in range(8)]
    scheduler, outputs = run(prompts, new_tokens=10, num_blocks=14, max_batch_size=4)
    assert outputs == expected(prompts, 10)
    stats = scheduler.stats()
    assert stats["prefix_hit_tokens"] > 0
    assert stats["preemptions"] > 0 or stats["block_evictions"] > 0, "the pool was too roomy"
    assert scheduler.pool.allocator.num_used == 0


def test_a_preempted_request_finds_its_own_blocks_again() -> None:
    prompts = [tokens(30 + i, 11 + i) for i in range(5)]
    scheduler, outputs = run(prompts, new_tokens=18, num_blocks=14, max_batch_size=4)
    assert outputs == expected(prompts, 18)
    assert scheduler.total_preemptions > 0
    # Recomputing after a preemption reuses the victim's own cached blocks where they survived.
    assert scheduler.stats()["prefix_hit_tokens"] > 0


def test_turning_it_off_changes_nothing_but_the_hits() -> None:
    shared = tokens(15, 16)
    prompts = [shared + tokens(16, 4), shared + tokens(17, 6)]
    off, off_outputs = run(prompts, prefix_caching=False, one_at_a_time=True)
    on, on_outputs = run(prompts, prefix_caching=True, one_at_a_time=True)
    assert off_outputs == on_outputs == expected(prompts)
    assert off.stats()["prefix_hit_tokens"] == 0 and off.pool.allocator.num_cached == 0
    assert on.stats()["prefix_hit_tokens"] == 16
    assert off.stats()["prefix_caching"] == 0 and on.stats()["prefix_caching"] == 1


def test_a_request_waiting_for_memory_does_not_leak_the_blocks_it_matched() -> None:
    shared = tokens(18, 8)
    model = make_model(CFG)
    pool = KVPool(CFG, num_blocks=6, block_size=BLOCK, dtype=torch.float32, device="cpu")
    scheduler = Scheduler(model, pool, max_batch_size=4)
    first = scheduler.add_request(0, shared + tokens(19, 4), 8)  # 3 blocks now, 5 by the end
    scheduler.step()
    second = scheduler.add_request(1, shared + tokens(21, 8), 8)  # matches 2 blocks, needs 3 more
    scheduler.step()
    assert second in scheduler.waiting  # it could not be admitted yet
    scheduler.run_to_completion()
    assert first.generated == expected([first.prompt], 8)[0]
    assert second.generated == expected([second.prompt], 8)[0]
    assert pool.allocator.num_used == 0
