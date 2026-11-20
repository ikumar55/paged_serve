"""Batched paged decoding and the scheduler, checked against one-sequence-at-a-time greedy decoding.

The reference is ``greedy_generate`` (single request, contiguous cache). Batching changes the order
of floating-point operations, so logits are compared with a tight tolerance and generated tokens
exactly; the random-weight model has token gaps far larger than that noise.
"""

import pytest
import torch

from engine.block_allocator import OutOfBlocksError
from engine.config import LlamaConfig
from engine.generate import greedy_generate
from engine.kv_cache import ContiguousKVCache
from engine.paged_cache import KVPool, PagedBatchCache, PagedSequenceCache
from engine.scheduler import Scheduler
from tests.test_paged_cache import make_model

CFG = LlamaConfig.tiny()
LOGIT_TOL = 1e-5


def make_prompts(lengths: list[int], seed: int = 10) -> list[list[int]]:
    generator = torch.manual_seed(seed)
    return [torch.randint(0, CFG.vocab_size, (n,), generator=generator).tolist() for n in lengths]


def new_pool(num_blocks: int, block_size: int = 4) -> KVPool:
    return KVPool(CFG, num_blocks, block_size, torch.float32, "cpu")


def test_batched_decode_matches_each_sequence_alone() -> None:
    model = make_model(CFG)
    prompts = make_prompts([3, 11, 6, 17])
    pool = new_pool(num_blocks=64)
    caches = [PagedSequenceCache(pool) for _ in prompts]
    refs = [ContiguousKVCache(CFG, 1, 40, model.dtype, model.device) for _ in prompts]
    with torch.no_grad():
        for prompt, cache, ref in zip(prompts, caches, refs, strict=True):
            ids = torch.tensor([prompt])
            assert torch.equal(model(ids, cache=cache), model(ids, cache=ref))

        generator = torch.manual_seed(11)
        for _ in range(8):  # eight decode steps over sequences at four different lengths
            tokens = torch.randint(0, CFG.vocab_size, (len(prompts), 1), generator=generator)
            step = PagedBatchCache(caches)
            batched = model(tokens, positions=step.positions(), cache=step)
            for row, ref in enumerate(refs):
                alone = model(tokens[row : row + 1], cache=ref)
                assert (batched[row] - alone[0]).abs().max() < LOGIT_TOL


def test_batch_step_claims_blocks_all_or_nothing() -> None:
    model = make_model(CFG)
    pool = new_pool(num_blocks=4, block_size=4)
    a, b = PagedSequenceCache(pool), PagedSequenceCache(pool)
    with torch.no_grad():
        model(torch.tensor([make_prompts([8])[0]]), cache=a)  # 2 blocks, both full
        model(torch.tensor([make_prompts([8], seed=3)[0]]), cache=b)  # 2 blocks, both full
    assert pool.allocator.num_free == 0
    with pytest.raises(OutOfBlocksError):
        PagedBatchCache([a, b])  # each needs a third block
    assert len(a.blocks) == 2 and len(b.blocks) == 2  # nothing half-claimed


def reference_outputs(
    requests: list[tuple[list[int], int]], eos: int | None = None
) -> list[list[int]]:
    model = make_model(CFG)
    return [greedy_generate(model, prompt, new, eos) for prompt, new in requests]


def test_scheduler_matches_single_request_decoding() -> None:
    model = make_model(CFG)
    lengths, new_tokens = [3, 12, 7, 20, 5, 9], [10, 4, 15, 6, 12, 8]
    requests = list(zip(make_prompts(lengths), new_tokens, strict=True))
    expected = reference_outputs(requests)

    pool = new_pool(num_blocks=200)
    scheduler = Scheduler(model, pool, max_batch_size=4)
    for index, (prompt, new) in enumerate(requests):
        scheduler.add_request(index, prompt, new)
    outputs = scheduler.run_to_completion()

    assert [outputs[i] for i in range(len(requests))] == expected
    assert scheduler.total_preemptions == 0
    assert pool.allocator.num_used == 0  # every block came back


def test_scheduler_never_runs_more_than_max_batch_size() -> None:
    model = make_model(CFG)
    scheduler = Scheduler(model, new_pool(200), max_batch_size=3)
    for index, prompt in enumerate(make_prompts([4] * 8)):
        scheduler.add_request(index, prompt, 6)
    biggest = 0
    while scheduler.has_work:
        result = scheduler.step()
        biggest = max(biggest, len(scheduler.running), result.decoded)
    assert biggest == 3


def test_preemption_by_recompute_still_gives_identical_output() -> None:
    model = make_model(CFG)
    lengths, new_tokens = [10, 9, 11, 8, 12], [20, 18, 16, 20, 14]
    requests = list(zip(make_prompts(lengths, seed=21), new_tokens, strict=True))
    expected = reference_outputs(requests)

    # Each request needs up to 8 blocks of 4 tokens; 14 blocks cannot hold three at once.
    pool = new_pool(num_blocks=14, block_size=4)
    scheduler = Scheduler(model, pool, max_batch_size=4)
    for index, (prompt, new) in enumerate(requests):
        scheduler.add_request(index, prompt, new)
    streamed: dict[int, list[int]] = {i: [] for i in range(len(requests))}
    while scheduler.has_work:
        for request_id, tokens in scheduler.step().new_tokens.items():
            streamed[request_id].extend(tokens)

    assert scheduler.total_preemptions > 0, "the pool was meant to be too small"
    # What a client would have received: each token exactly once, in order, despite preemption.
    assert [streamed[i] for i in range(len(requests))] == expected
    assert pool.allocator.num_used == 0


def test_requests_arriving_between_steps_are_batched_in() -> None:
    model = make_model(CFG)
    requests = list(zip(make_prompts([6, 9, 4], seed=31), [12, 12, 12], strict=True))
    expected = reference_outputs(requests)

    pool = new_pool(100)
    scheduler = Scheduler(model, pool, max_batch_size=8)
    submitted = [scheduler.add_request(0, *requests[0])]
    for _ in range(3):
        scheduler.step()
    submitted.append(scheduler.add_request(1, *requests[1]))
    scheduler.step()
    submitted.append(scheduler.add_request(2, *requests[2]))
    saw_mixed = False
    while scheduler.has_work:
        result = scheduler.step()
        saw_mixed |= bool(result.prefilled) and result.decoded > 0
    assert saw_mixed  # one step prefilled a new request while others were decoding
    assert [r.generated for r in submitted] == expected
    assert all(r.finished for r in submitted)


def test_eos_stops_a_request_early_and_frees_its_blocks() -> None:
    model = make_model(CFG)
    prompt = make_prompts([7], seed=41)[0]
    full = greedy_generate(model, prompt, 12)
    eos = full[3]
    expected = greedy_generate(model, prompt, 12, eos)
    assert len(expected) <= 4  # the reference stops at the first occurrence of that token

    pool = new_pool(50)
    scheduler = Scheduler(model, pool, max_batch_size=2)
    request = scheduler.add_request(0, prompt, 12, eos_token_id=eos)
    scheduler.run_to_completion()
    assert request.generated == expected
    assert pool.allocator.num_used == 0


def test_requests_are_validated_when_added() -> None:
    model = make_model(CFG)
    scheduler = Scheduler(model, new_pool(num_blocks=4, block_size=4))  # 16 token slots
    scheduler.add_request(0, [1, 2, 3], 5)
    with pytest.raises(ValueError, match="already in use"):
        scheduler.add_request(0, [1], 1)
    with pytest.raises(ValueError, match="at least one token"):
        scheduler.add_request(1, [], 1)
    with pytest.raises(ValueError, match="positive"):
        scheduler.add_request(2, [1], 0)
    with pytest.raises(ValueError, match="only has 4"):
        scheduler.add_request(3, [1] * 10, 10)  # 20 tokens can never fit in 16 slots
    with pytest.raises(ValueError, match="exceed the model's maximum"):
        scheduler.add_request(4, [1] * 250, 20)
