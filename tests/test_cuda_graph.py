"""The CUDA-graph decode path (D-020), without a GPU: the static-buffer, padded, bucketed step.

``CudaGraphDecoder(capture=False)`` runs the identical static-buffer code eagerly, so what a CPU can
check is everything except the capture itself: padding rows must not touch live sequences, real
rows must get the logits they would get eagerly, and the scheduler must give the same tokens.
The capture, and graph-against-eager bit equality, are checked on the GPU
(``tests/test_cuda_graph_gpu.py``).
"""

from __future__ import annotations

import pytest
import torch

from engine.config import LlamaConfig
from engine.cuda_graph import CudaGraphDecoder, default_buckets
from engine.paged_attention import decode_attention_reference
from engine.paged_cache import KVPool, PagedBatchCache, PagedSequenceCache
from engine.scheduler import Scheduler
from tests.test_paged_cache import make_model

CFG = LlamaConfig.tiny()
PROMPT_LENGTHS = [3, 11, 6, 17, 9, 5]


def new_pool(num_blocks: int = 128) -> KVPool:
    return KVPool(CFG, num_blocks, 4, torch.float32, "cpu")


def prefilled(pool: KVPool, count: int, seed: int = 1) -> list[PagedSequenceCache]:
    model = make_model(CFG)
    generator = torch.manual_seed(seed)
    caches = []
    with torch.no_grad():
        for length in PROMPT_LENGTHS[:count]:
            ids = torch.randint(0, CFG.vocab_size, (1, length), generator=generator)
            cache = PagedSequenceCache(pool)
            model(ids, cache=cache)
            caches.append(cache)
    return caches


def eager_step(
    pool: KVPool, caches: list[PagedSequenceCache], tokens: torch.Tensor
) -> torch.Tensor:
    step = PagedBatchCache(caches, attention=decode_attention_reference)
    with torch.no_grad():
        return make_model(CFG)(tokens, positions=step.positions(), cache=step)


def test_default_buckets() -> None:
    assert default_buckets(64) == [1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64]
    assert default_buckets(6) == [1, 2, 4, 6]
    assert default_buckets(1) == [1]


def test_bucket_for_picks_the_smallest_that_fits() -> None:
    decoder = CudaGraphDecoder(
        make_model(CFG), new_pool(), decode_attention_reference, 8, buckets=[2, 4, 8], capture=False
    )
    assert [decoder.bucket_for(n) for n in (1, 2, 3, 4, 5, 8, 9)] == [2, 2, 4, 4, 8, 8, None]


def test_the_decoder_reserves_a_scratch_block_for_good() -> None:
    pool = new_pool(16)
    before = pool.allocator.num_free
    decoder = CudaGraphDecoder(make_model(CFG), pool, decode_attention_reference, 4, capture=False)
    assert pool.allocator.num_free == before - 1
    assert pool.allocator.ref_count(decoder.scratch_block) == 1


@pytest.mark.parametrize("buckets", [[4], [8], [2, 4, 8]])
def test_real_rows_get_the_logits_they_get_eagerly(buckets: list[int]) -> None:
    model = make_model(CFG)
    tokens = torch.randint(0, CFG.vocab_size, (4, 1), generator=torch.manual_seed(5))
    pool_a, pool_b = new_pool(), new_pool()
    caches_a, caches_b = prefilled(pool_a, 4), prefilled(pool_b, 4)
    decoder = CudaGraphDecoder(
        model, pool_b, decode_attention_reference, 8, buckets=buckets, capture=False
    )
    expected = eager_step(pool_a, caches_a, tokens)
    with torch.no_grad():
        got = decoder.run(PagedBatchCache(caches_b, attention=decode_attention_reference), tokens)
    assert got.shape == expected.shape
    assert (got - expected).abs().max() < 1e-5
    assert [c.length for c in caches_b] == [c.length for c in caches_a]  # lengths advanced


def test_padding_rows_never_touch_a_live_sequence() -> None:
    """Step with 4 rows, then 2: rows 2 and 3 are padding but their static tables still hold the
    ids of live blocks from the first step. They must write to the scratch block only."""
    model = make_model(CFG)
    pool_a, pool_b = new_pool(), new_pool()
    caches_a, caches_b = prefilled(pool_a, 4), prefilled(pool_b, 4)
    decoder = CudaGraphDecoder(
        model, pool_b, decode_attention_reference, 4, buckets=[4], capture=False
    )
    generator = torch.manual_seed(8)
    for count in (4, 2):
        tokens = torch.randint(0, CFG.vocab_size, (count, 1), generator=generator)
        expected = eager_step(pool_a, caches_a[:count], tokens)
        with torch.no_grad():
            got = decoder.run(
                PagedBatchCache(caches_b[:count], attention=decode_attention_reference), tokens
            )
        assert (got - expected).abs().max() < 1e-5
    # Every block any live sequence owns holds what the eager run put there.
    for a, b in zip(caches_a, caches_b, strict=True):
        for block_a, block_b in zip(a.blocks, b.blocks, strict=True):
            assert (pool_a.k[:, block_a] - pool_b.k[:, block_b]).abs().max() < 1e-5
            assert (pool_a.v[:, block_a] - pool_b.v[:, block_b]).abs().max() < 1e-5
    # And the padding rows did write somewhere: the scratch block is no longer all zeros.
    assert pool_b.k[:, decoder.scratch_block].abs().max() > 0


def test_a_step_too_wide_for_the_static_table_is_refused_and_falls_back() -> None:
    model = make_model(CFG)
    pool = new_pool()
    decoder = CudaGraphDecoder(
        model, pool, decode_attention_reference, 4, max_blocks=2, buckets=[4], capture=False
    )
    caches = prefilled(pool, 2)  # 11 tokens need 3 blocks of 4: wider than 2
    step = PagedBatchCache(caches, attention=decode_attention_reference)
    assert not decoder.can_run(step)
    with pytest.raises(ValueError):
        decoder.run(step, torch.zeros(2, 1, dtype=torch.long))


def make_prompts() -> list[list[int]]:
    generator = torch.manual_seed(21)
    prefix = torch.randint(0, CFG.vocab_size, (12,), generator=generator).tolist()
    tails = [
        torch.randint(0, CFG.vocab_size, (n,), generator=generator).tolist() for n in (2, 5, 9)
    ]
    lone = torch.randint(0, CFG.vocab_size, (7,), generator=generator).tolist()
    return [prefix + tail for tail in tails] + [lone, prefix + tails[0]]


def serve(use_graphs: bool, num_blocks: int, max_blocks: int = 64) -> tuple[Scheduler, dict]:  # type: ignore[type-arg]
    model = make_model(CFG)
    # The decoder holds one block as scratch, so give that pool one more: equal usable capacity.
    pool = new_pool(num_blocks + (1 if use_graphs else 0))
    decoder = (
        CudaGraphDecoder(
            model, pool, decode_attention_reference, 4, max_blocks=max_blocks, capture=False
        )
        if use_graphs
        else None
    )
    scheduler = Scheduler(
        model,
        pool,
        max_batch_size=4,
        decode_attention=decode_attention_reference,
        graph_decoder=decoder,
    )
    for index, prompt in enumerate(make_prompts()):
        scheduler.add_request(index, prompt, 12)
    return scheduler, scheduler.run_to_completion()


@pytest.mark.parametrize("num_blocks", [200, 15])  # plenty of room, and forced preemption
def test_scheduler_tokens_do_not_change_with_the_graph_decoder(num_blocks: int) -> None:
    base, expected = serve(False, num_blocks)
    graphed, got = serve(True, num_blocks)
    assert got == expected
    assert graphed.graph_decoder is not None
    assert graphed.graph_decoder.stats.replays == graphed.decode_steps > 0
    assert graphed.stats()["graph_replays"] == graphed.decode_steps
    assert graphed.prefix_hit_tokens == base.prefix_hit_tokens > 0
    if num_blocks == 15:
        assert graphed.total_preemptions == base.total_preemptions > 0


def test_wide_steps_run_eagerly_and_are_counted() -> None:
    base, expected = serve(False, 200)
    graphed, got = serve(True, 200, max_blocks=6)  # sequences grow past 6 blocks of 4 tokens
    assert got == expected
    stats = graphed.graph_decoder.stats  # type: ignore[union-attr]
    assert stats.eager_fallbacks > 0 and stats.replays > 0
    assert stats.replays + stats.eager_fallbacks == base.decode_steps


def test_more_sequences_than_the_largest_bucket_are_refused() -> None:
    pool = new_pool()
    decoder = CudaGraphDecoder(
        make_model(CFG), pool, decode_attention_reference, 2, buckets=[2], capture=False
    )
    step = PagedBatchCache(prefilled(pool, 3), attention=decode_attention_reference)
    assert not decoder.can_run(step)


def test_benchmark_tools_pass_the_decoder_through_and_refuse_nonsense() -> None:
    from bench import profile_decode as pd
    from bench import scheduler_eval as se
    from engine.server import main

    model = make_model(CFG)
    pool = new_pool(256)
    decoder = CudaGraphDecoder(model, pool, decode_attention_reference, 3, capture=False)
    with torch.inference_mode():
        pd.profile_batch(
            model,
            pool,
            batch_size=0,
            prompt_len=[3, 12, 7],
            steps=2,
            warmup=1,
            decode_attention=decode_attention_reference,
            graph_decoder=decoder,
        )
    assert decoder.stats.replays > 0

    config = se.SchedulerConfig("served", num_blocks=64, max_batch_size=2, prefix_caching=True)
    with pytest.raises(RuntimeError, match="CUDA"):  # a CPU cannot capture a graph
        se.run_scheduler(
            model, [[1, 2, 3]], 3, config, torch.float32, decode_attention_reference, True
        )
    with pytest.raises(ValueError, match="kernel"):
        se.run_scheduler(model, [[1, 2, 3]], 3, config, torch.float32, None, True)
    base = ["--model-dir", "m", "--out", "o", "--label", "x"]
    with pytest.raises(SystemExit):
        se.parse_args([*base, "--cuda-graphs"])
    with pytest.raises(SystemExit, match="paged-attention-kernel"):
        main(["--backend", "mock", "--cuda-graphs"])


def test_the_evaluations_leak_check_allows_the_scratch_block_and_still_catches_leaks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from bench import scheduler_eval as se

    model = make_model(CFG)
    config = se.SchedulerConfig("no-cache", num_blocks=64, max_batch_size=2, prefix_caching=False)
    prompts = [[1, 2, 3, 4, 5], [6, 7, 8]]
    with torch.inference_mode():
        plain, _ = se.run_scheduler(model, prompts, 4, config, torch.float32)
        graphed, _ = se.run_scheduler(
            model,
            prompts,
            4,
            config,
            torch.float32,
            decode_attention_reference,
            True,
            graph_capture=False,
        )
    assert graphed == plain  # the one scratch block is not reported as a leak

    original = Scheduler.run_to_completion

    def leaky(self: Scheduler) -> dict[int, list[int]]:
        outputs = original(self)
        self.pool.allocator.allocate()  # a block nobody gives back
        return outputs

    monkeypatch.setattr(Scheduler, "run_to_completion", leaky)
    with torch.inference_mode(), pytest.raises(RuntimeError, match="leaked"):
        se.run_scheduler(
            model,
            prompts,
            4,
            config,
            torch.float32,
            decode_attention_reference,
            True,
            graph_capture=False,
        )
