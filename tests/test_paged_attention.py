"""Decode attention that reads the blocks in place (D-025): the CPU-testable parts.

The Triton kernel itself runs only on a GPU (``tests/test_paged_attention_gpu.py``). What is checked
here is everything around it, using the float64 reference implementation as the in-place function:
the reference agrees with the engine's gather path, it cannot read slots no sequence owns, and the
cache, model, scheduler and server flag route to it correctly, with the same outputs as the gather
path.
"""

import pytest
import torch

from engine.config import LlamaConfig
from engine.paged_attention import decode_attention_reference, resolve_kernel
from engine.paged_cache import KVPool, PagedBatchCache, PagedSequenceCache
from engine.scheduler import Scheduler
from engine.server import main
from tests.paged_attention_support import gather_attention, make_case
from tests.test_paged_cache import make_model

CFG = LlamaConfig.tiny()
RAGGED = [0, 15, 16, 31, 1, 40, 7]  # cached tokens; with the new one: 1, 16, 17, 32, 2, 41, 8


def truth(case):  # type: ignore[no-untyped-def]
    return decode_attention_reference(
        case.q, case.k_pool, case.v_pool, case.tables, case.lengths, dtype=torch.float64
    )


# -- the reference against the existing gather path -----------------------------------------


@pytest.mark.parametrize("block_size", [4, 16])
def test_reference_agrees_with_the_gather_path(block_size: int) -> None:
    case = make_case(RAGGED, heads=8, kv_heads=2, head_dim=16, block_size=block_size)
    gathered = gather_attention(case.q, case.k_pool, case.v_pool, case.tables, case.lengths)
    error = (gathered.double() - truth(case)).abs().max().item()
    assert error < 1e-5, error


def test_a_wrong_reference_would_be_caught() -> None:
    # Break the length by one (attend over one position too few): the comparison must notice.
    case = make_case(RAGGED, heads=8, kv_heads=2, head_dim=16)
    gathered = gather_attention(case.q, case.k_pool, case.v_pool, case.tables, case.lengths)
    short = decode_attention_reference(
        case.q, case.k_pool, case.v_pool, case.tables, case.lengths - 1, dtype=torch.float64
    )
    # (length 0 would index -1 + 1 = 0 tokens; only compare rows that still have a token)
    assert (gathered.double()[1:] - short[1:]).abs().max().item() > 1e-3


def test_reference_never_reads_slots_no_sequence_owns() -> None:
    clean = make_case(RAGGED, heads=8, kv_heads=2, head_dim=16, seed=3)
    poisoned = make_case(RAGGED, heads=8, kv_heads=2, head_dim=16, seed=3, poison=float("nan"))
    assert not poisoned.valid.all()  # there is something to poison
    # The valid slots are identical in the two cases, so the answers must be too.
    expected, got = truth(clean), truth(poisoned)
    assert torch.isfinite(got).all()
    assert torch.equal(expected, got)


def test_padded_entries_pointing_at_an_owned_block_zero_are_not_read() -> None:
    case = make_case(
        [30, 2, 9], heads=8, kv_heads=2, head_dim=16, block_size=4, own_block_zero=True
    )
    assert case.tables[0, 0].item() == 0 and (case.tables[1:, -1] == 0).all()
    gathered = gather_attention(case.q, case.k_pool, case.v_pool, case.tables, case.lengths)
    assert (gathered.double() - truth(case)).abs().max().item() < 1e-5


def test_gather_path_is_unharmed_by_large_stale_values() -> None:
    clean = make_case(RAGGED, heads=8, kv_heads=2, head_dim=16, seed=4)
    stale = make_case(RAGGED, heads=8, kv_heads=2, head_dim=16, seed=4, poison=1e4)
    args = ("q", "k_pool", "v_pool", "tables", "lengths")
    a = gather_attention(*(getattr(clean, name) for name in args))
    b = gather_attention(*(getattr(stale, name) for name in args))
    assert torch.equal(a, b)


# -- the cache and the model route to the in-place function ---------------------------------


def new_pool(num_blocks: int = 64, block_size: int = 4) -> KVPool:
    return KVPool(CFG, num_blocks, block_size, torch.float32, "cpu")


def test_batch_cache_reads_in_place_only_when_given_a_function() -> None:
    pool = new_pool()
    caches = [PagedSequenceCache(pool)]
    assert not PagedBatchCache(caches).reads_in_place
    assert PagedBatchCache(caches, attention=decode_attention_reference).reads_in_place
    with pytest.raises(RuntimeError):
        zeros = torch.zeros(1, CFG.num_heads, 1, CFG.head_dim)
        PagedBatchCache(caches).attend_in_place(0, zeros, zeros[:, : CFG.num_kv_heads], zeros)


def test_in_place_step_gives_the_same_logits_as_the_gather_step() -> None:
    model = make_model(CFG)
    prompts = [torch.randint(0, CFG.vocab_size, (n,)).tolist() for n in (3, 11, 6, 17)]
    pool_a, pool_b = new_pool(), new_pool()
    caches_a = [PagedSequenceCache(pool_a) for _ in prompts]
    caches_b = [PagedSequenceCache(pool_b) for _ in prompts]
    with torch.no_grad():
        for prompt, a, b in zip(prompts, caches_a, caches_b, strict=True):
            ids = torch.tensor([prompt])
            model(ids, cache=a)
            model(ids, cache=b)
        tokens = torch.randint(0, CFG.vocab_size, (len(prompts), 1))
        for _ in range(6):
            gather_step = PagedBatchCache(caches_a)
            place_step = PagedBatchCache(caches_b, attention=decode_attention_reference)
            expected = model(tokens, positions=gather_step.positions(), cache=gather_step)
            got = model(tokens, positions=place_step.positions(), cache=place_step)
            assert (expected - got).abs().max() < 1e-5
            tokens = expected[:, -1].argmax(dim=-1, keepdim=True)


def test_in_place_step_does_not_gather() -> None:
    pool = new_pool()
    cache = PagedSequenceCache(pool)
    model = make_model(CFG)
    with torch.no_grad():
        model(torch.tensor([[1, 2, 3]]), cache=cache)
        step = PagedBatchCache([cache], attention=decode_attention_reference)

        def forbidden(*args: object) -> None:
            raise AssertionError("the in-place path must not gather")

        step._gather = forbidden  # type: ignore[method-assign, assignment]
        model(torch.tensor([[4]]), positions=step.positions(), cache=step)


# -- the scheduler gives the same tokens with the in-place function --------------------------


def run_scheduler(
    decode_attention: object, *, num_blocks: int, prompts: list[list[int]], new_tokens: int
) -> tuple[Scheduler, dict[int, list[int]]]:
    model = make_model(CFG)
    scheduler = Scheduler(
        model,
        new_pool(num_blocks),
        max_batch_size=4,
        decode_attention=decode_attention,  # type: ignore[arg-type]
    )
    for index, prompt in enumerate(prompts):
        scheduler.add_request(index, prompt, new_tokens)
    return scheduler, scheduler.run_to_completion()


def shared_prefix_prompts() -> list[list[int]]:
    generator = torch.manual_seed(21)
    prefix = torch.randint(0, CFG.vocab_size, (12,), generator=generator).tolist()
    tails = [
        torch.randint(0, CFG.vocab_size, (n,), generator=generator).tolist() for n in (2, 5, 9)
    ]
    lone = torch.randint(0, CFG.vocab_size, (7,), generator=generator).tolist()
    return [prefix + tail for tail in tails] + [lone, prefix + tails[0]]


@pytest.mark.parametrize("num_blocks", [200, 14])  # plenty of room, and forced preemption
def test_scheduler_tokens_do_not_change(num_blocks: int) -> None:
    prompts = shared_prefix_prompts()
    base, expected = run_scheduler(None, num_blocks=num_blocks, prompts=prompts, new_tokens=12)
    placed, got = run_scheduler(
        decode_attention_reference, num_blocks=num_blocks, prompts=prompts, new_tokens=12
    )
    assert got == expected
    assert placed.prefix_hit_tokens == base.prefix_hit_tokens > 0  # the cache was really hit
    if num_blocks == 14:
        assert placed.total_preemptions == base.total_preemptions > 0


def test_scheduler_decode_steps_really_call_the_function() -> None:
    calls: list[int] = []

    def counting(*args: torch.Tensor) -> torch.Tensor:
        calls.append(1)
        return decode_attention_reference(*args)

    prompts = shared_prefix_prompts()
    scheduler, _ = run_scheduler(counting, num_blocks=200, prompts=prompts, new_tokens=6)
    assert len(calls) == scheduler.decode_steps * CFG.num_layers > 0


# -- choosing the implementation ------------------------------------------------------------


def test_resolve_kernel() -> None:
    assert resolve_kernel("reference") is decode_attention_reference
    with pytest.raises(ValueError):
        resolve_kernel("nonsense")


@pytest.mark.skipif(torch.cuda.is_available(), reason="checks the no-GPU message")
def test_triton_kernel_without_a_gpu_says_so() -> None:
    with pytest.raises(RuntimeError, match="CUDA"):
        resolve_kernel("triton")


def test_server_flag_needs_the_llama_backend_and_a_gpu() -> None:
    with pytest.raises(SystemExit, match="llama backend"):
        main(["--backend", "mock", "--paged-attention-kernel"])
    if not torch.cuda.is_available():
        with pytest.raises(SystemExit, match="paged-attention-kernel"):
            main(["--backend", "llama", "--model-dir", "unused", "--paged-attention-kernel"])


# -- the benchmark tools can switch the kernel on --------------------------------------------


def test_a_triton_kernel_name_counts_as_attention_in_the_profile() -> None:
    from bench.profile_decode import categorize

    assert categorize("_decode_kernel") == "attention"


def test_profile_batch_and_scheduler_eval_pass_the_function_through() -> None:
    from bench import profile_decode as pd
    from bench import scheduler_eval as se

    calls: list[int] = []

    def counting(*args: torch.Tensor) -> torch.Tensor:
        calls.append(1)
        return decode_attention_reference(*args)

    model = make_model(CFG)
    with torch.inference_mode():
        pd.profile_batch(
            model,
            new_pool(256),
            batch_size=0,
            prompt_len=[3, 12, 7],
            steps=2,
            warmup=1,
            decode_attention=counting,
        )
    assert calls  # the profiled steps went through the function

    calls.clear()
    prompts = [[1, 2, 3, 4, 5], [6, 7, 8]]
    config = se.SchedulerConfig("served", num_blocks=64, max_batch_size=2, prefix_caching=True)
    with torch.inference_mode():
        plain, _ = se.run_scheduler(model, prompts, 5, config, torch.float32)
        placed, _ = se.run_scheduler(model, prompts, 5, config, torch.float32, counting)
    assert calls and placed == plain
