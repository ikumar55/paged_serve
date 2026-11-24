"""GPU checks of the CUDA-graph decode step (D-020 amendment 2, criterion (a)(iv)).

Run by hand on the GPU machine: ``pytest -m gpu tests/test_cuda_graph_gpu.py -s``. Skipped without a
CUDA device or Triton. A tiny random-weight Llama (heads, head size and block size are powers of
two, so the Triton kernel applies) is run in fp32 with TF32 off:

* a captured step with no padding is bit-equal to the eager kernel path;
* a captured step with padding rows gives the real rows the same logits, bit for bit, as the eager
  kernel path at the real batch size, over several steps in which the batch shrinks, and leaves
  every live sequence's blocks bit-equal to the eager run's.
"""

from __future__ import annotations

import pytest
import torch

from engine.config import LlamaConfig
from engine.cuda_graph import CudaGraphDecoder
from engine.model import Llama
from engine.paged_cache import KVPool, PagedBatchCache, PagedSequenceCache

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU"),
]
pytest.importorskip("triton")

CFG = LlamaConfig.tiny()
LENGTHS = [3, 11, 6, 17, 9, 5, 14, 2]


@pytest.fixture(autouse=True)
def strict_fp32() -> object:
    old = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    yield
    torch.backends.cuda.matmul.allow_tf32 = old


def kernel() -> object:
    from engine.triton_paged_attention import paged_decode_attention

    return paged_decode_attention


def make_model() -> Llama:
    torch.manual_seed(0)
    model = Llama(CFG, dtype=torch.float32, device="cuda")
    for parameter in model.parameters():
        torch.nn.init.normal_(parameter, std=0.2)
    return model.eval()


def prefilled(model: Llama, count: int) -> tuple[KVPool, list[PagedSequenceCache]]:
    pool = KVPool(CFG, 128, 4, torch.float32, "cuda")
    generator = torch.Generator().manual_seed(1)
    caches = []
    with torch.inference_mode():
        for length in LENGTHS[:count]:
            ids = torch.randint(0, CFG.vocab_size, (1, length), generator=generator).cuda()
            cache = PagedSequenceCache(pool)
            model(ids, cache=cache)
            caches.append(cache)
    return pool, caches


def eager(model: Llama, caches: list[PagedSequenceCache], tokens: torch.Tensor) -> torch.Tensor:
    step = PagedBatchCache(caches, attention=kernel())  # type: ignore[arg-type]
    with torch.inference_mode():
        return model(tokens, positions=step.positions(), cache=step).clone()


def graphed(
    decoder: CudaGraphDecoder, caches: list[PagedSequenceCache], tokens: torch.Tensor
) -> torch.Tensor:
    step = PagedBatchCache(caches, attention=kernel())  # type: ignore[arg-type]
    with torch.inference_mode():
        out = decoder.run(step, tokens).clone()
    torch.cuda.synchronize()
    return out


def test_a_captured_step_without_padding_is_bit_equal_to_eager() -> None:
    model = make_model()
    _pool_a, caches_a = prefilled(model, 4)
    pool_b, caches_b = prefilled(model, 4)
    decoder = CudaGraphDecoder(model, pool_b, kernel(), 4, buckets=[4])  # type: ignore[arg-type]
    assert decoder.buckets[4].graph is not None  # really captured
    tokens = torch.randint(0, CFG.vocab_size, (4, 1), generator=torch.Generator().manual_seed(5))
    tokens = tokens.cuda()
    assert torch.equal(eager(model, caches_a, tokens), graphed(decoder, caches_b, tokens))


FP32_NOISE = 1e-6  # a few ulps of fp32 on logits of this size


def test_padded_steps_leave_real_rows_and_live_blocks_equal_to_eager() -> None:
    """Padding changes the matrix shape the GEMMs see (a batch of 1 is padded to 2), and cuBLAS may
    pick a different kernel for a different shape, so the comparison is bit-exact only when the
    shapes agree. Session 1 of D-020 found it bit-exact for 7 of 8 steps and 1.8e-7 off for a
    single real row padded to two, so this checks the fp32 noise level, not bit equality."""
    model = make_model()
    pool_a, caches_a = prefilled(model, 6)
    pool_b, caches_b = prefilled(model, 6)
    decoder = CudaGraphDecoder(model, pool_b, kernel(), 8, buckets=[2, 4, 8])  # type: ignore[arg-type]
    generator = torch.Generator().manual_seed(8)
    worst, exact = 0.0, 0
    for count in (6, 6, 5, 4, 3, 3, 2, 1):  # the batch shrinks, so the padding changes each time
        tokens = torch.randint(0, CFG.vocab_size, (count, 1), generator=generator).cuda()
        expected = eager(model, caches_a[:count], tokens)
        got = graphed(decoder, caches_b[:count], tokens)
        worst = max(worst, (got - expected).abs().max().item())
        exact += int(torch.equal(expected, got))
        assert (got - expected).abs().max().item() <= FP32_NOISE, count
    for a, b in zip(caches_a, caches_b, strict=True):
        for block_a, block_b in zip(a.blocks, b.blocks, strict=True):
            assert (pool_a.k[:, block_a] - pool_b.k[:, block_b]).abs().max().item() <= FP32_NOISE
            assert (pool_a.v[:, block_a] - pool_b.v[:, block_b]).abs().max().item() <= FP32_NOISE
    print(f"padded steps bit-equal to eager: {exact} of 8; worst difference {worst}")
    assert decoder.stats.replays == 8 and decoder.stats.padded_rows > 0


def test_padding_rows_cannot_influence_real_rows_at_all() -> None:
    """The scratch block holds NaN in one run and its natural contents in the other; the real rows'
    logits and every live block must be bit-identical, since real rows never read it."""
    model = make_model()
    results = []
    for poison in (False, True):
        pool, caches = prefilled(model, 3)
        decoder = CudaGraphDecoder(model, pool, kernel(), 8, buckets=[8])  # type: ignore[arg-type]
        if poison:
            pool.k[:, decoder.scratch_block] = float("nan")
            pool.v[:, decoder.scratch_block] = float("nan")
        generator = torch.Generator().manual_seed(3)
        logits = []
        for _ in range(3):
            tokens = torch.randint(0, CFG.vocab_size, (3, 1), generator=generator).cuda()
            logits.append(graphed(decoder, caches, tokens))
        results.append((logits, [(pool.k[:, c.blocks], pool.v[:, c.blocks]) for c in caches]))
    for a, b in zip(results[0][0], results[1][0], strict=True):
        assert torch.isfinite(b).all() and torch.equal(a, b)
    for (ka, va), (kb, vb) in zip(results[0][1], results[1][1], strict=True):
        assert torch.equal(ka, kb) and torch.equal(va, vb)


def test_eager_itself_depends_on_the_batch_shape_at_the_noise_level() -> None:
    """Documents the cause of the 1.8e-7: eager row 0 at batch 1 against batch 2, no graph."""
    model = make_model()
    _, caches_one = prefilled(model, 1)
    _, caches_two = prefilled(model, 2)
    tokens = torch.randint(0, CFG.vocab_size, (2, 1), generator=torch.Generator().manual_seed(2))
    tokens = tokens.cuda()
    one = eager(model, caches_one, tokens[:1])
    two = eager(model, caches_two, tokens)[:1]
    print(
        f"eager batch 1 against batch 2, row 0: bit-equal {torch.equal(one, two)}, "
        f"max difference {(one - two).abs().max().item()}"
    )
    assert (one - two).abs().max().item() <= FP32_NOISE
