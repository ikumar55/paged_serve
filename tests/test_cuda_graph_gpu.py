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


def test_padded_steps_leave_real_rows_and_live_blocks_bit_equal() -> None:
    model = make_model()
    pool_a, caches_a = prefilled(model, 6)
    pool_b, caches_b = prefilled(model, 6)
    decoder = CudaGraphDecoder(model, pool_b, kernel(), 8, buckets=[2, 4, 8])  # type: ignore[arg-type]
    generator = torch.Generator().manual_seed(8)
    worst = 0.0
    for count in (6, 6, 5, 4, 3, 3, 2, 1):  # the batch shrinks, so the padding changes each time
        tokens = torch.randint(0, CFG.vocab_size, (count, 1), generator=generator).cuda()
        expected = eager(model, caches_a[:count], tokens)
        got = graphed(decoder, caches_b[:count], tokens)
        worst = max(worst, (got - expected).abs().max().item())
        assert torch.equal(expected, got), (count, worst)
    for a, b in zip(caches_a, caches_b, strict=True):
        for block_a, block_b in zip(a.blocks, b.blocks, strict=True):
            assert torch.equal(pool_a.k[:, block_a], pool_b.k[:, block_b])
            assert torch.equal(pool_a.v[:, block_a], pool_b.v[:, block_b])
    print(f"worst difference over the padded steps: {worst}")
    assert decoder.stats.replays == 8 and decoder.stats.padded_rows > 0
