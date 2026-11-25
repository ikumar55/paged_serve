"""GPU correctness gate for the split-KV Triton kernel (D-031). Run by hand on the GPU machine.

``pytest -m gpu tests/test_split_attention_gpu.py -s``. Skipped without a CUDA device or Triton.
Same gate as the unsplit kernel (``tests/test_paged_attention_gpu.py``): against the float64
reference, the error is at most twice the gather path's, in fp32 and bf16, for every split count
and on the ragged shapes that include one-block sequences (so most splits are empty).
"""

from __future__ import annotations

import pytest
import torch

from engine.paged_attention import decode_attention_reference
from tests.paged_attention_support import gather_attention, make_case

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU"),
]

triton = pytest.importorskip("triton")

CASES = {
    "boundaries": [0, 15, 16, 31],
    "ragged": [0, 15, 16, 31, 1, 40, 7, 200, 443, 1048],
    "long": [2184, 5, 1023],
    "single": [100],
    "batch60": [(i * 37) % 900 for i in range(60)],
}


@pytest.fixture(autouse=True)
def strict_fp32() -> object:
    old = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    yield
    torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = old


@pytest.mark.parametrize("splits", [1, 2, 4, 8])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
@pytest.mark.parametrize("name", sorted(CASES))
def test_split_kernel_error_is_at_most_twice_the_gather_paths(
    name: str, dtype: torch.dtype, splits: int
) -> None:
    from engine.triton_split_attention import paged_split_decode_attention

    for seed in range(3):
        case = make_case(CASES[name], dtype=dtype, device="cuda", seed=seed, poison=1e4)
        args = (case.q, case.k_pool, case.v_pool, case.tables, case.lengths)
        expected = decode_attention_reference(*args, dtype=torch.float64)
        gather_error = (gather_attention(*args).double() - expected).abs().max().item()
        out = paged_split_decode_attention(*args, splits=splits)
        torch.cuda.synchronize()
        error = (out.double() - expected).abs().max().item()
        label = f"{name} {dtype} splits {splits} seed {seed}"
        print(f"{label}: gather {gather_error:.3e} split {error:.3e}")
        assert error <= 2 * gather_error + 1e-7, (error, gather_error)


def test_the_split_kernel_never_reads_blocks_past_a_sequences_end() -> None:
    from engine.triton_split_attention import paged_split_decode_attention

    case = make_case(CASES["ragged"], device="cuda", poison=float("nan"))
    out = paged_split_decode_attention(
        case.q, case.k_pool, case.v_pool, case.tables, case.lengths, splits=8
    )
    torch.cuda.synchronize()
    assert torch.isfinite(out).all()
