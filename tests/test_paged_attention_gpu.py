"""GPU correctness gates for the Triton decode-attention kernel (D-025, criterion (a)(i)).

Run by hand on the GPU machine: ``pytest -m gpu tests/test_paged_attention_gpu.py -s``. Skipped
without a CUDA device or without Triton, so CI and the CPU-only machines never run it.

The gate: against the float64 reference, on random data with ragged lengths (including 1, 16, 17
and 32 tokens), scrambled block tables and padded entries pointing at block 0, the kernel's
maximum absolute error is at most twice the existing gather path's, in fp32 and in bf16. Both
errors are printed, so the margin is on record. TF32 is switched off so "fp32" means fp32.
"""

from __future__ import annotations

import pytest
import torch

from engine.paged_attention import decode_attention_reference
from tests.paged_attention_support import Case, gather_attention, make_case

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU"),
]

triton = pytest.importorskip("triton")

# Cached tokens before the new one, so the sequences attend over 1, 16, 17 and 32 positions.
BOUNDARY = [0, 15, 16, 31]
CASES = {
    "boundaries": BOUNDARY,
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


def kernel() -> object:
    from engine.triton_paged_attention import paged_decode_attention

    return paged_decode_attention


def truth(case: Case) -> torch.Tensor:
    return decode_attention_reference(
        case.q, case.k_pool, case.v_pool, case.tables, case.lengths, dtype=torch.float64
    )


def run(fn: object, case: Case) -> torch.Tensor:
    out = fn(case.q, case.k_pool, case.v_pool, case.tables, case.lengths)  # type: ignore[operator]
    torch.cuda.synchronize()
    return out  # type: ignore[no-any-return]


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
@pytest.mark.parametrize("name", sorted(CASES))
def test_kernel_error_is_at_most_twice_the_gather_paths(name: str, dtype: torch.dtype) -> None:
    worst_ratio = 0.0
    for seed in range(5):  # five random draws of every shape
        case = make_case(CASES[name], dtype=dtype, device="cuda", seed=seed, poison=1e4)
        expected = truth(case)
        gather_error = (run(gather_attention, case).double() - expected).abs().max().item()
        kernel_error = (run(kernel(), case).double() - expected).abs().max().item()
        print(f"{name} {dtype} seed {seed}: gather {gather_error:.3e} kernel {kernel_error:.3e}")
        assert kernel_error <= 2 * gather_error, (kernel_error, gather_error)
        worst_ratio = max(worst_ratio, kernel_error / gather_error)
    print(f"{name} {dtype}: worst kernel/gather error ratio {worst_ratio:.2f}")


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
def test_kernel_never_reads_slots_no_sequence_owns(dtype: torch.dtype) -> None:
    # NaN in every unused slot, block 0 included when nobody owns it. If the kernel read any of
    # them as data (or as a padded table entry), the output would be NaN or would change.
    clean = make_case(CASES["ragged"], dtype=dtype, device="cuda", seed=9)
    poisoned = make_case(CASES["ragged"], dtype=dtype, device="cuda", seed=9, poison=float("nan"))
    assert not poisoned.valid.all()
    got, expected = run(kernel(), poisoned), run(kernel(), clean)
    assert torch.isfinite(got).all()
    assert torch.equal(got, expected)


def test_padded_entries_pointing_at_an_owned_block_zero_are_not_read() -> None:
    # Block 0 belongs to the longest sequence; the short ones are padded with 0 as in the engine,
    # so a kernel that read a padded entry would mix that sequence's data into theirs.
    case = make_case([300, 3, 20], device="cuda", seed=1, own_block_zero=True)
    assert case.tables[0, 0].item() == 0 and (case.tables[1:, -1] == 0).all()
    expected = truth(case)
    error = (run(kernel(), case).double() - expected).abs().max().item()
    assert error < 1e-5, error


def test_kernel_accepts_the_queries_the_model_produces() -> None:
    # The model hands over q as a transposed view, [batch, heads, 1, head_dim] with unusual strides.
    case = make_case(CASES["ragged"], device="cuda", seed=2)
    # A slice of a wider tensor: same values, but the batch and head strides are not those of a
    # packed [B, H, 1, D] tensor (a plain transpose of a size-1 dimension would still be packed).
    wide = torch.zeros(*case.q.shape[:2], 3, case.q.shape[3], device="cuda", dtype=case.q.dtype)
    wide[:, :, 1:2] = case.q
    view = wide[:, :, 1:2]
    assert view.shape == case.q.shape and not view.is_contiguous()
    assert torch.equal(
        kernel()(view, case.k_pool, case.v_pool, case.tables, case.lengths),  # type: ignore[operator]
        run(kernel(), case),
    )


def test_kernel_rejects_what_it_cannot_do() -> None:
    case = make_case(BOUNDARY, device="cuda")
    fn = kernel()
    with pytest.raises(ValueError):
        fn(case.q.cpu(), case.k_pool, case.v_pool, case.tables, case.lengths)  # type: ignore[operator]
    with pytest.raises(ValueError):
        fn(case.q, case.k_pool, case.v_pool, case.tables.int(), case.lengths)  # type: ignore[operator]
    two_tokens = case.q.expand(-1, -1, 2, -1)
    with pytest.raises(ValueError):
        fn(two_tokens, case.k_pool, case.v_pool, case.tables, case.lengths)  # type: ignore[operator]
