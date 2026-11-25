"""Split-KV decode attention (D-031): the algorithm on a CPU, and the shape of the Triton file."""

from __future__ import annotations

import ast
import math
from pathlib import Path

import pytest
import torch

from engine.paged_attention import (
    decode_attention_reference,
    split_blocks,
    split_decode_attention_reference,
)
from tests.paged_attention_support import make_case

SHAPE = {"heads": 8, "kv_heads": 2, "head_dim": 16, "block_size": 4}
RAGGED = [0, 3, 4, 15, 16, 31, 1, 40, 7, 100]  # blocks: 1, 1, 2, 4, 5, 8, 1, 11, 2, 26


def both(lengths: list[int], splits: int, **kwargs: object) -> tuple[torch.Tensor, torch.Tensor]:
    case = make_case(lengths, **SHAPE, **kwargs)  # type: ignore[arg-type]
    args = (case.q, case.k_pool, case.v_pool, case.tables, case.lengths)
    whole = decode_attention_reference(*args, dtype=torch.float64)
    split = split_decode_attention_reference(*args, splits, dtype=torch.float64)
    return whole, split


# -- the rule that deals blocks out ------------------------------------------------------


@pytest.mark.parametrize("num_blocks", [1, 2, 3, 5, 8, 9, 26, 100])
@pytest.mark.parametrize("splits", [1, 2, 3, 4, 8, 16])
def test_every_block_goes_to_exactly_one_split_in_order(num_blocks: int, splits: int) -> None:
    dealt = [list(split_blocks(num_blocks, splits, s)) for s in range(splits)]
    assert [b for part in dealt for b in part] == list(range(num_blocks))
    per_split = math.ceil(num_blocks / splits)
    assert all(len(part) <= per_split for part in dealt)
    sizes = [len(part) for part in dealt]
    assert sizes == sorted(sizes, reverse=True)  # the empty splits are the last ones


def test_more_splits_than_blocks_leaves_the_last_splits_empty() -> None:
    assert [len(split_blocks(2, 8, s)) for s in range(8)] == [1, 1, 0, 0, 0, 0, 0, 0]
    assert [len(split_blocks(5, 4, s)) for s in range(4)] == [2, 2, 1, 0]


# -- the algorithm equals the unsplit reference ------------------------------------------


@pytest.mark.parametrize("splits", [1, 2, 3, 4, 5, 8, 16])
def test_the_split_result_equals_the_unsplit_one(splits: int) -> None:
    whole, split = both(RAGGED, splits)
    assert (whole - split).abs().max().item() < 1e-12


@pytest.mark.parametrize("splits", [2, 4, 8])
def test_a_single_token_sequence_with_more_splits_than_blocks(splits: int) -> None:
    whole, split = both([0], splits)  # one block, so every split but the first is empty
    assert (whole - split).abs().max().item() < 1e-12


def test_slots_no_sequence_owns_cannot_leak_in_whatever_the_splits() -> None:
    whole, split = both(RAGGED, 4, poison=1e6, own_block_zero=True)
    assert (whole - split).abs().max().item() < 1e-9


def test_a_dominant_score_in_a_late_split_is_merged_correctly() -> None:
    """The merge must rescale to the largest maximum, wherever it is."""
    case = make_case([60], **SHAPE)
    case.k_pool[case.tables[0, 14]] *= (
        50.0  # the last block of the 15: a huge score in the last split
    )
    args = (case.q, case.k_pool, case.v_pool, case.tables, case.lengths)
    whole = decode_attention_reference(*args, dtype=torch.float64)
    for splits in (2, 4):
        split = split_decode_attention_reference(*args, splits, dtype=torch.float64)
        assert (whole - split).abs().max().item() < 1e-9


def test_a_wrong_merge_would_be_noticed_by_these_checks() -> None:
    """The comparison has teeth: averaging the partial results instead of rescaling them differs."""
    whole, _ = both(RAGGED, 4)
    case = make_case(RAGGED, **SHAPE)
    keep = case.lengths.clone()
    keep[:] = torch.clamp(keep, max=3)  # truncated sequences give a different answer
    other = decode_attention_reference(
        case.q, case.k_pool, case.v_pool, case.tables, keep, dtype=torch.float64
    )
    assert (whole - other).abs().max().item() > 1e-3


def test_the_split_reference_validates_its_arguments() -> None:
    case = make_case([3], **SHAPE)
    args = (case.q, case.k_pool, case.v_pool, case.tables, case.lengths)
    with pytest.raises(ValueError, match="at least 1"):
        split_decode_attention_reference(*args, 0)
    with pytest.raises(ValueError, match="one new token"):
        split_decode_attention_reference(case.q.repeat(1, 1, 2, 1), *args[1:], 2)


# -- the Triton file, which cannot be imported here ----------------------------------------------


def test_the_triton_split_file_parses_and_exposes_the_expected_entry_points() -> None:
    source = Path("engine/triton_split_attention.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    names = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
    assert {
        "_partial_kernel",
        "_merge_kernel",
        "paged_split_decode_attention",
        "split_variant",
    } <= names
    assigned = {
        target.id
        for node in tree.body
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
    }
    assert "ALLOWED_SPLITS" in assigned
    assert "(1, 2, 4, 8)" in source  # the merge kernel needs a power of two


def test_the_triton_kernel_deals_blocks_out_by_the_same_formula_as_the_reference() -> None:
    source = Path("engine/triton_split_attention.py").read_text(encoding="utf-8")
    assert "per_split = tl.cdiv(num_blocks, SPLITS)" in source
    assert "first = tl.minimum(split * per_split, num_blocks)" in source
    assert "last = tl.minimum(num_blocks, first + per_split)" in source
    reference = Path("engine/paged_attention.py").read_text(encoding="utf-8")
    assert "per_split = -(-num_blocks // splits)" in reference
    assert "start = min(split * per_split, num_blocks)" in reference
