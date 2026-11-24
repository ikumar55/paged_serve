"""The fp32 probe's arithmetic and that it runs on the CPU."""

from __future__ import annotations

import pytest
import torch

from bench import fp32_probe as fp


def test_max_relative_error_is_scaled_by_the_largest_true_value() -> None:
    truth = torch.tensor([10.0, -2.0], dtype=torch.float64)
    actual = torch.tensor([10.5, -2.0])
    assert fp.max_relative_error(actual, truth) == pytest.approx(0.05)


def test_an_all_zero_truth_gives_the_absolute_error() -> None:
    truth = torch.zeros(3, dtype=torch.float64)
    assert fp.max_relative_error(torch.tensor([0.0, 0.25, 0.0]), truth) == pytest.approx(0.25)


def test_the_cpu_probe_covers_every_operation_with_small_errors() -> None:
    errors = fp.probe("cpu")
    assert set(errors) == set(fp.operations())
    assert all(0 <= error < 1e-4 for error in errors.values()), errors  # fp32 on the CPU is close


def test_settings_report_the_precision_controls() -> None:
    found = fp.settings("cpu")
    assert {"float32_matmul_precision", "allow_tf32_matmul", "torch"} <= set(found)
