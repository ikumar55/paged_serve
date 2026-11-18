"""The GPU self-test: it must pass on a healthy device and fail on one that corrupts results."""

from __future__ import annotations

import json

import pytest
import torch

from bench.gpu_selftest import failed_checks, main, run_selftest


def test_a_healthy_device_passes_every_check() -> None:
    report = run_selftest("cpu", repeats=5)
    assert report["passed"], failed_checks(report)
    assert report["device"] == "cpu"
    assert {"repeat_silu", "repeat_gemm", "silu_vs_float64", "gemm_vs_float64"} <= report[
        "checks"
    ].keys()
    json.dumps(report)  # the report is stored in result files


def test_an_intermittently_corrupting_operation_is_caught_and_localised() -> None:
    """Simulate the observed fault: a few fixed rows come back wrong on some repeats."""
    base = torch.randn(40, 16, generator=torch.Generator().manual_seed(0))
    calls = {"n": 0}

    def flaky() -> torch.Tensor:
        calls["n"] += 1
        out = base.clone()
        if calls["n"] % 3 == 0:  # every third call, rows 7 and 31 are off by one ulp-ish amount
            out[[7, 31]] += 1e-3
        return out

    report = run_selftest("cpu", repeats=9, ops={"stable": lambda: base * 2, "flaky": flaky})
    assert not report["passed"]
    assert failed_checks(report) == ["repeat_flaky"]
    flaky_report = report["checks"]["repeat_flaky"]
    assert flaky_report["differing_repeats"] > 0
    assert flaky_report["distinct_rows_affected"] == 2
    assert report["checks"]["repeat_stable"]["passed"]


def test_an_inaccurate_result_fails_the_accuracy_check(monkeypatch: pytest.MonkeyPatch) -> None:
    real_silu = torch.nn.functional.silu
    monkeypatch.setattr(
        torch.nn.functional,
        "silu",
        lambda t: real_silu(t) * (1 + 1e-3) if t.dtype == torch.float32 else real_silu(t),
    )
    report = run_selftest("cpu", repeats=2, ops={})
    assert failed_checks(report) == ["silu_vs_float64"]


def test_cli_exit_code_reflects_the_result(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--device", "cpu", "--repeats", "2"]) == 0
    assert json.loads(capsys.readouterr().out)["passed"] is True
