"""Determinism and accuracy self-test for a CUDA device (it also runs on CPU for testing).

A GPU that returns different results for identical inputs makes every number measured on it
meaningless. This test exists because one did: on a GPU with clean ECC counters, ``silu(x)``
differed between repeats on every iteration, always in the same few rows, while multiply, add and
tanh were fine. The error showed up only as unexplained logit differences of up to 1.8 in a
model-level comparison.

The checks:

* each elementwise operation, repeated on the same input, must return bit-identical results;
* a matrix multiply, repeated, must be bit-identical;
* ``silu`` and a matrix multiply must agree with a float64 CPU result to fp32 rounding error.

Usage:
    python -m bench.gpu_selftest            # prints JSON, exits 1 if the device fails
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Mapping
from typing import Any

import torch
import torch.nn.functional as F

ROWS, COLS = 189, 8192  # the shape of a Llama MLP activation for a ~190-token sequence
DEFAULT_REPEATS = 300
ACCURACY_TOLERANCE = 1e-5  # relative to the largest value; fp32 rounding is about 1e-7

Op = Callable[[], torch.Tensor]


def _repeat_check(op: Op, repeats: int) -> dict[str, Any]:
    reference = op()
    differing_repeats = differing_elements = 0
    rows: set[int] = set()
    for _ in range(repeats):
        mismatch = op() != reference
        count = int(mismatch.sum())
        if count:
            differing_repeats += 1
            differing_elements += count
            flat = (
                mismatch.reshape(mismatch.shape[0], -1) if mismatch.ndim > 1 else mismatch[:, None]
            )
            rows.update(flat.any(-1).nonzero().flatten().tolist())
    return {
        "passed": differing_repeats == 0,
        "differing_repeats": differing_repeats,
        "repeats": repeats,
        "differing_elements": differing_elements,
        "distinct_rows_affected": len(rows),
    }


def default_ops(device: str, seed: int) -> dict[str, Op]:
    generator = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(ROWS, COLS, device=device, generator=generator)
    y = torch.randn(ROWS, COLS, device=device, generator=generator)
    activations = torch.randn(ROWS, 2048, device=device, generator=generator)
    weights = torch.randn(COLS, 2048, device=device, generator=generator) / 45
    return {
        "mul": lambda: x * y,
        "exp": lambda: torch.exp(x),
        "sigmoid": lambda: torch.sigmoid(x),
        "silu": lambda: F.silu(x),
        "silu_mul": lambda: F.silu(x) * y,
        "tanh": lambda: torch.tanh(x),
        "rsqrt": lambda: torch.rsqrt(x.abs() + 0.1),
        "gemm": lambda: F.linear(activations, weights),
    }


def _accuracy_checks(device: str, seed: int) -> dict[str, dict[str, Any]]:
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(ROWS, COLS, generator=generator)
    activations = torch.randn(ROWS, 2048, generator=generator)
    weights = torch.randn(COLS, 2048, generator=generator) / 45
    cases = {
        "silu_vs_float64": (F.silu(x.to(device)), F.silu(x.double())),
        "gemm_vs_float64": (
            F.linear(activations.to(device), weights.to(device)),
            F.linear(activations.double(), weights.double()),
        ),
    }
    results: dict[str, dict[str, Any]] = {}
    for name, (got, truth) in cases.items():
        error = ((got.double().cpu() - truth).abs().max() / truth.abs().max()).item()
        results[name] = {"passed": error < ACCURACY_TOLERANCE, "relative_error": error}
    return results


def run_selftest(
    device: str = "cuda",
    repeats: int = DEFAULT_REPEATS,
    seed: int = 0,
    ops: Mapping[str, Op] | None = None,
) -> dict[str, Any]:
    """Run every check and return a JSON-serializable report with an overall ``passed`` flag."""
    # Make fp32 mean fp32: TF32 would otherwise turn the accuracy check into a precision test.
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    checks: dict[str, dict[str, Any]] = {}
    for name, op in (ops if ops is not None else default_ops(device, seed)).items():
        checks[f"repeat_{name}"] = _repeat_check(op, repeats)
    checks.update(_accuracy_checks(device, seed))
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    return {
        "passed": all(c["passed"] for c in checks.values()),
        "device": torch.cuda.get_device_name(device) if device.startswith("cuda") else "cpu",
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "repeats": repeats,
        "checks": checks,
    }


def failed_checks(report: Mapping[str, Any]) -> list[str]:
    return [name for name, check in report["checks"].items() if not check["passed"]]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0] if __doc__ else None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    args = parser.parse_args(argv)
    report = run_selftest(args.device, args.repeats)
    sys.stdout.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if not report["passed"]:
        sys.stderr.write(f"SELF-TEST FAILED: {', '.join(failed_checks(report))}\n")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
