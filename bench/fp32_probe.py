"""Which operation makes "fp32" less accurate than expected on this GPU?

The float64 truth report (``tests/fp64_truth_report.py``) found that on an RTX 5090 the model's
fp32 logits are up to about 1e-3 from float64, against 1.8e-4 for the same model on the CPU. This
probe prints the fp32 precision settings in force and measures, for the operations the model is
made of, each fp32 implementation's worst relative error against float64 computed from the same
fp32 inputs: a square matmul, the model's largest linear layer (the output head), softmax
attention, and RMS normalisation. If one operation is far worse on the GPU than on the CPU, that
operation is where the accuracy goes (a tensor-core path with a shorter mantissa, or an
emulation), and the setting that controls it can be named.

Usage:
    python -m bench.fp32_probe --device cuda
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Callable

import torch
import torch.nn.functional as F

Operation = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def max_relative_error(actual: torch.Tensor, truth: torch.Tensor) -> float:
    """Largest ``|actual - truth|`` divided by the largest ``|truth|`` (a scale-free worst case)."""
    scale = float(truth.abs().max())
    if scale == 0:
        return float((actual - truth).abs().max())
    return float((actual.double() - truth).abs().max()) / scale


def operations() -> dict[str, tuple[Callable[[], tuple[torch.Tensor, ...]], Operation]]:
    """name -> (a function making fp32 inputs, the operation on them); inputs are seeded."""

    def matmul_inputs() -> tuple[torch.Tensor, ...]:
        generator = torch.Generator().manual_seed(0)
        return (
            torch.randn(2048, 2048, generator=generator),
            torch.randn(2048, 2048, generator=generator),
        )

    def head_inputs() -> tuple[torch.Tensor, ...]:  # one hidden state through the output head
        generator = torch.Generator().manual_seed(1)
        return (
            torch.randn(8, 2048, generator=generator),
            torch.randn(128256, 2048, generator=generator) * 0.02,
        )

    def attention_inputs() -> tuple[torch.Tensor, ...]:
        generator = torch.Generator().manual_seed(2)
        q = torch.randn(1, 8, 64, 64, generator=generator)
        kv = torch.randn(1, 8, 512, 64, generator=generator)
        return (q, kv)

    def norm_inputs() -> tuple[torch.Tensor, ...]:
        generator = torch.Generator().manual_seed(3)
        return (
            torch.randn(64, 2048, generator=generator) * 3,
            torch.rand(2048, generator=generator),
        )

    return {
        "matmul 2048x2048": (matmul_inputs, lambda a, b: a @ b),
        "output head (linear 2048 -> 128256)": (head_inputs, lambda x, w: F.linear(x, w)),
        "softmax attention": (
            attention_inputs,
            lambda q, kv: F.scaled_dot_product_attention(q, kv, kv),
        ),
        "rms norm": (
            norm_inputs,
            lambda x, w: x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5) * w,
        ),
    }


def settings(device: str) -> dict[str, object]:
    found: dict[str, object] = {
        "torch": torch.__version__,
        "cuda_build": torch.version.cuda,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "allow_tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "allow_tf32_cudnn": torch.backends.cudnn.allow_tf32,
        "cublas_env": {k: v for k, v in os.environ.items() if k.startswith("CUBLAS")},
    }
    fp32_precision = getattr(torch.backends.cuda.matmul, "fp32_precision", None)
    if fp32_precision is not None:
        found["matmul_fp32_precision"] = fp32_precision
    if device.startswith("cuda"):
        found["device"] = torch.cuda.get_device_name(0)
    return found


def probe(device: str) -> dict[str, float]:
    """Worst relative error against float64, per operation, with the settings as they are."""
    errors: dict[str, float] = {}
    with torch.inference_mode():
        for name, (make, op) in operations().items():
            inputs = make()
            truth = op(*(t.double() for t in inputs))
            errors[name] = max_relative_error(op(*(t.to(device) for t in inputs)).cpu(), truth)
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0] if __doc__ else None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(argv)
    report: dict[str, object] = {"settings": settings(args.device), "cpu": probe("cpu")}
    report["device_default"] = probe(args.device)
    if args.device.startswith("cuda"):
        previous = (torch.backends.cuda.matmul.allow_tf32, torch.get_float32_matmul_precision())
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        report["device_tf32_off_highest"] = probe(args.device)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        report["device_tf32_on_high"] = probe(args.device)
        torch.backends.cuda.matmul.allow_tf32, _ = previous
        torch.set_float32_matmul_precision(previous[1])
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
