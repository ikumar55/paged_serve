"""How accurate is each fp32 implementation, measured against a float64 reference?

Comparing two fp32 implementations to each other cannot say which one is right, or whether both
are off by the same amount. This script computes the real model's logits in float64 on CPU (the
closest thing to ground truth we have) for the 50 prompts and the 2048-token prompt used by the
real-weights tests, then reports each fp32 implementation's error against it: Hugging Face on
CPU, Hugging Face on the GPU and this engine on the GPU.

Run by hand where the weights and a CUDA GPU are available (needs about 12 GB of RAM):

    python -m tests.fp64_truth_report
"""

from __future__ import annotations

import gc
import statistics
from collections.abc import Callable

import torch
from transformers import AutoModelForCausalLM

from tests.real_model_support import MODEL_DIR, build_prompts, load_engine, long_prompt_ids

Logits = Callable[[torch.Tensor], torch.Tensor]


def hf_logits(model: torch.nn.Module, device: str) -> Logits:
    def run(ids: torch.Tensor) -> torch.Tensor:
        out = model(ids.to(device)).logits[0]
        return out.double().cpu()

    return run


def main() -> None:
    torch.set_float32_matmul_precision("highest")  # fp32 must mean fp32 on the GPU
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    prompts = [torch.tensor([p]) for p in build_prompts()]
    long_ids = long_prompt_ids()

    with torch.inference_mode():
        truth_model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.float64).eval()
        truth = [hf_logits(truth_model, "cpu")(ids) for ids in prompts]
        long_truth = truth_model(long_ids, logits_to_keep=1).logits[0, -1].double()
        del truth_model
        gc.collect()

        largest = max(t.abs().max().item() for t in truth)
        print(f"{len(prompts)} prompts; largest |logit| in the float64 reference: {largest:.1f}")
        print("max abs error against it, over the prompts:")

        implementations: dict[str, tuple[Logits, Logits]] = {}
        cpu = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.float32).eval()
        implementations["hf fp32, CPU"] = (
            hf_logits(cpu, "cpu"),
            lambda ids: cpu(ids, logits_to_keep=1).logits[0, -1].double(),
        )
        if torch.cuda.is_available():
            gpu = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.float32)
            gpu = gpu.to("cuda").eval()
            implementations["hf fp32, GPU"] = (
                hf_logits(gpu, "cuda"),
                lambda ids: gpu(ids.cuda(), logits_to_keep=1).logits[0, -1].double().cpu(),
            )
            engine = load_engine(torch.float32, "cuda")
            implementations["engine fp32, GPU"] = (
                lambda ids: engine(ids.cuda())[0].double().cpu(),
                lambda ids: engine(ids.cuda(), last_only=True)[0, -1].double().cpu(),
            )

        for name, (full, last) in implementations.items():
            errors = [
                (full(ids) - t).abs().max().item() for ids, t in zip(prompts, truth, strict=True)
            ]
            long_error = (last(long_ids) - long_truth).abs().max().item()
            print(
                f"  {name:18s} max {max(errors):.3e}  median {statistics.median(errors):.3e}"
                f"  | 2048-token prompt, last position: {long_error:.3e}"
            )


if __name__ == "__main__":
    main()
