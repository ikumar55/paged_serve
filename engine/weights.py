"""Loading public checkpoints (Hugging Face parameter naming) into the engine model."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import torch
from safetensors.torch import load_file
from torch import Tensor

from engine.config import LlamaConfig
from engine.model import Llama


def load_hf_state_dict(model: Llama, state: Mapping[str, Tensor]) -> None:
    """Copy a Hugging Face-named state dict into ``model``, converting dtype as needed.

    Raises ``ValueError`` on missing keys, unexpected keys or shape mismatches rather than loading
    partially.
    """
    # Older checkpoints store derived rotary frequencies; we recompute them instead.
    cleaned = {k: v for k, v in state.items() if not k.endswith("rotary_emb.inv_freq")}
    # Tied checkpoints may omit the output head; it shares the embedding matrix.
    if model.cfg.tie_word_embeddings and "lm_head.weight" not in cleaned:
        cleaned["lm_head.weight"] = cleaned["model.embed_tokens.weight"]

    expected = model.state_dict()
    missing = sorted(expected.keys() - cleaned.keys())
    unexpected = sorted(cleaned.keys() - expected.keys())
    if missing or unexpected:
        raise ValueError(f"state dict mismatch: missing={missing}, unexpected={unexpected}")
    for name, tensor in cleaned.items():
        if tensor.shape != expected[name].shape:
            raise ValueError(
                f"shape mismatch for {name}: got {tuple(tensor.shape)}, "
                f"expected {tuple(expected[name].shape)}"
            )
    model.load_state_dict(cleaned, strict=True)


def load_model(
    model_dir: Path, dtype: torch.dtype, device: torch.device | str
) -> tuple[Llama, list[int]]:
    """Load a single-file Hugging Face Llama checkpoint; also return its end-of-sequence ids."""
    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    model = Llama(LlamaConfig.from_hf_dict(config), dtype=dtype, device=device)
    load_hf_state_dict(model, load_file(model_dir / "model.safetensors"))
    model.eval()

    eos = config.get("eos_token_id")
    generation = model_dir / "generation_config.json"
    if generation.exists():  # this file is what ``generate`` uses, so it wins
        eos = json.loads(generation.read_text(encoding="utf-8")).get("eos_token_id", eos)
    eos_ids = [] if eos is None else [eos] if isinstance(eos, int) else [int(i) for i in eos]
    return model, eos_ids
