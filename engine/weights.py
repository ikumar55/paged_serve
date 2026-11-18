"""Loading public checkpoints (Hugging Face parameter naming) into the engine model."""

from __future__ import annotations

from collections.abc import Mapping

from torch import Tensor

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
