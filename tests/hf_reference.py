"""Hugging Face reference models for tests and benchmarks.

The engine never imports this module: Hugging Face is only the ground truth we compare against.
"""

from __future__ import annotations

from typing import Any

import torch
import transformers
from transformers import LlamaForCausalLM

from engine.config import LlamaConfig


def to_hf_config(cfg: LlamaConfig) -> transformers.LlamaConfig:
    kwargs: dict[str, Any] = {
        "vocab_size": cfg.vocab_size,
        "hidden_size": cfg.hidden_size,
        "intermediate_size": cfg.intermediate_size,
        "num_hidden_layers": cfg.num_layers,
        "num_attention_heads": cfg.num_heads,
        "num_key_value_heads": cfg.num_kv_heads,
        "head_dim": cfg.head_dim,
        "max_position_embeddings": cfg.max_position_embeddings,
        "rms_norm_eps": cfg.rms_norm_eps,
        "tie_word_embeddings": cfg.tie_word_embeddings,
        "attention_bias": False,
        "mlp_bias": False,
    }
    scaling = cfg.rope_scaling
    scaling_dict: dict[str, Any] | None = None
    if scaling is not None:
        scaling_dict = {
            "rope_type": "llama3",
            "factor": scaling.factor,
            "low_freq_factor": scaling.low_freq_factor,
            "high_freq_factor": scaling.high_freq_factor,
            "original_max_position_embeddings": scaling.original_max_position_embeddings,
        }

    # Hugging Face moved rope settings into ``rope_parameters``; support both layouts.
    if "rope_parameters" in transformers.LlamaConfig().to_dict():
        params: dict[str, Any] = {"rope_theta": cfg.rope_theta, "rope_type": "default"}
        if scaling_dict is not None:
            params.update(scaling_dict)
        kwargs["rope_parameters"] = params
    else:
        kwargs["rope_theta"] = cfg.rope_theta
        kwargs["rope_scaling"] = scaling_dict
    return transformers.LlamaConfig(**kwargs)


def randomize_weights(model: LlamaForCausalLM, seed: int) -> None:
    """Give every weight a scale that makes logits order one.

    Hugging Face's default initialization (std 0.02) produces near-zero logits, which would let
    real bugs hide inside a loose absolute tolerance.
    """
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, param in model.named_parameters():
            noise = torch.randn(param.shape, generator=generator)
            if param.ndim == 1:  # RMSNorm scales
                param.copy_(1.0 + 0.1 * noise)
            elif "embed_tokens" in name:
                param.copy_(noise)
            else:
                param.copy_(noise / param.shape[1] ** 0.5)


def build_reference_model(cfg: LlamaConfig, seed: int = 0) -> LlamaForCausalLM:
    """A deterministic random-weight fp32 reference model on CPU, in eval mode."""
    torch.manual_seed(seed)
    model = LlamaForCausalLM(to_hf_config(cfg)).to(torch.float32)
    randomize_weights(model, seed)
    model.eval()
    return model
