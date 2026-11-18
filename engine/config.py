"""Model configuration for Llama-architecture decoder-only transformers.

Only the architecture lives here. The forward pass is implemented separately, and Hugging Face
classes are used purely as a test-time reference, never by the engine itself.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RopeScaling:
    """Llama-3-style RoPE frequency scaling (``rope_type == "llama3"``)."""

    factor: float
    low_freq_factor: float
    high_freq_factor: float
    original_max_position_embeddings: int

    def __post_init__(self) -> None:
        if self.factor <= 0:
            raise ValueError("rope scaling factor must be positive")
        if not 0 < self.low_freq_factor < self.high_freq_factor:
            raise ValueError("need 0 < low_freq_factor < high_freq_factor")
        if self.original_max_position_embeddings <= 0:
            raise ValueError("original_max_position_embeddings must be positive")


@dataclass(frozen=True)
class LlamaConfig:
    """Architecture hyperparameters of a Llama-style model."""

    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_layers: int
    num_heads: int
    num_kv_heads: int
    max_position_embeddings: int
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    rope_scaling: RopeScaling | None = None
    tie_word_embeddings: bool = False

    def __post_init__(self) -> None:
        for name in (
            "vocab_size",
            "hidden_size",
            "intermediate_size",
            "num_layers",
            "num_heads",
            "num_kv_heads",
            "max_position_embeddings",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.hidden_size % self.num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for rotary embeddings")

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads

    def kv_bytes_per_token(self, dtype_bytes: int = 2) -> int:
        """Bytes of KV cache one token occupies: 2 (K and V) x layers x KV heads x head dim."""
        return 2 * self.num_layers * self.num_kv_heads * self.head_dim * dtype_bytes

    def num_parameters(self) -> int:
        """Total parameter count (a tied embedding and output head are counted once)."""
        attn = (
            self.hidden_size * self.num_heads * self.head_dim  # q_proj
            + 2 * self.hidden_size * self.num_kv_heads * self.head_dim  # k_proj, v_proj
            + self.num_heads * self.head_dim * self.hidden_size  # o_proj
        )
        mlp = 3 * self.hidden_size * self.intermediate_size  # gate, up, down
        norms = 2 * self.hidden_size  # input and post-attention RMSNorm
        per_layer = attn + mlp + norms
        embed = self.vocab_size * self.hidden_size
        head = 0 if self.tie_word_embeddings else embed
        return embed + self.num_layers * per_layer + self.hidden_size + head

    @classmethod
    def from_hf_dict(cls, raw: Mapping[str, Any]) -> LlamaConfig:
        """Build from the ``config.json`` of a Hugging Face Llama-architecture model."""
        num_heads = int(raw["num_attention_heads"])
        hidden = int(raw["hidden_size"])
        declared_head_dim = raw.get("head_dim")
        if declared_head_dim is not None and int(declared_head_dim) * num_heads != hidden:
            raise ValueError("head_dim does not equal hidden_size / num_attention_heads")

        # Older config.json files use ``rope_theta`` and ``rope_scaling``; newer Hugging Face
        # versions fold both into ``rope_parameters``.
        rope_theta = float(raw.get("rope_theta", 10000.0))
        scaling_raw = raw.get("rope_scaling")
        rope_parameters = raw.get("rope_parameters")
        if rope_parameters is not None:
            rope_theta = float(rope_parameters.get("rope_theta", rope_theta))
            if rope_parameters.get("rope_type", "default") != "default":
                scaling_raw = rope_parameters
        scaling: RopeScaling | None = None
        if scaling_raw is not None and scaling_raw.get("rope_type", "default") != "default":
            kind = scaling_raw.get("rope_type", scaling_raw.get("type"))
            if kind != "llama3":
                raise ValueError(f"unsupported rope scaling type: {kind!r}")
            scaling = RopeScaling(
                factor=float(scaling_raw["factor"]),
                low_freq_factor=float(scaling_raw["low_freq_factor"]),
                high_freq_factor=float(scaling_raw["high_freq_factor"]),
                original_max_position_embeddings=int(
                    scaling_raw["original_max_position_embeddings"]
                ),
            )

        return cls(
            vocab_size=int(raw["vocab_size"]),
            hidden_size=hidden,
            intermediate_size=int(raw["intermediate_size"]),
            num_layers=int(raw["num_hidden_layers"]),
            num_heads=num_heads,
            num_kv_heads=int(raw.get("num_key_value_heads", num_heads)),
            max_position_embeddings=int(raw["max_position_embeddings"]),
            rms_norm_eps=float(raw.get("rms_norm_eps", 1e-5)),
            rope_theta=rope_theta,
            rope_scaling=scaling,
            tie_word_embeddings=bool(raw.get("tie_word_embeddings", False)),
        )

    @classmethod
    def tiny(
        cls, *, tie_word_embeddings: bool = False, llama3_rope_scaling: bool = False
    ) -> LlamaConfig:
        """A random-weight-sized model small enough for CPU CI (about 40K parameters).

        ``llama3_rope_scaling`` and ``tie_word_embeddings`` exercise the two features that make
        Llama 3.2 differ from older Llama checkpoints.
        """
        scaling = (
            RopeScaling(
                factor=4.0,
                low_freq_factor=1.0,
                high_freq_factor=4.0,
                original_max_position_embeddings=64,
            )
            if llama3_rope_scaling
            else None
        )
        return cls(
            vocab_size=128,
            hidden_size=64,
            intermediate_size=128,
            num_layers=2,
            num_heads=4,
            num_kv_heads=2,
            max_position_embeddings=256,
            rope_theta=10000.0,
            rope_scaling=scaling,
            tie_word_embeddings=tie_word_embeddings,
        )

    @classmethod
    def llama_3_2_1b(cls) -> LlamaConfig:
        """Llama-3.2-1B hyperparameters as published on the model card.

        Verified equal to the published ``config.json`` by ``tests/test_real_llama.py``.
        """
        return cls(
            vocab_size=128256,
            hidden_size=2048,
            intermediate_size=8192,
            num_layers=16,
            num_heads=32,
            num_kv_heads=8,
            max_position_embeddings=131072,
            rms_norm_eps=1e-5,
            rope_theta=500000.0,
            rope_scaling=RopeScaling(
                factor=32.0,
                low_freq_factor=1.0,
                high_freq_factor=4.0,
                original_max_position_embeddings=8192,
            ),
            tie_word_embeddings=True,
        )
