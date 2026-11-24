"""Llama forward pass in plain PyTorch: RMSNorm, RoPE, grouped-query attention and SwiGLU.

Parameter names match the Hugging Face checkpoint layout so public weights load directly, but
nothing here imports Hugging Face.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from engine.config import LlamaConfig, RopeScaling
from engine.kv_cache import KVCache


def rope_inv_freq(head_dim: int, theta: float, scaling: RopeScaling | None) -> Tensor:
    """Rotary frequencies, one per pair of channels, optionally with Llama-3 scaling.

    Llama 3 slows down the low-frequency (long-wavelength) rotations by ``factor`` so the model
    can address longer contexts, leaves high-frequency ones alone, and blends smoothly between
    the two bands.
    """
    exponents = torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim
    inv_freq = 1.0 / (theta**exponents)
    if scaling is None:
        return inv_freq

    original = scaling.original_max_position_embeddings
    low_wavelength = original / scaling.low_freq_factor
    high_wavelength = original / scaling.high_freq_factor
    wavelength = 2 * math.pi / inv_freq

    slowed = inv_freq / scaling.factor
    blend = (original / wavelength - scaling.low_freq_factor) / (
        scaling.high_freq_factor - scaling.low_freq_factor
    )
    blended = (1 - blend) * slowed + blend * inv_freq
    return torch.where(
        wavelength > low_wavelength,
        slowed,
        torch.where(wavelength < high_wavelength, inv_freq, blended),
    )


def rotate_half(x: Tensor) -> Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def apply_rope(q: Tensor, k: Tensor, cos: Tensor, sin: Tensor) -> tuple[Tensor, Tensor]:
    """Rotate ``q`` and ``k`` of shape ``[batch, heads, tokens, head_dim]`` by position."""
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin


class Rotary:
    """Computes cos and sin tables for given positions.

    Deliberately not an ``nn.Module``: the frequencies must stay float32 even when the model is
    cast to bf16, and they are not part of the checkpoint.
    """

    def __init__(self, cfg: LlamaConfig) -> None:
        self.inv_freq = rope_inv_freq(cfg.head_dim, cfg.rope_theta, cfg.rope_scaling)

    def __call__(self, positions: Tensor, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        """Return cos and sin tables of shape ``[batch, tokens, head_dim]`` for the positions."""
        inv_freq = self.inv_freq.to(positions.device)
        angles = positions.unsqueeze(-1).float() * inv_freq
        table = torch.cat((angles, angles), dim=-1)
        return table.cos().to(dtype), table.sin().to(dtype)


class RMSNorm(nn.Module):
    def __init__(self, size: int, eps: float, dtype: torch.dtype, device: torch.device | str):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size, dtype=dtype, device=device))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        # Normalize in float32 for stability, then return to the activation dtype.
        x32 = x.float()
        x32 = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.eps)
        out: Tensor = self.weight * x32.to(x.dtype)
        return out


def attend(q: Tensor, k: Tensor, v: Tensor, past: int, mask: Tensor | None = None) -> Tensor:
    """Causal attention of ``q`` over ``past`` cached tokens plus its own.

    ``q`` is ``[batch, heads, new, head_dim]``; ``k`` and ``v`` are
    ``[batch, kv_heads, past + new, head_dim]``. Each query attends to every cached token and to
    the new tokens up to and including itself. An explicit boolean ``mask`` (True = may attend)
    replaces that rule; it is used when the rows of a batch have different lengths.
    """
    new = q.shape[2]
    if mask is not None:
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, enable_gqa=True)
    if past == 0:
        return F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
    if new == 1:
        return F.scaled_dot_product_attention(q, k, v, enable_gqa=True)
    total = k.shape[2]
    rows = torch.arange(new, device=q.device).unsqueeze(1) + past
    cols = torch.arange(total, device=q.device).unsqueeze(0)
    return F.scaled_dot_product_attention(q, k, v, attn_mask=cols <= rows, enable_gqa=True)


class Attention(nn.Module):
    def __init__(self, cfg: LlamaConfig, dtype: torch.dtype, device: torch.device | str):
        super().__init__()
        self.num_heads = cfg.num_heads
        self.num_kv_heads = cfg.num_kv_heads
        self.head_dim = cfg.head_dim
        kwargs = {"bias": False, "dtype": dtype, "device": device}
        self.q_proj = nn.Linear(cfg.hidden_size, cfg.num_heads * cfg.head_dim, **kwargs)  # type: ignore[arg-type]
        self.k_proj = nn.Linear(cfg.hidden_size, cfg.num_kv_heads * cfg.head_dim, **kwargs)  # type: ignore[arg-type]
        self.v_proj = nn.Linear(cfg.hidden_size, cfg.num_kv_heads * cfg.head_dim, **kwargs)  # type: ignore[arg-type]
        self.o_proj = nn.Linear(cfg.num_heads * cfg.head_dim, cfg.hidden_size, **kwargs)  # type: ignore[arg-type]

    def forward(
        self,
        x: Tensor,
        cos: Tensor,
        sin: Tensor,
        cache: KVCache | None,
        layer: int,
    ) -> Tensor:
        batch, tokens, _ = x.shape
        q: Tensor = self.q_proj(x).view(batch, tokens, self.num_heads, self.head_dim)
        k: Tensor = self.k_proj(x).view(batch, tokens, self.num_kv_heads, self.head_dim)
        v: Tensor = self.v_proj(x).view(batch, tokens, self.num_kv_heads, self.head_dim)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        q, k = apply_rope(q, k, cos, sin)

        past = 0
        mask = None
        if cache is not None and getattr(cache, "reads_in_place", False):
            # A paged batch cache with a decode-attention kernel (D-025): it writes the new key
            # and value into the pool and attends over the blocks there, with no gather.
            out = cache.attend_in_place(layer, q, k, v)  # type: ignore[attr-defined]
        else:
            if cache is not None:
                past = cache.length
                k, v = cache.update(layer, k, v)
                mask = cache.attention_mask(tokens)
            out = attend(q, k, v, past, mask)
        merged = out.transpose(1, 2).reshape(batch, tokens, self.num_heads * self.head_dim)
        result: Tensor = self.o_proj(merged)
        return result


class MLP(nn.Module):
    """SwiGLU feed-forward block."""

    def __init__(self, cfg: LlamaConfig, dtype: torch.dtype, device: torch.device | str):
        super().__init__()
        kwargs = {"bias": False, "dtype": dtype, "device": device}
        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, **kwargs)  # type: ignore[arg-type]
        self.up_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, **kwargs)  # type: ignore[arg-type]
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, **kwargs)  # type: ignore[arg-type]

    def forward(self, x: Tensor) -> Tensor:
        out: Tensor = self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
        return out


class DecoderLayer(nn.Module):
    def __init__(self, cfg: LlamaConfig, dtype: torch.dtype, device: torch.device | str):
        super().__init__()
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps, dtype, device)
        self.self_attn = Attention(cfg, dtype, device)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps, dtype, device)
        self.mlp = MLP(cfg, dtype, device)

    def forward(
        self,
        x: Tensor,
        cos: Tensor,
        sin: Tensor,
        cache: KVCache | None,
        layer: int,
    ) -> Tensor:
        x = x + self.self_attn(self.input_layernorm(x), cos, sin, cache, layer)
        out: Tensor = x + self.mlp(self.post_attention_layernorm(x))
        return out


class Backbone(nn.Module):
    def __init__(self, cfg: LlamaConfig, dtype: torch.dtype, device: torch.device | str):
        super().__init__()
        self.embed_tokens = nn.Embedding(
            cfg.vocab_size, cfg.hidden_size, dtype=dtype, device=device
        )
        self.layers = nn.ModuleList(DecoderLayer(cfg, dtype, device) for _ in range(cfg.num_layers))
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps, dtype, device)


class Llama(nn.Module):
    """A Llama-architecture causal language model."""

    def __init__(
        self,
        cfg: LlamaConfig,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str = "cpu",
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.model = Backbone(cfg, dtype, device)
        self.lm_head = nn.Linear(
            cfg.hidden_size, cfg.vocab_size, bias=False, dtype=dtype, device=device
        )
        if cfg.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight
        self.rotary = Rotary(cfg)

    @property
    def dtype(self) -> torch.dtype:
        return self.model.embed_tokens.weight.dtype

    @property
    def device(self) -> torch.device:
        return self.model.embed_tokens.weight.device

    def forward(
        self,
        input_ids: Tensor,
        positions: Tensor | None = None,
        cache: KVCache | None = None,
        last_only: bool = False,
    ) -> Tensor:
        """Return logits of shape ``[batch, tokens, vocab]`` (``tokens`` is 1 if ``last_only``).

        With a cache, ``input_ids`` holds only the tokens not yet cached and positions default to
        continuing from the cache length.
        """
        batch, tokens = input_ids.shape
        if positions is None:
            start = cache.length if cache is not None else 0
            positions = torch.arange(start, start + tokens, device=input_ids.device)
            positions = positions.unsqueeze(0).expand(batch, tokens)

        hidden: Tensor = self.model.embed_tokens(input_ids)
        cos, sin = self.rotary(positions, hidden.dtype)
        for index, layer in enumerate(self.model.layers):
            hidden = layer(hidden, cos, sin, cache, index)
        if cache is not None:
            cache.advance(tokens)

        hidden = self.model.norm(hidden)
        if last_only:
            hidden = hidden[:, -1:, :]
        logits: Tensor = self.lm_head(hidden)
        return logits
