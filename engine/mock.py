"""A model that computes nothing and only takes time, for router, deployment and CI work (D-008).

``MockModel`` plugs into the real scheduler, block pool and prefix cache in place of the Llama
forward pass. It writes zeros into the KV pool so blocks are claimed, shared and evicted exactly as
with a real model, sleeps for the time a :class:`TimingModel` says a prefill or decode step takes
(tokens served from the prefix cache are not in the step, so they cost nothing), and returns a
deterministic next token that depends only on the last token and its position. Output therefore
does not depend on batching, preemption or which replica served a request, which lets tests check
that nothing in between corrupted a stream.

**What the mock is not.** Its timing defaults are placeholders, not measurements: only the decode
step is loosely based on the 9 to 11 ms inter-token latency recorded for the Hugging Face baseline
on an RTX A6000 (D-015). Anything measured against the mock describes the router and the plumbing,
and says nothing about a real GPU until the timing model is calibrated against real runs.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass

import torch
from torch import Tensor

from engine.config import LlamaConfig
from engine.kv_cache import KVCache

OUTPUT_VOCAB = 1024  # generated tokens are drawn from this many ids (prompts may use any id)


@dataclass(frozen=True)
class TimingModel:
    """Seconds a forward pass takes. Uncalibrated placeholders (see the module docstring)."""

    prefill_base_ms: float = 3.0
    prefill_ms_per_token: float = 0.03
    decode_base_ms: float = 9.0
    decode_ms_per_sequence: float = 0.1
    time_scale: float = 1.0  # multiply every duration; 0 makes the mock instantaneous (tests)

    def prefill_s(self, tokens: int) -> float:
        return self.time_scale * (self.prefill_base_ms + self.prefill_ms_per_token * tokens) / 1000

    def decode_s(self, sequences: int) -> float:
        return (
            self.time_scale * (self.decode_base_ms + self.decode_ms_per_sequence * sequences) / 1000
        )

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def mock_next_token(last_token: int, position: int) -> int:
    """The token the mock model emits after ``last_token`` at ``position`` (always the same)."""
    return (last_token * 31 + position * 17 + 7) % OUTPUT_VOCAB


def mock_config(vocab_size: int = 128256, max_positions: int = 131072) -> LlamaConfig:
    """The smallest legal configuration: a pool entry is a few bytes, so many blocks are cheap.

    ``vocab_size`` only bounds the prompt token ids the server accepts.
    """
    return LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=4,
        intermediate_size=4,
        num_layers=1,
        num_heads=2,
        num_kv_heads=1,
        max_position_embeddings=max_positions,
    )


class MockModel:
    """Stands in for :class:`engine.model.Llama` in the scheduler."""

    def __init__(self, cfg: LlamaConfig | None = None, timing: TimingModel | None = None) -> None:
        self.cfg = cfg or mock_config()
        self.timing = timing or TimingModel()
        self.dtype = torch.float32
        self.device = torch.device("cpu")
        self.prefill_tokens = 0  # tokens run through a prefill (cached prefix tokens are not)
        self.decode_sequence_steps = 0  # sequence-steps run through batched decode

    @torch.inference_mode()
    def __call__(
        self,
        input_ids: Tensor,
        positions: Tensor | None = None,
        cache: KVCache | None = None,
        last_only: bool = False,
    ) -> Tensor:
        batch, tokens = input_ids.shape
        if positions is None:
            start = cache.length if cache is not None else 0
            positions = torch.arange(start, start + tokens).unsqueeze(0).expand(batch, tokens)
        if cache is not None:
            keys = torch.zeros(batch, self.cfg.num_kv_heads, tokens, self.cfg.head_dim)
            for layer in range(self.cfg.num_layers):
                cache.update(layer, keys, keys)
            cache.attention_mask(tokens)
            cache.advance(tokens)

        if tokens == 1:
            duration = self.timing.decode_s(batch)
            self.decode_sequence_steps += batch
        else:
            duration = self.timing.prefill_s(tokens)
            self.prefill_tokens += tokens
        if duration > 0:
            time.sleep(duration)

        successor = (input_ids[:, -1] * 31 + positions[:, -1] * 17 + 7) % OUTPUT_VOCAB
        logits = torch.zeros(batch, 1, OUTPUT_VOCAB)
        logits.scatter_(2, successor.view(batch, 1, 1), 1.0)
        return logits
