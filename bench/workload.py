"""Synthetic workloads: Poisson arrivals with lognormal prompt and output lengths.

Everything here is pure and seeded, so a workload can be regenerated exactly from its config.
Prompts are lists of token ids, which keeps lengths exact and needs no tokenizer. Arrivals,
prompt lengths, output lengths and token contents each draw from an independent random stream,
so changing one distribution does not reshuffle the others.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class LengthDist:
    """Lognormal token-count distribution, rounded and clipped to ``[minimum, maximum]``.

    ``median`` is the median before clipping and ``sigma`` the standard deviation of the log.
    ``sigma = 0`` gives a constant length.
    """

    median: float
    sigma: float
    minimum: int = 1
    maximum: int = 4096

    def __post_init__(self) -> None:
        if self.median <= 0:
            raise ValueError("median must be positive")
        if self.sigma < 0:
            raise ValueError("sigma must be non-negative")
        if not 1 <= self.minimum <= self.maximum:
            raise ValueError("need 1 <= minimum <= maximum")

    def sample(self, rng: np.random.Generator, count: int) -> npt.NDArray[np.int64]:
        raw = self.median * np.exp(self.sigma * rng.standard_normal(count))
        clipped = np.clip(np.rint(raw), self.minimum, self.maximum)
        return clipped.astype(np.int64)


@dataclass(frozen=True)
class WorkloadConfig:
    """An open-loop workload: ``num_requests`` arrivals at ``rate`` requests per second."""

    rate: float
    num_requests: int
    prompt: LengthDist
    output: LengthDist
    seed: int = 0
    vocab_size: int = 32000
    min_token_id: int = 1000  # skip the lowest ids, which are often special or control tokens

    def __post_init__(self) -> None:
        if self.rate <= 0:
            raise ValueError("rate must be positive")
        if self.num_requests <= 0:
            raise ValueError("num_requests must be positive")
        if not 0 <= self.min_token_id < self.vocab_size:
            raise ValueError("need 0 <= min_token_id < vocab_size")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> WorkloadConfig:
        return cls(
            rate=float(raw["rate"]),
            num_requests=int(raw["num_requests"]),
            prompt=LengthDist(**raw["prompt"]),
            output=LengthDist(**raw["output"]),
            seed=int(raw["seed"]),
            vocab_size=int(raw["vocab_size"]),
            min_token_id=int(raw["min_token_id"]),
        )


@dataclass(frozen=True)
class RequestSpec:
    id: int
    arrival_s: float  # seconds after the run starts at which the request should be sent
    prompt_token_ids: list[int]
    max_tokens: int


def generate_workload(cfg: WorkloadConfig) -> list[RequestSpec]:
    """Return the requests of ``cfg`` in arrival order."""
    arrivals_rng, prompt_len_rng, output_len_rng, token_rng = (
        np.random.default_rng(stream) for stream in np.random.SeedSequence(cfg.seed).spawn(4)
    )
    gaps = arrivals_rng.exponential(1.0 / cfg.rate, cfg.num_requests)
    arrival_times = np.cumsum(gaps)
    prompt_lengths = cfg.prompt.sample(prompt_len_rng, cfg.num_requests)
    output_lengths = cfg.output.sample(output_len_rng, cfg.num_requests)

    specs: list[RequestSpec] = []
    for index in range(cfg.num_requests):
        tokens = token_rng.integers(cfg.min_token_id, cfg.vocab_size, int(prompt_lengths[index]))
        specs.append(
            RequestSpec(
                id=index,
                arrival_s=float(arrival_times[index]),
                prompt_token_ids=[int(t) for t in tokens],
                max_tokens=int(output_lengths[index]),
            )
        )
    return specs
