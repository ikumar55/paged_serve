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
    # Shared prompt prefixes (for prefix caching and prefix-aware routing). With
    # ``shared_prefix_groups`` > 0 every prompt is one of that many fixed prefixes of
    # ``shared_prefix_len`` tokens followed by a unique suffix whose length comes from ``prompt``.
    # Group ``g`` is chosen with probability proportional to ``1 / (g + 1) ** shared_prefix_skew``:
    # 0 is uniform, 1 is a Zipf-like popularity where a few prefixes are much hotter than the rest.
    shared_prefix_groups: int = 0
    shared_prefix_len: int = 0
    shared_prefix_skew: float = 0.0

    def __post_init__(self) -> None:
        if self.rate <= 0:
            raise ValueError("rate must be positive")
        if self.num_requests <= 0:
            raise ValueError("num_requests must be positive")
        if not 0 <= self.min_token_id < self.vocab_size:
            raise ValueError("need 0 <= min_token_id < vocab_size")
        if self.shared_prefix_groups < 0 or self.shared_prefix_len < 0:
            raise ValueError("shared prefix settings must be non-negative")
        if (self.shared_prefix_groups > 0) != (self.shared_prefix_len > 0):
            raise ValueError("set both shared_prefix_groups and shared_prefix_len, or neither")
        if self.shared_prefix_skew < 0:
            raise ValueError("shared_prefix_skew must be non-negative")

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
            # Result files written before shared prefixes existed do not have these keys.
            shared_prefix_groups=int(raw.get("shared_prefix_groups", 0)),
            shared_prefix_len=int(raw.get("shared_prefix_len", 0)),
            shared_prefix_skew=float(raw.get("shared_prefix_skew", 0.0)),
        )


@dataclass(frozen=True)
class RequestSpec:
    id: int
    arrival_s: float  # seconds after the run starts at which the request should be sent
    prompt_token_ids: list[int]
    max_tokens: int
    prefix_group: int | None = None  # which shared prefix the prompt starts with, if any


def prefix_probabilities(groups: int, skew: float) -> npt.NDArray[np.float64]:
    """Chance of each shared-prefix group: proportional to ``1 / (rank + 1) ** skew``."""
    weights = 1.0 / np.arange(1, groups + 1, dtype=np.float64) ** skew
    probabilities: npt.NDArray[np.float64] = weights / weights.sum()
    return probabilities


def generate_workload(cfg: WorkloadConfig) -> list[RequestSpec]:
    """Return the requests of ``cfg`` in arrival order.

    The first four random streams are the ones this function has always used, so a workload
    without shared prefixes is unchanged; the two extra streams are used only when they are asked
    for.
    """
    arrivals_rng, prompt_len_rng, output_len_rng, token_rng, prefix_rng, group_rng = (
        np.random.default_rng(stream) for stream in np.random.SeedSequence(cfg.seed).spawn(6)
    )
    gaps = arrivals_rng.exponential(1.0 / cfg.rate, cfg.num_requests)
    arrival_times = np.cumsum(gaps)
    prompt_lengths = cfg.prompt.sample(prompt_len_rng, cfg.num_requests)
    output_lengths = cfg.output.sample(output_len_rng, cfg.num_requests)

    prefixes: list[list[int]] = []
    group_of: list[int | None] = [None] * cfg.num_requests
    if cfg.shared_prefix_groups > 0:
        for _ in range(cfg.shared_prefix_groups):
            drawn = prefix_rng.integers(cfg.min_token_id, cfg.vocab_size, cfg.shared_prefix_len)
            prefixes.append([int(t) for t in drawn])
        probabilities = prefix_probabilities(cfg.shared_prefix_groups, cfg.shared_prefix_skew)
        chosen = group_rng.choice(cfg.shared_prefix_groups, size=cfg.num_requests, p=probabilities)
        group_of = [int(g) for g in chosen]

    specs: list[RequestSpec] = []
    for index in range(cfg.num_requests):
        suffix = token_rng.integers(cfg.min_token_id, cfg.vocab_size, int(prompt_lengths[index]))
        group = group_of[index]
        head = prefixes[group] if group is not None else []
        specs.append(
            RequestSpec(
                id=index,
                arrival_s=float(arrival_times[index]),
                prompt_token_ids=[*head, *(int(t) for t in suffix)],
                max_tokens=int(output_lengths[index]),
                prefix_group=group,
            )
        )
    return specs
