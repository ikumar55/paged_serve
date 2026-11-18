import dataclasses

import numpy as np
import pytest

from bench.workload import LengthDist, WorkloadConfig, generate_workload


def config(**overrides: object) -> WorkloadConfig:
    base = WorkloadConfig(
        rate=10.0,
        num_requests=2000,
        prompt=LengthDist(median=100, sigma=0.5, minimum=4, maximum=1000),
        output=LengthDist(median=50, sigma=0.4, minimum=1, maximum=300),
        seed=7,
        vocab_size=5000,
    )
    return dataclasses.replace(base, **overrides)  # type: ignore[arg-type]


def test_same_seed_gives_identical_workload() -> None:
    assert generate_workload(config()) == generate_workload(config())


def test_different_seed_gives_different_workload() -> None:
    assert generate_workload(config(seed=1)) != generate_workload(config(seed=2))


def test_arrivals_are_increasing_and_poisson() -> None:
    specs = generate_workload(config(num_requests=20000))
    arrivals = np.array([s.arrival_s for s in specs])
    gaps = np.diff(arrivals, prepend=0.0)
    assert np.all(gaps > 0)
    assert gaps.mean() == pytest.approx(1 / 10.0, rel=0.03)
    # An exponential distribution has standard deviation equal to its mean.
    assert gaps.std() / gaps.mean() == pytest.approx(1.0, rel=0.03)


def test_lengths_respect_bounds_and_median() -> None:
    specs = generate_workload(config(num_requests=20000))
    prompt_lengths = np.array([len(s.prompt_token_ids) for s in specs])
    output_lengths = np.array([s.max_tokens for s in specs])
    assert prompt_lengths.min() >= 4 and prompt_lengths.max() <= 1000
    assert output_lengths.min() >= 1 and output_lengths.max() <= 300
    assert np.median(prompt_lengths) == pytest.approx(100, rel=0.05)
    assert np.median(output_lengths) == pytest.approx(50, rel=0.05)


def test_zero_sigma_gives_constant_lengths() -> None:
    specs = generate_workload(config(prompt=LengthDist(64, 0.0), output=LengthDist(16, 0.0)))
    assert {len(s.prompt_token_ids) for s in specs} == {64}
    assert {s.max_tokens for s in specs} == {16}


def test_token_ids_stay_in_range() -> None:
    cfg = config(min_token_id=2000)
    tokens = [t for s in generate_workload(cfg) for t in s.prompt_token_ids]
    assert min(tokens) >= 2000 and max(tokens) < 5000


def test_streams_are_independent() -> None:
    """Changing the prompt distribution must not reshuffle arrivals or output lengths."""
    base = generate_workload(config(num_requests=200))
    changed = generate_workload(config(num_requests=200, prompt=LengthDist(500, 0.1, 1, 2000)))
    assert [s.arrival_s for s in base] == [s.arrival_s for s in changed]
    assert [s.max_tokens for s in base] == [s.max_tokens for s in changed]
    assert [s.prompt_token_ids for s in base] != [s.prompt_token_ids for s in changed]


def test_config_roundtrips_through_dict() -> None:
    cfg = config()
    assert WorkloadConfig.from_dict(cfg.to_dict()) == cfg


@pytest.mark.parametrize(
    "build",
    [
        lambda: LengthDist(0, 0.5),
        lambda: LengthDist(10, -1),
        lambda: LengthDist(10, 0.5, minimum=5, maximum=2),
        lambda: config(rate=0),
        lambda: config(num_requests=0),
        lambda: config(min_token_id=5000),
    ],
)
def test_invalid_parameters_are_rejected(build: object) -> None:
    with pytest.raises(ValueError):
        build()  # type: ignore[operator]
