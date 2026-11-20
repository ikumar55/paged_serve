"""Shared prompt prefixes in the synthetic workload; old workloads must stay unchanged."""

import dataclasses
import hashlib
import json
from collections import Counter
from pathlib import Path

import pytest

from bench.loadgen import main
from bench.results_io import read_results
from bench.workload import LengthDist, WorkloadConfig, generate_workload, prefix_probabilities
from tests.fake_server import UvicornServer


def fingerprint(cfg: WorkloadConfig) -> str:
    specs = generate_workload(cfg)
    blob = json.dumps([[s.id, s.arrival_s, s.prompt_token_ids, s.max_tokens] for s in specs])
    return hashlib.sha256(blob.encode()).hexdigest()


def base(**overrides: object) -> WorkloadConfig:
    cfg = WorkloadConfig(
        rate=4.0,
        num_requests=300,
        prompt=LengthDist(median=20, sigma=0.4, minimum=2, maximum=60),
        output=LengthDist(median=8, sigma=0.3, minimum=1, maximum=20),
        seed=3,
        vocab_size=5000,
    )
    return dataclasses.replace(cfg, **overrides)  # type: ignore[arg-type]


def test_workloads_without_shared_prefixes_are_exactly_what_they_always_were() -> None:
    """These hashes were taken from the code before shared prefixes existed.

    Recorded results were produced by those workloads; if this fails, a recorded result can no
    longer be regenerated from its stored configuration.
    """
    old_style = [
        (0, 32000, 4.0, 50, "ffde905bd4f023625ee705bafde9b6e7eb3fe5f96cea85f5d8e8e9a7db926128"),
        (7, 100000, 1.5, 30, "c3823ea271396831e059d58e187575a588466d0ed5716dad3645c0cca5d2ae9a"),
    ]
    for seed, vocab, rate, count, expected in old_style:
        cfg = WorkloadConfig(
            rate=rate,
            num_requests=count,
            prompt=LengthDist(256, 0.7, 1, 2048),
            output=LengthDist(128, 0.5, 1, 512),
            seed=seed,
            vocab_size=vocab,
        )
        assert fingerprint(cfg) == expected


def test_adding_shared_prefixes_does_not_reshuffle_arrivals_or_output_lengths() -> None:
    plain = generate_workload(base())
    shared = generate_workload(base(shared_prefix_groups=4, shared_prefix_len=16))
    assert [s.arrival_s for s in plain] == [s.arrival_s for s in shared]
    assert [s.max_tokens for s in plain] == [s.max_tokens for s in shared]
    # The unique suffixes are the old prompts: the same random stream, now after a prefix.
    assert [s.prompt_token_ids for s in plain] == [s.prompt_token_ids[16:] for s in shared]


def test_every_prompt_starts_with_its_groups_prefix_followed_by_a_unique_suffix() -> None:
    cfg = base(shared_prefix_groups=5, shared_prefix_len=12)
    specs = generate_workload(cfg)
    prefixes: dict[int, list[int]] = {}
    for spec in specs:
        assert spec.prefix_group is not None and 0 <= spec.prefix_group < 5
        head = spec.prompt_token_ids[:12]
        assert prefixes.setdefault(spec.prefix_group, head) == head  # one prefix per group
        assert 12 + 2 <= len(spec.prompt_token_ids) <= 12 + 60  # prefix plus the suffix bounds
    assert len({tuple(p) for p in prefixes.values()}) == len(prefixes)  # groups differ
    suffixes = {tuple(s.prompt_token_ids[12:]) for s in specs}
    assert len(suffixes) > 0.95 * len(specs)  # suffixes are essentially all different


def test_the_same_seed_gives_the_same_shared_workload_and_another_seed_does_not() -> None:
    kwargs = {"shared_prefix_groups": 3, "shared_prefix_len": 8, "shared_prefix_skew": 1.0}
    assert fingerprint(base(**kwargs)) == fingerprint(base(**kwargs))
    assert fingerprint(base(**kwargs)) != fingerprint(base(seed=4, **kwargs))


def test_skew_makes_low_numbered_groups_hotter() -> None:
    cfg = base(num_requests=4000, shared_prefix_groups=4, shared_prefix_len=8, shared_prefix_skew=1)
    counts = Counter(s.prefix_group for s in generate_workload(cfg))
    expected = prefix_probabilities(4, 1.0)
    for group in range(4):
        assert counts[group] / 4000 == pytest.approx(expected[group], abs=0.03)
    assert counts[0] > counts[1] > counts[2] > counts[3]

    uniform = base(num_requests=4000, shared_prefix_groups=4, shared_prefix_len=8)
    flat = Counter(s.prefix_group for s in generate_workload(uniform))
    assert all(abs(flat[g] / 4000 - 0.25) < 0.03 for g in range(4))


def test_prefix_probabilities_sum_to_one() -> None:
    for skew in (0.0, 0.5, 1.0, 2.0):
        assert sum(prefix_probabilities(7, skew)) == pytest.approx(1.0)


def test_config_with_prefixes_roundtrips_and_old_dicts_still_load() -> None:
    cfg = base(shared_prefix_groups=3, shared_prefix_len=10, shared_prefix_skew=0.5)
    assert WorkloadConfig.from_dict(cfg.to_dict()) == cfg
    old = base().to_dict()
    for key in ("shared_prefix_groups", "shared_prefix_len", "shared_prefix_skew"):
        del old[key]  # a result file written before these existed
    assert WorkloadConfig.from_dict(old) == base()


@pytest.mark.parametrize(
    "overrides",
    [
        {"shared_prefix_groups": 3},  # a prefix length is needed too
        {"shared_prefix_len": 8},  # and a number of groups
        {"shared_prefix_groups": -1, "shared_prefix_len": 8},
        {"shared_prefix_groups": 2, "shared_prefix_len": 8, "shared_prefix_skew": -0.5},
    ],
)
def test_inconsistent_prefix_settings_are_rejected(overrides: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        base(**overrides)


def test_the_load_generator_records_and_sends_shared_prefix_workloads(tmp_path: Path) -> None:
    from tests.test_engine_server import make_app

    app, scheduler = make_app()
    out = tmp_path / "shared.jsonl"
    argv = [
        "run", "--model", "tiny", "--rate", "100", "--num-requests", "12", "--warmup", "0",
        "--vocab-size", "128", "--min-token-id", "3",
        "--prompt-median", "4", "--prompt-sigma", "0.2", "--prompt-max", "6",
        "--output-median", "3", "--output-sigma", "0.2", "--output-max", "4",
        "--shared-prefix-groups", "2", "--shared-prefix-len", "16",
        "--fetch-server-info", "--out", str(out),
    ]  # fmt: skip
    with UvicornServer(app) as url:
        assert main([*argv, "--url", url]) == 0

    meta, results, _ = read_results(out)
    workload = meta["workload"]
    assert workload["shared_prefix_groups"] == 2 and workload["shared_prefix_len"] == 16
    assert len(results) == 12 and all(r.status == "ok" for r in results)
    # Sixteen of the tokens in every prompt are shared, so the engine's prefix cache must hit
    # (the block size in this test server is 4 tokens).
    stats = meta["server_info"]["after"]["stats"]
    assert stats["prefix_hit_tokens"] > 0
    assert scheduler.pool.allocator.num_used == 0
