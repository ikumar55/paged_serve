"""The greedy-agreement tool: prompt construction, statistics, and that it can actually fail."""

from __future__ import annotations

from pathlib import Path

import torch

from bench.greedy_eval import (
    Prompt,
    PromptResult,
    build_prompt_set,
    counts_for,
    evaluate_prompt,
    format_summary,
    prompt_set_hash,
    summarize,
    wilson_interval,
)
from engine.config import LlamaConfig
from tests.test_model import make_pair

N_TOKENS = 12


def encode(text: str) -> list[int]:
    return [ord(c) % 200 + 300 for c in text]  # a stand-in tokenizer: one token per character


def make_repo(root: Path) -> None:
    (root / "docs").mkdir()
    (root / "engine").mkdir()
    (root / "bench").mkdir()
    (root / "README.md").write_text("alpha beta gamma delta " * 80)
    (root / "docs" / "decisions.md").write_text("epsilon zeta eta theta " * 80)
    (root / "engine" / "a.py").write_text("def f(x):\n    return x + 1\n" * 80)
    (root / "bench" / "b.py").write_text("import os\nprint(os.name)\n" * 80)


def test_counts_split_forty_thirty_thirty() -> None:
    assert counts_for(200) == {"prose": 80, "code": 60, "random": 60}
    assert sum(counts_for(7).values()) == 7


def test_prompt_set_is_deterministic_and_well_formed(tmp_path: Path) -> None:
    make_repo(tmp_path)
    build = lambda seed: build_prompt_set(encode, 1, tmp_path, 50, seed, 128256)  # noqa: E731
    first, again, other = build(0), build(0), build(1)
    assert prompt_set_hash(first) == prompt_set_hash(again)
    assert prompt_set_hash(first) != prompt_set_hash(other)
    assert [p.kind for p in first].count("prose") == 20
    for prompt in first:
        if prompt.kind == "random":
            assert 8 <= len(prompt.ids) <= 64
            assert all(1000 <= i < 100000 for i in prompt.ids)
        else:
            assert prompt.ids[0] == 1  # BOS
            assert 17 <= len(prompt.ids) <= 129


def test_random_token_ids_respect_a_small_vocabulary(tmp_path: Path) -> None:
    make_repo(tmp_path)
    prompts = build_prompt_set(encode, 1, tmp_path, 30, 0, vocab_size=1500)
    assert all(i < 1500 for p in prompts if p.kind == "random" for i in p.ids)


def test_wilson_interval() -> None:
    assert wilson_interval(0, 0) == (0.0, 1.0)
    low, high = wilson_interval(200, 200)
    assert 0.98 < low < 1.0 and high > 0.999999
    low, high = wilson_interval(100, 200)
    assert 0.43 < low < 0.5 < high < 0.57


def result(index: int, kind: str, divergence: int | None, max_diff: float = 1e-5) -> PromptResult:
    return PromptResult(
        index=index,
        kind=kind,
        prompt_ids=[1, 2, 3],
        reference_tokens=[0] * N_TOKENS,
        engine_tokens=[0] * N_TOKENS,
        first_divergence=divergence,
        gap_at_first_divergence=None if divergence is None else 0.001,
        teacher_forced_disagreements=(
            []
            if divergence is None
            else [
                {
                    "step": divergence,
                    "reference_token": 0,
                    "engine_token": 1,
                    "reference_logit_gap": 0.001,
                    "step_max_logit_diff": 1e-3,
                }
            ]
        ),
        reference_self_mismatches=0,
        max_logit_diff=max_diff,
        ref_logit_gaps=[0.5] * (N_TOKENS - 1) + [1e-6],
    )


def test_summary_counts_and_reports_risk() -> None:
    results = [result(0, "prose", None), result(1, "prose", 6), result(2, "code", None)]
    summary = summarize(results, N_TOKENS)
    free, forced = summary["free_running"], summary["teacher_forced"]
    assert free["exact_match"] == 2
    assert free["by_kind"] == {
        "code": {"prompts": 1, "exact_match": 1},
        "prose": {"prompts": 2, "exact_match": 1},
    }
    assert free["first_divergence_steps"] == [6]
    assert abs(free["mean_matched_prefix_fraction"] - (1 + 0.5 + 1) / 3) < 1e-9
    assert forced["steps"] == 3 * N_TOKENS and forced["top1_disagreements"] == 1
    assert forced["steps_at_risk"] == 3  # one near-tie step per prompt, gap 1e-6 <= 2e-5
    assert "free-running exact match: 2/3" in format_summary(summary)


def tiny_prompts(count: int, vocab: int) -> list[Prompt]:
    generator = torch.Generator().manual_seed(7)
    return [
        Prompt("random", torch.randint(0, vocab, (int(n),), generator=generator).tolist())
        for n in torch.randint(4, 12, (count,), generator=generator)
    ]


def test_matching_models_agree_on_every_token() -> None:
    cfg = LlamaConfig.tiny(llama3_rope_scaling=True)
    reference, ours = make_pair(cfg)
    for index, prompt in enumerate(tiny_prompts(8, cfg.vocab_size)):
        outcome = evaluate_prompt(index, prompt, reference, ours, N_TOKENS, "cpu", pad_id=0)
        assert len(outcome.reference_tokens) == N_TOKENS  # EOS handling did not stop generation
        assert outcome.first_divergence is None
        assert outcome.teacher_forced_disagreements == []
        assert outcome.max_logit_diff < 1e-4


def test_a_broken_engine_is_detected() -> None:
    """The tool must be able to fail: damaging one attention projection must show up."""
    cfg = LlamaConfig.tiny()
    reference, ours = make_pair(cfg)
    with torch.no_grad():
        ours.model.layers[0].self_attn.q_proj.weight.mul_(1.5)
    outcomes = [
        evaluate_prompt(i, p, reference, ours, N_TOKENS, "cpu", pad_id=0)
        for i, p in enumerate(tiny_prompts(8, cfg.vocab_size))
    ]
    assert any(o.first_divergence is not None for o in outcomes)
    assert max(o.max_logit_diff for o in outcomes) > 1e-2
    # For every flipped step the reference's gap is bounded by twice that step's logit difference.
    for outcome in outcomes:
        for d in outcome.teacher_forced_disagreements:
            assert d["reference_logit_gap"] <= 2 * d["step_max_logit_diff"] + 1e-6
