"""The scheduler evaluation's comparison logic, and a CPU run of the three configurations."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from bench import scheduler_eval as se
from engine.config import LlamaConfig
from engine.generate import greedy_generate
from tests.test_paged_cache import make_model

CFG = LlamaConfig.tiny()


def test_first_divergence_finds_the_first_differing_token_or_a_length_gap() -> None:
    assert se.first_divergence([1, 2, 3], [1, 2, 3]) is None
    assert se.first_divergence([1, 2, 3], [1, 9, 3]) == 1
    assert se.first_divergence([1, 2, 3], [1, 2]) == 2
    assert se.first_divergence([], []) is None


def test_compare_counts_identical_outputs_and_lists_the_rest() -> None:
    result = se.compare([[1, 2], [3, 4], [5, 6]], [[1, 2], [3, 9], [5, 6]])
    assert result == {
        "prompts": 3,
        "identical": 2,
        "diverged": [{"prompt": 1, "first_divergence": 1}],
    }


def test_compare_rejects_a_missing_output() -> None:
    with pytest.raises(ValueError):
        se.compare([[1]], [[1], [2]])


def write_greedy_record(path: Path, *, tamper: bool = False) -> list[list[int]]:
    from bench.greedy_eval import Prompt, prompt_set_hash

    prompts = [Prompt("prose", [1, 2, 3]), Prompt("random", [9, 8])]
    digest = prompt_set_hash(prompts)
    if tamper:
        prompts = [Prompt("prose", [1, 2, 4]), prompts[1]]  # ids changed after hashing
    record = {
        "meta": {"prompt_set": {"seed": 0, "counts": {"prose": 1, "random": 1}, "sha256": digest}},
        "prompts": [{"kind": p.kind, "prompt_ids": p.ids} for p in prompts],
    }
    path.write_text(json.dumps(record))
    return [p.ids for p in prompts]


def test_recorded_prompts_are_read_back_exactly_with_their_hash(tmp_path: Path) -> None:
    path = tmp_path / "greedy-eval-x.json"
    expected = write_greedy_record(path)
    ids, meta = se.load_recorded_prompts(path)
    assert ids == expected == [[1, 2, 3], [9, 8]]
    assert meta["source"] == "greedy-eval-x.json"
    assert meta["counts"] == {"prose": 1, "random": 1}
    assert len(meta["sha256"]) == 64


def test_recorded_prompts_that_do_not_match_their_hash_are_refused(tmp_path: Path) -> None:
    path = tmp_path / "greedy-eval-x.json"
    write_greedy_record(path, tamper=True)
    with pytest.raises(ValueError, match="does not match"):
        se.load_recorded_prompts(path)


def test_logit_gap_record_reports_the_gap_between_the_two_choices_and_the_top_two() -> None:
    logits = torch.tensor([0.0, 3.0, 2.875, -1.0])
    record = se.logit_gap_record(logits, reference_token=1, scheduler_token=2)
    assert record == {
        "reference_token_minus_scheduler_token_logit": 0.125,
        "reference_top1_minus_top2": 0.125,
        "reference_token_is_top1": True,
    }
    flipped = se.logit_gap_record(logits, reference_token=2, scheduler_token=1)
    assert flipped["reference_token_minus_scheduler_token_logit"] == -0.125
    assert flipped["reference_token_is_top1"] is False


def test_configs_have_an_ample_pool_and_a_small_one_that_still_fits_one_sequence() -> None:
    served, no_cache, preempting = se.configs_for(
        num_prompts=20, tokens=8, max_prompt=40, max_batch=8
    )
    per_sequence = -(-(40 + 8) // se.BLOCK_SIZE) + 1
    assert served.prefix_caching and not no_cache.prefix_caching and preempting.prefix_caching
    assert served.num_blocks == no_cache.num_blocks >= per_sequence * 20
    assert per_sequence < preempting.num_blocks < per_sequence * 8  # fits one, not the batch


def test_all_three_configurations_reproduce_single_request_decoding_on_a_tiny_model() -> None:
    model = make_model(CFG)
    generator = torch.manual_seed(3)
    prompts = [
        torch.randint(0, CFG.vocab_size, (n,), generator=generator).tolist()
        for n in (3, 11, 6, 17, 9)
    ]
    n_tokens = 40  # long enough that five sequences cannot share the small pool
    with torch.inference_mode():
        reference = [greedy_generate(model, p, n_tokens) for p in prompts]
        for config in se.configs_for(len(prompts), n_tokens, 17, max_batch=5):
            produced, stats = se.run_scheduler(model, prompts, n_tokens, config, torch.float32)
            result = se.compare(reference, produced)
            assert result["identical"] == len(prompts), (config.name, result)
            if config.name == "preempting":
                assert stats["preemptions"] > 0, "the small pool was meant to force preemption"
