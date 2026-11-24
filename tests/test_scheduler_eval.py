"""The scheduler evaluation's comparison logic, and a CPU run of the three configurations."""

from __future__ import annotations

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
