"""The KL and perplexity accuracy tool (D-033), on a tiny CPU model.

The float64 reference attention stands in for the Triton kernel and a stand-in object for Hugging
Face, so what is checked is the tool: the text and its windows, the metrics, the bootstrap, the
three-state verdicts, the teacher-forced lockstep, and that a deliberately broken attention path is
detected as such.
"""

from __future__ import annotations

import itertools
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from bench import kl_perplexity as kp
from bench.bf16_logit_error import build_variants
from engine.config import LlamaConfig
from engine.paged_attention import decode_attention_reference
from tests.test_bf16_logit_error import bf16_copy
from tests.test_paged_cache import make_model

CFG = LlamaConfig.tiny()
WINDOW, PROMPT = 24, 8  # 15 teacher-forced decode steps per window


# -- the text and its windows ------------------------------------------------------------------


def test_load_text_normalises_line_endings_and_hashes_the_text(tmp_path: Path) -> None:
    (tmp_path / "a.md").write_bytes(b"one\r\ntwo\r\n")
    (tmp_path / "b.py").write_bytes(b"x = 1\n")
    text, digest = kp.load_text(tmp_path, ["a.md", "b.py"])
    assert text == "one\ntwo\n\n\nx = 1\n"
    (tmp_path / "a.md").write_bytes(b"one\ntwo\n")  # the same text with Unix endings
    assert kp.load_text(tmp_path, ["a.md", "b.py"]) == (text, digest)
    (tmp_path / "a.md").write_bytes(b"one\ntwo!\n")
    assert kp.load_text(tmp_path, ["a.md", "b.py"])[1] != digest
    with pytest.raises(FileNotFoundError):
        kp.load_text(tmp_path, ["missing.md"])


def test_the_fixed_text_files_exist_in_the_repository() -> None:
    for name in kp.DEFAULT_TEXT_FILES:
        assert (kp.REPO_ROOT / name).exists(), name


def test_windows_start_with_the_start_token_do_not_overlap_and_are_reproducible() -> None:
    stream = list(range(1000, 3000))
    windows = kp.make_windows(stream, 1, count=20, window_len=64, seed=3)
    assert len(windows) == 20 and all(len(w) == 64 and w[0] == 1 for w in windows)
    assert windows == kp.make_windows(stream, 1, count=20, window_len=64, seed=3)
    assert windows != kp.make_windows(stream, 1, count=20, window_len=64, seed=4)
    tokens = [t for w in windows for t in w[1:]]
    assert len(tokens) == len(set(tokens))  # no token of the text is in two windows
    assert all(a[1] < b[1] for a, b in itertools.pairwise(windows))  # in text order


def test_too_little_text_for_the_windows_is_an_error() -> None:
    with pytest.raises(ValueError, match="not 30"):
        kp.make_windows(list(range(500)), 1, count=30, window_len=64, seed=0)


# -- the metrics --------------------------------------------------------------------------------


def test_kl_nll_and_top1_match_hand_computed_values() -> None:
    truth = torch.log(torch.tensor([[0.5, 0.5, 0.0 + 1e-30]]))
    truth = truth - torch.logsumexp(truth, dim=-1, keepdim=True)
    variant = torch.log(torch.tensor([[0.25, 0.75, 1e-30]]))
    variant = variant - torch.logsumexp(variant, dim=-1, keepdim=True)
    kl, nll, top1 = kp.position_metrics(truth, variant, torch.tensor([1]))
    expected_kl = 0.5 * math.log(0.5 / 0.25) + 0.5 * math.log(0.5 / 0.75)
    assert kl.item() == pytest.approx(expected_kl, rel=1e-5)
    assert nll.item() == pytest.approx(-math.log(0.75), rel=1e-5)
    assert (
        top1.item() is False
    )  # the truth's argmax is entry 0 (a tie broken first), the variant's is 1
    same_kl, same_nll, same_top1 = kp.position_metrics(truth, truth, torch.tensor([0]))
    assert same_kl.item() == pytest.approx(0.0, abs=1e-7) and bool(same_top1)
    assert same_nll.item() == pytest.approx(-math.log(0.5), rel=1e-5)


def test_the_blocked_sequence_metrics_equal_the_direct_ones() -> None:
    generator = torch.Generator().manual_seed(2)
    truth = torch.log_softmax(torch.randn(3, 11, 17, generator=generator), dim=-1)
    variant = torch.log_softmax(torch.randn(3, 11, 17, generator=generator), dim=-1)
    target = torch.randint(0, 17, (3, 11), generator=generator)
    direct = kp.position_metrics(truth, variant, target)
    blocked = kp.sequence_metrics(truth, variant, target, step_block=4)
    for a, b in zip(direct, blocked, strict=True):
        assert a.shape == (3, 11) and torch.equal(a, b)


# -- statistics -----------------------------------------------------------------------------------


def test_window_stats_and_summary() -> None:
    metrics = kp.VariantMetrics()
    metrics.add(
        torch.tensor([[1.0, 3.0], [2.0, 2.0]]),
        torch.tensor([[1.0, 1.0], [3.0, 3.0]]),
        torch.tensor([[1, 0], [1, 1]]),
    )
    metrics.add(torch.tensor([[4.0, 4.0]]), torch.tensor([[2.0, 2.0]]), torch.tensor([[0, 0]]))
    stats = kp.window_stats(metrics)  # three windows from two chunks
    assert stats.kl.tolist() == [2.0, 2.0, 4.0] and stats.nll.tolist() == [1.0, 3.0, 2.0]
    assert stats.top1.tolist() == [0.5, 1.0, 0.0] and stats.kl_positions.shape == (6,)
    summary = kp.summarise(stats)
    assert summary["mean_kl"] == pytest.approx(8 / 3) and summary["perplexity"] == pytest.approx(
        math.exp(2.0)
    )
    assert summary["windows"] == 3 and summary["positions"] == 6
    assert summary["top1_agreement"] == pytest.approx(0.5)


def test_the_verdict_has_three_states() -> None:
    assert kp.verdict(1.0, 1.2, limit=1.10, bound=1.25) == "met"
    assert kp.verdict(1.05, 1.30, limit=1.10, bound=1.25) == "inconclusive"
    assert kp.verdict(1.11, 1.12, limit=1.10, bound=1.25) == "failed"
    assert kp.verdict(1.10, 1.25, limit=1.10, bound=1.25) == "met"  # exactly at the limits


def test_bootstrap_indices_are_paired_and_reproducible() -> None:
    a, b = kp.bootstrap_indices(50, 100, seed=1), kp.bootstrap_indices(50, 100, seed=1)
    assert a.shape == (100, 50) and np.array_equal(a, b)
    assert not np.array_equal(a, kp.bootstrap_indices(50, 100, seed=2))
    assert a.min() >= 0 and a.max() < 50


def test_a_ratio_of_one_with_little_noise_is_met_and_a_bad_ratio_fails() -> None:
    rng = np.random.default_rng(0)
    base = rng.uniform(0.9, 1.1, 200)
    index = kp.bootstrap_indices(200, 500, seed=0)
    same = kp.ratio_criterion(base * 1.0, base, index, 1.10, 1.25)
    assert same["verdict"] == "met" and same["value"] == pytest.approx(1.0)
    worse = kp.ratio_criterion(base * 1.3, base, index, 1.10, 1.25)
    assert worse["verdict"] == "failed" and worse["value"] == pytest.approx(1.3)
    noisy_num = rng.uniform(0.0, 2.2, 6)  # six windows: a point estimate near 1 but a wide interval
    noisy_den = rng.uniform(0.0, 2.0, 6)
    wide = kp.ratio_criterion(noisy_num, noisy_den, kp.bootstrap_indices(6, 500, seed=0), 5.0, 6.0)
    assert wide["verdict"] in {"inconclusive", "met"} and wide["ci"][1] > wide["ci"][0]


def test_the_perplexity_criterion_is_the_absolute_relative_difference() -> None:
    rng = np.random.default_rng(1)
    reference = rng.uniform(1.8, 2.2, 200)
    index = kp.bootstrap_indices(200, 500, seed=0)
    ok = kp.perplexity_criterion(reference + 0.0005, reference, index, 0.002, 0.003)
    assert ok["verdict"] == "met" and ok["signed"] == pytest.approx(math.exp(0.0005) - 1)
    below = kp.perplexity_criterion(
        reference - 0.01, reference, index, 0.002, 0.003
    )  # lower is also a difference
    assert below["verdict"] == "failed" and below["signed"] < 0 and below["value"] > 0.002


def synthetic_stats(
    kernel_factor: float = 1.0, graphs_factor: float = 1.0, hf_bf16_scale: float = 1.0
) -> dict[str, kp.WindowStats]:
    """Equal paths by default; ``hf_bf16_scale`` makes Hugging Face bf16 less or more noisy."""
    rng = np.random.default_rng(5)
    windows = 200
    gather_kl = rng.uniform(0.9e-3, 1.1e-3, windows)
    hf_bf16_kl = gather_kl * hf_bf16_scale * rng.uniform(0.97, 1.03, windows)
    nll = rng.uniform(1.9, 2.1, windows)

    def stats(kl: np.ndarray[Any, Any], nll_shift: float = 0.0) -> kp.WindowStats:
        return kp.WindowStats(kl, nll + nll_shift, np.full(windows, 0.9), kl)

    return {
        kp.HF_TRUTH: stats(np.zeros(windows)),
        kp.HF_BF16: stats(hf_bf16_kl),
        kp.GATHER: stats(gather_kl),
        kp.KERNEL: stats(gather_kl * kernel_factor),
        kp.GRAPHS: stats(gather_kl * graphs_factor),
        "engine-fp32-gather": stats(np.full(windows, 1e-9)),
        "engine-fp32-kernel": stats(np.full(windows, 2e-9)),
    }


def test_evaluate_meets_every_criterion_for_equal_paths() -> None:
    result = kp.evaluate(synthetic_stats(), resamples=300, seed=0)
    assert result["overall"] == "met" and result["counts"] == {
        "met": 11,
        "inconclusive": 0,
        "failed": 0,
    }
    assert set(result["i"]) == {kp.KERNEL, kp.GRAPHS} and set(result["iii"]) == {
        kp.GATHER,
        kp.KERNEL,
        kp.GRAPHS,
    }
    assert set(result["iv"]) == {"engine-fp32-gather", "engine-fp32-kernel"}


def test_evaluate_fails_a_kernel_that_is_twenty_percent_worse() -> None:
    result = kp.evaluate(synthetic_stats(kernel_factor=1.2), resamples=300, seed=0)
    assert result["overall"] == "failed"
    assert (
        result["i"][kp.KERNEL]["verdict"] == "failed" and result["i"][kp.GRAPHS]["verdict"] == "met"
    )


def test_evaluate_fails_when_a_path_is_noisier_than_hugging_face_by_more_than_twice() -> None:
    result = kp.evaluate(synthetic_stats(graphs_factor=2.5), resamples=300, seed=0)
    assert (
        result["iii"][kp.GRAPHS]["verdict"] == "failed"
        and result["iii"][kp.GATHER]["verdict"] == "met"
    )


def test_evaluate_fails_a_wrong_fp32_path_and_skips_absent_variants() -> None:
    stats = synthetic_stats()
    stats["engine-fp32-kernel"] = kp.WindowStats(
        np.full(200, 1e-4), stats[kp.GATHER].nll, np.full(200, 1.0), np.full(200, 1e-4)
    )
    del stats[kp.GRAPHS]  # as with --no-graphs
    result = kp.evaluate(stats, resamples=300, seed=0)
    assert result["iv"]["engine-fp32-kernel"]["mean_kl"]["verdict"] == "failed"
    assert set(result["i"]) == {kp.KERNEL} and kp.GRAPHS not in result["iii"]


# -- the lockstep on real (here: random) text, end to end -------------------------------------


class TinyHuggingFace:
    """Stands in for a Hugging Face model: ``model(ids, logits_to_keep=n).logits``."""

    def __init__(self, model: Any) -> None:
        self.model = model
        self.kept: list[int] = []

    def __call__(self, ids: torch.Tensor, logits_to_keep: int) -> Any:
        self.kept.append(logits_to_keep)
        return SimpleNamespace(logits=self.model(ids)[:, -logits_to_keep:])


def tiny_setup(
    attention: Any = decode_attention_reference,
) -> tuple[Any, list[Any], kp.HuggingFaceReference, TinyHuggingFace]:
    fp32 = make_model(CFG)
    reference = kp.HuggingFaceReference(TinyHuggingFace(fp32), TinyHuggingFace(bf16_copy(fp32)))
    truth, others = build_variants(
        fp32, bf16_copy(fp32), attention, num_blocks=200, block_size=4, device="cpu",
        graphs=True, graph_max_batch=8, graph_capture=False,
    )  # fmt: skip
    return truth, others, reference, reference.models["truth"]


def tiny_windows(count: int = 4) -> torch.Tensor:
    generator = torch.Generator().manual_seed(9)
    return torch.randint(1, CFG.vocab_size, (count, WINDOW), generator=generator)


def run(attention: Any = decode_attention_reference) -> dict[str, kp.VariantMetrics]:
    truth, others, reference, _ = tiny_setup(attention)
    metrics: dict[str, kp.VariantMetrics] = {}
    with torch.inference_mode():
        kp.run_chunk(tiny_windows(), reference, [truth, *others], PROMPT, metrics)
    return metrics


def test_the_huggingface_adapter_slices_the_positions_the_steps_predict() -> None:
    _, _, reference, fp32 = tiny_setup()
    windows = tiny_windows(3)
    out = reference(windows, PROMPT)
    steps = WINDOW - 1 - PROMPT
    assert set(out) == {"truth", kp.HF_BF16} and out["truth"].shape == (3, steps, CFG.vocab_size)
    assert fp32.kept == [WINDOW - PROMPT]  # asked for the last 24 - 8 positions
    assert torch.allclose(
        out["truth"].exp().sum(-1), torch.ones(3, steps), atol=1e-5
    )  # log-probabilities
    expected = torch.log_softmax(make_model(CFG)(windows)[:, PROMPT : WINDOW - 1].float(), dim=-1)
    assert torch.allclose(out["truth"], expected, atol=1e-5)


def test_the_lockstep_measures_every_variant_at_every_decode_position() -> None:
    metrics = run()
    assert set(metrics) == {kp.HF_TRUTH, kp.HF_BF16, *kp.ENGINE_NAMES.values()}
    steps = WINDOW - 1 - PROMPT
    for name, m in metrics.items():
        kl, nll, top1 = m.arrays()
        assert kl.shape == nll.shape == top1.shape == (4, steps), name
        assert (kl > -1e-6).all() and np.isfinite(nll).all()
    assert metrics[kp.HF_TRUTH].arrays()[0].max() == pytest.approx(
        0.0, abs=1e-7
    )  # truth against itself


def test_an_fp32_engine_path_matches_the_truth_and_bf16_is_measurably_off() -> None:
    metrics = run()
    fp32 = kp.window_stats(metrics["engine-fp32-gather"]).kl.mean()
    kernel = kp.window_stats(metrics["engine-fp32-kernel"]).kl.mean()
    graphs = kp.window_stats(metrics["engine-fp32-graphs"]).kl.mean()
    assert fp32 < 1e-7 and kernel < 1e-7 and graphs < 1e-7  # fp32 noise only
    bf16 = kp.window_stats(metrics[kp.GATHER]).kl.mean()
    assert bf16 > 100 * max(fp32, kernel, graphs)  # bf16 rounding is plainly visible next to it
    assert kp.window_stats(metrics[kp.HF_BF16]).kl.mean() > 100 * fp32


def test_the_forced_tokens_are_the_real_text_not_the_models_own_choice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fed: list[torch.Tensor] = []
    original = kp.decode_step

    def spy(variant: Any, tokens: torch.Tensor) -> torch.Tensor:
        fed.append(tokens.clone())
        return original(variant, tokens)

    monkeypatch.setattr(kp, "decode_step", spy)
    run()
    windows = tiny_windows()
    first_variant_calls = fed[:: len(kp.ENGINE_NAMES)]  # variant 0 of each of the decode steps
    assert len(first_variant_calls) == WINDOW - 1 - PROMPT
    for step, tokens in enumerate(first_variant_calls):
        assert torch.equal(tokens, windows[:, PROMPT + step])


def test_the_targets_are_the_next_real_tokens() -> None:
    """The truth's likelihoods are those of tokens 129 and later; a shifted target would differ."""
    truth, others, reference, _ = tiny_setup()
    windows = tiny_windows()
    metrics: dict[str, kp.VariantMetrics] = {}
    with torch.inference_mode():
        kp.run_chunk(windows, reference, [truth, *others], PROMPT, metrics)
    with torch.inference_mode():
        lp = reference(windows, PROMPT)["truth"]
    steps = WINDOW - 1 - PROMPT
    expected = -lp.gather(-1, windows[:, PROMPT + 1 : PROMPT + 1 + steps].unsqueeze(-1)).squeeze(-1)
    assert np.allclose(metrics[kp.HF_TRUTH].arrays()[1], expected.double().numpy(), atol=1e-6)


def test_a_broken_attention_path_is_detected_end_to_end() -> None:
    def broken(q: Any, k_pool: Any, v_pool: Any, tables: Any, lengths: Any) -> Any:
        return torch.zeros_like(q)  # attends to nothing

    metrics = run(broken)
    stats = {name: kp.window_stats(m) for name, m in metrics.items()}
    # the gather variants do not use the function; the kernel and graph variants do
    assert stats[kp.GATHER].kl.mean() < 1e-3 < stats[kp.KERNEL].kl.mean()
    result = kp.evaluate(stats, resamples=200, seed=0)
    assert result["overall"] == "failed"
    assert (
        result["i"][kp.KERNEL]["verdict"] == "failed"
        and result["iv"]["engine-fp32-kernel"]["mean_kl"]["verdict"] == "failed"
    )


def test_the_command_line_checks_the_window_split() -> None:
    base = ["--model-dir", "m", "--out", "o", "--label", "x"]
    assert kp.parse_args(base).windows == 200 and kp.parse_args(base).prompt_len == 128
    with pytest.raises(SystemExit):
        kp.parse_args([*base, "--prompt-len", "255"])  # leaves no token to predict
    with pytest.raises(SystemExit):
        kp.parse_args([*base, "--prompt-len", "0"])


def test_running_the_windows_in_chunks_gives_the_same_numbers() -> None:
    """State left over between chunks (caches, blocks) would change the second chunk's results."""
    windows = tiny_windows(4)
    whole: dict[str, kp.VariantMetrics] = {}
    truth, others, reference, _ = tiny_setup()
    with torch.inference_mode():
        kp.run_chunk(windows, reference, [truth, *others], PROMPT, whole)
    chunked: dict[str, kp.VariantMetrics] = {}
    truth, others, reference, _ = tiny_setup()
    with torch.inference_mode():
        for start in (0, 2):
            kp.run_chunk(windows[start : start + 2], reference, [truth, *others], PROMPT, chunked)
    for name in whole:
        for a, b in zip(whole[name].arrays(), chunked[name].arrays(), strict=True):
            assert a.shape == b.shape == (4, WINDOW - 1 - PROMPT)
            assert np.allclose(a, b, atol=1e-6), name
    # and the pools are empty again after a chunk: nothing leaked from the decode steps
    assert all(
        v.pool.allocator.num_used <= 1 for v in [truth, *others]
    )  # at most a graph scratch block


def test_criterion_iii_compares_with_hugging_face_bf16_not_with_the_gather_path() -> None:
    """Hugging Face bf16 three times less noisy than the engine: the engine is more than twice as
    noisy as the yardstick, although it equals its own gather path."""
    quiet_hf = kp.evaluate(synthetic_stats(hf_bf16_scale=1 / 3), resamples=300, seed=0)
    assert all(c["verdict"] == "failed" for c in quiet_hf["iii"].values())
    assert all(c["verdict"] == "met" for c in quiet_hf["i"].values())  # equal to the gather path
    noisy_hf = kp.evaluate(synthetic_stats(hf_bf16_scale=3.0), resamples=300, seed=0)
    assert all(c["verdict"] == "met" for c in noisy_hf["iii"].values())


def test_the_extended_text_set_contains_the_default_one_and_exists() -> None:
    assert kp.EXTENDED_TEXT_FILES[: len(kp.DEFAULT_TEXT_FILES)] == kp.DEFAULT_TEXT_FILES
    assert len(set(kp.EXTENDED_TEXT_FILES)) == len(kp.EXTENDED_TEXT_FILES)
    for name in kp.EXTENDED_TEXT_FILES:
        assert (kp.REPO_ROOT / name).exists(), name
    base = ["--model-dir", "m", "--out", "o", "--label", "x"]
    assert kp.parse_args(base).text_set == "default"
    assert kp.parse_args([*base, "--text-set", "extended"]).text_set == "extended"
