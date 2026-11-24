"""The decode profiler's arithmetic and categories, plus an end-to-end run on a tiny CPU model."""

from __future__ import annotations

import pytest
import torch

from bench import profile_decode as pd
from engine.config import LlamaConfig
from engine.paged_cache import KVPool
from tests.test_paged_cache import make_model

CFG = LlamaConfig.tiny()


@pytest.mark.parametrize(
    ("name", "category"),
    [
        ("void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_64x64>", "matmul"),
        ("nvjet_tst_128x64_64x6_2x1_v_bz_coopA_TNT", "matmul"),
        ("void pytorch_flash::flash_fwd_kernel<Flash_fwd_kernel_traits<64>>", "attention"),
        ("fmha_cutlassF_bf16_aligned_64x64_rf_sm80", "attention"),
        ("void at::native::index_elementwise_kernel<128, 4>", "gather/copy"),
        ("void at::native::elementwise_kernel<128, 2, direct_copy_kernel_cuda>", "gather/copy"),
        (
            "void at::native::vectorized_elementwise_kernel<4, MulFunctor<bf16>>",
            "elementwise/other",
        ),
        ("void at::native::reduce_kernel<512, 1, ReduceOp<float, MeanOps>>", "elementwise/other"),
    ],
)
def test_kernel_names_fall_into_the_expected_category(name: str, category: str) -> None:
    assert pd.categorize(name) == category


def test_attention_wins_over_copy_when_a_name_mentions_both() -> None:
    assert pd.categorize("attention_copy_kernel") == "attention"


def test_summary_divides_by_steps_and_shares_sum_to_one() -> None:
    rows = [
        ("gemm_a", 160, 8000.0),  # 8 ms over 4 steps = 2 ms per step
        ("index_kernel", 64, 2000.0),
        ("mul_kernel", 320, 2000.0),
    ]
    summary = pd.summarise_kernels(rows, steps=4)
    assert summary["gpu_busy_ms_per_step"] == pytest.approx(3.0)
    assert summary["kernels_per_step"] == pytest.approx(136.0)
    shares = summary["busy_share_by_category"]
    assert shares == {
        "matmul": pytest.approx(2 / 3),
        "gather/copy": pytest.approx(1 / 6),
        "elementwise/other": pytest.approx(1 / 6),
    }
    assert sum(shares.values()) == pytest.approx(1.0)
    assert list(shares) == ["matmul", "gather/copy", "elementwise/other"]  # largest first


def test_top_kernels_are_ranked_and_limited() -> None:
    rows = [(f"k{i}", 4, float(i * 100)) for i in range(1, 6)]
    top = pd.summarise_kernels(rows, steps=2, top=2)["top_kernels"]
    assert [k["name"] for k in top] == ["k5", "k4"]
    assert top[0]["launches_per_step"] == 2.0
    assert top[0]["ms_per_step"] == pytest.approx(500 / 2 / 1000)
    assert top[0]["share"] == pytest.approx(500 / 1500)


def test_an_empty_profile_is_zero_not_an_error() -> None:
    summary = pd.summarise_kernels([], steps=3)
    assert summary["gpu_busy_ms_per_step"] == 0.0
    assert summary["busy_share_by_category"] == {}
    assert summary["top_kernels"] == []


def test_zero_steps_is_rejected() -> None:
    with pytest.raises(ValueError):
        pd.summarise_kernels([], steps=0)


def test_profile_batch_runs_end_to_end_on_a_tiny_cpu_model() -> None:
    model = make_model(CFG)
    pool = KVPool(CFG, 64, 4, torch.float32, "cpu")
    with torch.inference_mode():
        result = pd.profile_batch(model, pool, batch_size=3, prompt_len=5, steps=3, warmup=1)
    assert result["batch_size"] == 3
    assert result["wall_ms_per_step_median"] > 0
    assert result["wall_ms_per_step_min"] <= result["wall_ms_per_step_median"]
    assert result["wall_ms_per_step_median"] <= result["wall_ms_per_step_max"]
    assert result["gpu_idle_fraction"] is None  # no GPU kernels on a CPU run
    assert result["context_len_at_end"] == 5 + 1 + 6


def test_offline_throughput_serves_every_request_and_splits_the_time() -> None:
    model = make_model(CFG)
    pool = KVPool(CFG, 128, 4, torch.float32, "cpu")
    with torch.inference_mode():
        result = pd.offline_throughput(
            model, pool, requests=6, prompt_len=5, new_tokens=7, max_batch_size=3
        )
    assert result["requests"] == 6
    assert result["tokens_per_s"] == pytest.approx(6 * 7 / result["elapsed_s"])
    assert result["requests_per_s"] == pytest.approx(6 / result["elapsed_s"])
    assert 0 < result["decode_share"] < 1
    assert 0 < result["admit_share"] < 1
    assert result["decode_share"] + result["admit_share"] + result["other_share"] == pytest.approx(
        1.0
    )
    assert result["mean_decode_batch"] > 1  # requests were decoded together
    assert result["preemptions"] == 0
    # every request got its tokens: prefill makes one, decode steps make the rest
    assert result["decode_steps"] >= 6


def test_standard_requests_are_the_load_generators_requests_for_a_seed() -> None:
    from bench.workload import LengthDist, WorkloadConfig, generate_workload

    specs = pd.standard_requests(seed=3, rate=4.0, duration_s=5.0)
    assert len(specs) == 20
    expected = generate_workload(
        WorkloadConfig(
            rate=4.0,
            num_requests=20,
            prompt=LengthDist(256, 0.7, 1, 2048),
            output=LengthDist(128, 0.5, 1, 512),
            seed=3,
            vocab_size=100000,
        )
    )
    assert specs == [(s.prompt_token_ids, s.max_tokens) for s in expected]
    assert specs == pd.standard_requests(seed=3, rate=4.0, duration_s=5.0)  # deterministic
    assert specs != pd.standard_requests(seed=4, rate=4.0, duration_s=5.0)
    assert all(1 <= len(prompt) <= 2048 and 1 <= new <= 512 for prompt, new in specs)


def test_offline_run_reports_the_cost_per_token_with_and_without_the_prefix_cache() -> None:
    model = make_model(CFG)
    shared = list(range(10, 42))  # eight full blocks of four tokens, common to every prompt
    specs = [([*shared, 60 + i, 70 + i], 6 + i % 3) for i in range(8)]
    with torch.inference_mode():
        results = {
            caching: pd.offline_run(
                model,
                KVPool(CFG, 256, 4, torch.float32, "cpu"),
                specs,
                max_batch_size=4,
                prefix_caching=caching,
            )
            for caching in (False, True)
        }
    total = sum(new for _, new in specs)
    for caching, result in results.items():
        assert result["prefix_caching"] is caching
        assert result["new_tokens_total"] == total
        assert result["prompt_tokens_total"] == sum(len(p) for p, _ in specs)
        assert result["ms_per_token"] == pytest.approx(1000 * result["elapsed_s"] / total)
        assert result["tokens_per_s"] == pytest.approx(total / result["elapsed_s"])
    assert results[False]["prefix_hit_tokens"] == 0
    assert results[True]["prefix_hit_tokens"] > 0  # the shared prefix was served from the cache


def test_render_offline_marks_workload_runs_and_still_renders_fixed_length_rows() -> None:
    workload_row = {
        "max_batch_size": 64,
        "workload_seed": 2,
        "prefix_caching": True,
        "requests_per_s": 20.0,
        "tokens_per_s": 2560.0,
        "ms_per_token": 0.391,
        "decode_share": 0.6,
        "admit_share": 0.3,
        "other_share": 0.1,
        "mean_decode_batch": 40.0,
        "preemptions": 0,
        "block_evictions": 1234,
    }
    fixed_row = {
        k: v
        for k, v in workload_row.items()
        if k not in ("workload_seed", "prefix_caching", "ms_per_token", "block_evictions")
    }
    text = pd.render_offline([workload_row, fixed_row])
    assert "| 64 | 2 | on | 20.0 | 2560 | 0.391 | 60% | 30% | 10% | 40.0 | 0 | 1234 |" in text
    assert "| 64 | fixed | - | 20.0 | 2560 | nan |" in text


def test_render_offline_shows_a_row_per_batch_size() -> None:
    record = {
        "max_batch_size": 64,
        "requests_per_s": 20.0,
        "tokens_per_s": 2560.0,
        "decode_share": 0.6,
        "admit_share": 0.3,
        "other_share": 0.1,
        "mean_decode_batch": 40.0,
        "preemptions": 0,
    }
    assert "| 64 | fixed | - | 20.0 | 2560 | nan | 60% | 30% | 10% | 40.0 | 0 | - |" in (
        pd.render_offline([record])
    )


def test_fill_batch_leaves_every_sequence_running_and_unfinished() -> None:
    model = make_model(CFG)
    pool = KVPool(CFG, 64, 4, torch.float32, "cpu")
    with torch.inference_mode():
        scheduler = pd.fill_batch(model, pool, batch_size=4, prompt_len=6, new_tokens=10)
    assert len(scheduler.running) == 4
    assert all(len(r.generated) == 1 for r in scheduler.running)  # prefill produced one token


def test_fill_batch_reports_a_pool_too_small_for_the_batch() -> None:
    model = make_model(CFG)
    pool = KVPool(CFG, 6, 4, torch.float32, "cpu")  # room for about one sequence
    with torch.inference_mode(), pytest.raises(RuntimeError, match="running"):
        pd.fill_batch(model, pool, batch_size=4, prompt_len=6, new_tokens=10)


def test_render_shows_a_row_per_batch_size() -> None:
    record = {
        "batch_size": 32,
        "wall_ms_per_step_median": 10.0,
        "gpu_busy_ms_per_step": 4.0,
        "gpu_idle_fraction": 0.6,
        "kernels_per_step": 700.0,
        "busy_share_by_category": {"matmul": 0.5, "attention": 0.25, "gather/copy": 0.25},
    }
    text = pd.render([record])
    assert "| 32 | 10.00 | 4.00 | 60% | 700 | 50% | 25% | 25% | 0% |" in text
