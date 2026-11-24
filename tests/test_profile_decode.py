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
