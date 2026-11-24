"""The decode profiler's arithmetic and categories, plus an end-to-end run on a tiny CPU model."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

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


def test_offline_run_leaves_nothing_alive_for_the_garbage_collector(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A reference cycle kept every run's KV pool alive; the fifth GPU run ran out of memory."""
    import gc
    import weakref

    created: list[weakref.ref[pd.Scheduler]] = []

    class Tracked(pd.Scheduler):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            created.append(weakref.ref(self))

    monkeypatch.setattr(pd, "Scheduler", Tracked)
    model = make_model(CFG)
    gc.collect()
    gc.disable()  # only reference counting may free the scheduler
    try:
        with torch.inference_mode():
            pd.offline_run(
                model,
                KVPool(CFG, 64, 4, torch.float32, "cpu"),
                [([1, 2, 3], 4), ([4, 5, 6, 7], 3)],
                max_batch_size=2,
                prefix_caching=True,
            )
        assert created and created[0]() is None
    finally:
        gc.enable()


def test_sample_mix_lengths_are_prompt_plus_progress_and_deterministic() -> None:
    specs = [([1] * (10 * (i + 1)), 5 + i) for i in range(12)]  # prompt 10..120, new 5..16
    lengths = pd.sample_mix_lengths(specs, batch_size=6, seed=4)
    assert lengths == pd.sample_mix_lengths(specs, batch_size=6, seed=4)
    assert lengths != pd.sample_mix_lengths(specs, batch_size=6, seed=5)
    assert len(lengths) == 6
    for length in lengths:
        assert any(len(p) <= length < len(p) + new for p, new in specs)  # inside some request
    assert len(pd.sample_mix_lengths(specs, batch_size=12, seed=0)) == 12  # the whole list
    with pytest.raises(ValueError):
        pd.sample_mix_lengths(specs, batch_size=13, seed=0)  # cannot draw more than there are


def test_padding_statistics_count_padded_positions_against_useful_ones() -> None:
    stats = pd.padding_statistics([10, 20, 30], block_size=16, extra_tokens=2)
    assert stats["context_mean"] == pytest.approx(22.0)
    assert stats["context_max"] == 32.0
    assert stats["padded_width"] == 32.0  # 32 tokens is exactly two blocks
    assert stats["padding_ratio"] == pytest.approx(3 * 32 / 66)
    assert pd.padding_statistics([16, 16], block_size=16)["padding_ratio"] == pytest.approx(1.0)
    assert pd.padding_statistics([17, 1], block_size=16)["padded_width"] == 32.0  # a partial block


def test_fill_batch_takes_a_list_of_lengths_and_sets_the_batch_size_from_it() -> None:
    model = make_model(CFG)
    pool = KVPool(CFG, 128, 4, torch.float32, "cpu")
    with torch.inference_mode():
        scheduler = pd.fill_batch(model, pool, batch_size=99, prompt_len=[3, 9, 6], new_tokens=8)
    assert len(scheduler.running) == 3
    assert sorted(len(r.prompt) for r in scheduler.running) == [3, 6, 9]  # the lengths given
    assert all(len(r.generated) == 1 for r in scheduler.running)  # prefilled, none finished


def test_profile_batch_on_a_mix_reports_the_padding_it_does() -> None:
    model = make_model(CFG)
    pool = KVPool(CFG, 256, 4, torch.float32, "cpu")
    with torch.inference_mode():
        result = pd.profile_batch(
            model, pool, batch_size=0, prompt_len=[3, 12, 7], steps=2, warmup=1
        )
    assert result["batch_size"] == 3 and result["prompt_len"] == "mix"
    mix = result["context_mix"]
    assert mix["padding_ratio"] >= 1.0 and mix["context_max"] == 12 + 1 + 2
    assert result["context_len_at_end"] is None


def test_main_profiles_a_mix_per_seed_and_records_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import engine.weights

    monkeypatch.setattr(engine.weights, "load_model", lambda *a, **k: (make_model(CFG), []))
    monkeypatch.setattr(
        pd, "standard_requests", lambda seed: [([1, 2, 3, 4, 5], 6), ([5, 6, 7], 4), ([8] * 7, 5)]
    )
    out = tmp_path / "mix.json"
    code = pd.main(
        [
            "--model-dir", "unused", "--device", "cpu", "--dtype", "float32",
            "--kv-cache-gb", "0.0005", "--block-size", "4", "--skip-profile",
            "--mix-seeds", "0,1", "--mix-batch", "2", "--steps", "2", "--out", str(out),
        ]
    )  # fmt: skip
    assert code == 0
    record = json.loads(out.read_text())
    assert [m["workload_seed"] for m in record["mix"]] == [0, 1]
    assert all(m["batch_size"] == 2 and "padding_ratio" in m["context_mix"] for m in record["mix"])
    assert not (tmp_path / "mix.json.partial.jsonl").exists()


def test_release_gpu_memory_is_harmless_on_the_cpu() -> None:
    pd.release_gpu_memory("cpu")
    pd.release_gpu_memory(torch.device("cpu"))


def test_write_record_never_overwrites(tmp_path: Path) -> None:
    path = tmp_path / "r.json"
    pd.write_record(path, {"a": 1})
    assert json.loads(path.read_text()) == {"a": 1}
    with pytest.raises(FileExistsError):
        pd.write_record(path, {"a": 2})


def run_main_with_a_tiny_model(
    monkeypatch: pytest.MonkeyPatch, out: Path, offline_run: Any = None
) -> int:
    import engine.weights

    monkeypatch.setattr(engine.weights, "load_model", lambda *a, **k: (make_model(CFG), []))
    monkeypatch.setattr(
        pd, "standard_requests", lambda seed: [([1 + seed, 2, 3, 4], 4), ([5, 6, 7], 3)]
    )
    if offline_run is not None:
        monkeypatch.setattr(pd, "offline_run", offline_run)
    return pd.main(
        [
            "--model-dir", "unused", "--device", "cpu", "--dtype", "float32",
            "--kv-cache-gb", "0.0005", "--block-size", "4", "--batch-sizes", "3",
            "--skip-profile", "--offline-workload-seeds", "0,1", "--out", str(out),
        ]
    )  # fmt: skip


def test_main_runs_each_seed_with_the_cache_off_then_on_and_removes_the_partial_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    out = tmp_path / "offline.json"
    assert run_main_with_a_tiny_model(monkeypatch, out) == 0
    record = json.loads(out.read_text())
    runs = record["offline"]
    assert [(r["workload_seed"], r["prefix_caching"]) for r in runs] == [
        (0, False), (0, True), (1, False), (1, True),
    ]  # fmt: skip
    assert not (tmp_path / "offline.json.partial.jsonl").exists()


def test_a_crash_partway_keeps_the_runs_that_finished(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    real = pd.offline_run
    calls = {"n": 0}

    def flaky(*args: Any, **kwargs: Any) -> dict[str, Any]:
        calls["n"] += 1
        if calls["n"] == 3:
            raise RuntimeError("CUDA out of memory (simulated)")
        return real(*args, **kwargs)

    out = tmp_path / "offline.json"
    with pytest.raises(RuntimeError, match="simulated"):
        run_main_with_a_tiny_model(monkeypatch, out, offline_run=flaky)
    assert not out.exists()  # the final record was never written
    lines = (tmp_path / "offline.json.partial.jsonl").read_text().splitlines()
    kept = [json.loads(line) for line in lines]
    assert [(k["workload_seed"], k["prefix_caching"]) for k in kept] == [(0, False), (0, True)]
    assert all(k["kind"] == "offline" for k in kept)


def test_main_refuses_to_start_over_an_existing_result_or_partial_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    out = tmp_path / "offline.json"
    out.write_text("{}")
    with pytest.raises(SystemExit):
        run_main_with_a_tiny_model(monkeypatch, out)


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
