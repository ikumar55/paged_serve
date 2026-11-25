"""The D-031 microbenchmark tool on a CPU: cases, bytes, gates, the lever test, the report."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch

from bench import kernel_microbench as kmb
from bench.kernel_microbench import (
    BASELINE,
    Config,
    CpuTimer,
    Gate,
    Implementation,
    Inputs,
    Shape,
    build_inputs,
    copy_bandwidth_gbs,
    copy_gbs,
    lever_test,
    merge_documents,
    minimum_bytes,
    mix_cases,
    prediction_rows,
    read_step_profile,
    registry,
    render_report,
    run_benchmark,
    run_gate,
    uniform_case,
)

TINY = Shape(heads=4, kv_heads=2, head_dim=8, block_size=4, layers=2)


# -- cases and bytes -------------------------------------------------------------------------------


def test_a_uniform_case_attends_over_the_stated_context() -> None:
    case = uniform_case(3, 100)
    assert case.lengths == (99, 99, 99) and case.batch == 3 and case.kind == "uniform"
    assert case.name == "uniform-b3-c100"


def test_the_minimum_bytes_are_every_position_of_every_head_plus_query_and_output() -> None:
    # sequences of 10 and 5 positions; keys and values of 2 heads x 8 dims at 2 bytes; q and out
    expected_kv = (10 + 5) * 2 * 2 * 8 * 2
    expected_qo = 2 * 2 * 4 * 8 * 2
    assert minimum_bytes([9, 4], TINY, 2) == expected_kv + expected_qo == 1216
    assert minimum_bytes([9, 4], TINY, 4) == 2 * 1216  # float32 reads twice the bytes


def test_the_mix_is_the_one_the_profile_draws_and_is_reproducible() -> None:
    from bench.profile_decode import sample_mix_lengths, standard_requests

    cases = mix_cases([0, 1], batch=10)
    assert [c.name for c in cases] == ["mix-s0", "mix-s1"]
    assert list(cases[1].lengths) == sample_mix_lengths(standard_requests(1), 10, 1)
    assert mix_cases([0], batch=10)[0].lengths == cases[0].lengths
    assert cases[0].lengths != cases[1].lengths and all(c.batch == 10 for c in cases)


def test_inputs_have_scrambled_disjoint_blocks_and_the_right_shapes() -> None:
    lengths = [0, 5, 17, 3]
    inputs = build_inputs(lengths, TINY, torch.float32, "cpu")
    assert inputs.q.shape == (4, 4, 1, 8) and inputs.k_pool.shape[1:] == (2, 4, 8)
    assert inputs.lengths.tolist() == lengths and inputs.tables.dtype == torch.long
    need = [-(-(n + 1) // 4) for n in lengths]
    used = [inputs.tables[row, :count].tolist() for row, count in enumerate(need)]
    flat = [block for blocks in used for block in blocks]
    assert len(flat) == len(set(flat))  # no block belongs to two sequences
    assert flat != sorted(flat)  # and the order is scrambled
    assert inputs.k_pool.shape[0] == sum(need) + 1


# -- the registry ----------------------------------------------------------------------------------


def test_every_default_implementation_is_registered_and_names_are_unique() -> None:
    known = registry()
    assert set(kmb.DEFAULT_IMPLS) <= set(known)
    assert len(known) == len({item.name for item in known.values()})
    assert BASELINE in known and "gather-sdpa" in known


def test_the_gpu_implementations_say_why_they_are_unavailable_on_a_cpu() -> None:
    known = registry()
    for name in ("triton-w2", "triton-split4-w2", "flash-attn-paged", "flashinfer"):
        assert known[name].unavailable("cpu") == "needs a CUDA GPU"
    for name in ("gather-sdpa", "sdpa-contiguous", "reference", "split-ref-p2"):
        assert known[name].unavailable("cpu") is None


# -- the correctness gate --------------------------------------------------------------------------


def gate(name: str) -> Gate:
    known = registry()
    return run_gate(known[name], known["gather-sdpa"], TINY, "cpu")


@pytest.mark.parametrize(
    "name", ["gather-sdpa", "sdpa-contiguous", "reference", "split-ref-p2", "split-ref-p3"]
)
def test_the_correct_implementations_pass_the_gate(name: str) -> None:
    result = gate(name)
    assert result.passed, result
    assert result.fp32_error is not None and result.fp32_error < 1e-5
    assert result.bf16_limit is not None and result.bf16_error is not None


def fake(name: str, build: Any) -> Implementation:
    return Implementation(name, "fake", build, lambda device: None)


def wrong_by(offset: float) -> Any:
    def build(inputs: Inputs) -> Any:
        from engine.paged_attention import decode_attention_reference

        return lambda: (
            decode_attention_reference(
                inputs.q, inputs.k_pool, inputs.v_pool, inputs.tables, inputs.lengths
            )
            + offset
        )

    return build


def test_a_slightly_wrong_implementation_fails_the_gate() -> None:
    known = registry()
    result = run_gate(fake("biased", wrong_by(1e-3)), known["gather-sdpa"], TINY, "cpu")
    assert not result.passed and result.fp32_error is not None and result.fp32_error > 1e-5


def test_an_implementation_wrong_only_in_bf16_fails_the_gate() -> None:
    def build(inputs: Inputs) -> Any:
        from engine.paged_attention import decode_attention_reference

        def run() -> Any:
            out = decode_attention_reference(
                inputs.q, inputs.k_pool, inputs.v_pool, inputs.tables, inputs.lengths
            )
            return out + (5.0 if inputs.q.dtype == torch.bfloat16 else 0.0)

        return run

    result = run_gate(fake("bf16-bug", build), registry()["gather-sdpa"], TINY, "cpu")
    assert not result.passed and result.fp32_error is not None and result.fp32_error < 1e-5


def test_an_implementation_that_attends_over_too_few_positions_fails_the_gate() -> None:
    def build(inputs: Inputs) -> Any:
        from engine.paged_attention import decode_attention_reference

        short = torch.clamp(inputs.lengths, max=2)
        return lambda: decode_attention_reference(
            inputs.q, inputs.k_pool, inputs.v_pool, inputs.tables, short
        )

    result = run_gate(fake("truncated", build), registry()["gather-sdpa"], TINY, "cpu")
    assert not result.passed


def test_an_implementation_that_raises_fails_with_the_error_text() -> None:
    def build(inputs: Inputs) -> Any:
        raise TypeError("flash_attn_with_kvcache() got an unexpected keyword argument 'x'")

    result = run_gate(fake("broken-api", build), registry()["gather-sdpa"], TINY, "cpu")
    assert not result.passed and "unexpected keyword" in (result.reason or "")


def test_an_implementation_with_the_wrong_output_shape_fails() -> None:
    def build(inputs: Inputs) -> Any:
        return lambda: inputs.q.squeeze(2)

    result = run_gate(fake("shape", build), registry()["gather-sdpa"], TINY, "cpu")
    assert not result.passed and "shape" in (result.reason or "")


def test_the_bf16_limit_scales_with_the_baselines_own_error() -> None:
    known = registry()
    first = run_gate(known["reference"], known["gather-sdpa"], TINY, "cpu")
    assert first.bf16_limit == pytest.approx(2 * max(first.bf16_limit / 2, 1e-6))
    assert first.bf16_limit >= 2e-6


# -- the roofline ----------------------------------------------------------------------------------


def test_a_copy_counts_the_bytes_read_and_the_bytes_written() -> None:
    assert copy_gbs(1_000_000_000, 1.0) == pytest.approx(2.0)
    assert copy_gbs(500_000_000, 0.25) == pytest.approx(4.0)


def test_the_copy_roofline_is_a_positive_bandwidth() -> None:
    assert copy_bandwidth_gbs("cpu", megabytes=8, repeats=2) > 0


# -- the benchmark on a CPU ------------------------------------------------------------------------


def small_config(**overrides: Any) -> Config:
    fields: dict[str, Any] = {
        "device": "cpu",
        "dtype": torch.float32,
        "impls": ("gather-sdpa", "sdpa-contiguous", "split-ref-p2", "triton-w2"),
        "uniform_batches": (1, 3),
        "uniform_contexts": (8, 20),
        "mix_seeds": (0, 1),
        "mix_batch": 4,
        "repeats": 2,
        "warmup": 0,
        "shape": TINY,
        "copy_megabytes": 8,
    }
    return Config(**{**fields, **overrides})


def test_the_benchmark_times_every_case_of_every_passing_implementation() -> None:
    doc = run_benchmark(small_config(), CpuTimer())
    rows = doc["results"]
    assert len(rows) == (2 * 2 + 2) * 3  # 4 uniform + 2 mix cases, three implementations that run
    assert {r["impl"] for r in rows} == {"gather-sdpa", "sdpa-contiguous", "split-ref-p2"}
    assert doc["unavailable"] == {"triton-w2": "needs a CUDA GPU"}
    assert all(g["passed"] for g in doc["gates"].values())
    row = next(r for r in rows if r["case"] == "uniform-b3-c20" and r["impl"] == "gather-sdpa")
    assert row["batch"] == 3 and row["positions"] == 60 and row["kind"] == "uniform"
    assert row["minimum_bytes"] == minimum_bytes([19] * 3, TINY, 4)
    assert row["cold_gbs"] == pytest.approx(
        row["minimum_bytes"] / (row["cold_p50_us"] * 1e-6) / 1e9
    )
    assert row["fraction_of_copy"] == pytest.approx(row["cold_gbs"] / doc["copy_bandwidth_gbs"])
    assert row["fraction_of_spec"] == pytest.approx(row["cold_gbs"] / kmb.SPEC_BANDWIDTH_GBS)
    assert row["cold_p10_us"] <= row["cold_p50_us"] <= row["cold_p90_us"]


def test_an_implementation_that_fails_its_gate_is_reported_and_never_timed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    known = registry()
    known["biased"] = fake("biased", wrong_by(1e-3))
    monkeypatch.setattr(kmb, "registry", lambda: known)
    doc = run_benchmark(small_config(impls=("gather-sdpa", "biased")), CpuTimer())
    assert doc["gates"]["biased"]["passed"] is False
    assert {r["impl"] for r in doc["results"]} == {"gather-sdpa"}


def test_an_implementation_that_breaks_while_timing_is_recorded_as_failed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def build(inputs: Inputs) -> Any:
        from engine.paged_attention import decode_attention_reference

        def run() -> Any:
            if inputs.lengths.shape[0] == 3:
                raise RuntimeError("out of memory")
            return decode_attention_reference(
                inputs.q, inputs.k_pool, inputs.v_pool, inputs.tables, inputs.lengths
            )

        return run

    known = registry()
    known["fragile"] = fake("fragile", build)
    monkeypatch.setattr(kmb, "registry", lambda: known)
    doc = run_benchmark(small_config(impls=("gather-sdpa", "fragile")), CpuTimer())
    assert doc["gates"]["fragile"]["passed"] is False
    assert "out of memory" in doc["gates"]["fragile"]["reason"]


def test_unknown_implementation_names_are_refused() -> None:
    with pytest.raises(ValueError, match="unknown implementations"):
        run_benchmark(small_config(impls=("nonsense",)), CpuTimer())


# -- the lever test --------------------------------------------------------------------------------


def doc_with(times: dict[str, dict[str, float]]) -> dict[str, Any]:
    rows = [
        {"kind": "mix", "impl": impl, "case": case, "cold_p50_us": us, "fraction_of_copy": 0.1}
        for impl, per_case in times.items()
        for case, us in per_case.items()
    ]
    return {"results": rows}


STEPS = {"mix-s0": 8.0, "mix-s1": 8.0}


def test_the_lever_test_reports_attentions_share_and_a_candidate_variant() -> None:
    doc = doc_with(
        {
            "triton-w2": {"mix-s0": 250.0, "mix-s1": 250.0},
            "triton-split4-w2": {"mix-s0": 150.0, "mix-s1": 150.0},
            "gather-sdpa": {"mix-s0": 10.0, "mix-s1": 10.0},  # faster, but not a candidate
        }
    )
    result = lever_test(doc, STEPS)
    assert result["share_of_step"] == pytest.approx(16 * 0.25 / 8.0)  # 0.5
    assert result["step_speedup_bound"] == pytest.approx(2.0)
    assert result["best_variant"] == "triton-split4-w2"  # the gather path is not a candidate
    assert result["best_speedup_over_current"] == pytest.approx(250 / 150)
    assert result["verdict"].startswith("candidate")


def test_a_small_share_means_attention_is_not_the_lever() -> None:
    doc = doc_with({"triton-w2": {"mix-s0": 50.0, "mix-s1": 50.0}})  # 16 x 0.05 / 8 = 0.1
    assert "NOT the lever" in lever_test(doc, STEPS)["verdict"]


def test_the_share_limit_is_inclusive() -> None:
    us = kmb.LEVER_SHARE_LIMIT * 8.0 / 16 * 1000
    doc = doc_with({"triton-w2": {"mix-s0": us, "mix-s1": us}})
    assert "NOT the lever" in lever_test(doc, STEPS)["verdict"]


def test_without_a_variant_at_least_fifteen_percent_faster_nothing_is_adopted() -> None:
    doc = doc_with(
        {
            "triton-w2": {"mix-s0": 250.0, "mix-s1": 250.0},
            "triton-w4": {"mix-s0": 230.0, "mix-s1": 230.0},  # 1.087 times
        }
    )
    verdict = lever_test(doc, STEPS)["verdict"]
    assert "no variant tried" in verdict and "1.15" in verdict


def test_the_adoption_threshold_is_inclusive() -> None:
    doc = doc_with(
        {
            "triton-w2": {"mix-s0": 230.0, "mix-s1": 230.0},
            "triton-w4": {"mix-s0": 200.0, "mix-s1": 200.0},  # exactly 1.15 times
        }
    )
    assert lever_test(doc, STEPS)["verdict"].startswith("candidate")


def test_a_variant_missing_some_mix_cases_is_not_compared() -> None:
    doc = doc_with(
        {
            "triton-w2": {"mix-s0": 250.0, "mix-s1": 250.0},
            "triton-w4": {"mix-s0": 10.0},  # only one case: its mean would be unfair
        }
    )
    assert lever_test(doc, STEPS)["best_variant"] == "triton-w2"


def test_the_lever_test_says_when_it_cannot_be_evaluated() -> None:
    assert (
        "not evaluable" in lever_test(doc_with({"gather-sdpa": {"mix-s0": 1.0}}), STEPS)["verdict"]
    )
    assert "not evaluable" in lever_test(doc_with({"triton-w2": {"mix-s0": 1.0}}), {})["verdict"]


def test_the_step_times_come_from_the_graph_entries_of_the_profile(tmp_path: Path) -> None:
    mix = [
        {"attention": "triton+graphs", "workload_seed": 0, "wall_ms_per_step_median": 7.9},
        {"attention": "triton", "workload_seed": 0, "wall_ms_per_step_median": 8.8},
        {"attention": "triton+graphs", "workload_seed": 2, "wall_ms_per_step_median": 7.7},
    ]
    path = tmp_path / "profile.json"
    path.write_text(json.dumps({"mix": mix}))
    assert read_step_profile(path) == {"mix-s0": 7.9, "mix-s2": 7.7}


# -- predictions and the report --------------------------------------------------------------------


def full_doc() -> dict[str, Any]:
    cases = {"mix-s0": 1, "mix-s1": 1, "mix-s2": 1}
    times = {
        "triton-w1": 300.0,
        "triton-w2": 250.0,
        "triton-w4": 240.0,
        "triton-w8": 260.0,
        "triton-split4-w2": 150.0,
        "gather-sdpa": 1000.0,
    }
    doc = doc_with({impl: {case: us for case in cases} for impl, us in times.items()})
    for row in doc["results"]:
        row["fraction_of_copy"] = 0.12 if row["impl"] == "triton-w2" else 0.3
    return doc


def test_the_predictions_are_held_or_missed_against_their_written_ranges() -> None:
    steps = {"mix-s0": 7.8, "mix-s1": 7.8, "mix-s2": 7.8}
    got = {
        name: (measured, verdict)
        for name, _, measured, verdict in prediction_rows(full_doc(), steps)
    }
    assert got["(1') current kernel, mix, microseconds per layer"] == ("250", "held")
    assert got["(1') current kernel, share of measured copy bandwidth"][1] == "held"
    assert got["(2) gather path over current kernel, mix"] == ("4.00", "held")
    assert got["(3') best split-KV over current kernel, mix"] == ("1.67", "held")
    assert got["(4) current kernel over best warp count, mix"] == ("1.04", "held")
    assert got["(6') attention's share of the decode step"] == ("0.51", "held")
    assert got["(5) current kernel over FlashAttention, mix"][1] == "not evaluable"


def test_predictions_outside_their_ranges_are_missed() -> None:
    doc = full_doc()
    for row in doc["results"]:
        if row["impl"] == "triton-split4-w2":
            row["cold_p50_us"] = 240.0  # barely faster than the current kernel
    steps = {"mix-s0": 7.8, "mix-s1": 7.8, "mix-s2": 7.8}
    verdicts = {name: v for name, _, _, v in prediction_rows(doc, steps)}
    assert verdicts["(3') best split-KV over current kernel, mix"] == "missed"


def test_the_report_has_every_section_and_names_failures_and_missing_libraries() -> None:
    doc = full_doc()
    doc.update(
        device="cuda",
        dtype="bfloat16",
        copy_bandwidth_gbs=1500.0,
        spec_bandwidth_gbs=1792.0,
        gates={
            "triton-w2": {
                "passed": True,
                "fp32_error": 1e-7,
                "bf16_error": 1e-3,
                "bf16_limit": 3e-3,
                "reason": None,
            },
            "triton-split4-w2": {
                "passed": False,
                "fp32_error": None,
                "bf16_error": None,
                "bf16_limit": None,
                "reason": "CompilationError: bad",
            },
        },
        unavailable={"flashinfer": "flashinfer is not installed in this environment"},
    )
    for row in doc["results"]:
        row.update(
            case=row["case"],
            batch=60,
            cold_p10_us=1.0,
            cold_p90_us=2.0,
            warm_p50_us=1.0,
            cold_gbs=200.0,
            fraction_of_spec=0.1,
        )
    text = render_report(doc, {"mix-s0": 7.8, "mix-s1": 7.8, "mix-s2": 7.8})
    for heading in (
        "## Implementations",
        "## The real length mix",
        "## The lever test",
        "## Predictions",
    ):
        assert heading in text
    assert "FAILED" in text and "CompilationError" in text
    assert "not available" in text and "flashinfer is not installed" in text
    assert "candidate" in text


# -- the command line ------------------------------------------------------------------------------

SMALL_ARGS = [
    "--device", "cpu", "--dtype", "float32", "--impls", "gather-sdpa,triton-w2,split-ref-p2",
    "--repeats", "2", "--warmup", "0", "--seeds", "0", "--mix-batch", "4",
    "--uniform-batches", "1,2", "--uniform-contexts", "8", "--copy-mb", "8",
]  # fmt: skip


def test_the_command_line_writes_the_json_and_the_report_and_never_overwrites(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "bench.json"
    assert kmb.main([*SMALL_ARGS, "--out", str(out)]) == 0
    doc = json.loads(out.read_text())
    assert doc["device"] == "cpu" and "environment" in doc and "package_versions" in doc
    assert doc["unavailable"] == {"triton-w2": "needs a CUDA GPU"}
    assert "lever_test" in doc and "not evaluable" in doc["lever_test"]["verdict"]
    assert (tmp_path / "bench.md").read_text().startswith("# D-031")
    original = out.read_text()
    assert kmb.main([*SMALL_ARGS, "--out", str(out)]) == 2
    assert "never overwritten" in capsys.readouterr().err and out.read_text() == original


def test_asking_for_a_gpu_that_is_not_there_is_an_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert kmb.main(["--device", "cuda", "--out", str(tmp_path / "x.json")]) == 2
    assert "no CUDA GPU" in capsys.readouterr().err and not (tmp_path / "x.json").exists()


# -- merging the vLLM environment's results --------------------------------------------------------


def two_documents() -> tuple[dict[str, Any], dict[str, Any]]:
    main = full_doc()
    main.update(gates={"triton-w2": {"passed": True}}, unavailable={"flashinfer": "not installed"})
    main["copy_bandwidth_gbs"] = 1500.0
    extra = doc_with({"flash-attn-paged": {"mix-s0": 100.0, "mix-s1": 100.0, "mix-s2": 100.0}})
    extra["gates"] = {"flash-attn-paged": {"passed": True, "reason": None}}
    extra["gates"]["triton-w2"] = {"passed": False}
    extra["unavailable"] = {"flashinfer": "installed there"}
    extra["copy_bandwidth_gbs"] = 1490.0
    other = {"kind": "mix", "impl": "triton-w2", "case": "mix-s9", "cold_p50_us": 1.0}
    extra["results"].append({**other, "fraction_of_copy": 1.0})
    return main, extra


def test_results_of_another_environment_are_added_and_labelled_never_overwriting() -> None:
    main, extra = two_documents()
    before = json.dumps(main, sort_keys=True)
    merged = merge_documents(main, extra, "vllm-env")
    assert json.dumps(main, sort_keys=True) == before  # the input is untouched
    added = [r for r in merged["results"] if r.get("environment") == "vllm-env"]
    assert {r["impl"] for r in added} == {"flash-attn-paged"} and len(added) == 3
    assert merged["gates"]["flash-attn-paged"]["passed"] is True
    assert merged["gates"]["flash-attn-paged"]["reason"] == "[vllm-env]"
    assert merged["gates"]["triton-w2"] == {"passed": True}  # the main run's gate stands
    assert merged["unavailable"]["flashinfer"] == "not installed"
    assert merged["merged_from"] == [{"label": "vllm-env", "copy_bandwidth_gbs": 1490.0}]
    steps = {"mix-s0": 7.8, "mix-s1": 7.8, "mix-s2": 7.8}
    verdicts = {name: v for name, _, _, v in prediction_rows(merged, steps)}
    assert verdicts["(5) current kernel over FlashAttention, mix"] == "held"  # 250 / 100 = 2.5


def complete(doc: dict[str, Any]) -> None:
    """Fill the fields a rendered report needs, which the synthetic documents leave out."""
    doc.update(device="cuda", dtype="bfloat16", spec_bandwidth_gbs=1792.0)
    for row in doc["results"]:
        row.update(batch=60, cold_p10_us=1.0, cold_p90_us=2.0, warm_p50_us=1.0, cold_gbs=1.0)
        row["fraction_of_spec"] = 0.1
    for gate in doc["gates"].values():
        for key in ("fp32_error", "bf16_error", "bf16_limit", "reason"):
            gate.setdefault(key, None)


def test_the_report_only_mode_renders_an_earlier_result_with_the_merge(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    main, extra = two_documents()
    complete(main)
    complete(extra)
    first, second, out = tmp_path / "a.json", tmp_path / "vllm.json", tmp_path / "report.md"
    first.write_text(json.dumps(main))
    second.write_text(json.dumps(extra))
    args = ["--report", str(first), "--merge", str(second), "--out", str(out)]
    assert kmb.main(args) == 0
    assert "flash-attn-paged" in out.read_text()
    assert kmb.main(args) == 2 and "never overwritten" in capsys.readouterr().err
