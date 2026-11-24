"""The bf16 logit-error comparison (D-026), on a tiny CPU model.

The float64 reference stands in for the Triton kernel (it is the in-place function that runs on a
CPU), so what is checked is the tool: the lockstep, the bookkeeping, the criteria, and that a
deliberately broken attention is detected as such.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from bench import bf16_logit_error as ble
from engine.config import LlamaConfig
from engine.model import Llama
from engine.paged_attention import decode_attention_reference
from tests.test_paged_cache import make_model

CFG = LlamaConfig.tiny()
PROMPTS = [[1, 2, 3, 4, 5], [9, 8, 7], [4, 4, 4, 4, 4, 4, 4, 4, 4], [2, 5, 11, 3]]


def bf16_copy(model: Llama) -> Llama:
    copy = Llama(CFG, dtype=torch.bfloat16)
    copy.load_state_dict(model.state_dict())
    return copy.eval()


def run(attention, steps: int = 6):  # type: ignore[no-untyped-def]
    truth_model = make_model(CFG)
    truth, others = ble.build_variants(
        truth_model, bf16_copy(truth_model), attention, num_blocks=200, block_size=4, device="cpu"
    )
    with torch.inference_mode():
        return ble.run_lockstep(truth, others, PROMPTS, steps)


def test_lockstep_records_every_position_of_every_variant() -> None:
    records = run(decode_attention_reference, steps=6)
    assert set(records) == set(ble.VARIANTS)
    for record in records.values():
        assert len(record.max_error) == len(PROMPTS) * 6
        assert sorted(set(record.step)) == [1, 2, 3, 4, 5, 6]
        assert sorted(set(record.prompt)) == [0, 1, 2, 3]
        assert all(margin >= 0 for margin in record.margin)
        assert max(record.margin) > 0.01  # real top-1 minus top-2 gaps, not zeros


def test_an_fp32_variant_with_a_correct_attention_agrees_with_truth() -> None:
    records = run(decode_attention_reference)
    fp32 = ble.summarise(records["fp32-kernel"])
    assert fp32["flips"] == 0 and fp32["error_max"] < 1e-4


def test_bf16_variants_are_measurably_off_but_close() -> None:
    records = run(decode_attention_reference)
    for name in ("bf16-gather", "bf16-kernel"):
        summary = ble.summarise(records[name])
        assert 0 < summary["error_mean"] < 1.0, summary["error_mean"]


def test_a_broken_attention_is_detected() -> None:
    def zeros(q, k_pool, v_pool, tables, lengths):  # type: ignore[no-untyped-def]
        return torch.zeros_like(q)  # attends to nothing

    records = run(zeros)
    broken = ble.summarise(records["fp32-kernel"])
    assert broken["error_mean"] > 1e-2 and broken["flips"] > 0
    # the gather variant does not use the function, so it is unaffected; the bf16 kernel variant is
    # as broken as the fp32 one
    assert ble.summarise(records["bf16-gather"])["error_mean"] < broken["error_mean"]
    assert ble.summarise(records["bf16-kernel"])["error_mean"] > 1e-2


def test_every_variant_is_fed_truths_own_greedy_tokens(monkeypatch: pytest.MonkeyPatch) -> None:
    fed: list[tuple[str, torch.Tensor]] = []
    truth_argmax: list[torch.Tensor] = []
    original = ble.decode_step

    def spy(variant: ble.Variant, tokens: torch.Tensor) -> torch.Tensor:
        logits = original(variant, tokens)
        fed.append((variant.name, tokens.clone()))
        if variant.name == "truth":
            truth_argmax.append(logits.argmax(dim=-1))
        return logits

    monkeypatch.setattr(ble, "decode_step", spy)
    run(decode_attention_reference, steps=4)
    per_step = [fed[i : i + 4] for i in range(0, len(fed), 4)]  # truth, then the three others
    assert len(per_step) == 4
    for step, group in enumerate(per_step):
        assert [name for name, _ in group] == ["truth", *ble.VARIANTS]
        assert all(torch.equal(group[0][1], tokens) for _, tokens in group)  # the same for all
        if step:  # and it is truth's own choice at the step before
            assert torch.equal(group[0][1], truth_argmax[step - 1])


def test_flips_are_split_by_the_truth_margin() -> None:
    record = ble.Records(
        max_error=[0.1, 0.2, 0.3, 0.4],
        flip=[True, True, False, True],
        margin=[0.05, 0.125, 0.01, 0.5],
        prompt=[0, 1, 2, 3],
        step=[1, 1, 1, 1],
    )
    summary = ble.summarise(record)
    assert summary["flips"] == 3
    assert summary["flips_near_tie"] == 1  # margin 0.05; 0.125 is not below the limit
    assert summary["flips_not_near_tie"] == 2
    assert [entry["prompt"] for entry in summary["flip_list"]] == [0, 1, 3]


def summary_of(mean: float, p99: float, not_near: int, flips: int = 0) -> dict[str, float]:
    return {
        "error_mean": mean,
        "error_p99": p99,
        "flips_not_near_tie": not_near,
        "flips": flips,
    }


def test_criteria_boundaries() -> None:
    fp32 = summary_of(1e-6, 1e-5, 0)
    gather = summary_of(0.10, 0.40, 10)

    def verdict(kernel: dict[str, float], fp32_summary: dict[str, float] = fp32) -> dict:  # type: ignore[type-arg]
        return ble.criteria(
            {"fp32-kernel": fp32_summary, "bf16-gather": gather, "bf16-kernel": kernel}
        )

    assert verdict(summary_of(0.10, 0.40, 10))["all_pass"]
    assert verdict(summary_of(0.110, 0.50, 14))["all_pass"]  # exactly at 1.10, 1.25, 1.25*10+2
    assert not verdict(summary_of(0.1101, 0.40, 10))["i_pass"]
    assert not verdict(summary_of(0.10, 0.5001, 10))["ii_pass"]
    assert not verdict(summary_of(0.10, 0.40, 15))["iii_pass"]
    assert not verdict(summary_of(0.10, 0.40, 10), summary_of(1e-6, 1e-5, 0, flips=1))["iv_pass"]
    assert not verdict(summary_of(0.10, 0.40, 10), summary_of(2e-4, 1e-5, 0))["iv_pass"]


def test_report_pools_files_and_keeps_prompt_ids_apart(tmp_path: Path) -> None:
    records = run(decode_attention_reference, steps=3)

    def write(name: str) -> Path:
        path = tmp_path / name
        document = {
            "records": {
                variant: {
                    "max_error": r.max_error,
                    "flip": [int(f) for f in r.flip],
                    "margin": r.margin,
                    "prompt": r.prompt,
                    "step": r.step,
                }
                for variant, r in records.items()
            }
        }
        path.write_text(json.dumps(document), encoding="utf-8")
        return path

    result = ble.report([write("a.json"), write("b.json")])
    assert result["summaries"]["bf16-kernel"]["positions"] == 2 * len(PROMPTS) * 3
    assert result["criteria"]["iv_pass"]
    pooled = ble.combine([records, records])
    assert sorted(set(pooled["bf16-kernel"].prompt)) == [0, 1, 2, 3, 1000, 1001, 1002, 1003]


def test_the_command_line_wants_exactly_one_prompt_source() -> None:
    base = ["--model-dir", "m", "--out", "o", "--label", "x"]
    with pytest.raises(SystemExit):
        ble.parse_args(base)
    with pytest.raises(SystemExit):
        ble.parse_args([*base, "--prompts-from", "p.json", "--shared-prefix-groups", "8"])
    assert ble.parse_args([*base, "--shared-prefix-groups", "8"]).shared_prefix_groups == 8
