"""The saturation report: paired ratios, the engine-thread accounting and the report itself."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest

from bench import saturation_report as sr
from bench.metrics import summarize
from bench.results_io import write_results
from tests.test_metrics import ok_result


def test_paired_ratios_use_only_seeds_present_in_both_and_skip_zero_denominators() -> None:
    ratios = sr.paired_ratios({0: 10.0, 1: 12.0, 2: 9.0}, {0: 5.0, 1: 0.0, 3: 4.0})
    assert ratios == {0: 2.0}


def test_ratio_summary_is_mean_min_max_and_none_when_empty() -> None:
    assert sr.ratio_summary({}) is None
    assert sr.ratio_summary({0: 2.0, 1: 4.0, 2: 3.0}) == (3.0, 2.0, 4.0)


def stats(
    step: float, route: float, inbox: float, idle: float, steps: int, tokens: int
) -> dict[str, Any]:
    return {
        "engine_thread_seconds": {"step": step, "route": route, "inbox": inbox, "idle": idle},
        "decode_steps": steps,
        "tokens_generated": tokens,
        "mean_decode_batch": 40.0,
    }


def test_thread_accounting_is_the_difference_of_two_snapshots() -> None:
    before = stats(1.0, 0.5, 0.1, 2.0, steps=10, tokens=100)
    after = stats(81.0, 4.5, 0.3, 10.0, steps=410, tokens=8100)
    result = sr.thread_accounting(before, after)
    assert result is not None
    assert result["step_s"] == pytest.approx(80.0) and result["idle_s"] == pytest.approx(8.0)
    total = 80.0 + 4.0 + 0.2 + 8.0
    assert result["step_share"] == pytest.approx(80.0 / total)
    assert result["route_share"] == pytest.approx(4.0 / total)
    assert result["ms_per_step"] == pytest.approx(1000 * 80.0 / 400)
    assert result["ms_per_token"] == pytest.approx(1000 * 80.0 / 8000)
    assert result["mean_decode_batch"] == 40.0


def test_thread_accounting_is_none_for_a_server_without_the_timers() -> None:
    assert sr.thread_accounting({"decode_steps": 1}, {"decode_steps": 5}) is None


def test_thread_accounting_with_no_steps_gives_nan_not_an_error() -> None:
    same = stats(1.0, 0.0, 0.0, 1.0, steps=5, tokens=50)
    result = sr.thread_accounting(same, same)
    assert result is not None and math.isnan(result["ms_per_step"])


def write_sweep(directory: Path, batch: int, rate: float, spacing: float, thread: bool) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    name = f"b{batch}-w0-r{rate:g}"
    results = [ok_result(i, sent_s=i * spacing, ttft_s=0.1) for i in range(20)]
    info: dict[str, Any] = {}
    if thread:
        info = {
            "before": {"stats": stats(0.0, 0.0, 0.0, 0.0, steps=0, tokens=0)},
            "after": {"stats": stats(50.0, 2.0, 0.1, 5.0, steps=500, tokens=1000)},
        }
    write_results(directory / f"{name}.jsonl", {"server_info": info}, results, summarize(results))
    entry = {
        "name": name,
        "batch_size": batch,
        "window_ms": 0.0,
        "rate": rate,
        "status": "ok",
        "result_file": f"{name}.jsonl",
        "contended": False,
    }
    with (directory / "manifest.jsonl").open("a") as handle:
        handle.write(json.dumps(entry) + "\n")


def test_the_report_pairs_systems_seed_by_seed_and_shows_the_thread_split(tmp_path: Path) -> None:
    for seed in (0, 1):
        spacing = 0.1 + 0.01 * seed
        write_sweep(tmp_path / f"engine{seed}", 32, 32, spacing, thread=True)
        write_sweep(tmp_path / f"engine{seed}", 64, 32, spacing, thread=True)
        write_sweep(tmp_path / f"hf{seed}", 64, 32, spacing * 2, thread=False)
        write_sweep(tmp_path / f"vllm{seed}", 32, 64, spacing / 2, thread=False)
    out = tmp_path / "report.md"
    code = sr.main(
        [
            "--engine", str(tmp_path / "engine{seed}"),
            "--hf", str(tmp_path / "hf{seed}"),
            "--vllm", str(tmp_path / "vllm{seed}"),
            "--seeds", "0,1", "--out", str(out),
        ]
    )  # fmt: skip
    assert code == 0
    text = out.read_text()
    assert "| engine b64 | 0 |" in text and "| hf-static b64 | 1 |" in text
    assert "engine b64 / hf b64" in text and "vllm / engine b64" in text
    assert "## Engine thread time (D-021)" in text and "| engine b64 | 0 | 88%" in text
    # the engine ran at half the spacing of hf, so about twice its throughput
    ratio_line = next(
        line for line in text.splitlines() if line.startswith("| engine b64 / hf b64 | req/s")
    )
    assert "2.0" in ratio_line or "1.9" in ratio_line


def test_the_report_refuses_to_overwrite(tmp_path: Path) -> None:
    out = tmp_path / "report.md"
    out.write_text("existing")
    with pytest.raises(SystemExit):
        sr.main(["--engine", "x{seed}", "--hf", "y{seed}", "--vllm", "z{seed}", "--out", str(out)])
