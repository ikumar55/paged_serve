"""The variation report: grouping across sweep directories and the spread statistics."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bench.metrics import summarize
from bench.results_io import write_results
from bench.sweep_variation import Spread, group_runs, main, render, spread
from tests.test_metrics import failed_result, ok_result


def test_spread_statistics() -> None:
    assert spread([]) is None
    s = spread([2.0, 4.0, 3.0])
    assert s == Spread(n=3, mean=3.0, low=2.0, high=4.0)
    assert s is not None and s.relative_range == pytest.approx(2 / 3)


def write_run(directory: Path, name: str, ttft: float, n_failed: int = 0, rate: float = 4) -> None:
    directory.mkdir(exist_ok=True)
    results = [ok_result(i, sent_s=i * 0.5, ttft_s=ttft) for i in range(10)]
    results += [failed_result(100 + i, "timeout", "deadline") for i in range(n_failed)]
    write_results(directory / f"{name}.jsonl", {}, results, summarize(results))
    entry = {
        "name": name,
        "batch_size": 16,
        "window_ms": 20,
        "rate": rate,
        "status": "ok",
        "result_file": f"{name}.jsonl",
        "contended": False,
    }
    with (directory / "manifest.jsonl").open("a") as handle:
        handle.write(json.dumps(entry) + "\n")


def test_groups_runs_across_directories_and_drops_invalid_ones(tmp_path: Path) -> None:
    write_run(tmp_path / "seed0", "b16-w20-r4", ttft=0.2)
    write_run(tmp_path / "seed1", "b16-w20-r4", ttft=0.4)
    write_run(tmp_path / "seed2", "b16-w20-r4", ttft=0.9, n_failed=5)  # 67% succeeded: invalid
    groups = group_runs([tmp_path / "seed0", tmp_path / "seed1", tmp_path / "seed2"])
    assert list(groups) == [(16, 20.0, 4.0)]
    assert len(groups[(16, 20.0, 4.0)]) == 2  # the failing run is excluded, not averaged in


def test_render_reports_mean_and_range_and_skips_single_runs(tmp_path: Path) -> None:
    write_run(tmp_path / "a", "b16-w20-r4", ttft=0.2)
    write_run(tmp_path / "b", "b16-w20-r4", ttft=0.4)
    write_run(tmp_path / "a", "b16-w20-r8", ttft=0.1, rate=8)  # only one run at this rate
    text = render(group_runs([tmp_path / "a", tmp_path / "b"]))
    assert "| 16 | 20 | 4 | 2 |" in text
    assert "300 (200 to 400)" in text  # TTFT p50 mean 0.3 s, range 0.2 to 0.4 s
    assert "| 8 |" not in text


def test_cli_never_overwrites(tmp_path: Path) -> None:
    write_run(tmp_path / "a", "b16-w20-r4", ttft=0.2)
    write_run(tmp_path / "b", "b16-w20-r4", ttft=0.3)
    out = tmp_path / "variation.md"
    assert main([str(tmp_path / "a"), str(tmp_path / "b"), "--out", str(out)]) == 0
    assert out.read_text().startswith("# Run-to-run variation")
    with pytest.raises(FileExistsError):
        main([str(tmp_path / "a"), str(tmp_path / "b"), "--out", str(out)])
