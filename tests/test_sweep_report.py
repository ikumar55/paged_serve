"""The sweep report: reading a sweep directory, validity rules, best/worst, rendering, plotting."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from bench.metrics import summarize
from bench.results_io import write_results
from bench.sweep_report import (
    Row,
    best_and_worst_per_rate,
    load_rows,
    main,
    plot_ttft,
    render_report,
)
from tests.test_metrics import failed_result, ok_result


def make_row(
    batch: int = 8,
    window: float = 20,
    rate: float = 2,
    ttft_p99: float | None = 1.0,
    **overrides: Any,
) -> Row:
    fields: dict[str, Any] = {
        "name": f"b{batch}-w{window:g}-r{rate:g}",
        "batch_size": batch,
        "window_ms": window,
        "rate": rate,
        "status": "ok",
        "contended": False,
        "num_ok": 100,
        "num_requests": 100,
        "throughput_rps": rate,
        "tokens_per_s": 100.0,
        "ttft_p50_s": 0.5,
        "ttft_p99_s": ttft_p99,
    }
    fields.update(overrides)
    return Row(**fields)


def test_validity_and_sustained_rules() -> None:
    assert make_row().valid and make_row().sustained
    assert make_row(contended=True).invalid_reason == "contended"
    assert make_row(status="server_failed_to_start").invalid_reason == "server_failed_to_start"
    assert make_row(generator_saturated=True).invalid_reason == "load generator saturated"
    assert "98%" in (make_row(num_ok=98).invalid_reason or "")
    overloaded = make_row(throughput_rps=1.0, rate=4)  # valid, but served only 25% of the load
    assert overloaded.valid and not overloaded.sustained
    assert make_row(rate=4, throughput_rps=3.7).sustained  # 92.5% of the offered rate


def test_best_and_worst_use_only_valid_runs() -> None:
    rows = [
        make_row(batch=1, rate=2, ttft_p99=3.0),
        make_row(batch=4, rate=2, ttft_p99=0.4),
        make_row(batch=8, rate=2, ttft_p99=9.0, contended=True),  # excluded: contended
        make_row(batch=16, rate=2, ttft_p99=1.5),
        make_row(batch=4, rate=8, ttft_p99=2.0),
    ]
    result = best_and_worst_per_rate(rows)
    best, worst, kept_up = result[2.0]
    assert (best.batch_size, worst.batch_size) == (4, 1)  # the contended 9.0 s run is ignored
    assert kept_up
    only_best, only_worst, _ = result[8.0]
    assert only_best is only_worst  # a single valid run is both


def test_sustained_runs_are_preferred_and_overload_is_called_out() -> None:
    rows = [
        # At rate 4: one run keeps up (slower TTFT), two are overloaded with better p99 TTFT.
        make_row(batch=16, rate=4, ttft_p99=6.0, throughput_rps=3.8),
        make_row(batch=4, rate=4, ttft_p99=1.0, throughput_rps=1.5),
        make_row(batch=8, rate=4, ttft_p99=2.0, throughput_rps=2.5),
        # At rate 8 nothing keeps up.
        make_row(batch=16, rate=8, ttft_p99=60.0, throughput_rps=3.8),
        make_row(batch=4, rate=8, ttft_p99=90.0, throughput_rps=1.5),
    ]
    result = best_and_worst_per_rate(rows)
    best, worst, kept_up = result[4.0]
    assert kept_up and best is worst and best.batch_size == 16  # overloaded runs are not ranked
    best, worst, kept_up = result[8.0]
    assert not kept_up and (best.batch_size, worst.batch_size) == (16, 4)
    report = render_report(rows)
    assert "none sustained (least bad)" in report and "| 4 | sustained runs |" in report


def write_sweep(directory: Path) -> None:
    runs = [
        ("b4-w20-r2", 4, 20, 2, 0.2, 10, 0, False),
        ("b4-w20-r4", 4, 20, 4, 0.9, 8, 2, False),  # two failures: invalid
        ("b8-w20-r2", 8, 20, 2, 0.3, 10, 0, True),  # contended
    ]
    lines = []
    for name, batch, window, rate, ttft, n_ok, n_failed, contended in runs:
        results = [ok_result(i, sent_s=i * 0.5, ttft_s=ttft) for i in range(n_ok)]
        results += [failed_result(100 + i, "timeout", "deadline") for i in range(n_failed)]
        meta = {
            "server_info": {
                "after": {"stats": {"mean_batch_size": 3.2, "wasted_slot_fraction": 0.4}}
            }
        }
        write_results(directory / f"{name}.jsonl", meta, results, summarize(results))
        lines.append(
            {
                "name": name,
                "batch_size": batch,
                "window_ms": window,
                "rate": rate,
                "status": "ok",
                "result_file": f"{name}.jsonl",
                "contended": contended,
            }
        )
    lines.append(
        {
            "name": "b1-w0-r1",
            "batch_size": 1,
            "window_ms": 0,
            "rate": 1,
            "status": "server_failed_to_start",
            "result_file": "b1-w0-r1.jsonl",
            "contended": False,
        }
    )
    (directory / "manifest.jsonl").write_text("".join(json.dumps(x) + "\n" for x in lines))


def test_load_rows_reads_summaries_and_keeps_failed_runs(tmp_path: Path) -> None:
    write_sweep(tmp_path)
    rows = load_rows(tmp_path)
    assert [r.name for r in rows] == ["b1-w0-r1", "b4-w20-r2", "b8-w20-r2", "b4-w20-r4"]
    by_name = {r.name: r for r in rows}
    good = by_name["b4-w20-r2"]
    assert good.valid and good.num_ok == 10 and good.ttft_p50_s == pytest.approx(0.2)
    assert good.mean_batch_size == 3.2 and good.wasted_slot_fraction == 0.4
    assert by_name["b4-w20-r4"].invalid_reason == "80% succeeded"
    assert by_name["b8-w20-r2"].invalid_reason == "contended"
    assert by_name["b1-w0-r1"].invalid_reason == "server_failed_to_start"


def test_report_lists_every_run_and_says_why_some_are_excluded(tmp_path: Path) -> None:
    write_sweep(tmp_path)
    report = render_report(load_rows(tmp_path))
    assert (
        report.count("\n| 4 | 20 |") == 2 and "\n| 8 | 20 |" in report and "\n| 1 | 0 |" in report
    )
    assert "4 runs, 1 valid" in report
    assert "b8-w20-r2 (contended)" in report and "b1-w0-r1 (server_failed_to_start)" in report
    assert "warm-up" in report


def test_cli_writes_report_and_plot_and_never_overwrites(tmp_path: Path) -> None:
    write_sweep(tmp_path)
    out, figure = tmp_path / "REPORT.md", tmp_path / "ttft.png"
    assert main(["--dir", str(tmp_path), "--out", str(out), "--plot", str(figure)]) == 0
    assert out.read_text().startswith("# Static batching sweep")
    assert figure.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    with pytest.raises(FileExistsError):
        main(["--dir", str(tmp_path), "--out", str(out)])


def test_plot_handles_a_single_batch_size(tmp_path: Path) -> None:
    plot_ttft([make_row(rate=1), make_row(rate=2)], tmp_path / "one.png")
    assert (tmp_path / "one.png").stat().st_size > 0
