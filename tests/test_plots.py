from pathlib import Path

from bench.metrics import summarize
from bench.plots import plot_run, plot_sweep
from tests.test_metrics import failed_result, ok_result

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def meta(label: str, rate: float) -> dict:  # type: ignore[type-arg]
    return {"label": label, "workload": {"rate": rate}}


def test_plot_run_writes_a_png(tmp_path: Path) -> None:
    results = [ok_result(i, sent_s=i * 0.1, ttft_s=0.05 + i * 0.01) for i in range(20)]
    results.append(failed_result(20, "timeout", "timeout"))
    out = plot_run(meta("demo", 10.0), results, tmp_path / "plots" / "run.png")
    assert out.read_bytes().startswith(PNG_SIGNATURE)


def test_plot_run_handles_a_run_where_everything_failed(tmp_path: Path) -> None:
    results = [failed_result(i, "http_error", "http_503", 503) for i in range(5)]
    out = plot_run(meta("down", 1.0), results, tmp_path / "run.png")
    assert out.read_bytes().startswith(PNG_SIGNATURE)


def test_plot_sweep_groups_runs_by_label(tmp_path: Path) -> None:
    runs = []
    for label in ("ours", "baseline"):
        for rate in (1.0, 2.0, 4.0):
            results = [ok_result(i, sent_s=i / rate, ttft_s=0.05 * rate) for i in range(10)]
            runs.append((meta(label, rate), summarize(results)))
    runs.append((meta("baseline", 8.0), summarize([failed_result(0, "timeout", "timeout")])))
    out = plot_sweep(runs, tmp_path / "sweep.png")
    assert out.read_bytes().startswith(PNG_SIGNATURE)


def test_log_scale_is_used_only_for_wide_ranges() -> None:
    from matplotlib.figure import Figure

    from bench.plots import _log_if_wide

    wide = Figure().subplots()
    _log_if_wide(wide, [10.0, 12.0, 900.0], "x")
    assert wide.get_xscale() == "log"

    narrow = Figure().subplots()
    _log_if_wide(narrow, [20.0, 21.0, 24.0], "y")
    assert narrow.get_yscale() == "linear"

    sparse = Figure().subplots()
    _log_if_wide(sparse, [5.0], "x")
    assert sparse.get_xscale() == "linear"


def test_plot_run_with_a_wide_latency_spread(tmp_path: Path) -> None:
    results = [ok_result(i, sent_s=i * 0.1, itl_s=0.01) for i in range(10)]
    results.append(ok_result(10, sent_s=1.0, itl_s=0.5))  # a stall: gaps span over a decade
    out = plot_run(meta("stall", 10.0), results, tmp_path / "run.png")
    assert out.read_bytes().startswith(PNG_SIGNATURE)
