"""The D-030 report: validity rules, offered rates, paired ratios, wording, predictions."""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

import pytest

from bench.client import RequestResult
from bench.matched_report import (
    Run,
    ceiling_problems,
    generator_check,
    load_all,
    main,
    main_rate,
    paired_ratios,
    parse_directory,
    prediction_rows,
    rate_decisions,
    ratio_wording,
    valid_main,
    window_rates,
)
from bench.metrics import summarize
from bench.results_io import write_results

GOOD_FD = {"soft": 65536, "hard": 65536}


def dist(p99: float) -> dict[str, float]:
    return {"count": 100.0, "mean": 0.0, "p50": 0.0, "p90": 0.0, "p99": p99, "max": p99}


def synthetic(
    offered: float, rps: float, seconds: float = 60.0, ttft: float = 0.05, itl: float = 0.01
) -> list[RequestResult]:
    """Arrivals at ``offered`` req/s; completions spread evenly at ``rps`` req/s (all succeed)."""
    results = []
    for k in range(round(offered * seconds)):
        sent = k / offered
        done = max(k / rps + 0.5, sent + 0.2)
        results.append(
            RequestResult(
                id=k,
                scheduled_s=sent,
                sent_s=sent,
                prompt_tokens=8,
                requested_output_tokens=4,
                status="ok",
                http_status=200,
                output_tokens=4,
                tokens_from_usage=True,
                ttft_s=ttft,
                e2e_s=done - sent,
                token_times_s=[ttft + i * itl for i in range(4)],
            )
        )
    return results


def make_dir(
    root: Path,
    stage: str,
    arm: str,
    seed: int,
    *,
    offered: float = 60.0,
    rps: float = 20.0,
    seconds: float = 60.0,
    tag: str = "d030",
    loop_lags: list[dict[str, float] | None] | None = None,
    results: list[RequestResult] | None = None,
    entry: dict[str, Any] | None = None,
    shard_stats: list[dict[str, Any]] | None = None,
    ttft: float = 0.05,
    itl: float = 0.01,
    infix: str = "",
) -> Path:
    directory = root / f"{tag}-{stage}-{arm}-rtx5090-{infix}seed{seed}"
    directory.mkdir(parents=True)
    runs = results if results is not None else synthetic(offered, rps, seconds, ttft, itl)
    lags = [dist(0.002)] if loop_lags is None else loop_lags
    summary = summarize(runs, lags)
    summary["shard_stats"] = shard_stats
    name = f"b64-w0-r{offered:g}"
    write_results(directory / f"{name}.jsonl", {"workload": {"rate": offered}}, runs, summary)
    manifest = {
        "name": name, "result_file": f"{name}.jsonl", "status": "ok", "contended": False,
        "server_fd_limits": GOOD_FD, "generator_fd_limits": GOOD_FD,
        "started_at_utc": "2026-12-01T10:00:00+00:00",
        "finished_at_utc": "2026-12-01T10:05:00+00:00",
        **(entry or {}),
    }  # fmt: skip
    (directory / "manifest.jsonl").write_text(json.dumps(manifest) + "\n")
    return directory


def load(root: Path, **kwargs: Any) -> list[Run]:
    return load_all(root, "d030", None, waived=True, **kwargs)


def only(runs: list[Run]) -> Run:
    assert len(runs) == 1
    return runs[0]


# -- names and the rate rule -----------------------------------------------------------------------


def test_directory_names_are_parsed_and_foreign_names_are_not() -> None:
    assert parse_directory("d030-main-V64-rtx5090-seed3") == ("main", "V64", 3)
    assert parse_directory("d030-probe-E-rtx5090-seed0") == ("probe", "E", 0)
    assert parse_directory("d030-lat-VE-seed12") == ("lat", "VE", 12)
    for name in (
        "d030-main-X1-rtx5090-seed0",
        "d025-sat-gather-rtx5090-seed0",
        "d030-main-E-rtx5090",
    ):
        assert parse_directory(name) is None


@pytest.mark.parametrize(
    ("delivered", "expected"), [(29.0, 44.0), (28.0, 44.0), (24.0, 36.0), (90.0, 136.0), (1.0, 4.0)]
)
def test_the_main_offered_rate_is_one_and_a_half_times_capacity_rounded_up_to_four(
    delivered: float, expected: float
) -> None:
    assert main_rate(delivered) == expected


# -- one run, and each reason it can be invalid ----------------------------------------------------


def test_a_good_run_is_read_and_valid(tmp_path: Path) -> None:
    make_dir(tmp_path, "main", "E", 0, offered=60, rps=20)
    run = only(load(tmp_path))
    assert (run.stage, run.arm, run.seed, run.offered) == ("main", "E", 0, 60.0)
    assert run.window_rps == pytest.approx(20.0, rel=0.02)
    assert run.whole_rps is not None and run.ok == run.sent == 3600
    assert run.valid, run.reasons
    assert run.window_tokens_per_s == pytest.approx(80.0, rel=0.02)
    assert run.ttft_p50_s == pytest.approx(0.05) and run.itl_p50_s == pytest.approx(0.01)


def reasons_of(tmp_path: Path, **kwargs: Any) -> str:
    stage = kwargs.pop("stage", "main")
    make_dir(tmp_path, stage, "E", 0, **kwargs)
    return " | ".join(only(load(tmp_path)).reasons)


def test_a_contended_run_or_one_that_did_not_finish_is_not_valid(tmp_path: Path) -> None:
    assert "contended" in reasons_of(tmp_path / "a", entry={"contended": True})
    assert "did not finish ok" in reasons_of(tmp_path / "b", entry={"status": "loadgen_failed"})


def test_a_generator_that_sent_late_or_read_late_invalidates_the_run(tmp_path: Path) -> None:
    late = synthetic(60, 20)
    for r in late[::10]:  # one request in ten was sent 0.3 s late
        r.sent_s += 0.3
    assert "send lag" in reasons_of(tmp_path / "a", results=late)
    assert "event-loop lag p99" in reasons_of(tmp_path / "b", loop_lags=[dist(0.001), dist(0.04)])
    assert "event-loop lag not recorded" in reasons_of(tmp_path / "c", loop_lags=[None])
    shards = [
        {"shard": 0, "requests": 10, "send_lag_p99_s": 0.001, "loop_lag_p99_s": 0.001},
        {"shard": 1, "requests": 10, "send_lag_p99_s": 0.2, "loop_lag_p99_s": 0.001},
    ]
    assert "shard 1 send lag" in reasons_of(tmp_path / "d", shard_stats=shards)


def test_a_run_near_the_open_file_limit_or_with_unknown_limits_is_not_valid(tmp_path: Path) -> None:
    # offered 60, delivered 20: about 2,400 in flight at the end of the arrivals
    assert "ceiling-limited" in reasons_of(
        tmp_path / "a", entry={"server_fd_limits": {"soft": 2000, "hard": 2000}}
    )
    assert "not recorded" in reasons_of(tmp_path / "b", entry={"server_fd_limits": None})
    assert "not recorded" in reasons_of(tmp_path / "c", entry={"generator_fd_limits": None})
    unlimited = {"soft": "unlimited", "hard": "unlimited"}
    assert (
        reasons_of(
            tmp_path / "d", entry={"server_fd_limits": unlimited, "generator_fd_limits": unlimited}
        )
        == ""
    )


def test_a_capacity_run_must_be_offered_well_above_what_was_delivered(tmp_path: Path) -> None:
    assert reasons_of(tmp_path / "a", offered=30, rps=20) == ""  # 1.5 times: fine
    assert "not saturated" in reasons_of(tmp_path / "b", offered=27, rps=20)  # under 1.4 times
    # the probe stage has no such rule: it exists to find out
    assert reasons_of(tmp_path / "c", stage="probe", offered=27, rps=20) == ""


def test_a_latency_run_must_have_sustained_its_load(tmp_path: Path) -> None:
    assert reasons_of(tmp_path / "a", stage="lat", offered=8, rps=8) == ""
    assert "not sustained" in reasons_of(tmp_path / "b", stage="lat", offered=8, rps=5)
    dropped = synthetic(8, 8)
    dropped[3].status, dropped[3].error = "timeout", "timeout"
    assert "not sustained" in reasons_of(tmp_path / "c", stage="lat", offered=8, results=dropped)


def test_a_run_whose_arrivals_end_before_the_window_is_not_valid(tmp_path: Path) -> None:
    assert "no steady-state window" in reasons_of(tmp_path / "a", seconds=40.0)


def test_the_machine_audit_is_required_unless_waived(tmp_path: Path) -> None:
    directory = make_dir(tmp_path, "main", "E", 0)
    name = directory.name
    assert "no machine audit given" in only(load_all(tmp_path, "d030", None)).reasons
    assert only(load_all(tmp_path, "d030", {(name, "b64-w0-r60"): ()})).valid
    flagged = only(load_all(tmp_path, "d030", {(name, "b64-w0-r60"): ("other-cpu", "slowdown")}))
    assert "machine audit: other-cpu, slowdown" in flagged.reasons
    assert "no machine audit for this run" in only(load_all(tmp_path, "d030", {})).reasons


def test_the_audit_of_a_real_monitor_file_flags_a_busy_machine(tmp_path: Path) -> None:
    from bench.matched_report import audit_map

    directory = make_dir(tmp_path, "main", "E", 0)
    samples = [
        {
            "t_utc": f"2026-12-01T10:0{minute}:00+00:00",
            "cpu_cores_other_users": cpu,
            "gpu_apps_other": 0,
        }
        for minute in range(5)
        for cpu in (4.0,)
    ]
    quiet = tmp_path / "quiet.jsonl"
    busy = tmp_path / "busy.jsonl"
    quiet.write_text("\n".join(json.dumps({**s, "cpu_cores_other_users": 0.0}) for s in samples))
    busy.write_text("\n".join(json.dumps(s) for s in samples))
    key = (directory.name, "b64-w0-r60")
    assert audit_map(tmp_path, "d030", [quiet])[key] == ()
    assert "other-cpu" in audit_map(tmp_path, "d030", [busy])[key]


# -- the offered rates from the probes -------------------------------------------------------------


def test_each_arm_main_rate_comes_from_its_own_probe(tmp_path: Path) -> None:
    make_dir(tmp_path, "probe", "E", 0, offered=64, rps=29)
    make_dir(tmp_path, "probe", "V64", 0, offered=192, rps=90)
    decisions = rate_decisions(load(tmp_path))
    assert decisions["E"]["verdict"] == "ok" and decisions["E"]["main_rate"] == 44.0
    assert decisions["V64"]["main_rate"] == 136.0
    assert decisions["G"]["verdict"] == "no probe yet"


def test_a_probe_that_was_not_saturating_is_repeated_at_twice_the_rate(tmp_path: Path) -> None:
    make_dir(tmp_path, "probe", "E", 0, offered=64, rps=44)  # 64 / 1.5 = 42.7 < 44
    (decision,) = [d for a, d in rate_decisions(load(tmp_path)).items() if a == "E"]
    assert "not saturating" in decision["verdict"] and decision["repeat_at"] == 128.0
    assert "main_rate" not in decision


def test_the_saturation_boundary_is_offered_over_one_and_a_half(tmp_path: Path) -> None:
    make_dir(tmp_path, "probe", "E", 0, offered=60, rps=40)  # exactly 60 / 1.5: still saturating
    assert rate_decisions(load(tmp_path))["E"]["verdict"] == "ok"


def test_a_probe_that_is_invalid_is_repeated_not_used(tmp_path: Path) -> None:
    make_dir(tmp_path, "probe", "E", 0, offered=64, rps=29, loop_lags=[dist(0.2)])
    decision = rate_decisions(load(tmp_path))["E"]
    assert "invalid" in decision["verdict"] and "main_rate" not in decision


# -- ratios and the wording rule -------------------------------------------------------------------


def runs_for(tmp_path: Path, arm: str, rates: list[float]) -> dict[int, Run]:
    for seed, rps in enumerate(rates):
        make_dir(tmp_path, "main", arm, seed, offered=round(rps * 1.6), rps=rps)
    return {r.seed: r for r in load(tmp_path) if r.arm == arm and r.valid}


def test_ratios_are_paired_by_seed_and_use_only_seeds_valid_in_both(tmp_path: Path) -> None:
    top = runs_for(tmp_path, "V64", [90.0, 80.0, 100.0, 50.0])
    bottom = runs_for(tmp_path, "E", [30.0, 40.0, 25.0])
    ratios = paired_ratios(top, bottom)
    assert len(ratios) == 3  # seed 3 has no engine run
    assert ratios == pytest.approx([3.0, 2.0, 4.0], rel=0.03)


@pytest.mark.parametrize(
    ("ratios", "text"),
    [
        ([1.0, 1.25], "within 25%"),
        ([1.0, 1.26], "within a factor of two"),
        ([1.3, 2.0], "within a factor of two"),
        ([1.9, 2.01], "more than twice"),
        ([2.9, 3.1, 3.0], "more than twice"),
        ([], "no ratio"),
    ],
)
def test_the_wording_rule_is_judged_on_the_worst_seed(ratios: list[float], text: str) -> None:
    assert text in ratio_wording(ratios)


# -- predictions -----------------------------------------------------------------------------------


def verdicts(runs: list[Run], with_d029: bool = False) -> dict[str, str]:
    return {name: verdict for name, _, _, verdict in prediction_rows(runs, with_d029)}


def full_session(tmp_path: Path, **rates: list[float]) -> list[Run]:
    for arm, values in rates.items():
        for seed, rps in enumerate(values):
            make_dir(tmp_path, "main", arm, seed, offered=round(rps * 1.6), rps=rps)
    return load(tmp_path)


def test_predictions_are_held_or_missed_against_the_written_ranges(tmp_path: Path) -> None:
    runs = full_session(
        tmp_path, V64=[90.0] * 4, V32=[60.0] * 4, VE=[50.0] * 4, E=[30.0] * 4, G=[15.0] * 4
    )
    got = verdicts(runs)
    assert got["(1) vLLM 64 sequences, req/s"] == "held"
    assert got["(2) engine, kernel and graphs, req/s"] == "held"
    assert got["(3) engine, gather path, req/s"] == "held"
    assert got["(4) vLLM 64 over engine"] == "held"  # 90 / 30 = 3
    assert got["(4) engine over gather"] == "held"  # 30 / 15 = 2
    assert got["(5) engine inter-token p50 at 8 req/s, ms"] == "not evaluable"  # no lat runs
    assert got["(6) vLLM runs flagged generator-saturated"] == "held"


def test_a_figure_outside_its_range_is_missed_and_too_few_seeds_are_not_evaluable(
    tmp_path: Path,
) -> None:
    runs = full_session(tmp_path, V64=[70.0] * 4, E=[30.0, 30.0])
    got = verdicts(runs)
    assert got["(1) vLLM 64 sequences, req/s"] == "missed"
    assert got["(2) engine, kernel and graphs, req/s"] == "not evaluable"  # two seeds only


def test_what_d029_changes_is_the_engine_range_and_the_ratio_range(tmp_path: Path) -> None:
    runs = full_session(tmp_path, V64=[90.0] * 4, E=[36.0] * 4)
    assert verdicts(runs, with_d029=False)["(2) engine, kernel and graphs, req/s"] == "missed"
    assert verdicts(runs, with_d029=True)["(2) engine, kernel and graphs, req/s"] == "held"
    assert verdicts(runs, with_d029=True)["(4) vLLM 64 over engine"] == "held"  # 2.5 in 2 to 3
    assert verdicts(runs, with_d029=False)["(4) vLLM 64 over engine"] == "held"  # and in 2.5 to 3.5


def test_latency_predictions_use_the_latency_stage(tmp_path: Path) -> None:
    for seed in range(3):
        make_dir(tmp_path, "lat", "E", seed, offered=8, rps=8, ttft=0.09, itl=0.010)
        make_dir(tmp_path, "lat", "V64", seed, offered=8, rps=8, ttft=0.04, itl=0.003)
    got = verdicts(load(tmp_path))
    assert got["(5) engine inter-token p50 at 8 req/s, ms"] == "held"
    assert got["(5) vLLM inter-token p50 at 8 req/s, ms"] == "held"
    assert got["(5) engine over vLLM first-token p50"] == "held"  # 2.25 <= 3


def test_a_flagged_vllm_run_misses_prediction_six(tmp_path: Path) -> None:
    make_dir(tmp_path, "main", "V64", 0, offered=150, rps=90, loop_lags=[dist(0.2)])
    assert verdicts(load(tmp_path))["(6) vLLM runs flagged generator-saturated"] == "missed"


# -- the command line ------------------------------------------------------------------------------


def test_the_report_is_written_with_every_section(tmp_path: Path) -> None:
    results = tmp_path / "results"
    results.mkdir()
    for seed in range(3):
        make_dir(results, "main", "E", seed, offered=48, rps=30)
        make_dir(results, "main", "V64", seed, offered=144, rps=90)
    out = tmp_path / "report.md"
    assert main(["report", "--results", str(results), "--out", str(out), "--waive-audit"]) == 0
    text = out.read_text()
    for heading in (
        "## Every run",
        "## Capacity",
        "## Paired ratios",
        "## Latency",
        "## Predictions",
    ):
        assert heading in text
    assert "THE MACHINE AUDIT WAS WAIVED" in text
    assert "more than twice behind vLLM" in text  # 90 / 30 = 3
    assert "unresolved" in text  # arms G, V32 and VE have no runs


def test_the_rates_command_exits_nonzero_until_every_arm_has_a_good_probe(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    make_dir(tmp_path, "probe", "E", 0, offered=64, rps=29)
    assert main(["rates", "--results", str(tmp_path), "--waive-audit"]) == 1
    printed = capsys.readouterr().out
    assert "E: ok: main offered rate 44 req/s" in printed and "G: no probe yet" in printed


# -- step 0: the generator against the null server -------------------------------------------


def write_step0(path: Path, loop: float, send: float, sharded: bool = True) -> None:
    results = synthetic(40, 40, seconds=3.0)
    summary = summarize(results, [dist(loop), dist(loop)])
    summary["shard_stats"] = (
        [
            {"shard": i, "requests": 60, "send_lag_p99_s": send, "loop_lag_p99_s": loop}
            for i in range(2)
        ]
        if sharded
        else None
    )
    write_results(path, {"workload": {"rate": 40.0}}, results, summary)


def test_step_zero_passes_only_when_every_shard_keeps_up(tmp_path: Path) -> None:
    path = tmp_path / "ok.jsonl"
    write_step0(path, loop=0.005, send=0.01)
    assert generator_check(path) == (True, [])
    write_step0(tmp_path / "loop.jsonl", loop=0.05, send=0.01)
    ok, problems = generator_check(tmp_path / "loop.jsonl")
    assert not ok and any("event-loop" in p for p in problems)
    write_step0(tmp_path / "send.jsonl", loop=0.005, send=0.2)
    ok, problems = generator_check(tmp_path / "send.jsonl")
    assert not ok and any("send lag" in p for p in problems)


def test_step_zero_needs_a_sharded_run(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = tmp_path / "single.jsonl"
    write_step0(path, loop=0.005, send=0.01, sharded=False)
    ok, problems = generator_check(path)
    assert not ok and any("not sharded" in p for p in problems)
    assert main(["generator", str(path)]) == 1
    assert "FAIL" in capsys.readouterr().out


def test_the_open_file_ceiling_is_eighty_percent_of_the_smaller_limit() -> None:
    entry = {"server_fd_limits": {"soft": 1024, "hard": 1024}, "generator_fd_limits": GOOD_FD}
    assert ceiling_problems(entry, 819) == []  # 0.8 x 1,024 = 819.2
    assert "ceiling-limited" in ceiling_problems(entry, 820)[0]
    swapped = {"server_fd_limits": GOOD_FD, "generator_fd_limits": {"soft": 1024, "hard": 1024}}
    assert "ceiling-limited" in ceiling_problems(swapped, 900)[0]  # the generator's counts too


@pytest.mark.parametrize("infix", ["repeat-", "zrepeat-"])  # sorts before and after the first probe
def test_a_repeated_probe_replaces_the_first_one(tmp_path: Path, infix: str) -> None:
    make_dir(tmp_path, "probe", "E", 0, offered=64, rps=44)  # not saturating
    make_dir(tmp_path, "probe", "E", 0, offered=128, rps=44, infix=infix)
    runs = load(tmp_path)
    assert len(runs) == 2 and {r.offered for r in runs} == {64.0, 128.0}
    decision = rate_decisions(runs)["E"]
    assert decision["verdict"] == "ok" and decision["main_rate"] == 68.0  # 1.5 x 44 = 66 -> 68


def test_an_invalid_run_is_left_out_of_the_capacity_and_the_ratios(tmp_path: Path) -> None:
    for seed in range(3):
        make_dir(tmp_path, "main", "E", seed, offered=48, rps=30)
    make_dir(tmp_path, "main", "E", 3, offered=48, rps=10, loop_lags=[dist(0.2)])  # slow, invalid
    runs = load(tmp_path)
    valid = valid_main(runs, "E")
    assert sorted(valid) == [0, 1, 2]
    assert statistics.fmean(window_rates(runs, "E")) == pytest.approx(30.0, rel=0.02)
