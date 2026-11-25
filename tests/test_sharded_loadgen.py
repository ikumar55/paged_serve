"""The sharded load generator and the null server (D-030). Real processes and real sockets."""

from __future__ import annotations

import asyncio
import json
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import httpx
import pytest

from bench import null_server
from bench.client import RequestResult
from bench.loadgen import main
from bench.results_io import read_results
from bench.sharded import (
    ShardOutcome,
    merge,
    monotonic_start,
    run_sharded,
    shard_of,
    shard_stats,
)
from bench.workload import LengthDist, WorkloadConfig, generate_workload

LEAD_S = 8.0  # worker processes are spawned (not forked) here, and importing takes a few seconds
SMALL = [
    "--prompt-median", "8", "--prompt-sigma", "0", "--prompt-max", "16",
    "--output-median", "4", "--output-sigma", "0", "--output-max", "8",
]  # fmt: skip


@contextmanager
def null_server_url() -> Iterator[str]:
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    server = asyncio.run_coroutine_threadsafe(null_server.start(0), loop).result(timeout=5)
    try:
        yield f"http://127.0.0.1:{server.sockets[0].getsockname()[1]}"
    finally:

        async def stop() -> None:
            server.close()
            await server.wait_closed()

        asyncio.run_coroutine_threadsafe(stop(), loop).result(timeout=5)
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=5)


def workload(count: int = 12, rate: float = 30.0) -> WorkloadConfig:
    return WorkloadConfig(
        rate=rate,
        num_requests=count,
        prompt=LengthDist(8, 0, 1, 16),
        output=LengthDist(4, 0, 1, 8),
        seed=3,
        vocab_size=5000,
    )


def fake(id: int, sent: float = 0.0, scheduled: float = 0.0) -> RequestResult:
    return RequestResult(
        id=id, scheduled_s=scheduled, sent_s=sent, prompt_tokens=1, requested_output_tokens=1
    )


# -- the pieces ------------------------------------------------------------------------------------


def test_requests_are_dealt_to_shards_by_id_modulo_the_shard_count() -> None:
    assert [shard_of(i, 3) for i in range(7)] == [0, 1, 2, 0, 1, 2, 0]
    assert all(shard_of(i, 1) == 0 for i in range(5))


def test_the_wall_clock_start_is_converted_to_the_processes_own_monotonic_clock() -> None:
    assert monotonic_start(start_wall=1005.0, now_wall=1000.0, now_monotonic=50.0) == 55.0
    assert monotonic_start(start_wall=999.0, now_wall=1000.0, now_monotonic=50.0) == 49.0


def test_merging_orders_by_id_and_demands_every_request_exactly_once() -> None:
    a = ShardOutcome(0, [fake(2), fake(0)], None)
    b = ShardOutcome(1, [fake(1), fake(3)], None)
    assert [r.id for r in merge([b, a], 4)] == [0, 1, 2, 3]
    with pytest.raises(RuntimeError, match="exactly once"):
        merge([a], 4)  # shard b's results are missing
    with pytest.raises(RuntimeError, match="exactly once"):
        merge([a, ShardOutcome(1, [fake(1), fake(2)], None)], 3)  # id 2 twice, no id 3 needed
    with pytest.raises(RuntimeError, match="exactly once"):
        merge([a, b], 5)  # one more request was expected


def test_the_per_process_statistics_report_each_shards_own_lags() -> None:
    slow = ShardOutcome(1, [fake(1, sent=0.30, scheduled=0.0)], {"p99": 0.04})
    quick = ShardOutcome(0, [fake(0, sent=0.01, scheduled=0.0)], None)
    stats = shard_stats([slow, quick])
    assert [s["shard"] for s in stats] == [0, 1]
    assert stats[0] == {"shard": 0, "requests": 1, "send_lag_p99_s": 0.01, "loop_lag_p99_s": None}
    assert stats[1]["send_lag_p99_s"] == pytest.approx(0.30) and stats[1]["loop_lag_p99_s"] == 0.04


def test_a_run_needs_at_least_one_shard() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        run_sharded("http://x", "m", workload(), 0, timeout_s=5, ignore_eos=True)


# -- the null server -------------------------------------------------------------------------------


def test_the_null_server_streams_exactly_the_requested_tokens() -> None:
    with null_server_url() as url:
        with httpx.Client(base_url=url, timeout=10) as client:
            response = client.post("/v1/completions", json={"max_tokens": 5})
        events = [line[6:] for line in response.text.splitlines() if line.startswith("data: ")]
    assert response.status_code == 200 and events[-1] == "[DONE]"
    parsed = [json.loads(e) for e in events[:-1]]
    assert sum(1 for p in parsed if p["choices"] and p["choices"][0]["text"] == "x") == 5
    assert parsed[-1]["usage"]["completion_tokens"] == 5


@pytest.mark.parametrize("tokens", [0, -3, null_server.MAX_TOKENS_LIMIT + 1])
def test_the_null_server_refuses_a_request_it_should_not_obey(tokens: int) -> None:
    with null_server_url() as url, httpx.Client(base_url=url, timeout=10) as client:
        assert client.post("/v1/completions", json={"max_tokens": tokens}).status_code == 400


def test_several_null_server_workers_need_a_platform_that_can_share_a_port(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert null_server.main(["--port", "1", "--workers", "0"]) == 2
    assert "at least 1" in capsys.readouterr().err


# -- real shards -----------------------------------------------------------------------------------


def test_shards_send_the_exact_requests_of_the_single_process_at_the_same_arrival_times() -> None:
    cfg = workload(count=12)
    with null_server_url() as url:
        outcomes = run_sharded(url, "m", cfg, 3, timeout_s=30, ignore_eos=True, lead_s=LEAD_S)
    merged = merge(outcomes, cfg.num_requests)
    expected = generate_workload(cfg)

    assert [o.shard for o in outcomes] == [0, 1, 2]
    for outcome in outcomes:
        assert all(shard_of(r.id, 3) == outcome.shard for r in outcome.results)
        assert len(outcome.results) == 4
    assert [r.scheduled_s for r in merged] == [spec.arrival_s for spec in expected]
    assert [r.prompt_tokens for r in merged] == [len(spec.prompt_token_ids) for spec in expected]
    assert all(r.status == "ok" and r.output_tokens == 4 for r in merged)
    # every shard measured its own event loop, and none was sent far behind its schedule
    assert all(o.loop_lag is not None and o.loop_lag["count"] > 10 for o in outcomes)
    assert all(r.sent_s - r.scheduled_s < 2.0 for r in merged)


def test_the_command_line_runs_shards_and_records_them(tmp_path: Path) -> None:
    out = tmp_path / "sharded.jsonl"
    with null_server_url() as url:
        argv = [
            "run", "--url", url, "--model", "m", "--rate", "30", "--num-requests", "12",
            "--warmup", "1", "--shards", "2", "--shard-lead", str(LEAD_S),
            "--out", str(out), *SMALL,
        ]  # fmt: skip
        code = main(argv)
    assert code == 0
    meta, results, summary = read_results(out)
    assert [r.id for r in results] == list(range(12)) and summary is not None
    assert meta["request_params"]["shards"] == 2
    assert "open_file_limit" in meta["request_params"]
    assert summary["shards"] == 2 and summary["num_ok"] == 12
    assert [s["shard"] for s in summary["shard_stats"]] == [0, 1]
    assert sum(s["requests"] for s in summary["shard_stats"]) == 12
    assert len(summary["event_loop_lag_s"]) == 2
    assert summary["event_loop_lag_p99_max_s"] is not None
    assert summary["peak_inflight"] >= 1


def test_a_single_process_run_records_one_event_loop(tmp_path: Path) -> None:
    out = tmp_path / "single.jsonl"
    with null_server_url() as url:
        argv = [
            "run", "--url", url, "--model", "m", "--rate", "50", "--num-requests", "10",
            "--warmup", "0", "--out", str(out), *SMALL,
        ]  # fmt: skip
        assert main(argv) == 0
    _, _, summary = read_results(out)
    assert summary is not None and summary["shards"] == 1 and summary["shard_stats"] is None
    assert len(summary["event_loop_lag_s"]) == 1


def test_shards_need_a_real_url_and_a_sensible_count(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    base = ["run", "--url", "http://x", "--model", "m", "--rate", "5", "--num-requests", "2",
            "--warmup", "0", *SMALL]  # fmt: skip
    transport = httpx.MockTransport(lambda request: httpx.Response(503))
    assert (
        main([*base, "--shards", "2", "--out", str(tmp_path / "a.jsonl")], transport=transport) == 2
    )
    assert "real server URL" in capsys.readouterr().err
    assert main([*base, "--shards", "0", "--out", str(tmp_path / "b.jsonl")]) == 2
    assert "at least 1" in capsys.readouterr().err
    assert not (tmp_path / "a.jsonl").exists() and not (tmp_path / "b.jsonl").exists()


def test_a_failed_warm_up_aborts_a_sharded_run(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "bad.jsonl"
    argv = [
        "run", "--url", "http://127.0.0.1:9", "--model", "m", "--rate", "5", "--num-requests", "2",
        "--warmup", "1", "--shards", "2", "--out", str(out), *SMALL,
    ]  # fmt: skip
    code = main(argv)
    assert code == 2 and "warm-up" in capsys.readouterr().err
    assert not out.exists()
