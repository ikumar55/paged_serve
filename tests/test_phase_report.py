"""The scheduler's phase accounting and the report built from it (D-028)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from bench import phase_report as pr
from engine.config import LlamaConfig
from engine.paged_cache import KVPool
from engine.scheduler import Scheduler
from tests.test_paged_cache import make_model

CFG = LlamaConfig.tiny()


def serve(prompts: list[list[int]], num_blocks: int = 200, prefix: bool = True) -> Scheduler:
    pool = KVPool(CFG, num_blocks, 4, torch.float32, "cpu")
    scheduler = Scheduler(make_model(CFG), pool, max_batch_size=4, enable_prefix_caching=prefix)
    for index, prompt in enumerate(prompts):
        scheduler.add_request(index, prompt, 8)
    scheduler.run_to_completion()
    return scheduler


def prompts() -> list[list[int]]:
    generator = torch.manual_seed(31)
    shared = torch.randint(0, CFG.vocab_size, (12,), generator=generator).tolist()
    return [
        shared + torch.randint(0, CFG.vocab_size, (n,), generator=generator).tolist()
        for n in (3, 5, 2, 7)
    ]


def test_every_admission_is_counted_and_only_computed_tokens_are() -> None:
    plain = serve(prompts(), prefix=False)
    assert plain.prefills == 4
    assert plain.prefill_tokens == sum(len(p) for p in prompts())
    cached = serve(prompts(), prefix=True)
    assert cached.prefills == 4
    assert cached.prefix_hit_tokens > 0
    assert cached.prefill_tokens == sum(len(p) for p in prompts()) - cached.prefix_hit_tokens


def test_a_preempted_request_is_prefilled_again_and_counted() -> None:
    crowded = serve(prompts(), num_blocks=14)
    assert crowded.total_preemptions > 0
    assert crowded.prefills == 4 + crowded.total_preemptions


def test_the_phases_are_timed_and_reported_in_stats() -> None:
    scheduler = serve(prompts())
    phases = scheduler.phase_seconds
    assert phases["decode"] > 0 and phases["admit"] > 0 and phases["room"] >= 0
    stats = scheduler.stats()
    assert stats["phase_decode_s"] == phases["decode"]
    assert stats["phase_admit_s"] == phases["admit"]
    assert stats["phase_room_s"] == phases["room"]
    assert stats["prefills"] == scheduler.prefills
    assert stats["prefill_tokens"] == scheduler.prefill_tokens


def snapshot(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "phase_room_s": 1.0,
        "phase_decode_s": 60.0,
        "phase_admit_s": 25.0,
        "prefills": 100,
        "prefill_tokens": 25_000,
        "decode_steps": 1000,
        "engine_thread_seconds": {"step": 100.0, "route": 2.0, "inbox": 1.0, "idle": 0.5},
    }
    base.update(overrides)
    return base


def test_accounting_splits_the_step_time_and_subtracts_the_start() -> None:
    before = snapshot(
        phase_room_s=0.5,
        phase_decode_s=10.0,
        phase_admit_s=5.0,
        prefills=20,
        prefill_tokens=5_000,
        decode_steps=200,
        engine_thread_seconds={"step": 20.0, "route": 0.5, "inbox": 0.5, "idle": 0.0},
    )
    row = pr.phase_accounting(before, snapshot())
    assert row is not None
    assert row["step_s"] == pytest.approx(80.0)
    assert row["decode_share"] == pytest.approx(50.0 / 80.0)
    assert row["admission_share"] == pytest.approx(20.0 / 80.0)
    assert row["other_share"] == pytest.approx((80.0 - 0.5 - 50.0 - 20.0) / 80.0)
    assert row["ms_per_prefill"] == pytest.approx(1000 * 20.0 / 80)
    assert row["prefills_per_decode_step"] == pytest.approx(80 / 800)
    assert row["prefill_tokens_per_prefill"] == pytest.approx(250.0)
    assert row["ms_per_decode_step"] == pytest.approx(1000 * 50.0 / 800)
    thread_total = 80.0 + 1.5 + 0.5 + 0.5
    assert row["route_inbox_share"] == pytest.approx(2.0 / thread_total)
    assert row["idle_share"] == pytest.approx(0.5 / thread_total)


def test_an_engine_without_the_timers_gives_no_accounting() -> None:
    old = {"decode_steps": 10, "engine_thread_seconds": {"step": 1.0}}
    assert pr.phase_accounting({}, old) is None
    no_thread = snapshot()
    del no_thread["engine_thread_seconds"]
    assert pr.phase_accounting({}, no_thread) is None


def test_predictions_are_checked_against_the_stated_ranges() -> None:
    row = pr.phase_accounting({}, snapshot())
    assert row is not None
    verdicts = {name: text for name, _, text in pr.check_predictions({0: row})}
    assert verdicts["admission share of step time"].endswith("inside")  # 25 of 99.5
    assert verdicts["decode share of step time"].endswith("inside")
    assert verdicts["ms per admitted prompt"].endswith("OUTSIDE")  # 250 ms is far above 5 to 12
    worse = pr.phase_accounting({}, snapshot(phase_admit_s=5.0))
    assert worse is not None
    assert pr.check_predictions({0: worse})[0][2].endswith("OUTSIDE")  # admission only 5%


def test_render_shows_the_table_and_the_predictions() -> None:
    row = pr.phase_accounting({}, snapshot())
    assert row is not None
    text = pr.render({0: row, 1: row}, {0: 28.5, 1: 29.5})
    assert "28.50" in text and "29.50" in text
    assert "Against the predictions fixed in D-028" in text
    assert text.count("| 0 |") == 1 and text.count("| 1 |") == 1


def test_main_reads_the_sweep_directories_and_refuses_to_overwrite(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    for seed in (0, 1):
        directory = tmp_path / f"run-seed{seed}"
        directory.mkdir()
        (directory / "manifest.jsonl").write_text("", encoding="utf-8")
    row = SimpleNamespace(throughput_rps=28.0)
    monkeypatch.setattr(pr, "load_rows", lambda directory: [row])
    monkeypatch.setattr(pr, "find_run", lambda rows, batch_size, rate: rows[0])
    monkeypatch.setattr(pr, "run_stats", lambda directory, row: {"before": {}, "after": snapshot()})
    out = tmp_path / "report.md"
    args = ["--engine", str(tmp_path / "run-seed{seed}"), "--seeds", "0,1", "--out", str(out)]
    assert pr.main(args) == 0
    assert "| 0 |" in out.read_text(encoding="utf-8")
    with pytest.raises(SystemExit, match="exists"):
        pr.main(args)
    monkeypatch.setattr(pr, "run_stats", lambda directory, row: {"before": {}, "after": {}})
    with pytest.raises(SystemExit, match="phase timers"):
        pr.main([*args[:-1], str(tmp_path / "other.md")])


def test_each_phase_gets_its_own_interval(monkeypatch: pytest.MonkeyPatch) -> None:
    """A fake clock that moves 1, 2 and 3 seconds between the four readings of one step."""
    readings = iter([10.0, 11.0, 13.0, 16.0])
    monkeypatch.setattr("engine.scheduler.time.perf_counter", lambda: next(readings))
    pool = KVPool(CFG, 50, 4, torch.float32, "cpu")
    scheduler = Scheduler(make_model(CFG), pool, max_batch_size=2)
    scheduler.add_request(0, [1, 2, 3], 4)
    scheduler.step()  # nothing is running yet, so the decode is skipped; the clock is still read
    assert scheduler.phase_seconds == {"room": 1.0, "decode": 2.0, "admit": 3.0}
