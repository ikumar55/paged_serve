"""Pairing a repeat sweep with its original and summarising the differences."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bench import repro_compare as rc


def write_sweep(
    directory: Path, runs: dict[str, tuple[float, bool]], *, repeat_of: str | None = None
) -> Path:
    """A sweep directory with a manifest and one result file per run (throughput, contended)."""
    directory.mkdir()
    lines = []
    for name, (throughput, contended) in runs.items():
        lines.append(
            json.dumps(
                {
                    "name": name,
                    "batch_size": 32,
                    "window_ms": 0.0,
                    "rate": 8.0,
                    "status": "ok",
                    "contended": contended,
                    "result_file": f"{name}.jsonl",
                    "_throughput": throughput,
                }
            )
        )
    (directory / "manifest.jsonl").write_text("\n".join(lines) + "\n")
    return directory


@pytest.fixture
def fake_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    """load_rows reads result files too; read the throughput from the manifest instead."""

    def load(directory: Path) -> list[rc.Row]:
        rows = []
        for line in (directory / "manifest.jsonl").read_text().splitlines():
            entry = json.loads(line)
            rows.append(
                rc.Row(
                    name=entry["name"],
                    batch_size=entry["batch_size"],
                    window_ms=entry["window_ms"],
                    rate=entry["rate"],
                    status=entry["status"],
                    contended=entry["contended"],
                    throughput_rps=entry["_throughput"],
                )
            )
        return rows

    monkeypatch.setattr(rc, "load_rows", load)


def test_pairs_match_by_name_and_skip_runs_missing_from_the_original(
    tmp_path: Path, fake_rows: None
) -> None:
    original = write_sweep(tmp_path / "o", {"a": (10.0, False), "b": (20.0, False)})
    repeat = write_sweep(tmp_path / "r", {"b": (21.0, False), "c": (5.0, False)})
    pairs = rc.pair_sweeps("engine", original, repeat)
    assert [p.repeat.name for p in pairs] == ["b"]
    assert pairs[0].original.throughput_rps == 20.0
    assert pairs[0].diff_percent == pytest.approx(5.0)


def test_a_contended_run_on_either_side_is_listed_but_not_summarised(
    tmp_path: Path, fake_rows: None
) -> None:
    original = write_sweep(
        tmp_path / "o", {"a": (10.0, False), "b": (10.0, True), "c": (10.0, False)}
    )
    repeat = write_sweep(
        tmp_path / "r", {"a": (11.0, False), "b": (50.0, False), "c": (90.0, True)}
    )
    pairs = rc.pair_sweeps("engine", original, repeat)
    assert [p.usable for p in pairs] == [True, False, False]
    count, mean_abs, max_abs, mean_signed = rc.summarise(pairs)["engine"]
    assert (count, mean_abs, max_abs, mean_signed) == (
        1,
        pytest.approx(10.0),
        pytest.approx(10.0),
        pytest.approx(10.0),
    )


def test_summary_separates_absolute_and_signed_differences(tmp_path: Path, fake_rows: None) -> None:
    original = write_sweep(tmp_path / "o", {"a": (10.0, False), "b": (10.0, False)})
    repeat = write_sweep(tmp_path / "r", {"a": (11.0, False), "b": (9.0, False)})
    count, mean_abs, max_abs, mean_signed = rc.summarise(rc.pair_sweeps("hf", original, repeat))[
        "hf"
    ]
    assert count == 2
    assert mean_abs == pytest.approx(10.0)
    assert max_abs == pytest.approx(10.0)
    assert mean_signed == pytest.approx(0.0)


def test_a_label_with_no_usable_pair_has_no_summary_line(tmp_path: Path, fake_rows: None) -> None:
    original = write_sweep(tmp_path / "o", {"a": (10.0, True)})
    repeat = write_sweep(tmp_path / "r", {"a": (10.0, False)})
    assert rc.summarise(rc.pair_sweeps("engine", original, repeat)) == {}


def test_cli_writes_the_table_and_the_summary(tmp_path: Path, fake_rows: None) -> None:
    original = write_sweep(tmp_path / "o", {"a": (10.0, False)})
    repeat = write_sweep(tmp_path / "r", {"a": (10.5, False)})
    out = tmp_path / "out.md"
    code = rc.main(["--pair", "engine", str(original), str(repeat), "--out", str(out)])
    assert code == 0
    text = out.read_text()
    assert "| engine | r | a | 10.00 | 10.50 | +5.0 |" in text
    assert "- engine: 1 pairs, mean |diff| 5.0%, max |diff| 5.0%, mean signed +5.0%" in text
