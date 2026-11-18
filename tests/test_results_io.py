import json
from pathlib import Path

import pytest

from bench.metrics import summarize
from bench.results_io import SCHEMA_VERSION, read_results, write_results
from tests.test_metrics import failed_result, ok_result

META = {"label": "demo", "workload": {"rate": 2.0}}


def sample_results() -> list:  # type: ignore[type-arg]
    return [ok_result(0, ttft_s=0.123456789), failed_result(1, "timeout", "timeout")]


def test_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "run.jsonl"
    results = sample_results()
    summary = summarize(results)
    write_results(path, META, results, summary)

    meta, loaded, loaded_summary = read_results(path)
    assert meta["label"] == "demo" and meta["schema_version"] == SCHEMA_VERSION
    assert [r.id for r in loaded] == [0, 1]
    assert loaded[0].status == "ok" and loaded[1].status == "timeout"
    assert loaded_summary is not None and loaded_summary["num_ok"] == 1


def test_floats_are_rounded_to_microseconds(tmp_path: Path) -> None:
    path = tmp_path / "run.jsonl"
    write_results(path, META, sample_results(), summarize(sample_results()))
    _, loaded, _ = read_results(path)
    assert loaded[0].ttft_s == 0.123457


def test_existing_results_are_never_overwritten(tmp_path: Path) -> None:
    path = tmp_path / "run.jsonl"
    write_results(path, META, sample_results(), summarize(sample_results()))
    original = path.read_text()
    with pytest.raises(FileExistsError):
        write_results(path, META, [], summarize([]))
    assert path.read_text() == original


def test_file_layout(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "run.jsonl"
    write_results(path, META, sample_results(), summarize(sample_results()))
    kinds = [json.loads(line)["type"] for line in path.read_text().splitlines()]
    assert kinds == ["meta", "request", "request", "summary"]


def test_rejects_unknown_schema_version(tmp_path: Path) -> None:
    path = tmp_path / "run.jsonl"
    path.write_text(json.dumps({"type": "meta", "schema_version": 999}) + "\n")
    with pytest.raises(ValueError, match="schema_version"):
        read_results(path)


def test_rejects_invalid_json_with_line_number(tmp_path: Path) -> None:
    path = tmp_path / "run.jsonl"
    path.write_text(json.dumps({"type": "meta", "schema_version": SCHEMA_VERSION}) + "\n{oops\n")
    with pytest.raises(ValueError, match="line 2"):
        read_results(path)


def test_rejects_unknown_record_and_missing_meta(tmp_path: Path) -> None:
    path = tmp_path / "a.jsonl"
    path.write_text(json.dumps({"type": "mystery"}) + "\n")
    with pytest.raises(ValueError, match="unknown record type"):
        read_results(path)

    only_requests = tmp_path / "b.jsonl"
    only_requests.write_text(json.dumps(ok_result(0).to_dict()) + "\n")
    with pytest.raises(ValueError, match="missing meta"):
        read_results(only_requests)
