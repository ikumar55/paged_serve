"""JSONL result files: one ``meta`` line, one line per request, and a closing ``summary`` line.

Result files are written once by the load generator and never edited by hand. Writing refuses to
overwrite an existing file.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from bench.client import RequestResult

SCHEMA_VERSION = 1


def _round_floats(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 6)
    if isinstance(value, list):
        return [_round_floats(v) for v in value]
    if isinstance(value, dict):
        return {k: _round_floats(v) for k, v in value.items()}
    return value


def write_results(
    path: Path,
    meta: dict[str, Any],
    results: Sequence[RequestResult],
    summary: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [{"type": "meta", "schema_version": SCHEMA_VERSION, **meta}]
    lines.extend(result.to_dict() for result in results)
    lines.append({"type": "summary", **summary})
    text = "".join(json.dumps(_round_floats(line), sort_keys=True) + "\n" for line in lines)
    with path.open("x") as handle:  # "x": fail rather than overwrite an existing result
        handle.write(text)


def read_results(path: Path) -> tuple[dict[str, Any], list[RequestResult], dict[str, Any] | None]:
    meta: dict[str, Any] | None = None
    summary: dict[str, Any] | None = None
    results: list[RequestResult] = []
    with path.open() as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}: line {number} is not valid JSON") from exc
            kind = record.get("type")
            if kind == "meta":
                if record.get("schema_version") != SCHEMA_VERSION:
                    raise ValueError(
                        f"{path}: unsupported schema_version {record.get('schema_version')!r}"
                    )
                meta = {k: v for k, v in record.items() if k != "type"}
            elif kind == "request":
                results.append(RequestResult.from_dict(record))
            elif kind == "summary":
                summary = {k: v for k, v in record.items() if k != "type"}
            else:
                raise ValueError(f"{path}: line {number} has unknown record type {kind!r}")
    if meta is None:
        raise ValueError(f"{path}: missing meta record")
    return meta, results, summary
