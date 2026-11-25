"""Open-file limits of a process (D-021, D-030).

Every in-flight request holds a connection, so a process whose soft limit is 1,024 (the usual Linux
default) cannot have much more than a thousand requests in flight. The gather arm of D-025 peaked at
exactly 997 in flight and failed there; the limit was never read. This records it: for the process
that runs the load generator (inherited by its children) and, from ``/proc``, for a server.

A limit is an integer, the string ``"unlimited"``, or the whole record is ``None`` when it could not
be read (not Linux, or a process that is gone). ``None`` is never taken to mean "no limit".
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

UNLIMITED = "unlimited"


def _value(text: str) -> int | str:
    return UNLIMITED if text == UNLIMITED else int(text)


def parse_proc_limits(text: str) -> dict[str, Any] | None:
    """The open-file soft and hard limits from the text of ``/proc/<pid>/limits``."""
    for line in text.splitlines():
        if line.startswith("Max open files"):
            parts = line.split()
            try:
                return {"soft": _value(parts[3]), "hard": _value(parts[4])}
            except (IndexError, ValueError):
                return None
    return None


def own_limits() -> dict[str, Any] | None:
    """This process's open-file limits, or None where the ``resource`` module does not exist."""
    if sys.platform == "win32":
        return None
    import resource

    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)

    def value(limit: int) -> int | str:
        return UNLIMITED if limit == resource.RLIM_INFINITY else limit

    return {"soft": value(soft), "hard": value(hard)}


def process_limits(pid: int, proc_root: Path = Path("/proc")) -> dict[str, Any] | None:
    """Another process's open-file limits, read from ``/proc``; None if that is not possible."""
    try:
        return parse_proc_limits((proc_root / str(pid) / "limits").read_text())
    except OSError:
        return None


def soft_limit(limits: dict[str, Any] | None) -> float | None:
    """The soft limit as a number (infinity if unlimited); None if it is not known."""
    if limits is None or "soft" not in limits:
        return None
    soft = limits["soft"]
    return math.inf if soft == UNLIMITED else float(soft)
