"""The matched, same-session comparison with vLLM (D-030): rates, validity, ratios, predictions.

Reads the sweep directories written by ``deploy/gpu_d030_a.sh`` and ``gpu_d030_b.sh``, named
``<tag>-<stage>-<arm>-rtx5090-seed<N>`` (stage ``probe``, ``main`` or ``lat``; arm ``E`` the engine
with the kernel and graphs, ``G`` its gather path, ``V64`` and ``V32`` vLLM with 64 and 32
sequences, ``VE`` vLLM with 64 sequences and ``--enforce-eager``). Every rule here is the one
written in D-030 before any code or result existed; nothing is tuned afterwards.

    python -m bench.matched_report generator results/d030-step0.jsonl
    python -m bench.matched_report rates --results results --waive-audit
    python -m bench.matched_report report --results results --out results/d030-report.md \\
        --monitor results/machine-monitor-d030-a.jsonl results/machine-monitor-d030-b.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from bench.fdlimits import soft_limit
from bench.machine_audit import audit_sweeps, load_samples
from bench.metrics import GENERATOR_LAG_WARN_S, LOOP_LAG_WARN_S, peak_inflight, steady_window
from bench.results_io import read_results

ARMS = ("E", "G", "V64", "V32", "VE")
ARM_NAMES = {
    "E": "engine, Triton kernel and CUDA graphs, batch 64",
    "G": "engine, gather path, batch 64",
    "V64": "vLLM, 64 sequences",
    "V32": "vLLM, 32 sequences",
    "VE": "vLLM, 64 sequences, eager (no CUDA graphs)",
}
VLLM_ARMS = ("V64", "V32", "VE")
PROBE_RATES = {"E": 64.0, "G": 32.0, "V64": 192.0, "V32": 128.0, "VE": 96.0}
SATURATION_FACTOR = 1.5  # the offered rate of a capacity run, in units of the arm's capacity
MIN_OFFERED_OVER_DELIVERED = 1.4  # the 1.5 target with a little room
ROUND_UP_TO = 4.0
FD_CEILING_SHARE = 0.8
LATENCY_RATE = 8.0
SUSTAINED_SHARE = 0.9
MIN_VALID_SEEDS = 3
NAME = re.compile(
    r"^(?P<tag>.+?)-(?P<stage>probe|main|lat)-(?P<arm>E|G|V64|V32|VE)-.*seed(?P<seed>\d+)$"
)
Audit = dict[tuple[str, str], tuple[str, ...]]


@dataclass
class Run:
    directory: str
    stage: str
    arm: str
    seed: int
    offered: float | None = None
    whole_rps: float | None = None
    window_rps: float | None = None
    tokens_per_s: float | None = None
    window_tokens_per_s: float | None = None
    ok: int = 0
    sent: int = 0
    peak_inflight: int = 0
    ttft_p50_s: float | None = None
    ttft_p99_s: float | None = None
    itl_p50_s: float | None = None
    itl_p99_s: float | None = None
    generator_saturated: bool | None = None
    reasons: list[str] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return not self.reasons

    @property
    def delivered(self) -> float | None:
        """The larger of the whole-run and steady-window rates (the stricter test of saturation)."""
        values = [v for v in (self.whole_rps, self.window_rps) if v is not None]
        return max(values) if values else None


def parse_directory(name: str) -> tuple[str, str, int] | None:
    """``(stage, arm, seed)`` of a sweep directory name, or None if it is not one of ours."""
    match = NAME.match(name)
    return None if match is None else (match["stage"], match["arm"], int(match["seed"]))


def _dist(summary: dict[str, Any], key: str, stat: str) -> float | None:
    value = summary.get(key)
    return None if not value else value.get(stat)


# -- validity ------------------------------------------------------------------------------------


def generator_problems(summary: dict[str, Any]) -> list[str]:
    """Why the load generator may have limited this run (empty: it did not)."""
    send_limit, loop_limit = GENERATOR_LAG_WARN_S * 1000, LOOP_LAG_WARN_S * 1000
    problems: list[str] = []
    lag = summary.get("client_lag_s")
    if lag is not None and lag["p99"] > GENERATOR_LAG_WARN_S:
        problems.append(f"send lag p99 {lag['p99'] * 1000:.0f} ms > {send_limit:.0f}")
    loop = summary.get("event_loop_lag_p99_max_s")
    if loop is None:
        problems.append("event-loop lag not recorded")
    elif loop > LOOP_LAG_WARN_S:
        problems.append(f"event-loop lag p99 {loop * 1000:.0f} ms > {loop_limit:.0f}")
    for shard in summary.get("shard_stats") or []:
        if (shard["send_lag_p99_s"] or 0.0) > GENERATOR_LAG_WARN_S:
            problems.append(f"shard {shard['shard']} send lag p99 above {send_limit:.0f} ms")
        if (shard["loop_lag_p99_s"] or 0.0) > LOOP_LAG_WARN_S:
            problems.append(f"shard {shard['shard']} event-loop lag p99 above {loop_limit:.0f} ms")
    return problems


def ceiling_problems(entry: dict[str, Any], peak: int) -> list[str]:
    """Whether the requests in flight came close to a process's open-file limit."""
    limits = [
        soft_limit(entry.get("server_fd_limits")),
        soft_limit(entry.get("generator_fd_limits")),
    ]
    known = [limit for limit in limits if limit is not None]
    if len(known) < len(limits):
        return ["open-file limit not recorded"]
    smallest = min(known)
    if peak > FD_CEILING_SHARE * smallest:
        return [f"ceiling-limited: {peak} in flight against an open-file limit of {smallest:.0f}"]
    return []


def rate_problems(run: Run) -> list[str]:
    """The rules about the offered rate (stage ``main``) or about sustaining it (``lat``)."""
    if run.offered is None:
        return []
    delivered = run.delivered
    if (
        run.stage == "main"
        and delivered is not None
        and run.offered < MIN_OFFERED_OVER_DELIVERED * delivered
    ):
        return [
            f"not saturated: offered {run.offered:g} is under {MIN_OFFERED_OVER_DELIVERED} "
            f"times the delivered {delivered:.1f}"
        ]
    if run.stage == "lat":
        sustained = run.window_rps is not None and run.window_rps >= SUSTAINED_SHARE * run.offered
        if run.ok < run.sent or not sustained:
            return ["not sustained at the common load"]
    return []


def audit_problems(name: tuple[str, str], audit: Audit | None, waived: bool) -> list[str]:
    if audit is None:
        return [] if waived else ["no machine audit given"]
    flags = audit.get(name)
    if flags is None:
        return ["no machine audit for this run"]
    return ["machine audit: " + ", ".join(flags)] if flags else []


def read_run(
    directory: Path, entry: dict[str, Any], run: Run, audit: Audit | None, waived: bool
) -> None:
    """Fill ``run`` from its result file and give it the reasons, if any, it is not valid."""
    path = directory / entry["result_file"]
    if entry.get("status") != "ok" or not path.exists():
        run.reasons.append(f"run did not finish ok ({entry.get('status')})")
        return
    meta, results, summary = read_results(path)
    if summary is None:
        run.reasons.append("no summary line")
        return
    window = steady_window(results)
    run.offered = float(meta["workload"]["rate"])
    run.whole_rps = summary.get("request_throughput_rps")
    run.tokens_per_s = summary.get("output_tokens_per_s")
    run.window_rps = None if window is None else window["completed_rps"]
    run.window_tokens_per_s = None if window is None else window["output_tokens_per_s"]
    run.ok, run.sent = summary["num_ok"], summary["num_requests"]
    run.peak_inflight = peak_inflight(results)
    run.ttft_p50_s, run.ttft_p99_s = (
        _dist(summary, "ttft_s", "p50"),
        _dist(summary, "ttft_s", "p99"),
    )
    run.itl_p50_s, run.itl_p99_s = _dist(summary, "itl_s", "p50"), _dist(summary, "itl_s", "p99")
    run.generator_saturated = summary.get("generator_saturated")

    if entry.get("contended"):
        run.reasons.append("contended: another user's GPU load before the run")
    run.reasons += audit_problems((directory.name, entry["name"]), audit, waived)
    run.reasons += generator_problems(summary)
    run.reasons += ceiling_problems(entry, run.peak_inflight)
    if window is None:
        run.reasons.append("no steady-state window (arrivals ended before 50 s)")
    run.reasons += rate_problems(run)


def load_directory(directory: Path, audit: Audit | None, waived: bool = False) -> list[Run]:
    """The run(s) of one sweep directory (none if its name is not one of ours)."""
    parsed = parse_directory(directory.name)
    if parsed is None:
        return []
    lines = (directory / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
    runs = []
    for entry in (json.loads(line) for line in lines if line.strip()):
        run = Run(directory.name, *parsed)
        read_run(directory, entry, run, audit, waived)
        runs.append(run)
    return runs


def sweep_dirs(root: Path, tag: str) -> list[Path]:
    return [d for d in sorted(root.glob(f"{tag}-*")) if (d / "manifest.jsonl").exists()]


def load_all(root: Path, tag: str, audit: Audit | None, waived: bool = False) -> list[Run]:
    return [run for d in sweep_dirs(root, tag) for run in load_directory(d, audit, waived)]


def audit_map(root: Path, tag: str, monitors: Sequence[Path]) -> Audit:
    audits = audit_sweeps(sweep_dirs(root, tag), load_samples(monitors))
    return {(a.sweep, a.name): a.flags for a in audits}


# -- the offered rates ---------------------------------------------------------------------------


def main_rate(delivered: float) -> float:
    """1.5 times the probe's delivered rate, rounded up to a multiple of 4."""
    return math.ceil(SATURATION_FACTOR * delivered / ROUND_UP_TO) * ROUND_UP_TO


def rate_decisions(runs: Sequence[Run]) -> dict[str, dict[str, Any]]:
    """Per arm: from its seed-0 probe, the offered rate of the main runs, or a repeat."""
    decisions: dict[str, dict[str, Any]] = {}
    for arm in ARMS:
        probes = [r for r in runs if r.stage == "probe" and r.arm == arm]
        if not probes:
            decisions[arm] = {"verdict": "no probe yet", "probe_at": PROBE_RATES[arm]}
            continue
        probe = max(probes, key=lambda r: r.offered or 0.0)  # a repeat is at twice the rate
        delivered = probe.delivered
        decision: dict[str, Any] = {"reasons": probe.reasons, "delivered": delivered}
        if probe.offered is None or delivered is None or probe.reasons:
            decision["verdict"] = "probe failed or invalid; repeat it"
        elif delivered > probe.offered / SATURATION_FACTOR:
            decision["verdict"] = "probe was not saturating; repeat once at twice the rate"
            decision["repeat_at"] = 2 * probe.offered
        else:
            decision["verdict"] = "ok"
            decision["main_rate"] = main_rate(delivered)
        decisions[arm] = decision
    return decisions


# -- the figures ---------------------------------------------------------------------------------


def valid_main(runs: Sequence[Run], arm: str) -> dict[int, Run]:
    return {r.seed: r for r in runs if r.stage == "main" and r.arm == arm and r.valid}


def window_rates(runs: Sequence[Run], arm: str) -> list[float]:
    return [r.window_rps for r in valid_main(runs, arm).values() if r.window_rps is not None]


def resolved_mean(values: Sequence[float]) -> float | None:
    """The mean, or None ("unresolved") when fewer than three valid seeds went into it."""
    return statistics.fmean(values) if len(values) >= MIN_VALID_SEEDS else None


def spread(values: Sequence[float], digits: int = 1) -> str:
    if not values:
        return "-"
    mean, low, high = statistics.fmean(values), min(values), max(values)
    return f"{mean:.{digits}f} ({low:.{digits}f} to {high:.{digits}f})"


def paired_ratios(numerator: dict[int, Run], denominator: dict[int, Run]) -> list[float]:
    """Per seed, one arm's steady-window rate over the other's, for seeds valid in both."""
    ratios = []
    for seed in sorted(set(numerator) & set(denominator)):
        top, bottom = numerator[seed].window_rps, denominator[seed].window_rps
        if top is not None and bottom:
            ratios.append(top / bottom)
    return ratios


def ratio_wording(ratios: Sequence[float]) -> str:
    """How D-030 allows the vLLM-over-engine ratio to be stated, judged on the worst seed."""
    if not ratios:
        return "no ratio (not enough valid seeds)"
    worst = max(ratios)
    if worst <= 1.25:
        return "within 25% of vLLM"
    if worst <= 2.0:
        return "within a factor of two of vLLM"
    return "more than twice behind vLLM"


def latency_mean(runs: Sequence[Run], arm: str, attr: str) -> float | None:
    values = [
        getattr(r, attr)
        for r in runs
        if r.stage == "lat" and r.arm == arm and r.valid and getattr(r, attr) is not None
    ]
    return resolved_mean(values)


Row = tuple[str, str, str, str]


def row(
    name: str, expected: str, measured: float | None, low: float, high: float, unit: str = ""
) -> Row:
    """A prediction row: held if the measurement is inside the written range."""
    if measured is None:
        return name, expected, "unresolved", "not evaluable"
    verdict = "held" if low <= measured <= high else "missed"
    return name, expected, f"{measured:.2f}{unit}", verdict


def prediction_rows(runs: Sequence[Run], with_d029: bool) -> list[Row]:
    """The six predictions of D-030, each held or missed."""
    mean = {arm: resolved_mean(window_rates(runs, arm)) for arm in ARMS}
    e_high = 38.0 if with_d029 else 33.0
    low, high = (2.0, 3.0) if with_d029 else (2.5, 3.5)
    v_over_e = resolved_mean(paired_ratios(valid_main(runs, "V64"), valid_main(runs, "E")))
    e_over_g = resolved_mean(paired_ratios(valid_main(runs, "E"), valid_main(runs, "G")))
    e_itl, v_itl = latency_mean(runs, "E", "itl_p50_s"), latency_mean(runs, "V64", "itl_p50_s")
    e_ttft, v_ttft = latency_mean(runs, "E", "ttft_p50_s"), latency_mean(runs, "V64", "ttft_p50_s")
    ttft_ratio = None if e_ttft is None or not v_ttft else e_ttft / v_ttft

    def to_ms(value: float | None) -> float | None:
        return None if value is None else value * 1000

    flagged = [r for r in runs if r.arm in VLLM_ARMS and r.generator_saturated]
    seen = [r for r in runs if r.arm in VLLM_ARMS and r.generator_saturated is not None]
    last: Row = (
        "(6) vLLM runs flagged generator-saturated",
        "none",
        f"{len(flagged)} of {len(seen)}",
        "not evaluable" if not seen else ("held" if not flagged else "missed"),
    )
    return [
        row("(1) vLLM 64 sequences, req/s", "85 to 100", mean["V64"], 85, 100),
        row("(1) vLLM 32 sequences, req/s", "58 to 66", mean["V32"], 58, 66),
        row("(1) vLLM 64 eager, req/s", "40 to 60", mean["VE"], 40, 60),
        row("(2) engine, kernel and graphs, req/s", f"28 to {e_high:g}", mean["E"], 28, e_high),
        row("(3) engine, gather path, req/s", "14 to 17", mean["G"], 14, 17),
        row("(4) vLLM 64 over engine", f"{low:g} to {high:g}", v_over_e, low, high),
        row("(4) engine over gather", "1.8 to 2.3", e_over_g, 1.8, 2.3),
        row("(5) engine inter-token p50 at 8 req/s, ms", "9 to 11", to_ms(e_itl), 9, 11),
        row("(5) vLLM inter-token p50 at 8 req/s, ms", "2 to 4", to_ms(v_itl), 2, 4),
        row("(5) engine over vLLM first-token p50", "at most 3", ttft_ratio, 0, 3),
        last,
    ]


# -- the report ----------------------------------------------------------------------------------


def num(value: float | None, digits: int = 1) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def runs_table(runs: Sequence[Run]) -> list[str]:
    lines = [
        "| stage | arm | seed | offered | whole-run req/s | window req/s | tokens/s | ok / sent "
        "| peak in flight | valid |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for r in sorted(runs, key=lambda r: (r.stage, ARMS.index(r.arm), r.seed)):
        verdict = "yes" if r.valid else "NO: " + "; ".join(r.reasons)
        lines.append(
            f"| {r.stage} | {r.arm} | {r.seed} | {num(r.offered, 0)} | {num(r.whole_rps)} "
            f"| {num(r.window_rps)} | {num(r.tokens_per_s, 0)} | {r.ok}/{r.sent} "
            f"| {r.peak_inflight} | {verdict} |"
        )
    return lines


def capacity_table(runs: Sequence[Run]) -> list[str]:
    lines = [
        "| arm | what | valid seeds | window req/s, mean (min to max) | tokens/s, mean |",
        "| --- | --- | ---: | --- | ---: |",
    ]
    for arm in ARMS:
        valid = valid_main(runs, arm)
        rates = window_rates(runs, arm)
        tokens = [r.window_tokens_per_s for r in valid.values() if r.window_tokens_per_s]
        resolved = len(valid) >= MIN_VALID_SEEDS
        lines.append(
            f"| {arm} | {ARM_NAMES[arm]} | {len(valid)} "
            f"| {spread(rates) if resolved else 'unresolved'} "
            f"| {num(statistics.fmean(tokens), 0) if resolved and tokens else '-'} |"
        )
    return lines


def ratio_section(runs: Sequence[Run]) -> list[str]:
    lines = ["| ratio | per seed | mean (min to max) |", "| --- | --- | --- |"]
    pairs = [("V64", "E"), ("E", "G"), ("V32", "E"), ("VE", "E"), ("V64", "V32"), ("V64", "VE")]
    for top, bottom in pairs:
        ratios = paired_ratios(valid_main(runs, top), valid_main(runs, bottom))
        each = ", ".join(f"{x:.2f}" for x in ratios) or "-"
        lines.append(f"| {top} / {bottom} | {each} | {spread(ratios, 2)} |")
    headline = paired_ratios(valid_main(runs, "V64"), valid_main(runs, "E"))
    lines += [
        "",
        f"**Rule for stating vLLM 64 over the engine (worst seed): {ratio_wording(headline)}.**",
    ]
    e_over_g = paired_ratios(valid_main(runs, "E"), valid_main(runs, "G"))
    if e_over_g and statistics.fmean(e_over_g) < 1.6:
        lines += [
            "",
            'Engine over gather is below 1.6: the "2.1 times" is withdrawn and replaced by it.',
        ]
    engine, eager = resolved_mean(window_rates(runs, "E")), resolved_mean(window_rates(runs, "VE"))
    if engine and eager:
        close = abs(eager - engine) <= 0.2 * engine
        verdict = "yes, so the gap is mostly vLLM's graphs" if close else "no"
        lines += ["", f"vLLM eager within 20% of the engine: {verdict}."]
    return lines


def latency_section(runs: Sequence[Run]) -> list[str]:
    lines = [
        "| arm | valid seeds | TTFT p50 ms | TTFT p99 ms | ITL p50 ms | ITL p99 ms |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for arm in ARMS:
        valid = [r for r in runs if r.stage == "lat" and r.arm == arm and r.valid]
        if not valid:
            continue
        cells = []
        for attr in ("ttft_p50_s", "ttft_p99_s", "itl_p50_s", "itl_p99_s"):
            values = [getattr(r, attr) * 1000 for r in valid if getattr(r, attr) is not None]
            cells.append(num(statistics.fmean(values)) if values else "-")
        lines.append(f"| {arm} | {len(valid)} | " + " | ".join(cells) + " |")
    return lines


def render_report(runs: Sequence[Run], *, with_d029: bool, audit_waived: bool) -> str:
    lines = ["# D-030: the matched comparison with vLLM", ""]
    if audit_waived:
        lines += ["**THE MACHINE AUDIT WAS WAIVED; the plan requires a clean audit.**", ""]
    lines += [
        "Every rule is the one fixed in D-030 before any code or result. The primary metric is "
        "successful requests per second completed between 15 s and 50 s (the steady-state "
        "window); the whole-run figure is next to it. A run that fails a validity rule is listed "
        "with its reasons and is not used.",
        "",
        "## Every run",
        "",
        *runs_table(runs),
        "",
        "## Capacity (valid main runs, steady-state window)",
        "",
        f"An arm with fewer than {MIN_VALID_SEEDS} valid seeds is unresolved.",
        "",
        *capacity_table(runs),
        "",
        "## Paired ratios (per seed, valid in both arms)",
        "",
        *ratio_section(runs),
        "",
        f"## Latency at {LATENCY_RATE:g} req/s (valid lat runs, mean over seeds)",
        "",
        *latency_section(runs),
        "",
        "## Predictions of D-030, each held or missed",
        "",
        "| prediction | written | measured | verdict |",
        "| --- | --- | --- | --- |",
    ]
    lines += [f"| {a} | {b} | {c} | **{d}** |" for a, b, c, d in prediction_rows(runs, with_d029)]
    lines.append("")
    return "\n".join(lines)


def generator_check(path: Path) -> tuple[bool, list[str]]:
    """Step 0 of the session: did the sharded generator keep up against the null server?"""
    _, _, summary = read_results(path)
    if summary is None:
        return False, ["no summary line in the result"]
    problems = generator_problems(summary)
    if not summary.get("shard_stats"):
        problems.append("the run was not sharded (no per-process statistics)")
    return not problems, problems


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0] if __doc__ else None)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser(
        "generator", help="step 0: check a sharded run against the null server"
    ).add_argument("path", type=Path)
    helps = {
        "rates": "the offered rate of each arm's main runs, from its probe",
        "report": "the report",
    }
    for name, text in helps.items():
        cmd = sub.add_parser(name, help=text)
        cmd.add_argument("--results", type=Path, default=Path("results"))
        cmd.add_argument("--tag", default="d030")
        cmd.add_argument("--monitor", type=Path, nargs="*", default=None)
        cmd.add_argument("--waive-audit", action="store_true", help="skip the machine audit")
        if name == "report":
            cmd.add_argument("--with-d029", action="store_true", help="the engine includes D-029")
            cmd.add_argument("--out", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "generator":
        ok, problems = generator_check(args.path)
        print("generator check: " + ("PASS" if ok else "FAIL: " + "; ".join(problems)))
        return 0 if ok else 1

    audit = audit_map(args.results, args.tag, args.monitor) if args.monitor else None
    runs = load_all(args.results, args.tag, audit, args.waive_audit)
    if args.command == "rates":
        decisions = rate_decisions(runs)
        for arm, decision in decisions.items():
            detail = decision["verdict"]
            if "main_rate" in decision:
                detail += f": main offered rate {decision['main_rate']:g} req/s"
            if "repeat_at" in decision:
                detail += f": {decision['repeat_at']:g} req/s"
            print(f"{arm}: {detail}")
        return 0 if all(d["verdict"] == "ok" for d in decisions.values()) else 1
    waived = audit is None and args.waive_audit
    args.out.write_text(
        render_report(runs, with_d029=args.with_d029, audit_waived=waived), encoding="utf-8"
    )
    print(f"wrote {args.out}: {len(runs)} runs, {sum(1 for r in runs if r.valid)} valid")
    return 0


if __name__ == "__main__":
    sys.exit(main())
