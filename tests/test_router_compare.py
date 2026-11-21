"""The router policy comparison on a mock fleet (bench/router_compare.py)."""

import json
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import torch

from bench.client import RequestResult
from bench.router_compare import (
    BANNER,
    FleetConfig,
    FleetEndpoints,
    FleetError,
    Variant,
    build_variants,
    format_table,
    imbalance,
    main,
    replica_command,
    router_command,
    run_comparison,
    run_variant,
    sanitize_argv,
    spread,
    subprocess_fleet,
    summarize_run,
    workload_digest,
    write_outputs,
)
from bench.workload import LengthDist, WorkloadConfig, generate_workload
from engine import server
from engine.mock import MockModel, TimingModel
from engine.paged_cache import KVPool
from engine.scheduler import Scheduler
from router.app import create_router_app
from router.policies import DEFAULT_MAX_ENTRIES, make_policy
from tests.fake_server import UvicornServer

BLOCK = 4


def small_workload(**overrides: Any) -> WorkloadConfig:
    fields: dict[str, Any] = {
        "rate": 200.0,
        "num_requests": 24,
        "prompt": LengthDist(6, 0.2, 1, 12),
        "output": LengthDist(3, 0.2, 1, 5),
        "vocab_size": 100000,
        "shared_prefix_groups": 3,
        "shared_prefix_len": 16,
        "shared_prefix_skew": 1.0,
    }
    return WorkloadConfig(**{**fields, **overrides})


def small_fleet(**overrides: Any) -> FleetConfig:
    fields: dict[str, Any] = {"replicas": 2, "block_size": BLOCK, "time_scale": 0.0}
    return FleetConfig(**{**fields, **overrides})


@contextmanager
def thread_fleet(variant: Variant, fleet: FleetConfig) -> Iterator[FleetEndpoints]:
    """The same fleet as ``subprocess_fleet`` but as threads of this process, which starts in
    milliseconds. The subprocess version has its own test below."""
    with ExitStack() as stack:
        urls = []
        for _ in range(fleet.replicas):
            model = MockModel(timing=TimingModel(time_scale=0))
            pool = KVPool(model.cfg, 300, fleet.block_size, torch.float32, "cpu")
            scheduler = Scheduler(model, pool, max_batch_size=8)
            urls.append(
                stack.enter_context(UvicornServer(server.create_app(scheduler, [], model_id="m")))
            )
        policy = make_policy(
            variant.policy,
            chunk_tokens=fleet.block_size,
            load_cap=variant.effective_load_cap(fleet) or fleet.load_cap,
            max_entries=variant.max_entries or DEFAULT_MAX_ENTRIES,
        )
        router = stack.enter_context(
            UvicornServer(create_router_app(urls, policy, poll_interval_s=0.05))
        )
        yield FleetEndpoints(router, urls)


# -- the arithmetic, on hand-built reports ---------------------------------------------------------


def request(index: int, ttft: float) -> RequestResult:
    return RequestResult(
        id=index,
        scheduled_s=index * 0.1,
        sent_s=index * 0.1,
        prompt_tokens=10,
        requested_output_tokens=2,
        status="ok",
        output_tokens=2,
        tokens_from_usage=True,
        ttft_s=ttft,
        e2e_s=ttft + 0.05,
        token_times_s=[ttft, ttft + 0.05],
    )


def report(
    policy: str = "approx-prefix",
    served: tuple[int, ...] = (6, 2),
    tokens: tuple[tuple[int, int], ...] = ((100, 60), (50, 10)),
) -> dict[str, Any]:
    """A router ``/info?fleet=true`` for replicas that served ``served`` requests and saw
    ``(prompt tokens, prefix hit tokens)`` each."""
    urls = [f"http://replica{i}" for i in range(len(served))]
    return {
        "policy": policy,
        "policy_stats": {"decisions": sum(served), "load_fallbacks": 1},
        "failovers": 0,
        "upstream_errors": 0,
        "no_replica": 0,
        "bad_requests": 0,
        "replicas": [{"url": u, "served": s} for u, s in zip(urls, served, strict=True)],
        "fleet": {
            "replicas": {
                u: {
                    "stats": {
                        "prefix_prompt_tokens": p,
                        "prefix_hit_tokens": h,
                        "preemptions": 0,
                        "block_evictions": 0,
                    }
                }
                for u, (p, h) in zip(urls, tokens, strict=True)
            }
        },
    }


def test_imbalance_is_the_busiest_replicas_share_of_an_even_split() -> None:
    assert imbalance([5, 5, 5]) == 1.0
    assert imbalance([6, 2]) == pytest.approx(1.5)  # 6 against a fair 4
    assert imbalance([9, 0, 0]) == pytest.approx(3.0)
    assert imbalance([0, 0]) is None and imbalance([]) is None


def test_summarize_run_computes_the_fleet_figures_from_the_replicas_own_counters() -> None:
    fleet = small_fleet()
    summary = summarize_run(
        [request(0, 0.2), request(1, 0.4)], report(), Variant("approx-prefix"), fleet
    )
    assert summary["prefix_hit_tokens"] == 70 and summary["prefix_prompt_tokens"] == 150
    assert summary["prefix_hit_rate"] == pytest.approx(70 / 150)
    assert summary["served_imbalance"] == pytest.approx(1.5)  # 6 and 2 requests
    # Computed tokens: 100-60 = 40 and 50-10 = 40, so perfectly even although served is not.
    assert [r["computed_prompt_tokens"] for r in summary["replicas"]] == [40, 40]
    assert summary["computed_tokens_imbalance"] == pytest.approx(1.0)
    assert summary["ok"] == 2 and summary["ttft_s"]["p50"] == pytest.approx(0.3)
    assert summary["policy_stats"]["load_fallbacks"] == 1


def test_a_report_that_cannot_be_trusted_is_refused_rather_than_counted_as_zero() -> None:
    fleet, variant = small_fleet(), Variant("approx-prefix")
    results = [request(0, 0.1)]

    missing = report()
    del missing["fleet"]["replicas"]["http://replica1"]["stats"]  # that replica did not answer
    with pytest.raises(ValueError, match="missing"):
        summarize_run(results, missing, variant, fleet)

    with pytest.raises(ValueError, match="policy"):
        summarize_run(results, report(policy="round-robin"), variant, fleet)

    with pytest.raises(ValueError, match="replicas"):
        summarize_run(results, report(served=(6, 2, 1), tokens=((1, 0),) * 3), variant, fleet)


def test_a_workload_digest_identifies_exactly_what_is_sent() -> None:
    specs = generate_workload(small_workload(seed=1))
    assert workload_digest(specs) == workload_digest(generate_workload(small_workload(seed=1)))
    assert workload_digest(specs) != workload_digest(generate_workload(small_workload(seed=2)))
    changed = [replace(specs[0], prompt_token_ids=[*specs[0].prompt_token_ids[:-1], 1]), *specs[1:]]
    assert workload_digest(changed) != workload_digest(specs)


def test_variants_are_validated_and_labelled() -> None:
    assert Variant("round-robin").label == "round-robin"
    assert Variant("approx-prefix", 64).label == "approx-prefix[max-entries=64]"
    with pytest.raises(ValueError, match="unknown policy"):
        Variant("random")
    with pytest.raises(ValueError, match="only applies to approx-prefix"):
        Variant("precise-prefix", 10)
    with pytest.raises(ValueError, match="positive"):
        Variant("approx-prefix", 0)
    labels = [v.label for v in build_variants(["round-robin", "approx-prefix"], [8])]
    assert labels == ["round-robin", "approx-prefix", "approx-prefix[max-entries=8]"]
    with pytest.raises(ValueError, match="needs approx-prefix"):
        build_variants(["round-robin"], [8])
    with pytest.raises(ValueError, match="duplicate"):
        build_variants(["approx-prefix"], [8, 8])


def test_a_load_cap_belongs_to_the_prefix_policies_only() -> None:
    assert Variant("precise-prefix", load_cap=8).label == "precise-prefix[cap=8]"
    assert Variant("approx-prefix", 64, 8).label == "approx-prefix[max-entries=64,cap=8]"
    for policy in ("round-robin", "least-outstanding"):
        with pytest.raises(ValueError, match="prefix policies"):
            Variant(policy, load_cap=8)
    with pytest.raises(ValueError, match="positive"):
        Variant("approx-prefix", load_cap=0)
    fleet = small_fleet(load_cap=16)
    assert Variant("approx-prefix").effective_load_cap(fleet) == 16  # the fleet's, by default
    assert Variant("approx-prefix", load_cap=4).effective_load_cap(fleet) == 4
    assert Variant("round-robin").effective_load_cap(fleet) is None  # it has no cap to report


def test_load_caps_multiply_the_prefix_policies_and_leave_the_others_alone() -> None:
    policies = ["round-robin", "least-outstanding", "approx-prefix", "precise-prefix"]
    labels = [v.label for v in build_variants(policies, [], [4, 32])]
    assert labels == [
        "round-robin",
        "least-outstanding",
        "approx-prefix[cap=4]",
        "approx-prefix[cap=32]",
        "precise-prefix[cap=4]",
        "precise-prefix[cap=32]",
    ]
    with pytest.raises(ValueError, match="needs a prefix policy"):
        build_variants(["round-robin"], [], [4])
    # A memory-limited variant keeps the fleet's cap: it is a different question.
    with_limit = build_variants(["approx-prefix"], [8], [4])
    assert [(v.max_entries, v.load_cap) for v in with_limit] == [(None, 4), (8, None)]


def test_spread_is_mean_with_min_and_max() -> None:
    assert spread([1.0, 2.0, 6.0]) == "3.0 (1.0 to 6.0)"
    assert spread([0.5, 0.25], scale=100, digits=0) == "38 (25 to 50)"


def test_commands_carry_the_settings_and_the_memory_limit_only_when_there_is_one() -> None:
    fleet = small_fleet(load_cap=7)
    plain = router_command(9000, ["http://a", "http://b"], Variant("approx-prefix"), fleet)
    assert plain[plain.index("--policy") + 1] == "approx-prefix"
    assert plain[plain.index("--chunk-tokens") + 1] == str(BLOCK)  # chunks match replica blocks
    assert plain[plain.index("--load-cap") + 1] == "7"
    assert [plain[i + 1] for i, a in enumerate(plain) if a == "--replica"] == [
        "http://a",
        "http://b",
    ]
    assert "--max-entries" not in plain
    limited = router_command(9000, ["http://a"], Variant("approx-prefix", 33), fleet)
    assert limited[limited.index("--max-entries") + 1] == "33"
    assert limited[limited.index("--load-cap") + 1] == "7"  # a memory limit leaves the cap alone
    capped = router_command(9000, ["http://a"], Variant("precise-prefix", load_cap=3), fleet)
    assert capped[capped.index("--load-cap") + 1] == "3"  # the variant's cap beats the fleet's


# -- whole comparisons over a (threaded) fleet ---------------------------------------------------


def test_every_variant_of_a_seed_gets_the_same_requests_and_serves_all_of_them() -> None:
    variants = [Variant("round-robin"), Variant("least-outstanding"), Variant("approx-prefix")]
    workload = small_workload()
    runs = run_comparison(thread_fleet, variants, small_fleet(), workload, [0, 1], timeout_s=30)
    assert [(r["seed"], r["variant"]) for r in runs] == [
        (seed, v.label) for seed in (0, 1) for v in variants
    ]
    assert all(r["status"] == "ok" for r in runs)
    # The cap each router really ran with is recorded: none for the policies without one.
    assert {r["variant"]: r["load_cap"] for r in runs} == {
        "round-robin": None,
        "least-outstanding": None,
        "approx-prefix": 16,
    }
    for seed in (0, 1):
        digests = {r["workload_digest"] for r in runs if r["seed"] == seed}
        assert len(digests) == 1
    assert len({r["workload_digest"] for r in runs}) == 2  # the two seeds differ
    for run in runs:
        summary = run["summary"]
        assert summary["requests"] == summary["ok"] == workload.num_requests
        assert sum(r["served"] for r in summary["replicas"]) == workload.num_requests
        assert summary["policy_stats"]["decisions"] == workload.num_requests
        assert 0 <= summary["prefix_hit_rate"] <= 1


def test_a_fleet_that_will_not_start_is_recorded_not_hidden() -> None:
    @contextmanager
    def broken(variant: Variant, fleet: FleetConfig) -> Iterator[FleetEndpoints]:
        raise FleetError("no")
        yield  # pragma: no cover

    entry = run_variant(
        broken, Variant("round-robin"), small_fleet(), generate_workload(small_workload()),
        timeout_s=5,
    )  # fmt: skip
    assert entry["status"] == "fleet_error" and "summary" not in entry


# -- the files -----------------------------------------------------------------------------------


def run_main(out: Path, *extra: str, fleet: Any = thread_fleet) -> int:
    return main(
        [
            "--out", str(out), "--label", "unit", "--policies", "round-robin,approx-prefix",
            "--approx-max-entries", "6", "--seeds", "0,1", "--replicas", "2",
            "--block-size", str(BLOCK), "--mock-time-scale", "0", "--rate", "200",
            "--num-requests", "24", "--shared-prefix-groups", "3", "--shared-prefix-len", "16",
            "--prompt-median", "6", "--prompt-max", "12", "--output-median", "3",
            "--output-max", "5", *extra,
        ],
        start_fleet=fleet,
    )  # fmt: skip


def test_main_writes_a_labelled_write_once_result_and_report(tmp_path: Path) -> None:
    out = tmp_path / "cmp.json"
    assert run_main(out) == 0
    doc = json.loads(out.read_text())
    text = (tmp_path / "cmp.md").read_text()

    assert doc["backend"] == "mock" and doc["calibrated"] is False and doc["banner"] == BANNER
    assert text.count(BANNER) == 1 and text.startswith("# Router policy comparison, mock fleet")
    assert doc["variant_labels"] == [
        "round-robin",
        "approx-prefix",
        "approx-prefix[max-entries=6]",
    ]
    assert len(doc["runs"]) == 6 and doc["seeds"] == [0, 1]
    assert doc["mock_timing"]["time_scale"] == 0.0  # the timing that was in force is recorded
    assert "seed" not in doc["workload"]  # the seeds are listed once, per run
    for label in doc["variant_labels"]:
        assert f"| {label} |" in text
    # The same file cannot be produced twice, and neither half may be replaced.
    before = out.read_text()
    assert run_main(out) == 2 and out.read_text() == before
    out.unlink()
    assert run_main(out) == 2 and not out.exists()  # the .md alone is enough to refuse


def test_writing_the_outputs_refuses_to_replace_either_file(tmp_path: Path) -> None:
    doc = {"label": "", "seeds": [], "fleet": {"replicas": 1}, "variant_labels": [], "runs": []}
    doc |= {"workload": {"num_requests": 1, "rate": 1.0}, "notes": []}
    json_path, md_path = tmp_path / "a.json", tmp_path / "a.md"
    write_outputs(json_path, md_path, doc)
    first = (json_path.read_text(), md_path.read_text())
    other = {**doc, "label": "something else"}  # so an overwrite would be visible
    with pytest.raises(FileExistsError):
        write_outputs(json_path, md_path, other)
    assert (json_path.read_text(), md_path.read_text()) == first
    json_path.unlink()
    with pytest.raises(FileExistsError):  # the report alone is enough...
        write_outputs(json_path, md_path, other)
    assert not json_path.exists()  # ...and nothing was written before refusing
    md_path.unlink()
    json_path.write_text("kept")
    with pytest.raises(FileExistsError):
        write_outputs(json_path, md_path, other)
    assert json_path.read_text() == "kept" and not md_path.exists()


def test_a_file_created_after_the_check_is_still_not_overwritten(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The open itself must refuse too, for a file that appears between the check and the write."""
    doc = {"label": "", "seeds": [], "fleet": {"replicas": 1}, "variant_labels": [], "runs": []}
    doc |= {"workload": {"num_requests": 1, "rate": 1.0}, "notes": []}
    for existing in ("a.json", "a.md"):
        for stale in tmp_path.iterdir():
            stale.unlink()
        (tmp_path / existing).write_text("kept")
        monkeypatch.setattr(Path, "exists", lambda self: False)  # the check is fooled
        with pytest.raises(FileExistsError):
            write_outputs(tmp_path / "a.json", tmp_path / "a.md", doc)
        assert (tmp_path / existing).read_text() == "kept"


def test_the_replica_command_carries_the_fleet_settings() -> None:
    fleet = small_fleet(max_batch_size=5, kv_cache_gb=0.25, time_scale=0.5)
    command = replica_command(1234, fleet)

    def value(flag: str) -> str:
        return command[command.index(flag) + 1]

    assert value("--backend") == "mock" and value("--port") == "1234"
    assert value("--max-batch-size") == "5" and value("--block-size") == str(BLOCK)
    assert value("--kv-cache-gb") == "0.25" and value("--mock-time-scale") == "0.5"


def test_the_result_files_name_no_machine_address_or_account(tmp_path: Path) -> None:
    out = tmp_path / "secret-account-dir" / "cmp.json"
    assert run_main(out, "--seeds", "0") == 0
    for path in (out, out.with_suffix(".md")):
        text = path.read_text()
        for leak in ("127.0.0.1", "localhost", "http://", "secret-account-dir", str(Path.home())):
            assert leak not in text, f"{leak!r} in {path.name}"
    assert "--out <redacted>" in json.loads(out.read_text())["command"]


def test_sanitize_argv_hides_the_output_path_and_the_home_directory() -> None:
    home = str(Path.home())
    clean = sanitize_argv(["--out", f"{home}/r.json", "--label", f"{home}/x", "--out=/a/b"])
    assert clean == ["--out", "<redacted>", "--label", "~/x", "--out=<redacted>"]


def test_a_run_that_failed_makes_the_exit_code_nonzero_but_is_still_written(
    tmp_path: Path,
) -> None:
    @contextmanager
    def flaky(variant: Variant, fleet: FleetConfig) -> Iterator[FleetEndpoints]:
        if variant.max_entries is not None:
            raise FleetError("no")
        with thread_fleet(variant, fleet) as endpoints:
            yield endpoints

    out = tmp_path / "cmp.json"
    assert run_main(out, fleet=flaky) == 1
    doc = json.loads(out.read_text())
    assert sorted({r["status"] for r in doc["runs"]}) == ["fleet_error", "ok"]
    assert "(+2 not ok)" in out.with_suffix(".md").read_text()


def test_the_table_reports_each_figure_as_mean_and_range_over_seeds() -> None:
    def run(seed: int, variant: str, hit_tokens: int, ttft: float) -> dict[str, Any]:
        summary = summarize_run(
            [request(0, ttft), request(1, ttft)],
            report(tokens=((100, hit_tokens), (100, hit_tokens))),
            Variant("approx-prefix"),
            small_fleet(),
        )
        return {
            "seed": seed, "variant": variant, "policy": "approx-prefix", "status": "ok",
            "summary": summary,
        }  # fmt: skip

    doc = {
        "label": "t",
        "seeds": [0, 1],
        "fleet": {"replicas": 2},
        "workload": {"num_requests": 2, "rate": 1.0},
        "variant_labels": ["approx-prefix"],
        "notes": [],
        "runs": [run(0, "approx-prefix", 20, 0.1), run(1, "approx-prefix", 60, 0.3)],
    }
    row = next(
        line for line in format_table(doc).splitlines() if line.startswith("| approx-prefix")
    )
    cells = [c.strip() for c in row.strip("|").split("|")]
    assert cells[1] == "2"
    assert cells[2] == "40.0 (20.0 to 60.0)"  # hit rate %, mean of 20% and 60%
    assert cells[3] == "200 (100 to 300)"  # TTFT p50 in ms
    assert cells[-2] == "0" and cells[-1] == "2"  # no failed requests; one fallback per run


# -- real processes ------------------------------------------------------------------------------


def test_the_real_fleet_of_processes_runs_a_variant_with_its_memory_limit() -> None:
    """Replica and router as separate processes started by the real command lines. Slower (each
    process imports torch), so this is the one test of that path."""
    fleet = small_fleet()
    variant = Variant("approx-prefix", 5)
    workload = small_workload(num_requests=12)
    specs = generate_workload(workload)

    def start(v: Variant, f: FleetConfig) -> Any:
        return subprocess_fleet(v, f, startup_timeout_s=120)

    entry = run_variant(start, variant, fleet, specs, timeout_s=60)
    assert entry["status"] == "ok", entry
    summary = entry["summary"]
    assert summary["ok"] == 12 and sum(r["served"] for r in summary["replicas"]) == 12
    stats = summary["policy_stats"]
    # The router really received --max-entries: it remembers at most that many chunks.
    assert 0 < stats["entries"] <= 5
