"""The router proxy in front of mock engine replicas, over real sockets."""

import json
import socket
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Any

import httpx
import pytest
import torch

from bench.loadgen import main as loadgen_main
from bench.results_io import read_results
from engine import server
from engine.mock import MockModel, TimingModel, mock_next_token
from engine.paged_cache import KVPool
from engine.scheduler import Scheduler
from router import server as router_server
from router.app import create_router_app
from router.policies import Policy, make_policy
from tests.fake_server import UvicornServer

BLOCK = 4
INSTANT = TimingModel(time_scale=0)
SLOW = TimingModel(prefill_base_ms=0, prefill_ms_per_token=0, decode_base_ms=25, time_scale=1.0)


_CLIENT: list[httpx.Client] = []


@contextmanager
def use_client() -> Iterator[httpx.Client]:
    """One shared HTTP client for a scenario. A fresh ``httpx.post`` builds a client every call,
    which costs about 0.2 s on Windows and would make timing-based tests unreliable."""
    shared = httpx.Client(timeout=30)
    _CLIENT.append(shared)
    try:
        yield shared
    finally:
        _CLIENT.pop()
        shared.close()


def client() -> httpx.Client:
    return _CLIENT[-1]


def prompt_of(length: int, seed: int) -> list[int]:
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(1000, 100000, (length,), generator=generator).tolist()


def expected_tokens(prompt: list[int], count: int) -> list[int]:
    tokens = list(prompt)
    for _ in range(count):
        tokens.append(mock_next_token(tokens[-1], len(tokens) - 1))
    return tokens[len(prompt) :]


def make_replica(timing: TimingModel = INSTANT, num_blocks: int = 300) -> tuple[Any, Scheduler]:
    model = MockModel(timing=timing)
    pool = KVPool(model.cfg, num_blocks, BLOCK, torch.float32, "cpu")
    scheduler = Scheduler(model, pool, max_batch_size=8)
    return server.create_app(scheduler, [], model_id="mock"), scheduler


@contextmanager
def running(
    policy: Policy | str,
    timings: list[TimingModel] | None = None,
    **router_kwargs: Any,
) -> Iterator[tuple[str, list[str], list[Scheduler]]]:
    """Replicas and a router in front of them; yields the router URL, replica URLs, schedulers."""
    timings = timings or [INSTANT, INSTANT]
    if isinstance(policy, str):
        policy = make_policy(policy, chunk_tokens=BLOCK, load_cap=router_kwargs.pop("load_cap", 16))
    router_kwargs.setdefault("poll_interval_s", 0.05)
    with ExitStack() as stack:
        stack.enter_context(use_client())
        urls, schedulers = [], []
        for timing in timings:
            app, scheduler = make_replica(timing)
            urls.append(stack.enter_context(UvicornServer(app)))
            schedulers.append(scheduler)
        router_url = stack.enter_context(
            UvicornServer(create_router_app(urls, policy, **router_kwargs))
        )
        yield router_url, urls, schedulers


def finished(url: str) -> int:
    return int(client().get(f"{url}/info").json()["stats"]["requests_finished"])


def wait_until(condition: Callable[[], bool], timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.02)
    return False


def complete(router: str, prompt: list[int], tokens: int = 4) -> dict[str, Any]:
    response = client().post(
        f"{router}/v1/completions", json={"prompt": prompt, "max_tokens": tokens}, timeout=30
    )
    assert response.status_code == 200
    return dict(response.json())


def test_a_response_through_the_router_is_exactly_what_the_replica_produces() -> None:
    prompt = prompt_of(11, 1)
    with running("round-robin") as (router, _urls, _):
        body = complete(router, prompt, 9)
        with client().stream(
            "POST",
            f"{router}/v1/completions",
            json={
                "prompt": prompt, "max_tokens": 6, "stream": True,
                "stream_options": {"include_usage": True},
            },
        ) as response:  # fmt: skip
            events = [ln[6:] for ln in response.iter_lines() if ln.startswith("data: ")]
            content_type = response.headers["content-type"]
    assert body["choices"][0]["token_ids"] == expected_tokens(prompt, 9)
    assert body["usage"] == {"prompt_tokens": 11, "completion_tokens": 9, "total_tokens": 20}
    assert content_type.startswith("text/event-stream")
    assert events[-1] == "[DONE]" and json.loads(events[-2])["usage"]["completion_tokens"] == 6
    streamed = [json.loads(e)["choices"][0].get("token_ids") for e in events[:-3]]
    assert [t[0] for t in streamed] == expected_tokens(prompt, 6)


def test_round_robin_alternates_between_replicas() -> None:
    with running("round-robin") as (router, urls, _):
        for index in range(6):
            complete(router, prompt_of(8, index))
        assert [finished(u) for u in urls] == [3, 3]
        info = client().get(f"{router}/info").json()
    assert info["policy"] == "round-robin" and [r["served"] for r in info["replicas"]] == [3, 3]
    assert all(r["outstanding"] == 0 and r["healthy"] for r in info["replicas"])


def test_least_outstanding_keeps_away_from_a_busy_replica() -> None:
    with running("least-outstanding", timings=[SLOW, INSTANT]) as (router, urls, _):
        long_request = threading.Thread(
            target=complete, args=(router, prompt_of(8, 1), 40), daemon=True
        )
        long_request.start()  # the first request goes to replica 0 (a tie) and takes about a second
        assert wait_until(
            lambda: client().get(f"{router}/info").json()["replicas"][0]["outstanding"] == 1
        )
        for index in range(4):
            complete(router, prompt_of(8, 10 + index))
        assert finished(urls[1]) == 4 and finished(urls[0]) == 0  # replica 0 was still busy
        long_request.join(timeout=10)
        assert finished(urls[0]) == 1


def shared_prefix_workload() -> list[list[int]]:
    """Twelve prompts in three groups of four, each group sharing a 16-token prefix.

    Three groups over two replicas on purpose: with four, round robin would send groups 0 and 2 to
    one replica and 1 and 3 to the other, keeping every group together by accident.
    """
    prefixes = [prompt_of(16, 100 + g) for g in range(3)]
    return [[*prefixes[i % 3], *prompt_of(3, 200 + i)] for i in range(12)]


def fleet_hit_rate(policy: str) -> tuple[float, list[int]]:
    with running(policy) as (router, _urls, _):
        for prompt in shared_prefix_workload():
            complete(router, prompt, 2)
        info = client().get(f"{router}/info", params={"fleet": "true"}).json()
        return float(info["fleet"]["prefix_hit_rate"]), [r["served"] for r in info["replicas"]]


def test_sending_a_prefix_back_to_its_holder_raises_the_fleets_hit_rate() -> None:
    """Round robin splits each prefix group over both replicas, so each must compute it."""
    round_robin, _ = fleet_hit_rate("round-robin")
    approximate, served = fleet_hit_rate("approx-prefix")
    assert approximate > round_robin  # round robin splits each group over both replicas
    # Nine of twelve prompts can reuse 16 of their 19 tokens when every group stays together.
    assert approximate == pytest.approx(9 * 16 / (12 * 19))
    assert round_robin == pytest.approx(6 * 16 / (12 * 19))  # each group computed on both replicas
    assert sum(served) == 12


def test_approximate_routing_sends_a_prefix_group_to_one_replica() -> None:
    prompts = shared_prefix_workload()
    with running("approx-prefix") as (router, _urls, _schedulers):
        for prompt in prompts:
            complete(router, prompt, 2)
        info = client().get(f"{router}/info").json()
    stats = info["policy_stats"]
    assert stats["decisions"] == 12 and stats["no_match"] == 3 and stats["matched"] == 9
    # Each of the three groups was first seen once, then found again three times.
    assert stats["matched_chunks"] == 9 * 4  # a 16-token prefix is four 4-token chunks


def test_precise_routing_uses_what_a_replica_actually_holds() -> None:
    prefix = prompt_of(16, 7)
    warm, probe = [*prefix, 1, 2, 3], [*prefix, 4, 5, 6]
    with running("precise-prefix") as (router, urls, schedulers):
        # Warm replica 1 directly, so the router has never seen this prefix go anywhere.
        client().post(
            f"{urls[1]}/v1/completions", json={"prompt": warm, "max_tokens": 2}, timeout=30
        )
        assert wait_until(
            lambda: client().get(f"{router}/info").json()["replicas"][1]["cached_blocks"] >= 4
        )
        complete(router, probe, 2)
        info = client().get(f"{router}/info").json()
        hits = [s.stats()["prefix_hit_tokens"] for s in schedulers]
    assert [r["served"] for r in info["replicas"]] == [0, 1]  # round robin would start at 0
    assert info["policy_stats"]["matched"] == 1 and info["policy_stats"]["matched_chunks"] >= 4
    assert hits == [0, 16]  # and replica 1 really did reuse the 16 shared tokens


def test_an_approximate_router_cannot_know_about_blocks_it_did_not_route() -> None:
    prefix = prompt_of(16, 7)
    with running("approx-prefix") as (router, urls, _):
        client().post(f"{urls[1]}/v1/completions", json={"prompt": [*prefix, 1], "max_tokens": 2})
        complete(router, [*prefix, 4], 2)
        info = client().get(f"{router}/info").json()
    assert info["policy_stats"]["no_match"] == 1  # it guessed, with nothing to guess from


def test_a_replica_over_the_load_cap_is_not_preferred() -> None:
    prefix = prompt_of(16, 9)
    with running("approx-prefix", timings=[SLOW, SLOW], load_cap=1) as (router, _urls, _):
        first = threading.Thread(target=complete, args=(router, [*prefix, 1], 30), daemon=True)
        first.start()
        assert wait_until(
            lambda: (
                sum(r["outstanding"] for r in client().get(f"{router}/info").json()["replicas"])
                == 1
            )
        )
        busy = next(
            i
            for i, r in enumerate(client().get(f"{router}/info").json()["replicas"])
            if r["outstanding"]
        )
        complete(router, [*prefix, 2], 2)  # shares the prefix, but its holder is at the cap
        info = client().get(f"{router}/info").json()
        first.join(timeout=10)
    assert info["policy_stats"]["load_fallbacks"] == 1
    assert info["replicas"][1 - busy]["served"] == 1


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def test_an_unreachable_replica_is_skipped_and_marked_down() -> None:
    dead = f"http://127.0.0.1:{free_port()}"
    app, _ = make_replica()
    with use_client(), UvicornServer(app) as live:
        router_app = create_router_app([dead, live], make_policy("round-robin"), cooldown_s=30)
        with UvicornServer(router_app) as router:
            bodies = [complete(router, prompt_of(6, i), 3) for i in range(4)]
            info = client().get(f"{router}/info").json()
            health = client().get(f"{router}/health")
    assert all(b["choices"][0]["finish_reason"] == "length" for b in bodies)
    assert [r["healthy"] for r in info["replicas"]] == [False, True]
    assert info["failovers"] == 1  # found out once; later requests did not even try it
    assert health.status_code == 200


def test_with_every_replica_down_the_router_says_503() -> None:
    dead = [f"http://127.0.0.1:{free_port()}", f"http://127.0.0.1:{free_port()}"]
    router_app = create_router_app(dead, make_policy("round-robin"))
    with use_client(), UvicornServer(router_app) as router:
        response = client().post(
            f"{router}/v1/completions", json={"prompt": [1, 2], "max_tokens": 2}
        )
        health = client().get(f"{router}/health")
    assert response.status_code == 503 and "no healthy replica" in response.text
    assert health.status_code == 503


def test_a_client_that_disconnects_mid_stream_frees_the_replica() -> None:
    with running("round-robin", timings=[SLOW, SLOW]) as (router, _urls, schedulers):
        body = {"prompt": prompt_of(8, 3), "max_tokens": 200, "stream": True, "ignore_eos": True}
        with (
            httpx.Client(timeout=30) as streamer,  # its own client: leaving closes the connection
            streamer.stream("POST", f"{router}/v1/completions", json=body) as response,
        ):
            lines = response.iter_lines()
            for _ in range(3):
                next(line for line in lines if line.startswith("data: "))
        assert wait_until(lambda: sum(s.aborted for s in schedulers) == 1, timeout=3.0)
        assert all(s.pool.allocator.num_used == 0 for s in schedulers)
        assert wait_until(
            lambda: all(
                r["outstanding"] == 0 for r in client().get(f"{router}/info").json()["replicas"]
            )
        )
        assert sum(s.steps for s in schedulers) < 100  # it was cut short, not run to the end


def test_bad_requests_are_rejected_by_the_router_and_replica_errors_pass_through() -> None:
    with running("round-robin") as (router, urls, _):
        for bad in ({"max_tokens": 3}, {"prompt": "text"}, {"prompt": []}, {"prompt": [1, True]}):
            response = client().post(f"{router}/v1/completions", json=bad)
            assert response.status_code == 400 and "list of token ids" in response.text
        assert client().post(f"{router}/v1/completions", content=b"not json").status_code == 400
        assert sum(finished(u) for u in urls) == 0  # nothing was forwarded
        # A request the replica itself refuses comes back with the replica's status and message.
        out_of_range = client().post(
            f"{router}/v1/completions", json={"prompt": [999999999], "max_tokens": 2}
        )
        assert out_of_range.status_code == 400 and "token ids" in out_of_range.text
        info = client().get(f"{router}/info").json()
    assert info["bad_requests"] == 5
    assert all(
        r["outstanding"] == 0 for r in info["replicas"]
    )  # a rejected request is not left counted


def test_the_load_generator_can_drive_the_router(tmp_path: Path) -> None:
    out = tmp_path / "via-router.jsonl"
    argv = [
        "run", "--model", "mock", "--rate", "100", "--num-requests", "16", "--warmup", "0",
        "--vocab-size", "100000", "--min-token-id", "1000",
        "--prompt-median", "6", "--prompt-sigma", "0.3", "--prompt-max", "10",
        "--output-median", "4", "--output-sigma", "0.2", "--output-max", "6",
        "--shared-prefix-groups", "3", "--shared-prefix-len", "16", "--out", str(out),
    ]  # fmt: skip
    with running("approx-prefix") as (router, urls, _):
        assert loadgen_main([*argv, "--url", router]) == 0
        served = [finished(u) for u in urls]
    _, results, summary = read_results(out)
    assert len(results) == 16 and all(r.status == "ok" for r in results)
    assert summary is not None and summary["failure_counts"] == {}
    assert sum(served) == 16


def test_main_builds_the_router_from_the_command_line(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}
    monkeypatch.setattr(
        router_server.uvicorn, "run", lambda app, **kw: captured.update(app=app, **kw)
    )
    code = router_server.main(
        [
            "--replica",
            "http://a:1",
            "--replica",
            "http://b:2",
            "--policy",
            "round-robin",
            "--port",
            "7",
        ]
    )
    assert code == 0 and captured["port"] == 7 and captured["host"] == "127.0.0.1"
    with use_client(), UvicornServer(captured["app"]) as url:
        info = client().get(f"{url}/info").json()
    assert info["policy"] == "round-robin"
    assert [r["url"] for r in info["replicas"]] == ["http://a:1", "http://b:2"]

    with pytest.raises(SystemExit):
        router_server.main(["--replica", "a:1"])  # not an http(s) URL
    with pytest.raises(SystemExit):
        router_server.main(["--policy", "round-robin"])  # no replica


def test_a_router_needs_at_least_one_replica() -> None:
    with pytest.raises(ValueError):
        create_router_app([], make_policy("round-robin"))
