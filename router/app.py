"""A streaming proxy in front of engine replicas, choosing the replica with a routing policy.

It speaks the same ``/v1/completions`` as the replicas and relays their response bytes unchanged,
so the load generator can drive it exactly as it drives one engine. Per request it:

1. reads the prompt (a list of token ids) from the body and asks the policy for a replica;
2. opens the request to that replica and streams the answer back as it arrives;
3. counts the request as in flight on that replica until the stream ends, however it ends.

If a replica cannot be reached before the response starts, the router marks it down for a cooldown
and tries another. If the client disconnects, the upstream connection is closed, which the engine
treats as an abort and frees the request's blocks. A failure *during* a stream cannot be retried
(tokens may already have gone to the client) and is passed on as a broken stream.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from typing import Any

import anyio
import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from router.policies import Policy, PrecisePrefix, ReplicaState


def create_router_app(
    replica_urls: Sequence[str],
    policy: Policy,
    *,
    poll_interval_s: float = 0.5,
    cooldown_s: float = 5.0,
    poll_cache: bool | None = None,
) -> FastAPI:
    """Build the router.

    ``poll_cache`` makes the router fetch every replica's ``/cache`` each ``poll_interval_s``; it
    defaults to on for the precise policy, which scores replicas by it, and off otherwise.
    """
    if not replica_urls:
        raise ValueError("at least one replica is needed")
    if poll_cache is None:
        poll_cache = isinstance(policy, PrecisePrefix)
    replicas = [ReplicaState(url=url.rstrip("/")) for url in replica_urls]
    down_until = [0.0] * len(replicas)
    counters = {"failovers": 0, "upstream_errors": 0, "no_replica": 0, "bad_requests": 0}
    state: dict[str, Any] = {}

    def refresh_health() -> None:
        now = time.monotonic()
        for index, replica in enumerate(replicas):
            replica.healthy = now >= down_until[index]

    async def poll_once(client: httpx.AsyncClient, index: int) -> None:
        replica = replicas[index]
        try:
            response = await client.get(f"{replica.url}/cache", timeout=2.0)
            body = response.json()
            replica.cached = frozenset(int(h, 16) for h in body["hashes"])
            replica.block_size = int(body["block_size"])
        except (httpx.HTTPError, ValueError, KeyError):
            pass  # keep the last report; a replica that is really down fails real requests

    async def poll_loop(client: httpx.AsyncClient) -> None:
        while True:
            await asyncio.gather(*(poll_once(client, i) for i in range(len(replicas))))
            await asyncio.sleep(poll_interval_s)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        timeout = httpx.Timeout(connect=5.0, read=None, write=30.0, pool=None)
        client = httpx.AsyncClient(timeout=timeout, limits=httpx.Limits(max_connections=None))
        state["client"] = client
        poller = asyncio.create_task(poll_loop(client)) if poll_cache else None
        try:
            yield
        finally:
            if poller is not None:
                poller.cancel()
            await client.aclose()

    app = FastAPI(title="pagedserve router", lifespan=lifespan)

    @app.get("/health")
    async def health() -> Response:
        refresh_health()
        if any(r.healthy for r in replicas):
            return JSONResponse({"status": "ok"})
        return JSONResponse({"status": "no healthy replica"}, status_code=503)

    @app.get("/info")
    async def info(fleet: bool = False) -> dict[str, Any]:
        refresh_health()
        result: dict[str, Any] = {
            "server": "pagedserve-router",
            "policy": policy.name,
            "policy_stats": policy.stats(),
            **counters,
            "replicas": [
                {
                    "url": r.url,
                    "outstanding": r.outstanding,
                    "served": r.served,
                    "healthy": r.healthy,
                    "cached_blocks": len(r.cached),
                    "block_size": r.block_size,
                }
                for r in replicas
            ],
        }
        if fleet:
            result["fleet"] = await _fleet(state["client"], replicas)
        return result

    @app.post("/v1/completions", response_model=None)
    async def completions(request: Request) -> Response:
        raw = await request.body()
        try:
            prompt = json.loads(raw)["prompt"]
            if not (
                isinstance(prompt, list)
                and prompt
                and all(isinstance(t, int) and not isinstance(t, bool) for t in prompt)
            ):
                raise ValueError
        except (ValueError, KeyError, TypeError):
            counters["bad_requests"] += 1
            return JSONResponse(
                {"error": {"message": "prompt must be a non-empty list of token ids"}},
                status_code=400,
            )

        client: httpx.AsyncClient = state["client"]
        tried: set[int] = set()
        while True:
            refresh_health()
            candidates = [i for i, r in enumerate(replicas) if r.healthy and i not in tried]
            if not candidates:
                counters["no_replica"] += 1
                return JSONResponse(
                    {"error": {"message": "no healthy replica available"}}, status_code=503
                )
            index = policy.route(prompt, replicas, candidates)
            replica = replicas[index]
            replica.outstanding += 1
            replica.served += 1
            try:
                upstream = await client.send(
                    client.build_request(
                        "POST",
                        f"{replica.url}/v1/completions",
                        content=raw,
                        headers={"content-type": "application/json"},
                    ),
                    stream=True,
                )
            except httpx.TransportError:
                replica.outstanding -= 1
                down_until[index] = time.monotonic() + cooldown_s
                tried.add(index)
                counters["failovers"] += 1
                continue
            break

        return StreamingResponse(
            _relay(upstream, replica, counters),
            status_code=upstream.status_code,
            media_type=upstream.headers.get("content-type"),
        )

    return app


async def _relay(
    upstream: httpx.Response, replica: ReplicaState, counters: dict[str, int]
) -> AsyncIterator[bytes]:
    try:
        async for chunk in upstream.aiter_raw():
            yield chunk
    except httpx.HTTPError:
        counters["upstream_errors"] += 1  # the client sees a stream that ends without [DONE]
    finally:
        replica.outstanding -= 1
        with anyio.CancelScope(shield=True):  # still runs if the client disconnected
            await upstream.aclose()


async def _fleet(client: httpx.AsyncClient, replicas: Sequence[ReplicaState]) -> dict[str, Any]:
    """Each replica's own ``/info``, plus totals of the prefix-cache counters across the fleet."""

    async def one(replica: ReplicaState) -> tuple[str, dict[str, Any]]:
        try:
            response = await client.get(f"{replica.url}/info", timeout=3.0)
            return replica.url, dict(response.json())
        except (httpx.HTTPError, ValueError) as error:
            return replica.url, {"error": type(error).__name__}

    infos = dict(await asyncio.gather(*(one(r) for r in replicas)))
    hits = prompts = 0
    for info in infos.values():
        stats = info.get("stats", {})
        hits += int(stats.get("prefix_hit_tokens", 0))
        prompts += int(stats.get("prefix_prompt_tokens", 0))
    return {
        "replicas": infos,
        "prefix_hit_tokens": hits,
        "prefix_prompt_tokens": prompts,
        "prefix_hit_rate": hits / prompts if prompts else 0.0,
    }
