"""OpenAI-style streaming ``/v1/completions`` over the paged engine and its scheduler.

    python -m engine.server --model-dir models/Llama-3.2-1B-Instruct --dtype bfloat16 --port 8000

It speaks the same API as the Hugging Face baseline (``bench.baselines.hf_server``), so the load
generator drives both unchanged: greedy decoding, token-id prompts, ``ignore_eos``, ``stream`` and
``stream_options.include_usage``. As in the baseline, token events carry the id as ``<id>`` and in
``token_ids`` (no detokenization), and an end-of-sequence token ends the stream without being
emitted.

One thread owns the scheduler and the model and runs it in a loop; the HTTP layer talks to it
through a queue, so a forward pass never blocks the event loop and the scheduler needs no locks.
"""

from __future__ import annotations

import argparse
import asyncio
import itertools
import json
import queue
import threading
import time
import uuid
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from engine.paged_cache import KVPool
from engine.scheduler import Scheduler, StepResult
from engine.weights import load_model

DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}


@dataclass
class Stream:
    """One request as the HTTP layer sees it: events arrive on ``events``.

    An event is ``("token", id)``, ``("done", finish_reason)`` or ``("error", message)``.
    """

    request_id: int
    prompt_len: int
    max_tokens: int
    ignore_eos: bool
    loop: asyncio.AbstractEventLoop
    events: asyncio.Queue[tuple[str, Any]] = field(default_factory=asyncio.Queue)
    emitted: int = 0
    done: bool = False

    def push(self, kind: str, payload: Any) -> None:
        self.loop.call_soon_threadsafe(self.events.put_nowait, (kind, payload))


class EngineRunner:
    """Runs the scheduler on its own thread and routes generated tokens to request streams."""

    def __init__(self, scheduler: Scheduler, eos_ids: Sequence[int]) -> None:
        self.scheduler = scheduler
        self.eos_ids = list(eos_ids)
        self._inbox: queue.Queue[tuple[str, Any]] = queue.Queue()
        self._ids = itertools.count()
        self._streams: dict[int, Stream] = {}  # touched only by the engine thread
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="engine", daemon=True)
        self.error: str | None = None
        self.tokens_generated = 0
        self.requests_finished = 0

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=10)

    def submit(
        self, prompt: list[int], max_tokens: int, ignore_eos: bool, loop: asyncio.AbstractEventLoop
    ) -> Stream:
        stream = Stream(next(self._ids), len(prompt), max_tokens, ignore_eos, loop)
        self._inbox.put(("add", (stream, prompt)))
        return stream

    def abort(self, stream: Stream) -> None:
        if not stream.done:
            self._inbox.put(("abort", stream.request_id))

    # -- engine thread ---------------------------------------------------------------------

    def _loop(self) -> None:
        try:
            while not self._stop.is_set():
                self._drain_inbox(block=not self.scheduler.has_work)
                if self.scheduler.has_work:
                    self._route(self.scheduler.step())
        except Exception as error:  # a broken engine must not leave clients hanging
            self.error = f"{type(error).__name__}: {error}"
            for stream in self._streams.values():
                stream.push("error", self.error)
            self._streams.clear()
            # Requests still in the inbox are failed by the HTTP layer once it sees ``error``.
            while True:
                try:
                    kind, payload = self._inbox.get_nowait()
                except queue.Empty:
                    break
                if kind == "add":
                    payload[0].push("error", self.error)

    def _drain_inbox(self, block: bool) -> None:
        try:
            item = self._inbox.get(timeout=0.05) if block else self._inbox.get_nowait()
        except queue.Empty:
            return
        while True:
            self._handle(*item)
            try:
                item = self._inbox.get_nowait()
            except queue.Empty:
                return

    def _handle(self, kind: str, payload: Any) -> None:
        if kind == "abort":
            stream = self._streams.pop(payload, None)
            if stream is not None:
                self.scheduler.abort(payload)
                stream.done = True
            return
        stream, prompt = payload
        eos = [] if stream.ignore_eos else self.eos_ids
        try:
            self.scheduler.add_request(stream.request_id, prompt, stream.max_tokens, eos)
        except ValueError as error:  # the HTTP layer validates first, so this is a safety net
            stream.push("error", str(error))
            return
        self._streams[stream.request_id] = stream

    def _route(self, result: StepResult) -> None:
        for request_id, tokens in result.new_tokens.items():
            stream = self._streams.get(request_id)
            if stream is None:
                continue
            reason = result.finish_reasons.get(request_id)
            if reason == "stop":
                tokens = tokens[:-1]  # the end-of-sequence token itself is not emitted
            for token in tokens:
                stream.push("token", token)
                stream.emitted += 1
            self.tokens_generated += len(tokens)
            if reason is not None:
                stream.push("done", reason)
                stream.done = True
                del self._streams[request_id]
                self.requests_finished += 1


class CompletionRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    model: str = ""
    prompt: list[int] = Field(min_length=1)
    max_tokens: int = Field(default=16, ge=1)
    temperature: float = 0.0
    stream: bool = False
    stream_options: dict[str, Any] | None = None
    ignore_eos: bool = False
    n: int = Field(default=1, ge=1, le=1)


def _sse(payload: dict[str, Any]) -> bytes:
    return b"data: " + json.dumps(payload).encode() + b"\n\n"


def _chunk(
    completion_id: str, created: int, model: str, token: int | None, finish_reason: str | None
) -> dict[str, Any]:
    choice: dict[str, Any] = {
        "index": 0,
        "text": "" if token is None else f"<{token}>",
        "finish_reason": finish_reason,
    }
    if token is not None:
        choice["token_ids"] = [token]
    return {
        "id": completion_id,
        "object": "text_completion",
        "created": created,
        "model": model,
        "choices": [choice],
    }


def _usage(prompt_tokens: int, completion_tokens: int) -> dict[str, int]:
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


def create_app(
    scheduler: Scheduler,
    eos_ids: Sequence[int],
    *,
    model_id: str,
    info: dict[str, Any] | None = None,
) -> FastAPI:
    runner = EngineRunner(scheduler, eos_ids)
    cfg = scheduler.model.cfg

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        runner.start()
        yield
        runner.stop()

    app = FastAPI(title="pagedserve engine", lifespan=lifespan)
    app.state.runner = runner
    static_info = {
        "server": "pagedserve",
        "model_id": model_id,
        "max_batch_size": scheduler.max_batch_size,
        "block_size": scheduler.pool.block_size,
        **(info or {}),
    }

    @app.get("/health")
    async def health() -> Response:
        if runner.error:
            return JSONResponse({"status": "failed", "error": runner.error}, status_code=503)
        return JSONResponse({"status": "ok"})

    @app.get("/info")
    async def server_info() -> dict[str, Any]:
        stats = {
            **scheduler.stats(),
            "tokens_generated": runner.tokens_generated,
            "requests_finished": runner.requests_finished,
        }
        return {**static_info, "stats": stats}

    @app.post("/v1/completions", response_model=None)
    async def completions(body: CompletionRequest) -> Response:
        if runner.error:
            raise HTTPException(503, f"engine failed: {runner.error}")
        if body.temperature != 0:
            raise HTTPException(400, "greedy decoding only (temperature 0)")
        if min(body.prompt) < 0 or max(body.prompt) >= cfg.vocab_size:
            raise HTTPException(400, f"prompt token ids must be in [0, {cfg.vocab_size})")
        try:
            scheduler.validate(len(body.prompt), body.max_tokens)
        except ValueError as error:
            raise HTTPException(400, str(error)) from error

        stream = runner.submit(
            list(body.prompt), body.max_tokens, body.ignore_eos, asyncio.get_running_loop()
        )
        completion_id = f"cmpl-{uuid.uuid4().hex[:16]}"
        created = int(time.time())
        name = body.model or model_id

        if body.stream:
            include_usage = bool((body.stream_options or {}).get("include_usage"))
            return StreamingResponse(
                _stream(runner, stream, completion_id, created, name, include_usage),
                media_type="text/event-stream",
            )

        tokens: list[int] = []
        try:
            while True:
                kind, payload = await stream.events.get()
                if kind == "token":
                    tokens.append(payload)
                elif kind == "done":
                    break
                else:
                    return JSONResponse({"error": {"message": payload}}, status_code=500)
        finally:
            runner.abort(stream)  # no-op once finished; frees the blocks if the client left
        return JSONResponse(
            {
                "id": completion_id,
                "object": "text_completion",
                "created": created,
                "model": name,
                "choices": [
                    {
                        "index": 0,
                        "text": "".join(f"<{t}>" for t in tokens),
                        "token_ids": tokens,
                        "finish_reason": payload,
                    }
                ],
                "usage": _usage(len(body.prompt), len(tokens)),
            }
        )

    return app


async def _stream(
    runner: EngineRunner,
    stream: Stream,
    completion_id: str,
    created: int,
    model: str,
    include_usage: bool,
) -> AsyncIterator[bytes]:
    try:
        while True:
            kind, payload = await stream.events.get()
            if kind == "token":
                yield _sse(_chunk(completion_id, created, model, payload, None))
            elif kind == "done":
                break
            else:  # cut short without [DONE]; clients treat that as a failure
                yield _sse({"error": {"message": payload}})
                return
        yield _sse(_chunk(completion_id, created, model, None, payload))
        if include_usage:
            usage = _usage(stream.prompt_len, stream.emitted)
            yield _sse({"id": completion_id, "choices": [], "usage": usage})
        yield b"data: [DONE]\n\n"
    finally:
        runner.abort(stream)  # a client that disconnects mid-stream must not keep its blocks


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0] if __doc__ else None)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--model-id", help="name to report in /info (default: directory name)")
    parser.add_argument("--dtype", choices=sorted(DTYPES), default="bfloat16")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-batch-size", type=int, default=32)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument(
        "--kv-cache-gb", type=float, default=4.0, help="memory for the paged KV cache"
    )
    parser.add_argument("--host", default="127.0.0.1", help="bind address (default: local only)")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)

    dtype = DTYPES[args.dtype]
    model, eos_ids = load_model(args.model_dir, dtype, args.device)
    per_token = model.cfg.kv_bytes_per_token(torch.empty((), dtype=dtype).element_size())
    num_blocks = int(args.kv_cache_gb * 2**30 // (per_token * args.block_size))
    if num_blocks < 1:
        raise SystemExit("--kv-cache-gb is too small for even one block")
    pool = KVPool(model.cfg, num_blocks, args.block_size, dtype, args.device)
    scheduler = Scheduler(model, pool, max_batch_size=args.max_batch_size)
    device_name = torch.cuda.get_device_name(model.device) if model.device.type == "cuda" else "cpu"
    app = create_app(
        scheduler,
        eos_ids,
        model_id=args.model_id or args.model_dir.name,
        info={
            "dtype": args.dtype,
            "device": device_name,
            "torch": torch.__version__,
            "cuda_build": torch.version.cuda,
            "kv_blocks": num_blocks,
            "kv_cache_gb": args.kv_cache_gb,
        },
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
