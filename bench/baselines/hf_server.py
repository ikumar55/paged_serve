"""Hugging Face ``generate`` behind static batching, served over an OpenAI-style streaming API.

This is baseline (1) of the project: the simplest thing a person would build to serve a model
with the standard library, with batching but without continuous batching or a paged cache.

    python -m bench.baselines.hf_server --model MODEL_ID_OR_PATH --dtype bfloat16 --port 8000

Notes that affect how its numbers should be read:

* Greedy decoding only, token-id prompts only (matching the load generator).
* Token events carry the token id as text (``<id>``) and in ``token_ids``; no detokenization is
  done, which slightly favors this baseline over servers that decode text.
* Tokens stream as each decode step finishes, so time to first token includes waiting for a batch
  slot, the batch window and prefill, but not the rest of the batch.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import time
import uuid
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, cast

import torch
import transformers
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field
from transformers import AutoModelForCausalLM, GenerationConfig, PreTrainedModel
from transformers.generation.streamers import BaseStreamer

from bench.baselines.batching import PendingRequest, StaticBatcher

DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}


class BatchStreamer(BaseStreamer):
    """Receives each decode step's tokens from ``generate`` and routes them to requests."""

    def __init__(self, batch: list[PendingRequest], eos_ids: set[int]) -> None:
        self._batch = batch
        self._eos_ids = eos_ids
        self.steps = 0

    def put(self, value: torch.Tensor) -> None:
        if value.dim() > 1:  # the first call carries the prompt, not generated tokens
            return
        self.steps += 1
        for request, token in zip(self._batch, value.tolist(), strict=True):
            if request.finished:
                continue  # already complete: this slot is padding until the batch ends
            if not request.ignore_eos and token in self._eos_ids:
                request.finish("stop")
            else:
                request.token(int(token))

    def end(self) -> None:
        pass


class HFGenerator:
    """Runs one static batch through ``model.generate`` with left-padded prompts."""

    def __init__(self, model: PreTrainedModel) -> None:
        self.model = model
        eos = model.generation_config.eos_token_id
        self.eos_ids: list[int] = (
            [] if eos is None else [eos] if isinstance(eos, int) else list(eos)
        )
        pad = model.generation_config.pad_token_id
        self.pad_id = int(pad) if pad is not None else (self.eos_ids[0] if self.eos_ids else 0)

    def _generation_config(self, max_new_tokens: int, stop_on_eos: bool) -> GenerationConfig:
        config = copy.deepcopy(self.model.generation_config)
        config.do_sample = False
        config.temperature = None  # sampling settings shipped with chat models are irrelevant
        config.top_p = None
        config.top_k = None
        config.max_new_tokens = max_new_tokens
        # Passing eos_token_id=None does not disable EOS: generate() fills in the model's own
        # default. An id outside the vocabulary can never be produced, so it reliably does.
        never = [int(self.model.config.vocab_size)]
        config.eos_token_id = self.eos_ids if stop_on_eos and self.eos_ids else never
        config.pad_token_id = self.pad_id
        return config

    @torch.inference_mode()
    def __call__(self, batch: list[PendingRequest]) -> int:
        longest = max(len(r.prompt_ids) for r in batch)
        input_ids = torch.full((len(batch), longest), self.pad_id, dtype=torch.long)
        attention_mask = torch.zeros_like(input_ids)
        for row, request in enumerate(batch):
            length = len(request.prompt_ids)
            input_ids[row, longest - length :] = torch.tensor(request.prompt_ids)
            attention_mask[row, longest - length :] = 1

        # Hugging Face stops early only when every sequence has finished, so let it watch for EOS
        # only if every request in the batch wants that. Otherwise the batch runs to its longest
        # request, which is the static-batching cost being measured.
        stop_on_eos = all(not r.ignore_eos for r in batch)
        streamer = BatchStreamer(batch, set(self.eos_ids))
        device = self.model.device
        # ``generate`` comes from a mixin that type checkers do not see on PreTrainedModel.
        cast("Any", self.model).generate(
            input_ids=input_ids.to(device),
            attention_mask=attention_mask.to(device),
            generation_config=self._generation_config(
                max(r.max_tokens for r in batch), stop_on_eos
            ),
            streamer=streamer,
        )
        return streamer.steps


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


def create_app(
    generate_batch: Any,
    *,
    model_id: str,
    vocab_size: int,
    max_positions: int,
    max_batch_size: int,
    batch_window_s: float,
    info: dict[str, Any] | None = None,
) -> FastAPI:
    """Build the HTTP app around any batch generator (see ``StaticBatcher``)."""
    batcher = StaticBatcher(
        generate_batch, max_batch_size=max_batch_size, batch_window_s=batch_window_s
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        batcher.start()
        yield
        batcher.stop()

    app = FastAPI(title="hf-static-batching baseline", lifespan=lifespan)
    app.state.batcher = batcher
    static_info = {
        "server": "hf-static-batching",
        "model_id": model_id,
        "max_batch_size": max_batch_size,
        "batch_window_ms": batch_window_s * 1000,
        **(info or {}),
    }

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/info")
    async def server_info() -> dict[str, Any]:
        return {**static_info, "stats": batcher.stats()}

    @app.post("/v1/completions", response_model=None)
    async def completions(body: CompletionRequest) -> Response:
        if body.temperature != 0:
            raise HTTPException(400, "this baseline supports greedy decoding only (temperature 0)")
        if min(body.prompt) < 0 or max(body.prompt) >= vocab_size:
            raise HTTPException(400, f"prompt token ids must be in [0, {vocab_size})")
        if len(body.prompt) + body.max_tokens > max_positions:
            raise HTTPException(400, f"prompt plus max_tokens exceeds the limit of {max_positions}")

        request = PendingRequest(
            list(body.prompt), body.max_tokens, body.ignore_eos, asyncio.get_running_loop()
        )
        batcher.submit(request)
        completion_id = f"cmpl-{uuid.uuid4().hex[:16]}"
        created = int(time.time())
        name = body.model or model_id

        if body.stream:
            include_usage = bool((body.stream_options or {}).get("include_usage"))
            return StreamingResponse(
                _stream(request, completion_id, created, name, include_usage),
                media_type="text/event-stream",
            )

        tokens: list[int] = []
        while True:
            kind, payload = await request.events.get()
            if kind == "token":
                tokens.append(payload)
            elif kind == "done":
                break
            else:
                return JSONResponse({"error": {"message": payload}}, status_code=500)
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


def _usage(prompt_tokens: int, completion_tokens: int) -> dict[str, int]:
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


async def _stream(
    request: PendingRequest, completion_id: str, created: int, model: str, include_usage: bool
) -> AsyncIterator[bytes]:
    while True:
        kind, payload = await request.events.get()
        if kind == "token":
            yield _sse(_chunk(completion_id, created, model, payload, None))
        elif kind == "done":
            break
        else:  # the stream is cut short without [DONE]; clients treat that as a failure
            yield _sse({"error": {"message": payload}})
            return
    yield _sse(_chunk(completion_id, created, model, None, payload))
    if include_usage:
        usage = _usage(len(request.prompt_ids), request.emitted)
        yield _sse({"id": completion_id, "choices": [], "usage": usage})
    yield b"data: [DONE]\n\n"


def build_app(
    model: PreTrainedModel,
    *,
    model_id: str,
    max_batch_size: int,
    batch_window_s: float,
) -> FastAPI:
    device_name = torch.cuda.get_device_name(model.device) if model.device.type == "cuda" else "cpu"
    return create_app(
        HFGenerator(model),
        model_id=model_id,
        vocab_size=int(model.config.vocab_size),
        max_positions=int(model.config.max_position_embeddings),
        max_batch_size=max_batch_size,
        batch_window_s=batch_window_s,
        info={
            "dtype": str(model.dtype).removeprefix("torch."),
            "device": device_name,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "cuda_build": torch.version.cuda,
        },
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0] if __doc__ else None)
    parser.add_argument("--model", required=True, help="Hugging Face model id or local directory")
    parser.add_argument(
        "--model-id", help="name to report in /info (default: model id or dir name)"
    )
    parser.add_argument("--dtype", choices=sorted(DTYPES), default="bfloat16")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-batch-size", type=int, default=8)
    parser.add_argument("--batch-window-ms", type=float, default=20.0)
    parser.add_argument("--host", default="127.0.0.1", help="bind address (default: local only)")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)

    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=DTYPES[args.dtype])
    model.to(args.device).eval()
    app = build_app(
        model,
        model_id=args.model_id or Path(args.model).name,
        max_batch_size=args.max_batch_size,
        batch_window_s=args.batch_window_ms / 1000,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
