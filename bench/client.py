"""Streaming client for OpenAI-style ``/v1/completions`` endpoints (engine, vLLM, router).

Timing uses ``time.perf_counter``. All times in a result are in seconds; token times are relative
to the moment the request was sent.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any

import httpx

from bench.workload import RequestSpec

COMPLETIONS_PATH = "/v1/completions"


class ProtocolError(Exception):
    """The server answered, but not with a well-formed stream."""


@dataclass
class RequestResult:
    id: int
    scheduled_s: float  # when the workload said to send, relative to run start
    sent_s: float  # when it was actually sent, relative to run start
    prompt_tokens: int
    requested_output_tokens: int
    status: str = "error"  # "ok", "http_error", "timeout" or "error"
    http_status: int | None = None
    error: str | None = None
    output_tokens: int = 0
    tokens_from_usage: bool = False  # False: counted stream events instead of server usage
    ttft_s: float | None = None
    e2e_s: float | None = None  # time until the stream ended or the request failed
    token_times_s: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"type": "request", **asdict(self)}

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> RequestResult:
        data = {k: v for k, v in raw.items() if k != "type"}
        return cls(**data)


def request_body(spec: RequestSpec, model: str, ignore_eos: bool) -> dict[str, Any]:
    return {
        "model": model,
        "prompt": spec.prompt_token_ids,
        "max_tokens": spec.max_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "ignore_eos": ignore_eos,
    }


async def _read_stream(
    client: httpx.AsyncClient,
    body: dict[str, Any],
    result: RequestResult,
    sent: float,
) -> None:
    async with client.stream("POST", COMPLETIONS_PATH, json=body) as response:
        result.http_status = response.status_code
        if response.status_code != 200:
            result.status = "http_error"
            result.error = f"http_{response.status_code}"
            return

        usage_tokens: int | None = None
        saw_done = False
        async for line in response.aiter_lines():
            if not line.startswith("data:"):
                continue
            payload = line[len("data:") :].strip()
            if payload == "[DONE]":
                saw_done = True
                break
            now = time.perf_counter()
            try:
                event = json.loads(payload)
            except json.JSONDecodeError:
                raise ProtocolError("malformed_event") from None
            if not isinstance(event, dict):
                raise ProtocolError("malformed_event")

            usage = event.get("usage")
            if isinstance(usage, dict) and usage.get("completion_tokens") is not None:
                usage_tokens = int(usage["completion_tokens"])
            choices = event.get("choices") or []
            if choices:
                choice = choices[0]
                # A chunk is a token event if it carries text, or is not the closing chunk.
                if choice.get("text") or choice.get("finish_reason") is None:
                    result.token_times_s.append(now - sent)
                    if result.ttft_s is None:
                        result.ttft_s = now - sent

        if not saw_done:
            raise ProtocolError("stream_ended_early")
        if not result.token_times_s:
            raise ProtocolError("no_tokens")
        result.status = "ok"
        result.tokens_from_usage = usage_tokens is not None
        result.output_tokens = (
            usage_tokens if usage_tokens is not None else len(result.token_times_s)
        )


async def execute_request(
    client: httpx.AsyncClient,
    model: str,
    spec: RequestSpec,
    run_start: float,
    *,
    timeout_s: float,
    ignore_eos: bool = True,
) -> RequestResult:
    """Send one request and measure it. Never raises: failures are recorded in the result."""
    sent = time.perf_counter()
    result = RequestResult(
        id=spec.id,
        scheduled_s=spec.arrival_s,
        sent_s=sent - run_start,
        prompt_tokens=len(spec.prompt_token_ids),
        requested_output_tokens=spec.max_tokens,
    )
    body = request_body(spec, model, ignore_eos)
    try:
        await asyncio.wait_for(_read_stream(client, body, result, sent), timeout_s)
    except asyncio.TimeoutError:
        result.status = "timeout"
        result.error = "timeout"
    except httpx.TimeoutException:
        result.status = "timeout"
        result.error = "timeout"
    except ProtocolError as exc:
        result.status = "error"
        result.error = str(exc)
    except httpx.HTTPError as exc:
        # Only the exception type is kept: messages can contain hostnames and URLs.
        result.status = "error"
        result.error = type(exc).__name__
    result.e2e_s = time.perf_counter() - sent
    return result
