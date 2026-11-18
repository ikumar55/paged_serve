"""In-process fake of an OpenAI-style streaming completions server, for load-generator tests."""

from __future__ import annotations

import asyncio
import json
import re
import threading
from collections.abc import AsyncIterator, Callable
from typing import Any

import httpx

Item = tuple[float, bytes]  # (seconds to wait before sending, bytes to send)


def sse(payload: dict[str, Any]) -> bytes:
    return b"data: " + json.dumps(payload).encode() + b"\n\n"


DONE = b"data: [DONE]\n\n"


def token_chunk(text: str = "x") -> bytes:
    return sse({"choices": [{"text": text, "finish_reason": None}]})


def closing_chunk() -> bytes:
    return sse({"choices": [{"text": "", "finish_reason": "length"}]})


def usage_chunk(completion_tokens: int) -> bytes:
    return sse({"choices": [], "usage": {"completion_tokens": completion_tokens}})


def well_formed_stream(tokens: int, delay: float = 0.0, *, usage: bool = True) -> list[Item]:
    items: list[Item] = [(delay, token_chunk()) for _ in range(tokens)]
    items.append((0.0, closing_chunk()))
    if usage:
        items.append((0.0, usage_chunk(tokens)))
    items.append((0.0, DONE))
    return items


async def _stream(items: list[Item]) -> AsyncIterator[bytes]:
    for delay, data in items:
        if delay:
            await asyncio.sleep(delay)
        yield data


def streaming_response(items: list[Item], status: int = 200) -> httpx.Response:
    return httpx.Response(
        status, headers={"content-type": "text/event-stream"}, content=_stream(items)
    )


def echo_handler(delay: float = 0.0) -> Callable[[httpx.Request], httpx.Response]:
    """Streams exactly ``max_tokens`` tokens for whatever each request asks."""

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        return streaming_response(well_formed_stream(body["max_tokens"], delay))

    return handler


def mock_client(handler: Callable[[httpx.Request], httpx.Response]) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://test")


class SseServer:
    """A real HTTP server on localhost streaming chunked SSE, run in a background thread.

    Unlike ``httpx.MockTransport`` this exercises real sockets, chunked transfer encoding and
    line buffering. Each request waits ``first_token_delay`` (simulated prefill), then emits
    ``max_tokens`` tokens ``token_delay`` apart.
    """

    def __init__(self, first_token_delay: float, token_delay: float) -> None:
        self.first_token_delay = first_token_delay
        self.token_delay = token_delay
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self.loop.run_forever, daemon=True)
        self.url = ""

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            header = await reader.readuntil(b"\r\n\r\n")
            match = re.search(rb"content-length:\s*(\d+)", header, re.IGNORECASE)
            assert match is not None
            body = json.loads(await reader.readexactly(int(match.group(1))))
            writer.write(
                b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n"
                b"Transfer-Encoding: chunked\r\nConnection: close\r\n\r\n"
            )

            async def send(data: bytes) -> None:
                writer.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")
                await writer.drain()

            count = body["max_tokens"]
            await asyncio.sleep(self.first_token_delay)
            for index in range(count):
                if index:
                    await asyncio.sleep(self.token_delay)
                await send(token_chunk())
            await send(closing_chunk())
            await send(usage_chunk(count))
            await send(DONE)
            writer.write(b"0\r\n\r\n")
            await writer.drain()
        finally:
            writer.close()

    def __enter__(self) -> str:
        self.thread.start()
        server = asyncio.run_coroutine_threadsafe(
            asyncio.start_server(self._handle, "127.0.0.1", 0), self.loop
        ).result(timeout=5)
        port = server.sockets[0].getsockname()[1]
        self.url = f"http://127.0.0.1:{port}"
        return self.url

    def __exit__(self, *exc_info: object) -> None:
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join(timeout=5)
