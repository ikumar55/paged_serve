"""A server that does no inference: it answers every completion at once (D-030, step 0).

To find out whether the load generator, and not a system under test, is the limit, it is pointed
at a server whose own speed is not in question. For every ``POST /v1/completions`` this one reads
the request and immediately streams ``max_tokens`` token events, a closing event, a usage event and
``[DONE]``, as one chunked body; nothing is computed and nothing waits. The generator therefore pays
the full cost of opening a connection and parsing every event, with no pacing from a server to hide
behind, which is more than it pays against a real server delivering the same tokens.

On Linux several worker processes can share one port (``SO_REUSEPORT``), so the server itself is not
the limit either:

    python -m bench.null_server --port 18437 --workers 4
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import multiprocessing
import re
import signal
import sys
from collections.abc import Sequence

MAX_TOKENS_LIMIT = 100_000  # a request asking for more is refused, not obeyed


def _event(payload: dict[str, object]) -> bytes:
    return b"data: " + json.dumps(payload).encode() + b"\n\n"


def completion_body(tokens: int) -> bytes:
    """The server-sent events of a completion of ``tokens`` tokens."""
    token = _event({"choices": [{"text": "x", "finish_reason": None}]})
    closing = _event({"choices": [{"text": "", "finish_reason": "length"}]})
    usage = _event({"choices": [], "usage": {"completion_tokens": tokens}})
    return token * tokens + closing + usage + b"data: [DONE]\n\n"


async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    try:
        header = await reader.readuntil(b"\r\n\r\n")
        length = re.search(rb"content-length:\s*(\d+)", header, re.IGNORECASE)
        if length is None:
            writer.write(b"HTTP/1.1 411 Length Required\r\nContent-Length: 0\r\n\r\n")
            return
        request = json.loads(await reader.readexactly(int(length.group(1))))
        tokens = int(request.get("max_tokens", 0))
        if not 1 <= tokens <= MAX_TOKENS_LIMIT:
            writer.write(b"HTTP/1.1 400 Bad Request\r\nContent-Length: 0\r\n\r\n")
            return
        body = completion_body(tokens)
        writer.write(
            b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n"
            b"Transfer-Encoding: chunked\r\nConnection: close\r\n\r\n"
            + f"{len(body):x}\r\n".encode()
            + body
            + b"\r\n0\r\n\r\n"
        )
        await writer.drain()
    except (asyncio.IncompleteReadError, ConnectionError, ValueError):
        pass  # a client that went away or sent nonsense is not this server's problem
    finally:
        writer.close()


async def start(port: int, *, host: str = "127.0.0.1", reuse_port: bool = False) -> asyncio.Server:
    """Listen on ``host:port`` (port 0: any free port, readable from ``server.sockets``)."""
    return await asyncio.start_server(
        handle, host, port, reuse_port=reuse_port or None, backlog=4096
    )


async def _serve_forever(port: int, reuse_port: bool) -> None:
    server = await start(port, reuse_port=reuse_port)
    async with server:
        await server.serve_forever()


def _interrupt(signum: int, frame: object) -> None:
    raise KeyboardInterrupt


def _worker(port: int) -> None:
    asyncio.run(_serve_forever(port, reuse_port=True))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0] if __doc__ else None)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args(argv)
    if args.workers < 1:
        print("error: --workers must be at least 1", file=sys.stderr)
        return 2
    if args.workers > 1 and not hasattr(__import__("socket"), "SO_REUSEPORT"):
        print("error: several workers need SO_REUSEPORT (Linux)", file=sys.stderr)
        return 2
    if args.workers == 1:
        with contextlib.suppress(KeyboardInterrupt):
            asyncio.run(_serve_forever(args.port, reuse_port=False))
        return 0
    context = multiprocessing.get_context("spawn")
    # daemon: the workers die with this process; SIGTERM becomes an interrupt so they are stopped.
    workers = [
        context.Process(target=_worker, args=(args.port,), daemon=True) for _ in range(args.workers)
    ]
    signal.signal(signal.SIGTERM, _interrupt)
    for worker in workers:
        worker.start()
    try:
        for worker in workers:
            worker.join()
    except KeyboardInterrupt:
        for worker in workers:
            worker.terminate()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
