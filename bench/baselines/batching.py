"""Static batching: the naive serving policy this project's engine is meant to beat.

A worker thread repeatedly takes up to ``max_batch_size`` queued requests (waiting at most
``batch_window_s`` after the first one arrives), runs the whole batch to completion, and only then
looks at the queue again. A request that arrives while a batch is running waits for all of it,
and every sequence in a batch occupies a slot until the longest one finishes. Continuous batching
removes exactly these two costs.

Model-agnostic: the generation itself is a callable, so the policy is testable without a model.
"""

from __future__ import annotations

import asyncio
import contextlib
import queue
import threading
import time
from collections import Counter
from collections.abc import Callable
from typing import Any, Literal

EventKind = Literal["token", "done", "error"]
Event = tuple[EventKind, Any]


class PendingRequest:
    """A request waiting for, or being served by, the batch worker.

    The worker thread calls :meth:`token`, :meth:`finish` and :meth:`fail`; the web handler
    awaits ``events`` on the event loop that created the request.
    """

    def __init__(
        self,
        prompt_ids: list[int],
        max_tokens: int,
        ignore_eos: bool,
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        self.prompt_ids = prompt_ids
        self.max_tokens = max_tokens
        self.ignore_eos = ignore_eos
        self.events: asyncio.Queue[Event] = asyncio.Queue()
        self.emitted = 0  # tokens delivered to the client
        self.finished = False
        self._loop = loop

    def _emit(self, event: Event) -> None:
        # If the event loop is gone (server shutting down) there is nobody to tell.
        with contextlib.suppress(RuntimeError):
            self._loop.call_soon_threadsafe(self.events.put_nowait, event)

    def token(self, token_id: int) -> None:
        if self.finished:
            return
        self.emitted += 1
        self._emit(("token", token_id))
        if self.emitted >= self.max_tokens:
            self.finish("length")

    def finish(self, reason: str) -> None:
        if not self.finished:
            self.finished = True
            self._emit(("done", reason))

    def fail(self, message: str) -> None:
        if not self.finished:
            self.finished = True
            self._emit(("error", message))


# Runs one batch to completion and returns how many decode steps it executed.
GenerateBatch = Callable[[list[PendingRequest]], int]


class StaticBatcher:
    def __init__(
        self, generate_batch: GenerateBatch, *, max_batch_size: int, batch_window_s: float
    ) -> None:
        if max_batch_size < 1:
            raise ValueError("max_batch_size must be at least 1")
        if batch_window_s < 0:
            raise ValueError("batch_window_s must be non-negative")
        self._generate_batch = generate_batch
        self._max_batch_size = max_batch_size
        self._window_s = batch_window_s
        self._queue: queue.Queue[PendingRequest] = queue.Queue()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="static-batcher", daemon=True)
        self._lock = threading.Lock()
        self._batch_sizes: Counter[int] = Counter()
        self._requests = 0
        self._slots = 0  # batch size x decode steps, summed over batches
        self._delivered = 0  # tokens that reached a client

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=30)
        while True:  # nobody will serve what is still queued
            try:
                self._queue.get_nowait().fail("shutdown")
            except queue.Empty:
                break

    def submit(self, request: PendingRequest) -> None:
        self._queue.put(request)

    def stats(self) -> dict[str, Any]:
        with self._lock:
            batches = sum(self._batch_sizes.values())
            return {
                "batches": batches,
                "requests": self._requests,
                "mean_batch_size": self._requests / batches if batches else None,
                "batch_size_counts": {str(k): v for k, v in sorted(self._batch_sizes.items())},
                # Decode slots spent on sequences that had already finished: the static-batching
                # waste that continuous batching reclaims.
                "wasted_slot_fraction": 1 - self._delivered / self._slots if self._slots else None,
            }

    def _collect(self) -> list[PendingRequest]:
        try:
            batch = [self._queue.get(timeout=0.1)]
        except queue.Empty:
            return []
        deadline = time.monotonic() + self._window_s
        while len(batch) < self._max_batch_size:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                batch.append(self._queue.get(timeout=remaining))
            except queue.Empty:
                break
        return batch

    def _run(self) -> None:
        while not self._stop.is_set():
            batch = self._collect()
            if not batch:
                continue
            steps = 0
            try:
                steps = self._generate_batch(batch)
            except Exception as exc:
                for request in batch:
                    request.fail(type(exc).__name__)
            for request in batch:
                request.finish("length")  # no-op for requests that already ended
            with self._lock:
                self._batch_sizes[len(batch)] += 1
                self._requests += len(batch)
                self._slots += len(batch) * steps
                self._delivered += sum(r.emitted for r in batch)
