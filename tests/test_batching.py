import asyncio
import time

import pytest

from bench.baselines.batching import Event, PendingRequest, StaticBatcher


class Recorder:
    """A fake batch generator: records batch composition and emits one token per step."""

    def __init__(self, step_s: float = 0.0, fail_on: int | None = None) -> None:
        self.batches: list[list[int]] = []
        self.step_s = step_s
        self.fail_on = fail_on

    def __call__(self, batch: list[PendingRequest]) -> int:
        self.batches.append([r.prompt_ids[0] for r in batch])
        if self.fail_on is not None and self.fail_on in self.batches[-1]:
            raise RuntimeError("out of memory")
        steps = max(r.max_tokens for r in batch)
        for step in range(steps):
            time.sleep(self.step_s)
            for request in batch:
                request.token(step)
        return steps


async def drain(request: PendingRequest) -> tuple[list[Event], float, float]:
    """Collect events until the request ends; also return first-token and end times."""
    events: list[Event] = []
    first = 0.0
    while True:
        event = await request.events.get()
        events.append(event)
        if event[0] == "token" and not first:
            first = time.monotonic()
        if event[0] in ("done", "error"):
            return events, first, time.monotonic()


def request(ident: int, max_tokens: int = 3, ignore_eos: bool = True) -> PendingRequest:
    return PendingRequest([ident], max_tokens, ignore_eos, asyncio.get_running_loop())


def test_requests_within_the_window_share_a_batch() -> None:
    async def go() -> None:
        recorder = Recorder()
        batcher = StaticBatcher(recorder, max_batch_size=8, batch_window_s=0.2)
        batcher.start()
        try:
            requests = [request(i) for i in range(3)]
            for r in requests:
                batcher.submit(r)
            for r in requests:
                events, _, _ = await drain(r)
                assert [e[0] for e in events] == ["token"] * 3 + ["done"]
        finally:
            batcher.stop()
        assert recorder.batches == [[0, 1, 2]]
        assert batcher.stats()["batch_size_counts"] == {"3": 1}

    asyncio.run(go())


def test_batch_size_is_capped_and_order_is_fifo() -> None:
    async def go() -> None:
        recorder = Recorder()
        batcher = StaticBatcher(recorder, max_batch_size=2, batch_window_s=0.05)
        batcher.start()
        try:
            requests = [request(i) for i in range(5)]
            for r in requests:
                batcher.submit(r)
            for r in requests:
                await drain(r)
        finally:
            batcher.stop()
        assert recorder.batches == [[0, 1], [2, 3], [4]]
        assert batcher.stats()["mean_batch_size"] == pytest.approx(5 / 3)

    asyncio.run(go())


def test_a_late_arrival_waits_for_the_running_batch() -> None:
    """The defining cost of static batching: no joining a batch that is already running."""

    async def go() -> None:
        recorder = Recorder(step_s=0.05)
        batcher = StaticBatcher(recorder, max_batch_size=8, batch_window_s=0.0)
        batcher.start()
        try:
            first = request(0, max_tokens=6)  # runs for about 0.3 s
            batcher.submit(first)
            await asyncio.sleep(0.1)  # the first batch is mid-flight
            late = request(1, max_tokens=2)
            batcher.submit(late)
            _, _, first_done = await drain(first)
            _, late_first_token, _ = await drain(late)
        finally:
            batcher.stop()
        assert recorder.batches == [[0], [1]]
        assert late_first_token >= first_done - 0.01

    asyncio.run(go())


def test_finished_sequences_waste_slots_until_the_batch_ends() -> None:
    async def go() -> None:
        recorder = Recorder()
        batcher = StaticBatcher(recorder, max_batch_size=8, batch_window_s=0.2)
        batcher.start()
        try:
            short, long_ = request(0, max_tokens=2), request(1, max_tokens=6)
            batcher.submit(short)
            batcher.submit(long_)
            short_events, _, _ = await drain(short)
            await drain(long_)
        finally:
            batcher.stop()
        assert sum(e[0] == "token" for e in short_events) == 2
        assert short_events[-1] == ("done", "length")
        # The batch ran 6 steps for 2 sequences (12 slots) but only 8 tokens were delivered.
        assert batcher.stats()["wasted_slot_fraction"] == pytest.approx(1 - 8 / 12)

    asyncio.run(go())


def test_a_failing_batch_fails_its_requests_and_the_worker_survives() -> None:
    async def go() -> None:
        recorder = Recorder(fail_on=99)
        batcher = StaticBatcher(recorder, max_batch_size=1, batch_window_s=0.0)
        batcher.start()
        try:
            bad, good = request(99), request(1)
            batcher.submit(bad)
            batcher.submit(good)
            bad_events, _, _ = await drain(bad)
            good_events, _, _ = await drain(good)
        finally:
            batcher.stop()
        assert bad_events == [("error", "RuntimeError")]
        assert good_events[-1] == ("done", "length")

    asyncio.run(go())


def test_stopping_fails_requests_nobody_will_serve() -> None:
    async def go() -> None:
        batcher = StaticBatcher(Recorder(), max_batch_size=4, batch_window_s=0.0)  # never started
        pending = [request(0), request(1)]
        for r in pending:
            batcher.submit(r)
        batcher.stop()
        for r in pending:
            events, _, _ = await drain(r)
            assert events == [("error", "shutdown")]

    asyncio.run(go())


def test_pending_request_ignores_tokens_after_it_finishes() -> None:
    async def go() -> None:
        r = request(0, max_tokens=2)
        for token in range(5):
            r.token(token)
        events, _, _ = await drain(r)
        assert events == [("token", 0), ("token", 1), ("done", "length")]

    asyncio.run(go())


@pytest.mark.parametrize("kwargs", [{"max_batch_size": 0, "batch_window_s": 0.0},
                                    {"max_batch_size": 1, "batch_window_s": -1.0}])  # fmt: skip
def test_invalid_settings_are_rejected(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        StaticBatcher(Recorder(), **kwargs)  # type: ignore[arg-type]
