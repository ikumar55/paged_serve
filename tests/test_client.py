import asyncio
import json
import time
from collections.abc import Callable
from typing import Any

import httpx

from bench.client import RequestResult, execute_request
from bench.workload import RequestSpec
from tests.fake_server import (
    DONE,
    Item,
    closing_chunk,
    mock_client,
    sse,
    streaming_response,
    token_chunk,
    usage_chunk,
    well_formed_stream,
)

Handler = Callable[[httpx.Request], httpx.Response]


def spec(prompt_len: int = 4, max_tokens: int = 5) -> RequestSpec:
    return RequestSpec(0, 0.0, list(range(1000, 1000 + prompt_len)), max_tokens)


def send(
    handler: Handler, *, timeout_s: float = 5.0, request: RequestSpec | None = None
) -> RequestResult:
    async def go() -> RequestResult:
        async with mock_client(handler) as client:
            return await execute_request(
                client, "m", request or spec(), time.perf_counter(), timeout_s=timeout_s
            )

    return asyncio.run(go())


def stream_handler(items: list[Item]) -> Handler:
    return lambda request: streaming_response(items)


def test_successful_stream_is_measured() -> None:
    result = send(stream_handler(well_formed_stream(5, delay=0.02)))
    assert result.status == "ok"
    assert result.http_status == 200
    assert result.output_tokens == 5 and result.tokens_from_usage
    assert len(result.token_times_s) == 5
    assert result.token_times_s == sorted(result.token_times_s)
    assert result.ttft_s is not None and result.ttft_s >= 0.8 * 0.02
    assert result.e2e_s is not None and result.e2e_s >= result.token_times_s[-1]
    # Each token waited 20 ms, so the gaps between tokens should be about 20 ms.
    gaps = [b - a for a, b in zip(result.token_times_s, result.token_times_s[1:], strict=False)]
    assert all(gap >= 0.8 * 0.02 for gap in gaps)


def test_tokens_are_counted_from_events_without_usage() -> None:
    result = send(stream_handler(well_formed_stream(4, usage=False)))
    assert result.status == "ok"
    assert result.output_tokens == 4
    assert not result.tokens_from_usage


def test_usage_overrides_event_count_when_chunks_hold_several_tokens() -> None:
    items: list[Item] = [(0.0, token_chunk("ab")), (0.0, closing_chunk()), (0.0, usage_chunk(2))]
    result = send(stream_handler([*items, (0.0, DONE)]))
    assert result.output_tokens == 2 and len(result.token_times_s) == 1


def test_request_body_matches_the_contract() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["body"] = json.loads(request.content)
        return streaming_response(well_formed_stream(3))

    send(handler, request=spec(prompt_len=3, max_tokens=3))
    assert seen["path"] == "/v1/completions"
    assert seen["body"]["prompt"] == [1000, 1001, 1002]
    assert seen["body"]["max_tokens"] == 3
    assert seen["body"]["stream"] is True
    assert seen["body"]["temperature"] == 0.0
    assert seen["body"]["ignore_eos"] is True
    assert seen["body"]["stream_options"] == {"include_usage": True}


def test_http_error_is_recorded() -> None:
    result = send(lambda request: httpx.Response(429, json={"error": "busy"}))
    assert result.status == "http_error"
    assert result.http_status == 429 and result.error == "http_429"
    assert result.ttft_s is None and result.e2e_s is not None


def test_connection_error_hides_host_details() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("could not reach db-internal-7.example.corp")

    result = send(handler)
    assert result.status == "error" and result.error == "ConnectError"
    assert "example.corp" not in json.dumps(result.to_dict())


def test_timeout_keeps_partial_progress() -> None:
    items: list[Item] = [(0.0, token_chunk()), (2.0, token_chunk()), (0.0, DONE)]
    result = send(stream_handler(items), timeout_s=0.2)
    assert result.status == "timeout"
    assert len(result.token_times_s) == 1 and result.ttft_s is not None
    assert result.e2e_s is not None and result.e2e_s < 1.0


def test_stream_that_ends_without_done_is_an_error() -> None:
    result = send(stream_handler([(0.0, token_chunk()), (0.0, token_chunk())]))
    assert result.status == "error" and result.error == "stream_ended_early"


def test_stream_without_tokens_is_an_error() -> None:
    result = send(stream_handler([(0.0, closing_chunk()), (0.0, DONE)]))
    assert result.status == "error" and result.error == "no_tokens"


def test_malformed_event_is_an_error() -> None:
    result = send(stream_handler([(0.0, b"data: {not json}\n\n"), (0.0, DONE)]))
    assert result.status == "error" and result.error == "malformed_event"


def test_non_object_event_is_an_error() -> None:
    result = send(stream_handler([(0.0, sse([1, 2]))]))  # type: ignore[arg-type]
    assert result.status == "error" and result.error == "malformed_event"


def test_result_roundtrips_through_dict() -> None:
    result = send(stream_handler(well_formed_stream(3)))
    assert RequestResult.from_dict(result.to_dict()) == result
