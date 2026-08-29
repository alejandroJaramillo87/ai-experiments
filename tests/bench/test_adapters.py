"""Request construction and response decoding, per engine.

The fixtures are the wire shapes each engine actually emits. They exist because the
failures this module guards against are silent: a missing stream option costs the usage
object, and a frame-counting token count is wrong only by a little, only on some models.
"""

from __future__ import annotations

import io
import json
from unittest import mock

import pytest

from bench import adapters, client
from common.serving import ServingError

# --- wire fixtures --------------------------------------------------------------

LLAMA_CHUNK = {
    "choices": [{"text": " the", "index": 0, "finish_reason": None}],
    "object": "text_completion",
}
# llama.cpp's terminal frame: no text, and it carries both usage and server timings.
LLAMA_FINAL = {
    "choices": [{"text": "", "index": 0, "finish_reason": "length"}],
    "usage": {"prompt_tokens": 128, "completion_tokens": 4, "total_tokens": 132},
    "timings": {
        "prompt_n": 128,
        "prompt_ms": 51.2,
        "predicted_n": 4,
        "predicted_ms": 114.3,
    },
}
VLLM_CHUNK = {"choices": [{"text": " the", "index": 0}], "object": "text_completion"}
# vLLM's usage frame arrives only with stream_options.include_usage, and it carries an
# empty choices list rather than an empty text.
VLLM_FINAL = {
    "choices": [],
    "usage": {"prompt_tokens": 128, "completion_tokens": 4, "total_tokens": 132},
}


def sse(*payloads: dict, done: bool = True) -> bytes:
    lines = [b"data: " + json.dumps(p).encode() + b"\n\n" for p in payloads]
    if done:
        lines.append(b"data: [DONE]\n\n")
    return b"".join(lines)


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


class TestRequestConstruction:
    def test_every_streaming_engine_asks_for_usage_in_the_stream(self):
        # Without this vLLM streams no usage at all and token counts fall back to
        # counting frames, which undercounts multi-token chunks.
        for name in ("llama-cpu", "llama-gpu", "vllm-gpu"):
            body = adapters.for_service(name).build_request("hi", 8, seed=1)
            assert body["stream"] is True
            assert body["stream_options"] == {"include_usage": True}

    def test_the_harness_is_declared_non_streaming_and_asks_for_no_stream(self):
        # schemas/harness_chat_endpoint.json: stream: true is refused with a 400 rather
        # than answered with a whole body, deliberately, so that a streaming client
        # cannot receive a non-streamed response and believe it timed one.
        harness = adapters.for_service("harness")
        assert harness.streams is False
        body = harness.build_request("hi", 8, seed=1)
        assert body["stream"] is False
        assert "stream_options" not in body

    def test_the_harness_uses_the_chat_path_and_a_messages_body(self):
        harness = adapters.for_service("harness")
        assert harness.url.endswith("/v1/chat/completions")
        body = harness.build_request("hi", 8, seed=1)
        assert body["messages"] == [{"role": "user", "content": "hi"}]
        assert "prompt" not in body

    def test_the_streaming_engines_are_declared_streaming(self):
        for name in ("llama-cpu", "llama-gpu", "vllm-gpu"):
            assert adapters.for_service(name).streams is True

    def test_llama_cpp_forces_length_and_disables_the_prompt_cache(self):
        body = adapters.for_service("llama-cpu").build_request("hi", 8, seed=1)
        assert body["ignore_eos"] is True
        assert body["cache_prompt"] is False
        assert body["max_tokens"] == 8

    def test_vllm_pins_both_ends_of_the_generation_length(self):
        body = adapters.for_service("vllm-gpu").build_request("hi", 8, seed=1)
        assert body["ignore_eos"] is True
        assert body["min_tokens"] == 8 == body["max_tokens"]

    def test_vllm_sends_no_cache_prompt_because_it_is_a_launch_flag(self):
        assert "cache_prompt" not in adapters.for_service("vllm-gpu").build_request(
            "hi", 8, seed=1
        )

    def test_the_harness_cannot_force_an_exact_length_and_says_so(self):
        harness = adapters.for_service("harness")
        assert harness.forces_exact_length is False
        assert "ignore_eos" not in harness.build_request("hi", 8, seed=1)

    def test_generation_is_deterministic(self):
        body = adapters.for_service("llama-gpu").build_request("hi", 8, seed=99)
        assert body["temperature"] == 0.0
        assert body["seed"] == 99

    def test_completions_endpoint_not_chat(self):
        # A chat template between the workload and the token count would make the
        # prompt length something other than what the sweep asked for.
        assert adapters.for_service("llama-cpu").url.endswith("/v1/completions")

    def test_ports_come_from_the_service_registry(self):
        assert "8001" in adapters.for_service("llama-cpu").url
        assert "8003" in adapters.for_service("vllm-gpu").url
        assert "8005" in adapters.for_service("harness").url

    def test_unknown_service_is_refused(self):
        with pytest.raises(KeyError):
            adapters.for_service("llama-hugepages")


class TestUsageAndTimings:
    def test_llama_cpp_reports_server_timings(self):
        adapter = adapters.for_service("llama-cpu")
        assert adapter.reports_server_timings is True
        timings = adapter.parse_timings(LLAMA_FINAL)
        assert timings.prompt_ms == pytest.approx(51.2)
        assert timings.predicted_ms == pytest.approx(114.3)

    def test_vllm_reports_none_rather_than_a_substitute(self):
        adapter = adapters.for_service("vllm-gpu")
        assert adapter.reports_server_timings is False
        assert adapter.parse_timings(VLLM_FINAL) is None

    def test_usage_parses_on_both_engines(self):
        for adapter, final in (
            (adapters.for_service("llama-cpu"), LLAMA_FINAL),
            (adapters.for_service("vllm-gpu"), VLLM_FINAL),
        ):
            usage = adapter.parse_usage(final)
            assert usage.prompt_tokens == 128
            assert usage.completion_tokens == 4

    def test_a_content_chunk_carries_no_usage(self):
        assert adapters.for_service("llama-cpu").parse_usage(LLAMA_CHUNK) is None


def run_stream(service: str, payload: bytes) -> client.StreamResult:
    adapter = adapters.for_service(service)
    with mock.patch.object(
        client.urllib.request, "urlopen", return_value=FakeResponse(payload)
    ):
        return client.stream(adapter, {"prompt": "hi"}, timeout=5.0)


class TestStreamDecoding:
    def test_only_text_chunks_count_as_token_arrivals(self):
        # Four tokens, then a usage frame, then [DONE]. A frame count would say six.
        payload = sse(*[LLAMA_CHUNK] * 4, LLAMA_FINAL)
        result = run_stream("llama-cpu", payload)
        assert len(result.arrivals) == 4
        assert result.chunks_seen == 5
        assert result.usage.completion_tokens == 4

    def test_the_vllm_usage_frame_with_empty_choices_is_not_an_arrival(self):
        result = run_stream("vllm-gpu", sse(*[VLLM_CHUNK] * 3, VLLM_FINAL))
        assert len(result.arrivals) == 3
        assert result.usage.completion_tokens == 4

    def test_token_count_comes_from_usage_not_from_the_arrival_count(self):
        # Three chunks, each carrying two tokens: usage is authoritative, and the
        # difference is what makes the ITL series approximate rather than per-token.
        final = dict(LLAMA_FINAL)
        final["usage"] = {"prompt_tokens": 128, "completion_tokens": 6}
        result = run_stream("llama-cpu", sse(*[LLAMA_CHUNK] * 3, final))
        assert len(result.arrivals) == 3
        assert result.usage.completion_tokens == 6

    def test_arrivals_are_monotonic(self):
        result = run_stream("llama-cpu", sse(*[LLAMA_CHUNK] * 5, LLAMA_FINAL))
        assert result.arrivals == sorted(result.arrivals)
        assert result.t_submit <= result.arrivals[0]

    def test_the_done_sentinel_is_not_a_chunk(self):
        result = run_stream("llama-cpu", sse(LLAMA_CHUNK, done=True))
        assert result.chunks_seen == 1

    def test_a_malformed_frame_is_skipped_not_timed(self):
        payload = b"data: {not json}\n\n" + sse(LLAMA_CHUNK, LLAMA_FINAL)
        result = run_stream("llama-cpu", payload)
        assert len(result.arrivals) == 1

    def test_a_response_with_no_text_is_an_error_not_an_empty_result(self):
        with pytest.raises(ServingError, match="no text chunks"):
            run_stream("llama-cpu", sse(LLAMA_FINAL))

    def test_non_sse_lines_are_ignored(self):
        payload = b": keep-alive\n\n" + sse(LLAMA_CHUNK, LLAMA_FINAL)
        assert len(run_stream("llama-cpu", payload).arrivals) == 1


class TestWholeResponseTransport:
    """The non-streaming path, which the harness is the only user of."""

    HARNESS_BODY = {
        "id": "x",
        "object": "chat.completion",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "hi"},
                "finish_reason": "length",
            }
        ],
        "usage": {"prompt_tokens": 128, "completion_tokens": 8, "total_tokens": 136},
    }

    def send(self, payload: bytes):
        adapter = adapters.for_service("harness")
        with mock.patch.object(
            client.urllib.request, "urlopen", return_value=FakeResponse(payload)
        ):
            return client.send(adapter, {"messages": []}, timeout=5.0)

    def test_send_routes_a_non_streaming_engine_to_the_whole_transport(self):
        result = self.send(json.dumps(self.HARNESS_BODY).encode())
        # One response, one timestamp: there is nothing else to observe.
        assert len(result.arrivals) == 1
        assert result.chunks_seen == 1
        assert result.usage.completion_tokens == 8

    def test_send_routes_a_streaming_engine_to_the_stream_transport(self):
        adapter = adapters.for_service("llama-cpu")
        payload = sse(*[LLAMA_CHUNK] * 3, LLAMA_FINAL)
        with mock.patch.object(
            client.urllib.request, "urlopen", return_value=FakeResponse(payload)
        ):
            result = client.send(adapter, {"prompt": "hi"}, timeout=5.0)
        assert len(result.arrivals) == 3

    def test_a_non_json_response_is_an_error_not_a_zero(self):
        with pytest.raises(ServingError, match="not JSON"):
            self.send(b"<html>502 Bad Gateway</html>")
