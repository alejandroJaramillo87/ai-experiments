"""Request construction and response decoding, per engine.

The fixtures are the wire shapes each engine actually emits. They exist because the
failures this module guards against are silent: a missing stream option costs the usage
object, and a frame-counting token count is wrong only by a little, only on some models.
"""

from __future__ import annotations

import io
import json
import pathlib
from unittest import mock

import pytest

from bench import adapters, client
from common.serving import ServingError

FIXTURES = pathlib.Path(__file__).parent / "fixtures"


def fixture(name: str) -> dict:
    """A wire frame as the pinned engine emits it; see fixtures/provenance.json."""
    return json.loads((FIXTURES / f"{name}.json").read_text())


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
    def test_every_engine_asks_for_usage_in_the_stream(self):
        # Without this vLLM streams no usage at all and token counts fall back to
        # counting frames, which undercounts multi-token chunks.
        for name in sorted(adapters.ENGINE_FOR_SERVICE):
            body = adapters.for_service(name).build_request("hi", 8, seed=1)
            assert body["stream"] is True
            assert body["stream_options"] == {"include_usage": True}

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
        assert "8002" in adapters.for_service("llama-gpu").url
        assert "8003" in adapters.for_service("vllm-gpu").url

    def test_unknown_service_is_refused(self):
        with pytest.raises(KeyError):
            adapters.for_service("llama-hugepages")

    def test_a_real_service_that_is_not_a_target_says_why(self):
        # The harness is in common/serving.py's registry and is not benchmarked here:
        # `make ladder` drives its runtime directly rather than over HTTP, and its
        # endpoint refuses streaming outright. Membership in SERVICES is necessary and
        # not sufficient, and the refusal names the alternative.
        with pytest.raises(KeyError, match="make ladder"):
            adapters.for_service("harness")


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

    def test_the_real_llama_cpp_partial_frame_is_text_and_nothing_else(self):
        # BQ-F23. The source-derived partial frame, previously read by no test. It is
        # what proves the inline chunk above is not an idealisation: the real frame
        # carries logprobs, a model name and a fingerprint, and still yields neither a
        # usage object nor timings, so only the terminal frame can supply either.
        adapter = adapters.for_service("llama-cpu")
        partial = fixture("llama_cpp_partial")
        assert adapter.parse_usage(partial) is None
        assert adapter.parse_timings(partial) is None

    def test_a_non_numeric_token_count_is_refused_rather_than_raised(self):
        # BQ-F10. No pinned engine emits this: llama.cpp and vLLM both serialize the
        # counts as integers. The guard exists because parse_usage runs inside
        # bench/client.py::stream, whose only caller handles the serving error type --
        # a ValueError here would end a run of up to an hour with a traceback, while an
        # absent usage object degrades to the runner's explained refusal.
        adapter = adapters.for_service("vllm-gpu")
        assert (
            adapter.parse_usage(
                {"usage": {"prompt_tokens": "n/a", "completion_tokens": 4}}
            )
            is None
        )


class TestVllmPerRequestTimings:
    """BQ-F19. vLLM attaches per-request timings to the frame already being parsed.

    `PerRequestTimingMetrics` rides on the same final usage chunk under the launch flag
    `--enable-per-request-metrics`, which docker-compose.inference.yaml does not pass.
    The parser is therefore inert against today's wire shape and lights up on the day
    the flag is set, rather than needing to be written then.
    """

    def test_todays_production_frame_still_yields_no_server_timings(self):
        # The shape the running service emits: usage, no metrics object.
        assert (
            adapters.for_service("vllm-gpu").parse_timings(fixture("vllm_final"))
            is None
        )

    def test_the_metrics_frame_is_parsed_when_the_flag_is_set(self):
        timings = adapters.for_service("vllm-gpu").parse_timings(
            fixture("vllm_final_with_metrics")
        )
        # Carried raw: vLLM stamps this from scheduling, not arrival, so it is already
        # queue-free and is the same phase llama.cpp reports as prompt_ms. The envelope
        # is slightly wider -- vLLM includes any preemption during prefill -- which is a
        # caveat on the comparison, not a term to subtract.
        assert timings.prompt_ms == pytest.approx(512.0)
        assert timings.predicted_ms == pytest.approx(240.0)
        assert timings.queue_ms == pytest.approx(18.0)

    def test_metrics_without_a_queue_field_still_yield_prefill(self):
        # Queue time is informational, not a term in the prefill arithmetic, so its
        # absence costs the contention signal and nothing else.
        body = {
            "usage": {},
            "metrics": {"time_to_first_token_ms": 512.0, "generation_time_ms": 240.0},
        }
        timings = adapters.for_service("vllm-gpu").parse_timings(body)
        assert timings.prompt_ms == pytest.approx(512.0)
        assert timings.queue_ms is None

    def test_llama_cpp_timings_carry_no_queue_wait(self):
        # llama.cpp's timings object has no queue field, and the runner must not read
        # an absent one as zero wait.
        assert (
            adapters.for_service("llama-cpu")
            .parse_timings(fixture("llama_cpp_final"))
            .queue_ms
            is None
        )


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

    def test_the_source_derived_frames_decode_like_the_inline_ones(self):
        # BQ-F23. The fixtures are the shapes the pinned llama.cpp serializers emit;
        # driving them through the real client is what keeps the hand-written chunks
        # above honest about the extra fields a real frame carries.
        result = run_stream(
            "llama-cpu",
            sse(fixture("llama_cpp_partial"), fixture("llama_cpp_final")),
        )
        assert len(result.arrivals) == 1
        assert result.usage.completion_tokens == 8
        assert result.timings.prompt_ms == pytest.approx(512.0)

    def test_non_sse_lines_are_ignored(self):
        payload = b": keep-alive\n\n" + sse(LLAMA_CHUNK, LLAMA_FINAL)
        assert len(run_stream("llama-cpu", payload).arrivals) == 1
