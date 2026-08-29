"""What the streaming client does with time, and with a server that misbehaves.

tests/bench/test_adapters.py already drives `bench.client.stream` through a BytesIO
standing in for the socket, which proves the decoding. A BytesIO yields every line at
once, so none of those tests can observe the property the whole suite rests on: a gap
between frames on the wire has to become a gap of the same size in the arrival series,
because every latency this repository publishes is a difference between two of those
stamps.

The fakes in tests/bench/fakes.py supply the gaps. Tolerances are wide on purpose --
these tests distinguish a 40 ms interval from an 80 ms one, and are not a measurement.
"""

from __future__ import annotations

import dataclasses
import json
import pathlib
import urllib.error

import pytest

from bench import adapters, client, metrics
from common.serving import SERVICES, ServingError, ServingTransientError
from tests.bench import fakes

FIXTURES = pathlib.Path(__file__).parent / "fixtures"

TTFT_S = 0.060
GAP_S = 0.020
TOKENS = 8


def fixture(name: str) -> dict:
    return json.loads((FIXTURES / f"{name}.json").read_text())


def llama_adapter(port: int | None = None) -> adapters.Adapter:
    spec = SERVICES["llama-cpu"]
    if port is not None:
        spec = dataclasses.replace(spec, port=port)
    return adapters.LlamaCppAdapter(spec)


def sample_from(result: client.StreamResult) -> metrics.Sample:
    """Build the Sample the runner would build, without its length assertion."""
    assert result.usage is not None
    return metrics.Sample(
        t_submit=result.t_submit,
        arrivals=tuple(result.arrivals),
        prompt_tokens=result.usage.prompt_tokens,
        completion_tokens=result.usage.completion_tokens,
    )


def generation(withhold: int | None = None) -> list[fakes.Frame]:
    """A full generation of TOKENS tokens, optionally withholding one token's text.

    A withheld token still takes its time on the server. llama.cpp sends an empty text
    field when a token is held for a pending stop-string match, and sends nothing at
    all when the token leaves an incomplete UTF-8 sequence. Both cost the client one
    arrival while the generation length reported in usage stays the same.
    """
    frames = []
    for index in range(TOKENS):
        delay = TTFT_S if index == 0 else GAP_S
        if index == withhold:
            frames.append(fakes.Frame.of(fixture("llama_cpp_withheld"), delay))
        else:
            frames.append(fakes.Frame.of(fakes.text_chunk(f" t{index}"), delay))
    frames.append(fakes.Frame.of(fixture("llama_cpp_final"), GAP_S))
    return frames


def stream_scripted(frames: list[fakes.Frame]) -> client.StreamResult:
    """Run the real client against the scripted response, no socket involved."""
    adapter = llama_adapter()
    import unittest.mock as mock

    with mock.patch.object(
        client.urllib.request, "urlopen", return_value=fakes.ScriptedResponse(frames)
    ):
        return client.stream(adapter, {"prompt": "x"}, timeout=10.0)


class TestGapsBecomeLatencies:
    def test_an_injected_gap_is_recovered_as_an_inter_token_latency(self):
        result = stream_scripted(generation())
        itls = sample_from(result).itl_ms()
        assert len(itls) == TOKENS - 1
        for observed in itls:
            assert GAP_S * 1000 * 0.5 < observed < GAP_S * 1000 * 2.0

    def test_time_to_first_token_reflects_the_first_gap_not_the_others(self):
        result = stream_scripted(generation())
        ttft = sample_from(result).ttft_ms()
        assert ttft is not None
        # The client cannot see the server's prefill, only the delay before the first
        # frame. That delay is TTFT_S, which is well clear of a single decode gap.
        assert TTFT_S * 1000 * 0.5 < ttft < TTFT_S * 1000 * 2.0
        assert ttft > GAP_S * 1000 * 2

    def test_arrivals_are_monotonic_under_real_delays(self):
        result = stream_scripted(generation())
        assert list(result.arrivals) == sorted(result.arrivals)

    def test_the_usage_frame_is_not_timed_as_an_arrival(self):
        # The final frame is sent GAP_S after the last token, carries usage, and has an
        # empty text field. Counting it would add a spurious interval and one arrival.
        result = stream_scripted(generation())
        assert len(result.arrivals) == TOKENS
        assert result.usage is not None
        assert result.usage.completion_tokens == TOKENS


class TestOverASocket:
    """The same properties through a real port, urllib, and the kernel.

    A substituted response object cannot show frames coalescing in a write buffer or a
    socket receive buffer. If several frames were delivered as one read, their stamps
    would collapse and the intervals between them would read as near zero.
    """

    def test_a_real_socket_preserves_the_gaps(self):
        with fakes.ScriptedServer(generation()) as base_url:
            port = int(base_url.rsplit(":", 1)[1])
            result = client.stream(llama_adapter(port), {"prompt": "x"}, timeout=10.0)
        itls = sample_from(result).itl_ms()
        assert len(itls) == TOKENS - 1
        # Coalescing would show up as intervals far below the injected gap.
        assert min(itls) > GAP_S * 1000 * 0.5

    def test_a_server_error_is_transient_so_the_cell_can_be_retried(self):
        with fakes.ScriptedServer([], status=503) as base_url:
            port = int(base_url.rsplit(":", 1)[1])
            with pytest.raises(ServingTransientError):
                client.stream(llama_adapter(port), {"prompt": "x"}, timeout=10.0)

    def test_a_client_error_is_permanent_because_retrying_cannot_fix_it(self):
        with fakes.ScriptedServer([], status=400) as base_url:
            port = int(base_url.rsplit(":", 1)[1])
            with pytest.raises(ServingError) as caught:
                client.stream(llama_adapter(port), {"prompt": "x"}, timeout=10.0)
        assert not isinstance(caught.value, ServingTransientError)

    def test_an_unreachable_service_is_transient(self):
        with fakes.ScriptedServer([]) as base_url:
            port = int(base_url.rsplit(":", 1)[1])
        # The server is closed by the time the request is made.
        with pytest.raises(ServingTransientError):
            client.stream(llama_adapter(port), {"prompt": "x"}, timeout=5.0)


class TestDecoding:
    def test_a_chat_shaped_delta_is_decoded_as_text(self):
        # bench/client.py reads choices[0].delta.content when text is absent, so an
        # engine served through the chat surface still yields arrivals. No fixture
        # reached that branch before this test.
        frames = [
            fakes.Frame.of({"choices": [{"delta": {"content": " hi"}, "index": 0}]}),
            fakes.Frame.of(fixture("llama_cpp_final")),
        ]
        result = stream_scripted(frames)
        assert len(result.arrivals) == 1

    def test_the_vllm_usage_frame_carries_no_arrival(self):
        frames = [
            fakes.Frame.of(fixture("vllm_chunk")),
            fakes.Frame.of(fixture("vllm_final")),
        ]
        result = stream_scripted(frames)
        assert len(result.arrivals) == 1
        assert result.usage is not None
        assert result.usage.completion_tokens == 8


class TestWithheldFrames:
    """BQ-F01. A token whose text never reaches the client costs one arrival.

    The effect depends on where in the generation it happens, and the two cases are
    different defects rather than one. An interior withhold leaves every aggregate
    correct and corrupts only the distribution. A withhold at either end shortens the
    span the rates divide by while the token count stays the same.
    """

    def test_an_interior_withhold_fabricates_a_latency_outlier(self):
        result = stream_scripted(generation(withhold=3))
        sample = sample_from(result)
        itls = sample.itl_ms()
        assert len(itls) == TOKENS - 2
        # One interval spans two token generations. Nothing in the series says which.
        assert max(itls) > GAP_S * 1000 * 1.5

    def test_an_interior_withhold_leaves_the_aggregate_rate_correct(self):
        # Recorded so a later fix does not "correct" a number that was never wrong.
        result = stream_scripted(generation(withhold=3))
        rate = sample_from(result).decode_tok_s()
        assert rate is not None
        assert abs(rate - 1.0 / GAP_S) / (1.0 / GAP_S) < 0.15

    @pytest.mark.xfail(
        strict=True,
        reason="BQ-F01: a withheld leading frame shortens the decode span while the "
        "token count stays at usage.completion_tokens, so the reported rate is "
        "high by one interval's worth",
    )
    def test_a_leading_withhold_does_not_overstate_the_decode_rate(self):
        result = stream_scripted(generation(withhold=0))
        rate = sample_from(result).decode_tok_s()
        assert rate is not None
        true_rate = 1.0 / GAP_S
        assert abs(rate - true_rate) / true_rate < 0.05

    @pytest.mark.xfail(
        strict=True,
        reason="BQ-F01: with the first token's text withheld, the first arrival the "
        "client sees is the second token, so time to first token is late by one "
        "inter-token interval",
    )
    def test_a_leading_withhold_does_not_overstate_time_to_first_token(self):
        result = stream_scripted(generation(withhold=0))
        ttft = sample_from(result).ttft_ms()
        assert ttft is not None
        assert abs(ttft - TTFT_S * 1000) / (TTFT_S * 1000) < 0.20


def test_a_malformed_frame_is_dropped_without_a_trace():
    """BQ-F09. The frame is skipped, and nothing anywhere records that it happened."""
    frames = [
        fakes.Frame.of(fakes.text_chunk(" a")),
        fakes.Frame.of(b"{not json", GAP_S),
        fakes.Frame.of(fakes.text_chunk(" b"), GAP_S),
        fakes.Frame.of(fixture("llama_cpp_final")),
    ]
    result = stream_scripted(frames)
    assert len(result.arrivals) == 2
    # chunks_seen counts it, but the count reaches no field of the report, so a run
    # cannot be told apart from one where the wire was clean.
    assert result.chunks_seen == 4


def test_a_response_carrying_no_text_is_refused():
    frames = [fakes.Frame.of(fixture("llama_cpp_final"))]
    with pytest.raises(ServingError, match="no text chunks"):
        stream_scripted(frames)


def test_urllib_error_types_are_mapped(monkeypatch):
    def boom(*args, **kwargs):
        raise urllib.error.URLError("scripted")

    monkeypatch.setattr(client.urllib.request, "urlopen", boom)
    with pytest.raises(ServingTransientError):
        client.stream(llama_adapter(), {"prompt": "x"}, timeout=1.0)
