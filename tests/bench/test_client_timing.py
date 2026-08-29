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

    The reachable llama.cpp branch -- a token leaving an incomplete UTF-8 sequence --
    sends no frame at all, so nothing here can recover the arrival. What is available
    is detection: the server states how many tokens it generated, and an arrival
    series shorter than that count does not resolve the generation it is being
    divided by.

    Where the withhold falls decides which way the error runs, and the client cannot
    tell which case it is in. At either end the decode rate reads high and the mean
    time per output token reads low; a leading withhold puts time to first token one
    interval late and a trailing one inflates the end-to-end rate. In the interior
    every aggregate is correct and one fabricated interval enters the latency series.
    Indistinguishable from arrival stamps alone, so all four arrival-derived
    quantities are refused together rather than a wrong one being published.
    """

    def test_an_interior_withhold_fabricates_a_latency_outlier(self):
        result = stream_scripted(generation(withhold=3))
        sample = sample_from(result)
        itls = sample.itl_ms()
        assert len(itls) == TOKENS - 2
        # One interval spans two token generations. Nothing in the series says which.
        assert max(itls) > GAP_S * 1000 * 1.5

    def test_an_interior_withhold_is_refused_with_the_ends_it_cannot_be_told_from(self):
        # The aggregate happens to be right here, and the client has no way to know
        # that: seven arrivals against eight tokens is the same observation whether
        # the missing frame was the first, the last, or the fourth.
        sample = sample_from(stream_scripted(generation(withhold=3)))
        assert sample.arrivals_resolve_generation is False
        assert sample.decode_tok_s() is None

    def test_a_leading_withhold_does_not_overstate_the_decode_rate(self):
        # 29.2 tok/s against a true 25.0 before the fix: the span loses one interval
        # from the numerator while the denominator stays at usage.completion_tokens.
        sample = sample_from(stream_scripted(generation(withhold=0)))
        assert sample.decode_tok_s() is None
        assert sample.tpot_ms() is None

    def test_a_leading_withhold_does_not_overstate_time_to_first_token(self):
        # The first arrival the client sees is the second token, so the stamp is one
        # inter-token interval late and nothing in the series says so.
        sample = sample_from(stream_scripted(generation(withhold=0)))
        assert sample.ttft_ms() is None

    def test_a_trailing_withhold_does_not_inflate_the_end_to_end_rate(self):
        sample = sample_from(stream_scripted(generation(withhold=TOKENS - 1)))
        assert sample.e2e_tok_s() is None

    def test_a_complete_generation_still_reports_every_metric(self):
        # The refusal must cost nothing on the clean path, which is every sample the
        # pinned engines produce for ASCII prose.
        sample = sample_from(stream_scripted(generation()))
        assert sample.arrivals_resolve_generation is True
        assert sample.ttft_ms() is not None
        assert sample.tpot_ms() is not None
        assert sample.decode_tok_s() is not None
        assert sample.e2e_tok_s() is not None


def test_a_malformed_frame_costs_an_arrival_and_is_detected():
    """BQ-F09, folded into BQ-F01: the same root, an unreconciled arrival count.

    A frame that fails to parse is skipped silently, merging two intervals into one.
    `chunks_seen` counts it and reaches no field of the report, so the only thing that
    can catch it is the server's own token count.
    """
    frames = [
        fakes.Frame.of(fakes.text_chunk(" a")),
        fakes.Frame.of(b"{not json", GAP_S),
        fakes.Frame.of(fakes.text_chunk(" b"), GAP_S),
        fakes.Frame.of(fakes.usage_frame(128, 3)),
    ]
    result = stream_scripted(frames)
    assert len(result.arrivals) == 2
    assert result.chunks_seen == 4
    sample = sample_from(result)
    assert sample.arrivals_resolve_generation is False
    assert sample.decode_tok_s() is None


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
