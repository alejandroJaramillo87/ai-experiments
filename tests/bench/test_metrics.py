"""Metric reduction against fixtures whose answers are known by construction."""

from __future__ import annotations

import math

import pytest

from bench.metrics import Sample, percentile, prefill_slope, reduce_samples, stats


def make_sample(
    ttft_s: float,
    itl_s: float,
    n_tokens: int,
    prompt_tokens: int = 128,
    chunks: int | None = None,
    server_prompt_ms: float | None = None,
) -> Sample:
    """A request whose first token lands at ttft_s and whose rest arrive itl_s apart."""
    count = n_tokens if chunks is None else chunks
    arrivals = tuple(ttft_s + i * itl_s for i in range(count))
    return Sample(
        t_submit=0.0,
        arrivals=arrivals,
        prompt_tokens=prompt_tokens,
        completion_tokens=n_tokens,
        server_prompt_ms=server_prompt_ms,
    )


class TestSample:
    def test_ttft_is_first_arrival_minus_submit(self):
        assert make_sample(0.25, 0.01, 5).ttft_ms() == pytest.approx(250.0)

    def test_itl_series_is_successive_differences(self):
        itl = make_sample(0.25, 0.01, 4).itl_ms()
        assert itl == pytest.approx([10.0, 10.0, 10.0])

    def test_itl_series_is_one_shorter_than_the_arrivals(self):
        assert len(make_sample(0.1, 0.01, 8).itl_ms()) == 7

    def test_tpot_excludes_prefill(self):
        # Three intervals of 10ms across four tokens: 30ms / 3 = 10ms per token, and
        # the 250ms spent before the first token must not appear in it.
        assert make_sample(0.25, 0.01, 4).tpot_ms() == pytest.approx(10.0)

    def test_decode_rate_excludes_prefill(self):
        # 3 tokens after the first, over 0.03s -> 100 tok/s, regardless of the TTFT.
        assert make_sample(0.25, 0.01, 4).decode_tok_s() == pytest.approx(100.0)
        assert make_sample(9.99, 0.01, 4).decode_tok_s() == pytest.approx(100.0)

    def test_e2e_rate_includes_prefill_and_so_differs_from_decode(self):
        sample = make_sample(0.25, 0.01, 4)
        # 4 tokens over 0.28s: the number the earlier tooling reported as tokens/second.
        assert sample.e2e_tok_s() == pytest.approx(4 / 0.28)
        assert sample.e2e_tok_s() < sample.decode_tok_s()

    def test_packing_cannot_move_the_end_to_end_rate(self):
        # BQ-F01, narrowed. Both endpoints of the end-to-end span are invariant under
        # chunk packing: the submit stamp is taken before the request goes out, and the
        # last token's text has to ride in the last content-bearing chunk. Eight tokens
        # in four chunks and eight tokens in eight chunks that end at the same instant
        # are the same rate, so refusing the packed one discards a figure packing
        # cannot bias.
        packed = Sample(
            t_submit=0.0,
            arrivals=(0.25, 0.26, 0.27, 0.28),
            prompt_tokens=128,
            completion_tokens=8,
        )
        unpacked = Sample(
            t_submit=0.0,
            arrivals=tuple(0.21 + i * 0.01 for i in range(8)),
            prompt_tokens=128,
            completion_tokens=8,
        )
        assert packed.arrivals_resolve_generation is False
        assert unpacked.arrivals_resolve_generation is True
        assert packed.e2e_tok_s() == pytest.approx(8 / 0.28)
        assert packed.e2e_tok_s() == pytest.approx(unpacked.e2e_tok_s())

    def test_packing_leaves_time_to_first_token_reported(self):
        # Packing can only move the first stamp later -- a first chunk carrying two
        # tokens arrives after the second was generated -- so the bias is pessimistic
        # and cannot flatter a result. Refusing it removed the prefill arm's only
        # client-side metric.
        packed = make_sample(0.25, 0.01, 8, chunks=4)
        assert packed.arrivals_resolve_generation is False
        assert packed.ttft_ms() == pytest.approx(250.0)

    def test_packing_still_refuses_the_two_span_derived_metrics(self):
        # The span between first and last arrival loses however many intervals the
        # packing swallowed while the denominator stays at the server's token count,
        # and no stamp says where they went. These two stay refused.
        packed = make_sample(0.25, 0.01, 8, chunks=4)
        assert packed.tpot_ms() is None
        assert packed.decode_tok_s() is None

    def test_a_sample_with_no_arrivals_has_no_arrival_derived_metrics(self):
        # The client refuses a response carrying no text, so this shape does not reach
        # the runner; the reduction is a pure function and must not raise on it.
        empty = Sample(
            t_submit=0.0, arrivals=(), prompt_tokens=128, completion_tokens=8
        )
        assert empty.ttft_ms() is None
        assert empty.e2e_tok_s() is None
        assert empty.tpot_ms() is None
        assert empty.decode_tok_s() is None

    def test_single_token_response_has_no_decode_metrics(self):
        # The prefill arm generates exactly one token; a duration-based fallback here
        # would silently reintroduce prefill into a decode metric.
        single = make_sample(0.25, 0.0, 1)
        assert single.tpot_ms() is None
        assert single.decode_tok_s() is None
        assert single.ttft_ms() == pytest.approx(250.0)


class TestStats:
    def test_reports_the_full_shape(self):
        block = stats([1.0, 2.0, 3.0, 4.0])
        assert block["median"] == 2.5
        assert block["min"] == 1.0
        assert block["max"] == 4.0
        assert block["mean"] == 2.5
        assert block["n"] == 4
        assert block["stdev"] == pytest.approx(1.2909944, rel=1e-6)

    def test_single_sample_has_no_spread(self):
        block = stats([7.0])
        assert block["stdev"] is None
        assert block["median"] == 7.0
        assert block["n"] == 1

    def test_two_samples_have_a_spread(self):
        assert stats([1.0, 3.0])["stdev"] == pytest.approx(math.sqrt(2))

    def test_empty_sample_is_absent_not_zero(self):
        assert stats([]) is None

    def test_p99_is_flagged_below_a_hundred_samples(self):
        assert stats([1.0, 2.0, 3.0])["p99_meaningful"] is False
        assert "p99_meaningful" not in stats([float(i) for i in range(100)])

    def test_percentile_is_nearest_rank(self):
        values = [float(i) for i in range(1, 101)]
        assert percentile(values, 0.95) == 95.0
        assert percentile(values, 0.99) == 99.0
        assert percentile(values, 1.0) == 100.0

    def test_percentile_of_empty_sample_raises(self):
        with pytest.raises(ValueError):
            percentile([], 0.5)


class TestReduceSamples:
    def test_reduces_a_point(self):
        samples = [make_sample(0.25, 0.01, 4) for _ in range(5)]
        block = reduce_samples(samples, concurrency=1)
        assert block["n"] == 5
        assert block["ttft_ms"]["median"] == pytest.approx(250.0)
        assert block["tpot_ms"]["median"] == pytest.approx(10.0)
        assert block["decode_tok_s"]["median"] == pytest.approx(100.0)
        assert block["includes_queue"] is False

    def test_ttft_is_marked_queue_inclusive_above_concurrency_one(self):
        samples = [make_sample(0.25, 0.01, 4)]
        assert reduce_samples(samples, concurrency=16)["includes_queue"] is True

    def test_one_token_per_chunk_is_not_approximate(self):
        samples = [make_sample(0.1, 0.01, 8)]
        block = reduce_samples(samples, concurrency=1)
        assert block["itl_approximate"] is False
        assert block["tokens_per_chunk_mean_max"] == pytest.approx(1.0)

    def test_multi_token_chunks_flag_the_itl_series_as_approximate(self):
        # 8 tokens delivered in 4 chunks: the gaps are chunk gaps, not token gaps.
        samples = [make_sample(0.1, 0.01, 8, chunks=4)]
        block = reduce_samples(samples, concurrency=1)
        assert block["itl_approximate"] is True
        assert block["tokens_per_chunk_mean_max"] == pytest.approx(2.0)

    def test_the_packing_field_is_a_max_over_per_sample_means(self):
        # BQ-F02. The worst single chunk is not observable from arrival stamps, so
        # the field names what it is: each sample's mean packing, maxed over samples.
        # Eight tokens in four chunks is a mean of 2.0 whether the chunks were
        # 2/2/2/2 or 5/1/1/1, and the field reports 2.0 for both.
        block = reduce_samples(
            [make_sample(0.1, 0.01, 8, chunks=4), make_sample(0.1, 0.01, 8, chunks=8)],
            concurrency=1,
        )
        assert block["tokens_per_chunk_mean_max"] == pytest.approx(2.0)
        assert "tokens_per_chunk_max" not in block

    def test_one_unresolved_sample_refuses_the_span_metrics_for_the_whole_point(self):
        # BQ-F01. Refusing per sample took the median over the samples that resolved,
        # and the ones that do not resolve are the ones the scheduler batched hardest
        # -- correlated with load, so systematically slower. Dropping them flatters the
        # published median, which is the direction the refusal exists to prevent. The
        # point refuses as a whole instead, and `n` against `arrivals_incomplete` is
        # what makes the absence reconcilable.
        block = reduce_samples(
            [make_sample(0.1, 0.01, 8, chunks=7)]
            + [make_sample(0.1, 0.01, 8) for _ in range(3)],
            concurrency=1,
        )
        assert block["n"] == 4
        assert block["arrivals_incomplete"] == 1
        assert block["decode_tok_s"] is None
        assert block["tpot_ms"] is None
        # Both endpoints of these two are packing-invariant, so they are reported --
        # over every sample of the point, never a subset of it.
        assert block["ttft_ms"]["n"] == 4
        assert block["e2e_tok_s"]["n"] == 4

    def test_a_resolved_point_reports_the_span_metrics_over_every_sample(self):
        block = reduce_samples([make_sample(0.1, 0.01, 8) for _ in range(4)], 1)
        assert block["arrivals_incomplete"] == 0
        assert block["decode_tok_s"]["n"] == 4
        assert block["tpot_ms"]["n"] == 4

    def test_the_packing_flag_and_the_incomplete_count_are_one_predicate(self):
        # BQ-F01. `itl_approximate` and `arrivals_incomplete` answer the same question
        # -- did some sample deliver fewer chunks than the server counted tokens -- and
        # were computed twice from the same data five lines apart. A sample carrying no
        # arrival at all diverged them: the per-sample packing mean skips it and falls
        # back to 1.0, while the reconciliation counts it.
        cases = [
            [make_sample(0.1, 0.01, 8)],
            [make_sample(0.1, 0.01, 8, chunks=4)],
            [make_sample(0.1, 0.01, 8), make_sample(0.1, 0.01, 8, chunks=7)],
            [make_sample(0.1, 0.01, 8, chunks=0)],
        ]
        for samples in cases:
            block = reduce_samples(samples, concurrency=1)
            assert block["itl_approximate"] == (block["arrivals_incomplete"] > 0)

    def test_a_clean_point_records_no_incomplete_arrivals(self):
        block = reduce_samples([make_sample(0.1, 0.01, 8)], concurrency=1)
        assert block["arrivals_incomplete"] == 0

    def test_prefill_arm_omits_decode_metrics_rather_than_zeroing_them(self):
        samples = [make_sample(0.25, 0.0, 1) for _ in range(3)]
        block = reduce_samples(samples, concurrency=1)
        assert block["tpot_ms"] is None
        assert block["decode_tok_s"] is None
        assert block["ttft_ms"]["n"] == 3

    def test_server_prefill_rate_present_only_when_the_server_reports_timings(self):
        with_timings = [make_sample(0.25, 0.01, 4, server_prompt_ms=50.0)]
        # 128 prompt tokens in 50ms -> 2560 tok/s
        rate = reduce_samples(with_timings, concurrency=1)["prefill_tok_s_server"]
        assert rate["median"] == pytest.approx(2560.0)

        without = [make_sample(0.25, 0.01, 4)]
        assert reduce_samples(without, concurrency=1)["prefill_tok_s_server"] is None

    def test_no_samples_raises(self):
        with pytest.raises(ValueError):
            reduce_samples([], concurrency=1)


class TestPrefillSlope:
    def test_recovers_rate_and_overhead_from_a_perfect_line(self):
        # 20ms fixed overhead plus 1ms per 10 prompt tokens => 10000 tok/s prefill.
        points = [(n, 20.0 + n * 0.1) for n in (128, 512, 2048, 8192)]
        fit = prefill_slope(points)
        assert fit["intercept_ms"] == pytest.approx(20.0)
        assert fit["slope_ms_per_token"] == pytest.approx(0.1)
        assert fit["prefill_tok_s_slope"] == pytest.approx(10_000.0)
        assert fit["r2"] == pytest.approx(1.0)

    def test_the_slope_beats_a_single_point_when_overhead_is_large(self):
        # A 200ms fixed cost dwarfs a 128-token prefill. The naive point estimate
        # charges that cost to prefill; the slope does not.
        points = [(n, 200.0 + n * 0.1) for n in (128, 512, 2048, 8192)]
        naive = 128 / ((200.0 + 12.8) / 1000.0)
        fit = prefill_slope(points)
        assert naive < 700  # badly understated
        assert fit["prefill_tok_s_slope"] == pytest.approx(10_000.0)

    def test_an_outlier_shows_up_as_a_degraded_fit(self):
        clean = prefill_slope([(n, 20.0 + n * 0.1) for n in (128, 512, 2048, 8192)])
        noisy = prefill_slope([(128, 32.8), (512, 71.2), (2048, 900.0), (8192, 839.2)])
        assert clean["r2"] == pytest.approx(1.0)
        assert noisy["r2"] < 0.9

    def test_a_single_point_cannot_be_fitted(self):
        assert prefill_slope([(128, 32.8)]) is None
        assert prefill_slope([]) is None

    def test_identical_prompt_lengths_cannot_be_fitted(self):
        assert prefill_slope([(512, 70.0), (512, 71.0)]) is None
