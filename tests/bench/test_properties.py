"""Properties of the reduction, over inputs nobody chose.

The example-based tests beside this one prove the cases their author thought of. A
property holds over a generated population, and the population is what catches the
case nobody thought of -- the empty sample, the two-element sample, the series where
every value is identical.

Seeded rather than random. A property test that cannot be re-run on the input that
broke it is a bug report with the evidence deleted, so every generator here takes the
module seed and any failure reproduces exactly.

No property library is installed, and adding one means regenerating a lockfile the
sandbox cannot resolve, so the generators are `random.Random` with a fixed seed. That
buys the population without the shrinking, which is the part worth having here: these
inputs are lists of floats, and a failing list of floats is already small.
"""

from __future__ import annotations

import random

import pytest

from bench import delta, metrics
from tests.bench import support

SEED = 20260829
CASES = 200


def samples(rng: random.Random, size: int) -> list[float]:
    return [rng.uniform(0.001, 5000.0) for _ in range(size)]


def arrival_series(rng: random.Random, count: int) -> metrics.Sample:
    """A plausible request: a prefill delay, then jittered inter-token gaps."""
    t_submit = rng.uniform(0.0, 1000.0)
    first = t_submit + rng.uniform(0.01, 2.0)
    arrivals = [first]
    for _ in range(count - 1):
        arrivals.append(arrivals[-1] + rng.uniform(0.001, 0.2))
    return metrics.Sample(
        t_submit=t_submit,
        arrivals=tuple(arrivals),
        prompt_tokens=rng.randint(1, 8192),
        completion_tokens=count,
    )


class TestStatsOrdering:
    def test_the_field_names_describe_the_order_of_the_values(self):
        rng = random.Random(SEED)
        for _ in range(CASES):
            values = samples(rng, rng.randint(1, 40))
            block = metrics.stats(values)
            assert block is not None
            assert block["min"] <= block["median"] <= block["max"]
            assert block["min"] <= block["p95"] <= block["max"]
            assert block["p95"] <= block["p99"] <= block["max"]
            assert block["min"] <= block["mean"] <= block["max"]
            assert block["n"] == len(values)

    def test_spread_is_absent_for_one_sample_and_present_above_it(self):
        rng = random.Random(SEED + 1)
        for size in range(1, 12):
            block = metrics.stats(samples(rng, size))
            assert block is not None
            if size == 1:
                assert block["stdev"] is None
            else:
                assert block["stdev"] is not None and block["stdev"] >= 0.0

    def test_an_empty_sample_is_absent_rather_than_a_zero(self):
        assert metrics.stats([]) is None

    def test_a_constant_series_has_no_spread_and_equal_order_statistics(self):
        block = metrics.stats([7.5] * 20)
        assert block is not None
        assert block["stdev"] == 0.0
        assert block["min"] == block["median"] == block["p95"] == block["max"] == 7.5


class TestPercentile:
    def test_a_quantile_never_leaves_the_sample(self):
        rng = random.Random(SEED + 2)
        for _ in range(CASES):
            values = samples(rng, rng.randint(1, 60))
            for q in (0.0, 0.5, 0.95, 0.99, 1.0):
                assert metrics.percentile(values, q) in values

    def test_quantiles_are_monotonic_in_q(self):
        rng = random.Random(SEED + 3)
        for _ in range(CASES):
            values = samples(rng, rng.randint(2, 60))
            previous = metrics.percentile(values, 0.0)
            for q in (0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0):
                current = metrics.percentile(values, q)
                assert current >= previous
                previous = current

    def test_a_quantile_below_one_stays_under_the_maximum_on_a_wide_sample(self):
        # On 100 distinct ascending values the 95th is the 95th, not the largest. The
        # nearest-rank implementation clamps out-of-range ranks, so a quantile larger
        # than one silently returns the maximum; this pins the intended value.
        values = [float(i) for i in range(100)]
        assert metrics.percentile(values, 0.95) == 94.0
        assert metrics.percentile(values, 0.99) == 98.0
        assert metrics.percentile(values, 1.0) == 99.0

    def test_an_empty_sample_has_no_quantile(self):
        with pytest.raises(ValueError):
            metrics.percentile([], 0.5)

    def test_the_zeroth_quantile_is_the_smallest_value(self):
        # The rank is clamped to at least one, and that floor is what makes q=0 name
        # the minimum rather than the second-smallest.
        rng = random.Random(SEED + 14)
        for _ in range(50):
            values = samples(rng, rng.randint(1, 40))
            assert metrics.percentile(values, 0.0) == min(values)

    def test_the_reported_block_uses_the_ninety_fifth_and_ninety_ninth(self):
        # Asserting through stats() rather than percentile() directly, because the
        # quantiles the report publishes are the constants written there.
        block = metrics.stats([float(i) for i in range(100)])
        assert block is not None
        assert block["p95"] == 94.0
        assert block["p99"] == 98.0
        assert block["max"] == 99.0


class TestSampleGuards:
    def test_decode_metrics_appear_exactly_when_a_second_token_exists(self):
        # The guard is `completion_tokens < 2 or chunks < 2`, and its boundary decides
        # whether the prefill arm reports a decode rate it has no basis for.
        rng = random.Random(SEED + 4)
        for count in range(1, 8):
            sample = arrival_series(rng, count)
            if count < 2:
                assert sample.tpot_ms() is None
                assert sample.decode_tok_s() is None
            else:
                assert sample.tpot_ms() is not None
                assert sample.decode_tok_s() is not None

    def test_the_queue_marker_turns_on_at_the_first_level_above_one(self):
        # Concurrency two is where a request can start waiting, so it is where the
        # marker has to change. Above that the flag is already on and a test at
        # sixteen would pass whatever boundary the code used.
        rng = random.Random(SEED + 15)
        one = metrics.reduce_samples([arrival_series(rng, 8)], concurrency=1)
        two = metrics.reduce_samples([arrival_series(rng, 8)], concurrency=2)
        assert one["includes_queue"] is False
        assert two["includes_queue"] is True

    def test_the_latency_series_is_one_shorter_than_the_arrivals(self):
        rng = random.Random(SEED + 5)
        for _ in range(CASES):
            count = rng.randint(1, 40)
            sample = arrival_series(rng, count)
            assert len(sample.itl_ms()) == count - 1

    def test_the_decode_rate_exceeds_the_end_to_end_rate_exactly_when_prefill_is_slow(
        self,
    ):
        """The separation between the two rates is conditional, not a law.

        End to end charges prefill to the denominator, so it is usually quoted where a
        decode rate was meant. Algebra says the gap opens only when time to first
        token exceeds the mean time per output token:

            decode > e2e  <=>  (n-1)/span > n/(ttft+span)  <=>  ttft > span/(n-1)

        On this workstation prefill is far faster per token than decode, so the
        condition holds for every real arm and the two rates do separate. It is still
        a condition. BQ-F20 records that bench.md entry B4 states the separation
        unconditionally, which makes its acceptance criterion an assumption about the
        hardware rather than a property of the measurement.
        """
        rng = random.Random(SEED + 6)
        separated = 0
        for _ in range(CASES):
            sample = arrival_series(rng, rng.randint(2, 40))
            decode = sample.decode_tok_s()
            e2e = sample.e2e_tok_s()
            ttft = sample.ttft_ms()
            tpot = sample.tpot_ms()
            assert None not in (decode, e2e, ttft, tpot)
            assert (decode > e2e) == (ttft > tpot)
            separated += decode > e2e
        # The generator deliberately produces both regimes, so neither branch of the
        # biconditional is left unexercised.
        assert 0 < separated < CASES

    def test_a_response_delivered_in_one_instant_has_no_rate(self):
        # Every arrival sharing a stamp leaves a zero-width span. Dividing by it would
        # be an infinity in a published field, so the guard returns an absent metric.
        sample = metrics.Sample(
            t_submit=10.0,
            arrivals=(11.0, 11.0, 11.0),
            prompt_tokens=8,
            completion_tokens=3,
        )
        assert sample.decode_tok_s() is None
        assert sample.tpot_ms() == 0.0

    def test_an_instant_end_to_end_response_has_no_rate(self):
        sample = metrics.Sample(
            t_submit=11.0,
            arrivals=(11.0, 11.0),
            prompt_tokens=8,
            completion_tokens=2,
        )
        assert sample.e2e_tok_s() is None

    def test_every_latency_is_positive_for_a_monotonic_arrival_series(self):
        rng = random.Random(SEED + 7)
        for _ in range(CASES):
            sample = arrival_series(rng, rng.randint(2, 30))
            assert all(value > 0 for value in sample.itl_ms())
            ttft = sample.ttft_ms()
            assert ttft is not None and ttft > 0


class TestClassifyIsTotal:
    VERDICTS = {
        delta.REGRESSION,
        delta.IMPROVEMENT,
        delta.WITHIN_NOISE,
        delta.BELOW_THRESHOLD,
        delta.INCONCLUSIVE,
    }

    def test_every_input_maps_to_exactly_one_verdict(self):
        rng = random.Random(SEED + 8)
        for _ in range(CASES * 5):
            delta_pct = rng.uniform(-200.0, 200.0)
            noise_pct = rng.choice([None, 0.0, rng.uniform(0.0, 60.0)])
            higher = rng.choice([True, False])
            assert delta.classify(delta_pct, noise_pct, higher) in self.VERDICTS

    def test_a_delta_inside_a_measured_floor_is_never_a_result(self):
        rng = random.Random(SEED + 9)
        for _ in range(CASES):
            floor = rng.uniform(0.1, 60.0)
            inside = rng.uniform(-floor, floor)
            verdict = delta.classify(inside, floor, rng.choice([True, False]))
            assert verdict == delta.WITHIN_NOISE

    def test_the_reporting_bar_is_five_percent_on_both_sides_of_it(self):
        assert delta.classify(4.999, 0.0, True) == delta.BELOW_THRESHOLD
        assert delta.classify(-4.999, 0.0, True) == delta.BELOW_THRESHOLD
        assert delta.classify(5.0, 0.0, True) == delta.IMPROVEMENT
        assert delta.classify(-5.0, 0.0, True) == delta.REGRESSION

    def test_the_sign_reverses_for_a_latency(self):
        rng = random.Random(SEED + 10)
        for _ in range(CASES):
            magnitude = rng.uniform(delta.REGRESSION_PCT + 0.1, 150.0)
            assert delta.classify(magnitude, 0.0, True) == delta.IMPROVEMENT
            assert delta.classify(magnitude, 0.0, False) == delta.REGRESSION
            assert delta.classify(-magnitude, 0.0, True) == delta.REGRESSION
            assert delta.classify(-magnitude, 0.0, False) == delta.IMPROVEMENT

    def test_a_large_delta_with_no_measured_floor_is_inconclusive(self):
        rng = random.Random(SEED + 11)
        for _ in range(CASES):
            magnitude = rng.uniform(delta.REGRESSION_PCT + 0.1, 150.0)
            assert delta.classify(magnitude, None, True) == delta.INCONCLUSIVE


class TestComparingAResultWithItself:
    def test_an_arm_against_itself_reports_no_change_anywhere(self):
        document = support.result()
        deltas, warnings = delta.compare(document, support.result())
        assert deltas
        for entry in deltas:
            assert entry.delta_pct == pytest.approx(0.0)
            assert entry.verdict in {delta.WITHIN_NOISE, delta.BELOW_THRESHOLD}
        assert any("no noise floor" in warning for warning in warnings)

    def test_a_floor_measured_from_an_identical_pair_is_zero(self):
        floor = delta.floor_from_pair(support.result(), support.result())
        assert floor["worst_pct"] == pytest.approx(0.0)
        assert all(value == pytest.approx(0.0) for value in floor["floor"].values())

    def test_a_blocked_run_warns_and_an_interleaved_one_does_not(self):
        # The order warning is the only thing standing between a blocked run and a
        # delta that quietly carries the session's thermal drift on one arm.
        interleaved = support.result()
        _, clean = delta.compare(interleaved, support.result())
        assert not any("interleaved" in warning for warning in clean)

        blocked = support.result()
        blocked["run"]["order"] = "blocked"
        _, warnings = delta.compare(blocked, support.result())
        assert any("interleaved" in warning for warning in warnings)

    def test_an_arm_against_itself_is_never_a_regression(self):
        rng = random.Random(SEED + 12)
        for _ in range(20):
            rate = rng.uniform(1.0, 500.0)
            ttft = rng.uniform(1.0, 5000.0)
            deltas, _ = delta.compare(
                support.result(decode_tok_s=rate, ttft_ms=ttft),
                support.result(decode_tok_s=rate, ttft_ms=ttft),
            )
            assert all(entry.verdict != delta.REGRESSION for entry in deltas)


class TestSlopeRecovery:
    def test_an_exact_line_yields_its_own_slope_and_intercept(self):
        rng = random.Random(SEED + 13)
        for _ in range(CASES):
            slope_ms = rng.uniform(0.01, 10.0)
            intercept = rng.uniform(0.0, 500.0)
            points = [(n, intercept + slope_ms * n) for n in (128, 512, 2048, 8192)]
            fit = metrics.prefill_slope(points)
            assert fit is not None
            assert fit["slope_ms_per_token"] == pytest.approx(slope_ms, rel=1e-6)
            assert fit["intercept_ms"] == pytest.approx(intercept, abs=1e-6)
            assert fit["r2"] == pytest.approx(1.0, abs=1e-9)
            assert fit["prefill_tok_s_slope"] == pytest.approx(
                1000.0 / slope_ms, rel=1e-6
            )

    def test_a_fit_needs_two_distinct_prompt_lengths(self):
        assert metrics.prefill_slope([(128, 10.0)]) is None
        assert metrics.prefill_slope([(128, 10.0), (128, 20.0)]) is None

    def test_scatter_degrades_the_reported_fit(self):
        clean = metrics.prefill_slope([(n, 2.0 * n) for n in (128, 512, 2048, 8192)])
        noisy = metrics.prefill_slope(
            [(128, 256.0), (512, 1024.0), (2048, 9000.0), (8192, 16384.0)]
        )
        assert clean is not None and noisy is not None
        assert clean["r2"] > noisy["r2"]
        assert noisy["r2"] < 0.99

    def test_the_goodness_of_fit_is_the_ratio_of_residual_to_total_variance(self):
        # Pins the arithmetic rather than a direction: squaring the deviations, and
        # dividing rather than multiplying, both change this number.
        points = [(0, 0.0), (1, 1.0), (2, 4.0)]
        fit = metrics.prefill_slope(points)
        assert fit is not None
        assert fit["slope_ms_per_token"] == pytest.approx(2.0)
        assert fit["intercept_ms"] == pytest.approx(-1.0 / 3.0)
        assert fit["r2"] == pytest.approx(12.0 / 13.0, abs=1e-9)

    def test_a_flat_sweep_reports_no_prefill_rate(self):
        # Identical latencies at every prompt length give a zero slope, so there is no
        # rate to report and the field is absent rather than an infinity.
        fit = metrics.prefill_slope([(128, 300.0), (512, 300.0), (2048, 300.0)])
        assert fit is not None
        assert fit["prefill_tok_s_slope"] is None
        assert fit["slope_ms_per_token"] == pytest.approx(0.0)
        # With no variance to explain there is no residual either, and the fit is
        # reported as perfect. Recorded rather than endorsed: a flat sweep is a
        # measurement that failed to resolve prompt length, not a clean line.
        assert fit["r2"] == 1.0

    @pytest.mark.xfail(
        strict=True,
        reason="BQ-F05: a two-point least-squares fit has no residual, so r2 is "
        "exactly 1.0 whatever the data. The quick profile sweeps two prefill "
        "lengths and bench.md B1 accepts r2 at or above 0.95, so that check "
        "cannot fail on the profile it is run with",
    )
    def test_a_two_point_fit_does_not_claim_a_perfect_correlation(self):
        fit = metrics.prefill_slope([(128, 300.0), (1024, 2400.0)])
        assert fit is not None
        assert fit["r2"] < 1.0
