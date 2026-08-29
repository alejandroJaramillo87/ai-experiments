"""Comparison verdicts, and the refusals that stop an unfounded one."""

from __future__ import annotations

import pytest

from bench import delta
from tests.bench.support import result


class TestClassify:
    def test_inside_the_floor_is_not_a_result(self):
        # The rig cannot tell this from having changed nothing.
        assert (
            delta.classify(3.0, noise_pct=8.0, higher_is_better=True)
            == delta.WITHIN_NOISE
        )
        assert (
            delta.classify(-3.0, noise_pct=8.0, higher_is_better=True)
            == delta.WITHIN_NOISE
        )

    def test_resolvable_but_under_the_reporting_bar(self):
        assert (
            delta.classify(3.0, noise_pct=1.0, higher_is_better=True)
            == delta.BELOW_THRESHOLD
        )

    def test_clears_both_thresholds(self):
        assert delta.classify(12.0, 2.0, higher_is_better=True) == delta.IMPROVEMENT
        assert delta.classify(-12.0, 2.0, higher_is_better=True) == delta.REGRESSION

    def test_latency_reverses_the_sign(self):
        # A larger time to first token is worse; a larger tokens/second is better.
        assert delta.classify(12.0, 2.0, higher_is_better=False) == delta.REGRESSION
        assert delta.classify(-12.0, 2.0, higher_is_better=False) == delta.IMPROVEMENT

    def test_a_big_delta_with_no_floor_measured_is_inconclusive(self):
        # This is the case the retired tooling reported as a win. Whether the machine
        # can resolve 12% was never established, and on this workstation a 44% spread
        # from an unchanged binary has been recorded.
        assert delta.classify(12.0, None, higher_is_better=True) == delta.INCONCLUSIVE

    def test_a_delta_exactly_on_the_floor_is_within_it(self):
        assert delta.classify(8.0, 8.0, higher_is_better=True) == delta.WITHIN_NOISE

    def test_the_threshold_is_the_skills_five_percent(self):
        assert delta.REGRESSION_PCT == 5.0
        assert delta.classify(4.99, 0.1, higher_is_better=True) == delta.BELOW_THRESHOLD
        assert delta.classify(5.01, 0.1, higher_is_better=True) == delta.IMPROVEMENT


class TestComparability:
    @pytest.mark.parametrize(
        "field,value",
        [
            ("service", "llama-gpu"),
            ("architecture", "dense"),
            ("quantization", "Q4_K_M"),
            ("prompt_source", "rng"),
            ("profile_sha256", "different"),
            ("cpuset", "0-7"),
        ],
    )
    def test_a_mismatch_that_changes_the_experiment_is_refused(self, field, value):
        with pytest.raises(delta.IncomparableError):
            delta.compare(result(), result(**{field: value}))

    def test_every_mismatch_is_named_at_once(self):
        with pytest.raises(delta.IncomparableError) as exc:
            delta.compare(result(), result(service="llama-gpu", quantization="Q4_K_M"))
        assert "service" in str(exc.value) and "quantization" in str(exc.value)

    def test_a_differing_git_sha_warns_rather_than_refuses(self):
        _, warnings = delta.compare(
            result(), result(git_sha="cafe"), noise={"floor": {"ttft_ms": 1.0}}
        )
        assert any("git_sha differs" in w for w in warnings)

    def test_a_noise_file_that_never_measured_this_metric_is_still_unmeasured(self):
        # Supplying a floor file does not make every metric in the comparison
        # resolvable; one the floor run never produced is as unknown as no file at all.
        deltas, _ = delta.compare(
            result(ttft_ms=100.0), result(arm="b", ttft_ms=150.0), noise={"floor": {}}
        )
        ttft = next(d for d in deltas if d.metric == "ttft_ms")
        assert ttft.noise_pct is None
        assert ttft.verdict == delta.INCONCLUSIVE

    def test_a_missing_noise_file_warns_and_names_the_command(self):
        _, warnings = delta.compare(result(), result())
        assert any("bench noise" in w for w in warnings)


class TestCompare:
    def test_differences_the_matching_points(self):
        deltas, _ = delta.compare(
            result(decode_tok_s=35.0), result(arm="b", decode_tok_s=42.0)
        )
        decode = next(d for d in deltas if d.metric == "decode_tok_s")
        assert decode.delta_pct == pytest.approx(20.0)
        assert decode.higher_is_better is True

    def test_a_faster_service_regresses_on_latency_and_improves_on_rate(self):
        floor = {"floor": {"ttft_ms": 1.0, "decode_tok_s": 1.0}}
        deltas, _ = delta.compare(
            result(ttft_ms=100.0, decode_tok_s=35.0),
            result(arm="b", ttft_ms=150.0, decode_tok_s=42.0),
            noise=floor,
        )
        by_metric = {d.metric: d for d in deltas}
        assert by_metric["ttft_ms"].verdict == delta.REGRESSION
        assert by_metric["decode_tok_s"].verdict == delta.IMPROVEMENT

    def test_the_point_key_comes_from_identity_not_from_a_statistic(self):
        deltas, _ = delta.compare(result(), result(arm="b"))
        assert {d.point for d in deltas} >= {
            "prefill/p128/g1/c1",
            "decode/p128/g256/c1",
        }

    def test_the_prefill_slope_is_compared(self):
        deltas, _ = delta.compare(result(), result(arm="b"))
        assert any(d.metric == "prefill_tok_s_slope" for d in deltas)


class TestNoiseFloor:
    def test_the_floor_is_the_spread_between_two_identical_arms(self):
        floor = delta.floor_from_pair(
            result(arm="a", decode_tok_s=35.0), result(arm="a-prime", decode_tok_s=38.5)
        )
        assert floor["floor"]["decode/p128/g256/c1:decode_tok_s"] == pytest.approx(10.0)
        assert floor["worst_pct"] == pytest.approx(10.0)

    def test_a_measured_floor_downgrades_a_delta_beneath_it(self):
        floor = delta.floor_from_pair(
            result(arm="a", decode_tok_s=35.0), result(arm="a-prime", decode_tok_s=38.5)
        )
        # An 8% "win" under a 10% floor is not a win.
        deltas, _ = delta.compare(
            result(decode_tok_s=35.0), result(arm="b", decode_tok_s=37.8), noise=floor
        )
        decode = next(d for d in deltas if d.metric == "decode_tok_s")
        assert decode.delta_pct == pytest.approx(8.0)
        assert decode.verdict == delta.WITHIN_NOISE

    def test_an_identical_pair_has_a_zero_floor(self):
        assert (
            delta.floor_from_pair(result(), result(arm="a-prime"))["worst_pct"] == 0.0
        )
