"""Comparison verdicts, and the refusals that stop an unfounded one."""

from __future__ import annotations

import pytest

from bench import delta
from tests.bench import support
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
            ("profile", "different"),
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
            result(),
            result(git_sha="cafe"),
            noise=support.noise({"ttft_ms": 1.0}),
        )
        assert any("git_sha differs" in w for w in warnings)

    def test_a_noise_file_that_never_measured_this_metric_is_still_unmeasured(self):
        # Supplying a floor file does not make every metric in the comparison
        # resolvable; one the floor run never produced is as unknown as no file at all.
        deltas, _ = delta.compare(
            result(ttft_ms=100.0),
            result(arm="b", ttft_ms=150.0),
            noise=support.noise({"decode/p128/g256/c1:decode_tok_s": 2.0}),
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
        floor = support.noise({"ttft_ms": 1.0, "decode_tok_s": 1.0})
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
            result(decode_tok_s=35.0),
            result(arm="b", decode_tok_s=37.8),
            noise=support.noise(floor["floor"]),
        )
        decode = next(d for d in deltas if d.metric == "decode_tok_s")
        assert decode.delta_pct == pytest.approx(8.0)
        assert decode.verdict == delta.WITHIN_NOISE

    def test_an_identical_pair_has_a_zero_floor(self):
        assert (
            delta.floor_from_pair(result(), result(arm="a-prime"))["worst_pct"] == 0.0
        )


def with_observed(document: dict, point: str, metric: str, median: float) -> dict:
    """Attach an observed token count to a named point, in place."""
    for body in document["phases"].values():
        for entry in body.get("points", []):
            if entry["point"] == point:
                entry["metrics"][metric] = support.stat(median)
    return document


class TestTheFloorMustHaveMeasuredThisWorkload:
    """BQ-F03. A floor from another workload is partly applied, not ignored.

    `quick` and `publish` share the point key prefill/p128/g1/c1, and the slope block
    is keyed on the phase alone, so a foreign floor lands on exactly the keys a real
    regression would be graded against.
    """

    def test_a_floor_from_another_profile_is_refused(self):
        with pytest.raises(delta.IncomparableError) as caught:
            delta.compare(
                result(),
                result(arm="b"),
                noise=support.noise({"ttft_ms": 1.0}, profile="publish"),
            )
        assert "profile" in str(caught.value)

    def test_a_floor_from_another_service_is_refused(self):
        with pytest.raises(delta.IncomparableError) as caught:
            delta.compare(
                result(),
                result(arm="b"),
                noise=support.noise({"ttft_ms": 1.0}, service="vllm-gpu"),
            )
        assert "service" in str(caught.value)

    def test_a_floor_naming_no_workload_at_all_is_refused(self):
        # floor_from_pair returns the bare mapping; only bench/commands/noise.py adds
        # the run block. A floor whose provenance is unknown cannot be checked, and an
        # unchecked floor is what this refusal exists to stop.
        with pytest.raises(delta.IncomparableError):
            delta.compare(result(), result(arm="b"), noise={"floor": {"ttft_ms": 1.0}})

    def test_a_foreign_floor_no_longer_absorbs_a_real_regression(self):
        # The reproduction from the ledger: a genuine +40% time to first token graded
        # within_noise against a floor measured at a different repeat count, exit 0,
        # warnings empty.
        foreign = support.noise({"prefill/p128/g1/c1:ttft_ms": 60.0}, profile="publish")
        with pytest.raises(delta.IncomparableError):
            delta.compare(
                result(ttft_ms=100.0), result(arm="b", ttft_ms=140.0), noise=foreign
            )

    def test_a_floor_from_this_workload_is_applied(self):
        deltas, _ = delta.compare(
            result(ttft_ms=100.0),
            result(arm="b", ttft_ms=140.0),
            noise=support.noise({"prefill/p128/g1/c1:ttft_ms": 60.0}),
        )
        ttft = next(d for d in deltas if d.metric == "ttft_ms")
        assert ttft.verdict == delta.WITHIN_NOISE

    def test_a_floor_that_measured_nothing_warns_as_loudly_as_no_floor(self):
        # The warning was guarded on the argument being absent, so a useless floor
        # printed strictly less than naming no floor at all.
        _, warnings = delta.compare(result(), result(arm="b"), noise=support.noise({}))
        assert any("no noise floor" in warning for warning in warnings)


class TestObservedTokenCountsAreNotPerformanceMetrics:
    """BQ-F25, with BQ-F04 folded in. One root: they flow through unfiltered.

    An observed token count records what the server tokenized. It is differenced and
    graded as a latency in `compare`, and it sets `worst_pct` in `floor_from_pair`,
    which the runbook reads as an acceptance signal for the rig.
    """

    POINT = "decode/p128/g256/c1"

    def test_an_observed_count_is_not_differenced_as_a_latency(self):
        deltas, _ = delta.compare(
            with_observed(result(), self.POINT, "prompt_tokens_observed", 128.0),
            with_observed(result(arm="b"), self.POINT, "prompt_tokens_observed", 192.0),
        )
        assert not [d for d in deltas if d.metric == "prompt_tokens_observed"]

    def test_a_pinned_completion_count_is_not_differenced_either(self):
        deltas, _ = delta.compare(
            with_observed(result(), self.POINT, "completion_tokens_observed", 256.0),
            with_observed(
                result(arm="b"), self.POINT, "completion_tokens_observed", 256.0
            ),
        )
        assert not [d for d in deltas if d.metric == "completion_tokens_observed"]

    def test_a_tokenization_difference_cannot_set_the_published_floor(self):
        floor = delta.floor_from_pair(
            with_observed(result(arm="a"), self.POINT, "prompt_tokens_observed", 128.0),
            with_observed(
                result(arm="a-prime"), self.POINT, "prompt_tokens_observed", 192.0
            ),
        )
        assert floor["worst_pct"] == pytest.approx(0.0)
        assert f"{self.POINT}:prompt_tokens_observed" not in floor["floor"]
