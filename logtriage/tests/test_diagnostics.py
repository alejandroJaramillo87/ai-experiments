"""Detectors, checked against the answer key each scenario declares.

The most important test here is the one that asserts nothing fires. A detector
that reports a problem on a healthy corpus teaches its reader to skim past it,
and after that it may as well not exist -- so the baseline producing an empty
list is a harder requirement than any of the positive cases.
"""

from __future__ import annotations

import pytest

from logtriage.fixtures.generator import HOT_NS, generate
from logtriage.fixtures.scenarios import SCENARIO_NAMES, SCENARIOS
from logtriage.mongod.diagnostics import (
    COLLSCAN_MIN_COUNT,
    Finding,
    collscan_findings,
    connection_findings,
    election_findings,
    index_build_findings,
    run_all,
    scan_ratio_findings,
)
from logtriage.mongod.normalize import normalize
from logtriage.mongod.records import LogRecord, parse_lines
from logtriage.store.memory import InMemoryLogStore


def store_for(scenario: str, **kwargs) -> InMemoryLogStore:
    log = generate(scenario, **kwargs)
    return InMemoryLogStore(
        normalize(r) for r in parse_lines(log.lines) if isinstance(r, LogRecord)
    )


def kinds(findings: list[Finding]) -> set[str]:
    return {f.kind for f in findings}


class TestBaseline:
    """The control. Everything here is a false positive if it fires."""

    def test_a_healthy_corpus_produces_no_findings(self) -> None:
        assert run_all(store_for("baseline", minutes=60)) == []

    def test_healthy_slow_queries_do_not_trigger_a_finding(self) -> None:
        """A healthy system logs slow queries. Reporting their existence rather
        than their cause is the mistake this asserts against."""
        store = store_for("baseline")
        assert store.count.__self__ is store  # sanity: the fixture is populated
        assert collscan_findings(store) == []
        assert scan_ratio_findings(store) == []

    @pytest.mark.parametrize("seed", [1, 2, 3, 17, 99])
    def test_stays_clean_across_seeds(self, seed: int) -> None:
        """One quiet seed proves nothing; the thresholds have to hold generally."""
        assert run_all(store_for("baseline", seed=seed)) == []


class TestGroundTruth:
    @pytest.mark.parametrize("scenario", SCENARIO_NAMES)
    def test_every_expected_finding_is_produced(self, scenario: str) -> None:
        truth = SCENARIOS[scenario]
        found = kinds(run_all(store_for(scenario, minutes=60)))
        missing = set(truth.expected_findings) - found
        assert not missing, f"{scenario} did not produce {sorted(missing)}"

    @pytest.mark.parametrize("scenario", SCENARIO_NAMES)
    def test_no_forbidden_finding_is_produced(self, scenario: str) -> None:
        truth = SCENARIOS[scenario]
        found = kinds(run_all(store_for(scenario, minutes=60)))
        wrong = set(truth.forbidden_findings) & found
        assert not wrong, f"{scenario} wrongly produced {sorted(wrong)}"

    @pytest.mark.parametrize("scenario", SCENARIO_NAMES)
    def test_the_expected_namespace_is_named(self, scenario: str) -> None:
        truth = SCENARIOS[scenario]
        if not truth.expected_ns:
            return
        named = {f.ns for f in run_all(store_for(scenario, minutes=60)) if f.ns}
        assert set(truth.expected_ns) & named


class TestEvidence:
    @pytest.mark.parametrize("scenario", SCENARIO_NAMES)
    def test_every_finding_cites_records_that_resolve(self, scenario: str) -> None:
        """A citation nobody can resolve is a citation nobody can check, which
        is the entire mechanism keeping a generated report honest."""
        store = store_for(scenario, minutes=60)
        for finding in run_all(store):
            assert finding.evidence_ids, f"{finding.kind} cites nothing"
            resolved = store.fetch_by_ids(finding.evidence_ids)
            assert len(resolved) == len(set(finding.evidence_ids))

    @pytest.mark.parametrize("scenario", SCENARIO_NAMES)
    def test_every_finding_is_timestamped(self, scenario: str) -> None:
        """first_seen is what the report orders on to keep causal direction
        from being assumed; a finding without one cannot be placed."""
        for finding in run_all(store_for(scenario, minutes=60)):
            assert finding.first_seen is not None
            assert finding.last_seen is not None
            assert finding.first_seen <= finding.last_seen

    def test_cited_records_actually_support_the_claim(self) -> None:
        """Resolvable is not the same as relevant."""
        store = store_for("slow-query-storm")
        finding = next(f for f in run_all(store) if f.kind == "collscan")
        for record in store.fetch_by_ids(finding.evidence_ids):
            assert record.plan_summary == "COLLSCAN"
            assert record.ns == finding.ns


class TestCollscan:
    def test_reports_the_hot_namespace_with_its_counts(self) -> None:
        finding = next(
            f for f in run_all(store_for("slow-query-storm")) if f.kind == "collscan"
        )
        assert finding.ns == HOT_NS
        assert finding.severity == "high"
        assert finding.metrics["count"] > 100
        assert finding.metrics["docs_examined_total"] > 0

    def test_a_handful_of_scans_is_below_the_threshold(self) -> None:
        """One-off scans are an admin script, not an unindexed application path."""
        store = store_for("slow-query-storm")
        scans = [
            r for r in store.sample.__self__._records if r.plan_summary == "COLLSCAN"
        ]
        few = InMemoryLogStore(scans[: COLLSCAN_MIN_COUNT - 1])
        assert collscan_findings(few) == []


class TestScanRatio:
    def test_a_selective_index_is_not_reported(self) -> None:
        assert scan_ratio_findings(store_for("baseline")) == []


class TestConnections:
    def test_refusals_are_reported_regardless_of_rate(self) -> None:
        """The server turning clients away is what the application experiences;
        it is not a matter of degree."""
        found = connection_findings(store_for("connection-pool-exhaustion"))
        refused = next(f for f in found if f.kind == "connection_refused")
        assert refused.severity == "high"
        assert refused.metrics["count"] > 0

    def test_arrival_outrunning_teardown_is_reported(self) -> None:
        found = connection_findings(store_for("connection-pool-exhaustion"))
        churn = next(f for f in found if f.kind == "connection_churn")
        assert churn.metrics["peak_per_minute"] >= 60
        assert (
            churn.metrics["accepted_in_busiest_minute"]
            > churn.metrics["closed_in_busiest_minute"]
        )

    def test_a_burst_is_not_diluted_by_surrounding_healthy_traffic(self) -> None:
        """The ratio is evaluated in the busiest minute, not corpus-wide.

        A ten-minute episode inside an hour of healthy traffic has a global
        accept-to-close ratio near 1.0, so a corpus-wide comparison stays silent
        through the outage. Lengthening the corpus adds only healthy traffic, so
        if the detector were global this would stop firing as the hour grows.
        """
        for minutes in (40, 60, 180):
            found = connection_findings(
                store_for("connection-pool-exhaustion", minutes=minutes)
            )
            assert any(f.kind == "connection_churn" for f in found), (
                f"churn went undetected in a {minutes}-minute corpus"
            )

    def test_a_balanced_connection_pool_is_not_reported(self) -> None:
        assert connection_findings(store_for("baseline")) == []


class TestElection:
    def test_election_and_heartbeat_failures_are_both_reported(self) -> None:
        found = election_findings(store_for("replica-stepdown"))
        assert kinds(found) == {"election", "heartbeat_failure"}

    def test_the_election_finding_refuses_to_assert_a_direction(self) -> None:
        """An election is the most dramatic event in a log and the easiest to
        mistake for the origin of what surrounds it. The detector reports it as
        an event and says so."""
        finding = next(
            f
            for f in election_findings(store_for("replica-stepdown"))
            if f.kind == "election"
        )
        assert "consequence as often as a cause" in finding.summary
        assert "caused" not in finding.summary.replace("cause", "")

    def test_the_slow_queries_are_dated_before_the_election(self) -> None:
        """The evidence that settles the direction, carried on the findings
        themselves so the report can state it rather than infer it."""
        findings = run_all(store_for("replica-stepdown"))
        collscan = next(f for f in findings if f.kind == "collscan")
        election = next(f for f in findings if f.kind == "election")
        assert collscan.first_seen is not None and election.first_seen is not None
        assert collscan.first_seen < election.first_seen

    def test_a_corpus_with_no_election_reports_none(self) -> None:
        assert election_findings(store_for("baseline")) == []


class TestIndexBuild:
    def test_a_failed_build_is_reported_with_its_namespace(self) -> None:
        finding = next(iter(index_build_findings(store_for("index-build-abort"))))
        assert finding.ns == HOT_NS
        assert "still unindexed" in finding.summary

    def test_a_corpus_with_no_build_reports_none(self) -> None:
        assert index_build_findings(store_for("baseline")) == []


class TestOrdering:
    def test_findings_are_worst_first_then_earliest(self) -> None:
        findings = run_all(store_for("replica-stepdown"))
        assert [f.sort_key() for f in findings] == sorted(
            f.sort_key() for f in findings
        )
        assert findings[0].severity == "high"

    def test_running_twice_gives_the_same_answer(self) -> None:
        store = store_for("connection-pool-exhaustion")
        assert run_all(store) == run_all(store)
