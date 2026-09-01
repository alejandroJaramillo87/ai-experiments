"""The synthetic corpus: determinism, realism, and the injected faults.

These are the tests that make every later diagnostic test meaningful. If the
generator is not deterministic, a failing diagnostic cannot be reproduced; if
its output does not parse as mongod output, the diagnostics are being tested
against a format nothing produces.
"""

from __future__ import annotations

from datetime import timedelta

import pytest

from logtriage.fixtures.generator import HOT_NS, generate
from logtriage.fixtures.scenarios import SCENARIO_NAMES, SCENARIOS
from logtriage.mongod.normalize import SLOW_QUERY_ID, normalize
from logtriage.mongod.records import LogRecord, ParseError, parse_lines


def _records(scenario: str, **kwargs) -> list[LogRecord]:
    log = generate(scenario, **kwargs)
    parsed = list(parse_lines(log.lines))
    assert not [p for p in parsed if isinstance(p, ParseError)]
    return [p for p in parsed if isinstance(p, LogRecord)]


@pytest.mark.parametrize("scenario", SCENARIO_NAMES)
def test_every_line_parses(scenario: str) -> None:
    """The generator's output must be indistinguishable from mongod's."""
    assert _records(scenario, minutes=45)


@pytest.mark.parametrize("scenario", SCENARIO_NAMES)
def test_same_seed_is_byte_identical(scenario: str) -> None:
    first = generate(scenario, seed=7, minutes=30)
    second = generate(scenario, seed=7, minutes=30)
    assert first.text() == second.text()


def test_different_seeds_differ() -> None:
    """Otherwise the seed is decorative and every corpus is the same corpus."""
    assert generate("baseline", seed=1).text() != generate("baseline", seed=2).text()


@pytest.mark.parametrize("scenario", SCENARIO_NAMES)
def test_records_are_in_timestamp_order(scenario: str) -> None:
    """mongod appends; a corpus out of order would let a detector cheat."""
    timestamps = [r.ts for r in _records(scenario, minutes=45)]
    assert timestamps == sorted(timestamps)


def test_unknown_scenario_raises_rather_than_falling_back() -> None:
    """A typo must not silently yield a fault-free corpus."""
    with pytest.raises(KeyError, match="unknown scenario"):
        generate("slow-query-strom")


def test_baseline_contains_slow_queries_that_are_not_faults() -> None:
    """The control is only a control if it contains the thing being detected.

    A healthy system logs slow queries. A detector keying on their presence
    rather than on why they were slow must fail here, not pass.
    """
    slow = [r for r in _records("baseline") if r.log_id == SLOW_QUERY_ID]
    assert slow, "baseline has no slow queries, so it cannot catch a naive detector"
    for record in slow:
        out = normalize(record)
        assert out.plan_summary is not None
        assert out.plan_summary.startswith("IXSCAN")
        assert out.docs_per_returned == 1.0


def test_baseline_has_no_collscans_or_errors() -> None:
    records = _records("baseline")
    assert not [r for r in records if normalize(r).plan_summary == "COLLSCAN"]
    assert not [r for r in records if r.severity in {"E", "F"}]


def test_slow_query_storm_injects_collscans_on_the_hot_namespace() -> None:
    normalized = [normalize(r) for r in _records("slow-query-storm")]
    collscans = [n for n in normalized if n.plan_summary == "COLLSCAN"]
    assert len(collscans) > 100
    assert {n.ns for n in collscans} == {HOT_NS}
    # Every one reads the collection whole and returns almost nothing.
    assert all(n.docs_per_returned and n.docs_per_returned > 1000 for n in collscans)


def test_slow_query_storm_durations_climb_across_the_window() -> None:
    """Cache eviction makes a real storm worsen; a fixed threshold would miss it."""
    collscans = [
        n
        for n in (normalize(r) for r in _records("slow-query-storm"))
        if n.plan_summary == "COLLSCAN" and n.duration_ms
    ]
    first, last = collscans[:20], collscans[-20:]
    assert sum(n.duration_ms or 0 for n in last) > 3 * sum(
        n.duration_ms or 0 for n in first
    )


def test_connection_pool_exhaustion_reaches_refusals() -> None:
    records = _records("connection-pool-exhaustion")
    refused = [r for r in records if "too many open connections" in r.msg]
    assert refused
    assert all(r.severity == "W" for r in refused)
    assert all(r.component == "NETWORK" for r in refused)


def test_connection_pool_exhaustion_accepts_outrun_ends() -> None:
    records = _records("connection-pool-exhaustion")
    accepted = sum(r.msg == "Connection accepted" for r in records)
    ended = sum(r.msg == "Connection ended" for r in records)
    assert accepted > ended


def test_replica_stepdown_slow_queries_precede_the_election() -> None:
    """The trap, asserted as a property of the corpus.

    If this ordering ever inverts, the scenario stops testing what it exists to
    test and every report that blames the election becomes correct.
    """
    records = _records("replica-stepdown")
    first_collscan = min(
        r.ts for r in records if normalize(r).plan_summary == "COLLSCAN"
    )
    first_heartbeat_failure = min(r.ts for r in records if r.component == "REPL_HB")
    election = min(r.ts for r in records if r.component == "ELECTION")

    assert first_collscan < first_heartbeat_failure < election
    # And by a margin large enough that timestamp granularity cannot blur it.
    assert election - first_collscan > timedelta(minutes=5)


def test_replica_stepdown_slow_queries_continue_after_the_election() -> None:
    """The other half of the disproof: the election fixed nothing."""
    records = _records("replica-stepdown")
    election = max(r.ts for r in records if r.component == "ELECTION")
    after = [
        r
        for r in records
        if normalize(r).plan_summary == "COLLSCAN" and r.ts > election
    ]
    assert len(after) > 50


def test_index_build_abort_starts_progresses_and_fails() -> None:
    records = [r for r in _records("index-build-abort") if r.component == "INDEX"]
    messages = [r.msg for r in records]
    assert any("starting" in m for m in messages)
    assert any("progress" in m for m in messages)
    failed = [r for r in records if "failed" in r.msg]
    assert len(failed) == 1
    assert failed[0].severity == "E"
    assert failed[0].attr["namespace"] == HOT_NS


@pytest.mark.parametrize("scenario", SCENARIO_NAMES)
def test_ground_truth_window_matches_the_injected_events(scenario: str) -> None:
    """The answer key is declared apart from the generator; keep them agreeing."""
    truth = SCENARIOS[scenario]
    if scenario == "baseline":
        assert truth.expected_findings == ()
        return
    log = generate(scenario, minutes=60)
    records = [p for p in parse_lines(log.lines) if isinstance(p, LogRecord)]
    fault_start = log.start + timedelta(minutes=truth.fault_start_min)
    fault_end = log.start + timedelta(minutes=truth.fault_end_min)
    in_window = [
        r for r in records if fault_start <= r.ts <= fault_end + timedelta(minutes=1)
    ]
    assert len(in_window) > 20, (
        f"{scenario} declares a window with almost nothing in it"
    )
