"""What each synthetic corpus is supposed to contain, declared independently.

Ground truth lives here, apart from the generator that produces the logs, for
one reason: a test that derives its expectation from the same code that produced
the data proves only that the code is self-consistent. These declarations are
written against the *intent* of each scenario, so a generator change that
quietly stops injecting a fault fails a test instead of moving the goalposts.

This module deliberately imports nothing from the package. It is data.
"""

from __future__ import annotations

from dataclasses import dataclass


# Minute offsets from the start of a generated corpus. Scenarios are declared in
# minutes rather than timestamps so the same declaration holds whatever start
# instant or duration the generator is asked for.
@dataclass(frozen=True, slots=True)
class GroundTruth:
    """The answer key for one scenario."""

    name: str
    summary: str

    # Finding.kind values a correct run must surface.
    expected_findings: tuple[str, ...]

    # Namespaces the fault touches; empty when the fault is not namespace-scoped.
    expected_ns: tuple[str, ...] = ()

    # Minute offsets bounding the injected fault, inclusive of the start.
    fault_start_min: int = 0
    fault_end_min: int = 0

    # Claims a report must not make. These encode the wrong reading that the
    # evidence invites, not merely a false statement -- a guard against a
    # plausible error is worth having, a guard against an arbitrary one is not.
    must_not_conclude: tuple[str, ...] = ()

    # Findings that must NOT appear. A detector that fires on a healthy corpus
    # is worse than no detector, and this is where that gets asserted.
    forbidden_findings: tuple[str, ...] = ()

    notes: str = ""


BASELINE = GroundTruth(
    name="baseline",
    summary="Healthy replica-set member under steady load.",
    expected_findings=(),
    forbidden_findings=(
        "collscan",
        "scan_ratio",
        "connection_churn",
        "election",
        "index_build_abort",
        "severity_spike",
    ),
    notes=(
        "The control. Contains the slow queries a healthy system logs anyway -- "
        "indexed, selective, a few hundred milliseconds -- so a detector that "
        "keys on 'slow query exists' rather than on why it was slow fails here."
    ),
)

SLOW_QUERY_STORM = GroundTruth(
    name="slow-query-storm",
    summary="Unindexed predicate on a large collection; every execution reads it whole.",
    expected_findings=("collscan",),
    expected_ns=("shop.orders",),
    fault_start_min=20,
    fault_end_min=40,
    must_not_conclude=(
        # The queries are slow because the plan is a collection scan, not
        # because the disk is slow; blaming storage sends the reader to the
        # wrong subsystem and the fix (an index) is never found.
        "disk",
        "hardware",
    ),
    notes="planSummary COLLSCAN, docsExamined in the millions, nreturned in single digits.",
)

UNSELECTIVE_INDEX = GroundTruth(
    name="unselective-index",
    summary="An index exists and is used, but barely narrows anything.",
    expected_findings=("scan_ratio",),
    expected_ns=("shop.sessions",),
    fault_start_min=12,
    fault_end_min=38,
    must_not_conclude=(
        # The distinguishing feature of this fault is that the plan looks
        # healthy. A report that says the collection needs an index is wrong:
        # it has one, on the wrong field.
        "no index",
        "missing index",
        "collection scan",
    ),
    notes=(
        "IXSCAN on a low-cardinality field: tens of thousands of keys examined "
        "to return a handful of documents. Nothing in planSummary says this is "
        "wrong, which is why a threshold on keysExamined/nreturned is the only "
        "thing that finds it."
    ),
)

CONNECTION_POOL_EXHAUSTION = GroundTruth(
    name="connection-pool-exhaustion",
    summary="Client connection rate outruns teardown until the server refuses new ones.",
    expected_findings=("connection_churn", "connection_refused", "severity_spike"),
    fault_start_min=25,
    fault_end_min=35,
    must_not_conclude=(
        # Refusals are the symptom of exhaustion, not of a network fault.
        "network partition",
        "firewall",
    ),
    notes="Accept rate far exceeds end rate; open-connection count climbs to the cap.",
)

REPLICA_STEPDOWN = GroundTruth(
    name="replica-stepdown",
    summary="Storage stall starves the primary until its peers elect a new one.",
    expected_findings=("collscan", "election"),
    expected_ns=("shop.orders",),
    fault_start_min=18,
    fault_end_min=45,
    must_not_conclude=(
        # The trap. The election is the most visually dramatic event in the
        # corpus and it lands in the middle of the slow queries, so the
        # available reading is that it caused them. It did not: the slow
        # queries begin six minutes earlier and are what starved the heartbeat
        # thread. A report that inverts this sends an engineer to investigate
        # replication while the actual fault is an unindexed query.
        "election caused",
        "stepdown caused",
        "caused by the election",
        "caused by the stepdown",
        "after the election, queries",
    ),
    notes=(
        "Slow queries start at minute 18; heartbeat failures at 22; election at 24. "
        "The correct causal direction is starvation -> election, and the ordering "
        "in the log is the only evidence that distinguishes it."
    ),
)

INDEX_BUILD_ABORT = GroundTruth(
    name="index-build-abort",
    summary="A foreground-competing index build consumes resources and then aborts.",
    expected_findings=("index_build_abort", "severity_spike"),
    expected_ns=("shop.orders",),
    fault_start_min=15,
    fault_end_min=28,
    must_not_conclude=(
        # The build aborting is the reported event; the queries were already
        # slow because the index it would have created does not exist yet.
        "index build caused the slow queries",
    ),
    notes="Build starts at 15, emits progress, aborts at 28 with an error record.",
)

SCENARIOS: dict[str, GroundTruth] = {
    truth.name: truth
    for truth in (
        BASELINE,
        SLOW_QUERY_STORM,
        UNSELECTIVE_INDEX,
        CONNECTION_POOL_EXHAUSTION,
        REPLICA_STEPDOWN,
        INDEX_BUILD_ABORT,
    )
}

SCENARIO_NAMES: tuple[str, ...] = tuple(SCENARIOS)
