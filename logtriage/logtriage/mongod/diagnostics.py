"""Deterministic detectors, run before the model sees anything.

The ordering is the design decision here. These run first and their output goes
into the prompt, so the model's job is to explain, rank and connect evidence
that has already been found rather than to discover it. That is a direct
response to the way a language model fails on this task: asked to find a cause
in a wall of log lines it will produce a fluent, confident, and occasionally
inverted account, because a plausible narrative is cheaper to generate than a
correct one. Arithmetic over the whole corpus is not something it should be
doing at all, and every number here is arithmetic.

Every finding carries the ids of the records it came from. That is what makes a
claim checkable, and it is why no detector reports a count it cannot cite.

Thresholds are named constants with their reasoning attached. They are the part
most likely to need tuning against a real deployment, so they are gathered where
someone can find and change them rather than buried at their use sites.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Any

from logtriage.store.base import LogFilter, LogStore

# A namespace needs this many collection scans before it is reported. One or two
# are a manual query or an admin script; a sustained count is an application
# path with no index behind it. Set low because the cost of a false positive
# here is a sentence in a report, and the cost of a false negative is the
# finding the whole tool exists to produce.
COLLSCAN_MIN_COUNT = 5

# Index keys read per document returned. 1.0 is a perfectly selective index;
# anything approaching 100 means the server walked a great deal of index to
# return very little, which is the signature of an index that exists but has the
# wrong leading field or the wrong order.
SCAN_RATIO_THRESHOLD = 100.0

# Documents read per document returned, for queries that are not already
# reported as collection scans.
DOCS_PER_RETURNED_THRESHOLD = 1000.0

# Connections accepted per minute above which the rate is called elevated. A
# steady application holds a pool and reconnects rarely; a rate this high means
# connections are being created per request, or a client is reconnect-looping.
CONNECTION_ACCEPT_RATE_PER_MIN = 60

# Accepts to closes. Sustained arrival outrunning teardown is what fills the
# connection cap; a healthy server sits near 1.0 over any long window.
ACCEPT_TO_CLOSE_RATIO = 2.0

# Warnings of one kind before the repetition is itself the signal. A single
# warning is noise; a hundred identical ones is a system in a state.
WARNING_SPIKE_MIN_COUNT = 20

# NETWORK connection lifecycle ids. These three are stable and well known, so
# they are matched by id; everything else below matches on component and message
# text, because the ids for those call sites vary across server versions and a
# detector keyed on one would silently stop firing after an upgrade.
CONNECTION_ACCEPTED_ID = 22943
CONNECTION_ENDED_ID = 22944

# Severity ranking for how loudly a finding is reported.
HIGH, MEDIUM, LOW = "high", "medium", "low"


@dataclass(frozen=True, slots=True)
class Finding:
    """One thing the corpus says, with the records that say it."""

    kind: str
    severity: str
    summary: str
    evidence_ids: tuple[str, ...]
    metrics: dict[str, Any] = field(default_factory=dict)
    ns: str | None = None
    first_seen: datetime | None = None
    last_seen: datetime | None = None

    def sort_key(self) -> tuple:
        """Most severe first, then earliest, then by kind for a total order."""
        rank = {HIGH: 0, MEDIUM: 1, LOW: 2}.get(self.severity, 3)
        return (rank, self.first_seen or datetime.max, self.kind, self.ns or "")


def _span(store: LogStore, flt: LogFilter) -> tuple[datetime | None, datetime | None]:
    """First and last timestamp matching a filter.

    Two one-record samples rather than a scan: the store can serve each from a
    sort, and the alternative is pulling every matching record back to compute a
    min and a max the database already knows how to find.
    """
    first = store.sample(flt, limit=1, order="time")
    last = store.sample(flt, limit=1, order="-time")
    return (first[0].ts if first else None, last[0].ts if last else None)


def collscan_findings(store: LogStore, flt: LogFilter | None = None) -> list[Finding]:
    """Namespaces being read whole to answer a query.

    The highest-value finding this tool produces, because it is the one with an
    unambiguous fix: the query has no index it can use.
    """
    base = flt or LogFilter()
    scan_filter = replace(base, plan_summary="COLLSCAN")
    out: list[Finding] = []
    for bucket in store.facet(scan_filter, ["ns"], top=20):
        ns = bucket.key.get("ns")
        if not ns or bucket.count < COLLSCAN_MIN_COUNT:
            continue
        ns_filter = replace(scan_filter, namespaces=(ns,))
        first, last = _span(store, ns_filter)
        shapes = store.facet(ns_filter, ["query_hash"], top=5)
        out.append(
            Finding(
                kind="collscan",
                severity=HIGH,
                summary=(
                    f"{bucket.count} collection scans on {ns}: the query has no "
                    f"usable index, so every execution reads the collection whole"
                ),
                evidence_ids=bucket.examples,
                metrics={
                    "count": bucket.count,
                    "max_duration_ms": bucket.max_duration_ms,
                    "docs_examined_total": bucket.total_docs_examined,
                    "distinct_query_shapes": len(shapes),
                },
                ns=ns,
                first_seen=first,
                last_seen=last,
            )
        )
    return out


def scan_ratio_findings(store: LogStore, flt: LogFilter | None = None) -> list[Finding]:
    """Indexed queries whose index barely narrows anything.

    Distinct from a collection scan, and easy to miss because the plan looks
    healthy: an IXSCAN that reads a hundred thousand keys to return three rows
    is an index on the wrong field, and nothing about the plan summary says so.
    """
    base = flt or LogFilter()
    out: list[Finding] = []
    # Sampled rather than faceted: the ratio is per record and no aggregation
    # here computes it, so the worst offenders are read back and examined. The
    # limit bounds the work; ordering by duration puts the ones worth seeing at
    # the front.
    candidates = store.sample(
        replace(base, min_duration_ms=1),
        limit=500,
        order="-duration",
    )
    worst: dict[str, list] = {}
    for record in candidates:
        if record.plan_summary == "COLLSCAN" or record.ns is None:
            continue
        ratio = record.scan_ratio
        docs_ratio = record.docs_per_returned
        if (ratio is not None and ratio >= SCAN_RATIO_THRESHOLD) or (
            docs_ratio is not None and docs_ratio >= DOCS_PER_RETURNED_THRESHOLD
        ):
            worst.setdefault(record.ns, []).append(record)

    for ns, records in sorted(worst.items()):
        ratios = [r.scan_ratio for r in records if r.scan_ratio is not None]
        out.append(
            Finding(
                kind="scan_ratio",
                severity=MEDIUM,
                summary=(
                    f"{len(records)} queries on {ns} examined far more index keys "
                    f"than they returned: the index is present but not selective"
                ),
                evidence_ids=tuple(r.record_id for r in records[:3]),
                metrics={
                    "count": len(records),
                    "worst_keys_per_returned": max(ratios) if ratios else None,
                    "plan_summaries": sorted(
                        {r.plan_summary for r in records if r.plan_summary}
                    )[:3],
                },
                ns=ns,
                first_seen=min(r.ts for r in records),
                last_seen=max(r.ts for r in records),
            )
        )
    return out


def connection_findings(store: LogStore, flt: LogFilter | None = None) -> list[Finding]:
    """Connection arrival outrunning teardown, and outright refusals.

    A refusal is reported whatever the rate: the server turning clients away is
    the symptom the application actually experiences, and it is not a matter of
    degree.
    """
    base = flt or LogFilter()
    out: list[Finding] = []

    refused = replace(base, components=("NETWORK",), text="too many open connections")
    refused_count = store.count(refused)
    if refused_count:
        first, last = _span(store, refused)
        out.append(
            Finding(
                kind="connection_refused",
                severity=HIGH,
                summary=(
                    f"{refused_count} connections refused: the server reached its "
                    f"connection limit and turned clients away"
                ),
                evidence_ids=tuple(
                    r.record_id for r in store.sample(refused, limit=3, order="time")
                ),
                metrics={"count": refused_count},
                first_seen=first,
                last_seen=last,
            )
        )

    accepted = replace(base, log_ids=(CONNECTION_ACCEPTED_ID,))
    closed = replace(base, log_ids=(CONNECTION_ENDED_ID,))
    accept_count = store.count(accepted)
    close_count = store.count(closed)
    if not accept_count:
        return out

    # The ratio is evaluated in the busiest minute, not over the whole corpus.
    # Exhaustion is a burst: a ten-minute episode inside an hour of healthy
    # traffic has a global accept-to-close ratio near 1.0, so a corpus-wide
    # comparison dilutes exactly the event worth reporting and the detector
    # stays silent through the outage it exists to catch.
    accepts_per_minute = store.timeline(accepted, 60)
    closes_per_minute = {b.start: b.count for b in store.timeline(closed, 60)}
    peak = max((b.count for b in accepts_per_minute), default=0)
    if not accepts_per_minute:
        return out

    busiest = max(accepts_per_minute, key=lambda b: (b.count, b.start))
    closes_then = closes_per_minute.get(busiest.start, 0)
    ratio = busiest.count / closes_then if closes_then else float("inf")

    if peak >= CONNECTION_ACCEPT_RATE_PER_MIN and ratio >= ACCEPT_TO_CLOSE_RATIO:
        first, last = _span(store, accepted)
        out.append(
            Finding(
                kind="connection_churn",
                severity=MEDIUM if not refused_count else HIGH,
                summary=(
                    f"connection arrival outran teardown: in the busiest minute "
                    f"({busiest.start.isoformat()}) {busiest.count} were accepted "
                    f"against {closes_then} closed"
                ),
                evidence_ids=tuple(
                    r.record_id
                    for r in store.sample(
                        replace(accepted, start=busiest.start), limit=3, order="time"
                    )
                ),
                metrics={
                    "accepted_total": accept_count,
                    "closed_total": close_count,
                    "peak_per_minute": peak,
                    "busiest_minute": busiest.start.isoformat(),
                    "accepted_in_busiest_minute": busiest.count,
                    "closed_in_busiest_minute": closes_then,
                },
                first_seen=first,
                last_seen=last,
            )
        )
    return out


def election_findings(store: LogStore, flt: LogFilter | None = None) -> list[Finding]:
    """Replica-set elections and the heartbeat failures around them.

    Reported as an event, deliberately not as a cause. An election is the most
    visually dramatic thing in a log and the easiest to mistake for the origin
    of whatever surrounds it, so this finding states when it happened and leaves
    the ordering against everything else to the report -- which prints
    first-occurrence times precisely so the direction cannot be assumed.
    """
    base = flt or LogFilter()
    out: list[Finding] = []

    elections = replace(base, components=("ELECTION",))
    count = store.count(elections)
    if count:
        first, last = _span(store, elections)
        out.append(
            Finding(
                kind="election",
                severity=HIGH,
                summary=(
                    f"{count} election records: the replica set changed primary. "
                    f"An election is a consequence as often as a cause -- compare "
                    f"its time against the other findings before concluding either"
                ),
                evidence_ids=tuple(
                    r.record_id for r in store.sample(elections, limit=3, order="time")
                ),
                metrics={"count": count},
                first_seen=first,
                last_seen=last,
            )
        )

    heartbeats = replace(base, components=("REPL_HB",), severities=("W", "E", "F"))
    hb_count = store.count(heartbeats)
    if hb_count:
        first, last = _span(store, heartbeats)
        out.append(
            Finding(
                kind="heartbeat_failure",
                severity=MEDIUM,
                summary=(
                    f"{hb_count} failed heartbeats: members could not reach each "
                    f"other, or a member was too busy to answer in time"
                ),
                evidence_ids=tuple(
                    r.record_id for r in store.sample(heartbeats, limit=3, order="time")
                ),
                metrics={"count": hb_count},
                first_seen=first,
                last_seen=last,
            )
        )
    return out


def index_build_findings(
    store: LogStore, flt: LogFilter | None = None
) -> list[Finding]:
    """Index builds that failed.

    Matched on component and message rather than on a log id: the ids in this
    area differ across server versions, and a detector keyed on one would stop
    firing silently after an upgrade -- the worst way for a detector to fail.
    """
    base = flt or LogFilter()
    failures = replace(base, components=("INDEX",), severities=("E", "F"))
    count = store.count(failures)
    if not count:
        return []
    records = store.sample(failures, limit=3, order="time")
    first, last = _span(store, failures)
    namespaces = sorted(
        {
            str(r.attr.get("namespace"))
            for r in store.sample(failures, limit=20, order="time")
            if r.attr.get("namespace")
        }
    )
    return [
        Finding(
            kind="index_build_abort",
            severity=HIGH,
            summary=(
                f"{count} index build failure(s)"
                + (f" on {', '.join(namespaces)}" if namespaces else "")
                + ": the index does not exist, so queries that needed it are "
                "still unindexed"
            ),
            evidence_ids=tuple(r.record_id for r in records),
            metrics={"count": count, "namespaces": namespaces},
            ns=namespaces[0] if namespaces else None,
            first_seen=first,
            last_seen=last,
        )
    ]


def severity_findings(store: LogStore, flt: LogFilter | None = None) -> list[Finding]:
    """Errors, and warnings repeated enough that the repetition is the signal.

    Grouped by call site, which for mongod is exact: every line carries the id
    of the code that wrote it, so this is a true grouping rather than a
    clustering of similar text.
    """
    base = flt or LogFilter()
    out: list[Finding] = []
    for severities, minimum, level in (
        (("E", "F"), 1, HIGH),
        (("W",), WARNING_SPIKE_MIN_COUNT, LOW),
    ):
        problem = replace(base, severities=severities)
        for bucket in store.facet(problem, ["component", "log_id"], top=10):
            if bucket.count < minimum:
                continue
            component = bucket.key.get("component")
            log_id = bucket.key.get("log_id")
            scoped = replace(
                problem,
                components=(str(component),),
                log_ids=(int(log_id),) if log_id is not None else (),
            )
            sampled = store.sample(scoped, limit=1, order="time")
            message = sampled[0].msg if sampled else "(unknown)"
            first, last = _span(store, scoped)
            out.append(
                Finding(
                    kind="severity_spike",
                    severity=level,
                    summary=f"{bucket.count}x [{severities[0]}] {component}: {message}",
                    evidence_ids=bucket.examples,
                    metrics={
                        "count": bucket.count,
                        "component": component,
                        "log_id": log_id,
                    },
                    first_seen=first,
                    last_seen=last,
                )
            )
    return out


# Order is presentation only; findings are sorted by severity and time before
# being reported. Declared as a tuple so adding a detector is one edit.
DETECTORS = (
    collscan_findings,
    scan_ratio_findings,
    connection_findings,
    election_findings,
    index_build_findings,
    severity_findings,
)


def run_all(store: LogStore, flt: LogFilter | None = None) -> list[Finding]:
    """Run every detector and return findings worst-first.

    A clean corpus returns an empty list, and that is a result rather than a
    failure to find one: a detector that fires on healthy logs teaches its
    reader to ignore it, which is worse than having no detector at all.
    """
    findings: list[Finding] = []
    for detector in DETECTORS:
        findings.extend(detector(store, flt))
    findings.sort(key=lambda f: f.sort_key())
    return findings
