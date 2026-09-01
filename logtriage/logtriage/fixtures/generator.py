"""Seeded synthetic mongod logs with injectable faults.

Nothing real is needed to develop against: this produces a corpus with a known
answer, which is what lets the diagnostics be tested rather than eyeballed. A
seed fixes every choice, so the same arguments always produce byte-identical
output and a diff between two runs means a code change and nothing else.

On the log ids below, reconciled against the Database Manual at commit
88ec8192b924 (manual 8.3), reference/log-messages.txt:

  51803  confirmed exactly -- component COMMAND, severity I, msg "Slow query".
  22943  id and component NETWORK confirmed.
  22944  id and component NETWORK confirmed.

The manual's example messages for the two connection ids are the 4.4-era
templated form, `connection accepted from {session_remote} ...`, which 5.0
replaced with a plain message plus attributes. This module emits the modern
form, so it is the published example that is stale rather than the fixture.

The reconciliation also settled what the corpus cannot supply. MongoDB
publishes no catalogue of log ids; the manual shows fourteen of them, inside
worked examples, and none covers connection refusal, heartbeats, elections or
index builds. The ids used here for those remain *representative* -- plausible
values in the right components, not values verified against a server build --
and that is why diagnostics.py keys on (component, msg) throughout. A detector
matching an id this module invented would be a test that only ever passes
against this module; one matching a documented id would still go quiet after
the call site moved. kb/log_ids.py extracts what the manual documents, and a
corpus-marked test holds this table against it.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from logtriage.fixtures.scenarios import SCENARIOS, GroundTruth

# A fixed instant so a corpus is reproducible across machines and time zones.
# Arbitrary but stable; the tool never reasons about absolute dates.
EPOCH = datetime(2026, 8, 30, 9, 0, 0, tzinfo=UTC)

# Ids the manual documents (see the module docstring).
SLOW_QUERY = 51803
CONNECTION_ACCEPTED = 22943
CONNECTION_ENDED = 22944
DOCUMENTED_IDS = frozenset({SLOW_QUERY, CONNECTION_ACCEPTED, CONNECTION_ENDED})

# Representative ids: the manual documents none of these, and no detector
# matches on them. Declared as a set so a corpus-marked test can hold every id
# this module emits against the two categories and fail on a new one that
# belongs to neither.
CONNECTION_REFUSED = 20883
HEARTBEAT_FAILED = 23974
ELECTION_STARTING = 21438
ELECTION_SUCCEEDED = 21450
STATE_TRANSITION = 21358
INDEX_BUILD_STARTING = 20384
INDEX_BUILD_SCAN_DONE = 20391
INDEX_BUILD_FAILED = 20449
WT_CHECKPOINT = 22430
REPRESENTATIVE_IDS = frozenset(
    {
        CONNECTION_REFUSED,
        HEARTBEAT_FAILED,
        ELECTION_STARTING,
        ELECTION_SUCCEEDED,
        STATE_TRANSITION,
        INDEX_BUILD_STARTING,
        INDEX_BUILD_SCAN_DONE,
        INDEX_BUILD_FAILED,
        WT_CHECKPOINT,
    }
)

# The synthetic deployment. One database, three collections, of which orders is
# the large one every scenario picks on.
DATABASE = "shop"
COLLECTIONS = ("orders", "customers", "sessions")
HOT_NS = "shop.orders"

# Row count for shop.orders, used to size docsExamined on a collection scan so
# the numbers stay internally consistent across a run.
ORDERS_DOC_COUNT = 1_240_000


@dataclass(frozen=True, slots=True)
class GeneratedLog:
    """A synthetic corpus and the answer key that goes with it."""

    lines: tuple[str, ...]
    truth: GroundTruth
    start: datetime
    minutes: int

    def text(self) -> str:
        return "".join(line + "\n" for line in self.lines)


class Builder:
    """Collects events and serialises them in timestamp order.

    Events are appended at arbitrary offsets by the baseline and by whichever
    scenario is active, then sorted once. Sorting on (offset, sequence) rather
    than offset alone keeps two events in the same millisecond in the order they
    were appended, which is what makes the output byte-stable.
    """

    def __init__(self, rng: random.Random, start: datetime) -> None:
        self._rng = rng
        self._start = start
        self._events: list[tuple[int, int, dict[str, Any]]] = []
        self._seq = 0

    @property
    def rng(self) -> random.Random:
        return self._rng

    def emit(
        self,
        offset_ms: int,
        severity: str,
        component: str,
        log_id: int,
        ctx: str,
        msg: str,
        attr: dict[str, Any] | None = None,
    ) -> None:
        ts = self._start + timedelta(milliseconds=offset_ms)
        doc: dict[str, Any] = {
            # Key order matches mongod's own output, so a generated line is
            # indistinguishable from a real one to anything reading it as text.
            "t": {"$date": ts.isoformat(timespec="milliseconds")},
            "s": severity,
            "c": component,
            "id": log_id,
            "ctx": ctx,
            "msg": msg,
        }
        if attr:
            doc["attr"] = attr
        self._events.append((offset_ms, self._seq, doc))
        self._seq += 1

    def lines(self) -> tuple[str, ...]:
        # separators without spaces: mongod writes compact JSON, and a corpus
        # that is 15% whitespace misrepresents every size and throughput number
        # measured against it.
        return tuple(
            json.dumps(doc, separators=(",", ":"))
            for _, _, doc in sorted(self._events, key=lambda e: (e[0], e[1]))
        )


def _client(rng: random.Random) -> str:
    return f"10.0.1.{rng.randint(10, 99)}:{rng.randint(49152, 65535)}"


def _hex(rng: random.Random, width: int = 8) -> str:
    return f"{rng.getrandbits(width * 4):0{width}X}"


def _slow_query_attr(
    rng: random.Random,
    *,
    ns: str,
    plan_summary: str,
    keys_examined: int,
    docs_examined: int,
    nreturned: int,
    duration_ms: int,
    filter_field: str,
) -> dict[str, Any]:
    """One slow-query attr document with internally consistent counters."""
    collection = ns.split(".", 1)[1]
    # workingMillis excludes time blocked on locks and flow control (8.0+), so
    # it is always the smaller of the pair. The gap is what separates a query
    # that is expensive from one that was merely waiting.
    working_ms = max(1, int(duration_ms * rng.uniform(0.82, 0.98)))
    return {
        "type": "command",
        "ns": ns,
        "command": {
            "find": collection,
            "filter": {filter_field: f"value-{rng.randint(1000, 9999)}"},
            "limit": 20,
            "$db": ns.split(".", 1)[0],
        },
        "planSummary": plan_summary,
        "keysExamined": keys_examined,
        "docsExamined": docs_examined,
        "nreturned": nreturned,
        "numYields": max(0, docs_examined // 1000),
        "queryHash": _hex(rng),
        "planCacheKey": _hex(rng),
        "reslen": 220 + nreturned * 180,
        "locks": {"Global": {"acquireCount": {"r": nreturned + 1}}},
        "readConcern": {"level": "local", "provenance": "implicitDefault"},
        "remote": _client(rng),
        "protocol": "op_msg",
        "durationMillis": duration_ms,
        "workingMillis": working_ms,
    }


def _healthy_slow_query(builder: Builder, offset_ms: int, conn: int) -> None:
    """The slow queries a healthy system logs anyway.

    Present in every corpus including the baseline, and the reason the control
    is a real control: a detector keying on "a slow query exists" fires here,
    and must not.
    """
    rng = builder.rng
    ns = f"{DATABASE}.{rng.choice(COLLECTIONS)}"
    nreturned = rng.randint(1, 20)
    builder.emit(
        offset_ms,
        "I",
        "COMMAND",
        SLOW_QUERY,
        f"conn{conn}",
        "Slow query",
        _slow_query_attr(
            rng,
            ns=ns,
            plan_summary="IXSCAN { customerId: 1 }",
            keys_examined=nreturned,
            docs_examined=nreturned,
            nreturned=nreturned,
            duration_ms=rng.randint(101, 240),
            filter_field="customerId",
        ),
    )


def _baseline(builder: Builder, minutes: int) -> None:
    """Steady traffic on a healthy member: connections, queries, checkpoints."""
    rng = builder.rng
    total_ms = minutes * 60_000
    conn = 1000

    # Connection lifecycle roughly every ten seconds, each held for a few
    # seconds. The open count is tracked so accepted/ended lines agree.
    open_count = 40
    for offset in range(0, total_ms, 10_000):
        conn += 1
        jitter = rng.randint(0, 3_000)
        remote = _client(rng)
        open_count += 1
        builder.emit(
            offset + jitter,
            "I",
            "NETWORK",
            CONNECTION_ACCEPTED,
            "listener",
            "Connection accepted",
            {"remote": remote, "connectionId": conn, "connectionCount": open_count},
        )
        held = rng.randint(2_000, 8_000)
        if offset + jitter + held < total_ms:
            open_count -= 1
            builder.emit(
                offset + jitter + held,
                "I",
                "NETWORK",
                CONNECTION_ENDED,
                f"conn{conn}",
                "Connection ended",
                {"remote": remote, "connectionId": conn, "connectionCount": open_count},
            )

    # A handful of legitimately slow queries, every half minute or so.
    for offset in range(0, total_ms, 30_000):
        _healthy_slow_query(
            builder, offset + rng.randint(0, 20_000), rng.randint(1000, conn)
        )

    # Storage checkpoints once a minute.
    for offset in range(0, total_ms, 60_000):
        builder.emit(
            offset,
            "I",
            "WTCHKPT",
            WT_CHECKPOINT,
            "Checkpointer",
            "Checkpoint completed",
            {"durationMillis": rng.randint(80, 400)},
        )


def _inject_slow_query_storm(builder: Builder, truth: GroundTruth) -> None:
    """An unindexed predicate on the large collection, executed repeatedly.

    Duration climbs across the window because the working set stops fitting in
    cache as the scans evict it -- the shape a real storm has, and the reason a
    detector should not key on a fixed duration threshold alone.
    """
    rng = builder.rng
    start_ms = truth.fault_start_min * 60_000
    end_ms = truth.fault_end_min * 60_000
    span = max(1, end_ms - start_ms)
    for offset in range(start_ms, end_ms, 2_000):
        progress = (offset - start_ms) / span
        duration = int(300 + progress * 4_200 + rng.randint(-80, 80))
        builder.emit(
            offset + rng.randint(0, 1_500),
            "I",
            "COMMAND",
            SLOW_QUERY,
            f"conn{rng.randint(1000, 1400)}",
            "Slow query",
            _slow_query_attr(
                rng,
                ns=HOT_NS,
                plan_summary="COLLSCAN",
                keys_examined=0,
                docs_examined=ORDERS_DOC_COUNT,
                nreturned=rng.randint(1, 4),
                duration_ms=duration,
                filter_field="customerEmail",
            ),
        )


def _inject_unselective_index(builder: Builder, truth: GroundTruth) -> None:
    """An index that is used and barely helps.

    The interesting property is that nothing in the plan looks wrong. The server
    reports IXSCAN, the query uses an index, and the only evidence of the fault
    is the ratio between keys examined and documents returned -- which is why a
    reader scanning planSummary values will not find this and a threshold will.
    """
    rng = builder.rng
    start_ms = truth.fault_start_min * 60_000
    end_ms = truth.fault_end_min * 60_000
    ns = f"{DATABASE}.sessions"
    for offset in range(start_ms, end_ms, 4_000):
        nreturned = rng.randint(1, 3)
        # A low-cardinality leading field: the index narrows to a large bucket
        # and the rest of the predicate is applied by reading every one of them.
        keys_examined = rng.randint(40_000, 90_000)
        builder.emit(
            offset + rng.randint(0, 2_000),
            "I",
            "COMMAND",
            SLOW_QUERY,
            f"conn{rng.randint(1000, 1400)}",
            "Slow query",
            _slow_query_attr(
                rng,
                ns=ns,
                plan_summary="IXSCAN { status: 1 }",
                keys_examined=keys_examined,
                docs_examined=keys_examined,
                nreturned=nreturned,
                duration_ms=rng.randint(400, 1_900),
                filter_field="status",
            ),
        )


def _inject_connection_pool_exhaustion(builder: Builder, truth: GroundTruth) -> None:
    """Accept rate outruns teardown until the server starts refusing."""
    rng = builder.rng
    start_ms = truth.fault_start_min * 60_000
    end_ms = truth.fault_end_min * 60_000
    conn = 50_000
    open_count = 200
    # The server's connection cap. Refusals begin once the count reaches it.
    cap = 800
    for offset in range(start_ms, end_ms, 400):
        conn += 1
        remote = _client(rng)
        if open_count < cap:
            open_count += 1
            builder.emit(
                offset,
                "I",
                "NETWORK",
                CONNECTION_ACCEPTED,
                "listener",
                "Connection accepted",
                {"remote": remote, "connectionId": conn, "connectionCount": open_count},
            )
        else:
            builder.emit(
                offset,
                "W",
                "NETWORK",
                CONNECTION_REFUSED,
                "listener",
                "Connection refused because there are too many open connections",
                {"remote": remote, "connectionCount": open_count, "limit": cap},
            )
        # Teardown happens, just far slower than arrival: one end per five
        # accepts is what turns a spike into exhaustion rather than churn.
        if rng.random() < 0.2 and open_count > 0:
            open_count -= 1
            builder.emit(
                offset + 120,
                "I",
                "NETWORK",
                CONNECTION_ENDED,
                f"conn{conn - rng.randint(1, 40)}",
                "Connection ended",
                {"remote": _client(rng), "connectionCount": open_count},
            )


def _inject_replica_stepdown(builder: Builder, truth: GroundTruth) -> None:
    """Storage starvation first, election second. The ordering is the point.

    The slow queries begin six minutes before the first heartbeat failure and
    continue after the new primary is elected. Any reading that makes the
    election the cause has to explain queries that were already slow, and the
    only way to see that is to compare first-occurrence times -- which is why
    the report renderer states them explicitly.
    """
    rng = builder.rng
    slow_start = truth.fault_start_min * 60_000
    slow_end = truth.fault_end_min * 60_000
    hb_start = 22 * 60_000
    election_ms = 24 * 60_000

    for offset in range(slow_start, slow_end, 3_000):
        builder.emit(
            offset + rng.randint(0, 2_000),
            "I",
            "COMMAND",
            SLOW_QUERY,
            f"conn{rng.randint(1000, 1400)}",
            "Slow query",
            _slow_query_attr(
                rng,
                ns=HOT_NS,
                plan_summary="COLLSCAN",
                keys_examined=0,
                docs_examined=ORDERS_DOC_COUNT,
                nreturned=rng.randint(1, 6),
                duration_ms=rng.randint(2_400, 9_800),
                filter_field="customerEmail",
            ),
        )

    for offset in range(hb_start, election_ms, 10_000):
        builder.emit(
            offset,
            "W",
            "REPL_HB",
            HEARTBEAT_FAILED,
            "ReplCoord-2",
            "Heartbeat failed after max retries",
            {
                "target": "mongo-2.internal:27017",
                "maxHeartbeatRetries": 2,
                "error": {"code": 202, "codeName": "NetworkInterfaceExceededTimeLimit"},
            },
        )

    builder.emit(
        election_ms,
        "I",
        "ELECTION",
        ELECTION_STARTING,
        "ReplCoord-3",
        "Starting an election, since we've seen no PRIMARY in the past 10000ms",
        {"electionTimeoutPeriodMillis": 10_000},
    )
    builder.emit(
        election_ms + 1_400,
        "I",
        "ELECTION",
        ELECTION_SUCCEEDED,
        "ReplCoord-3",
        "Election succeeded, assuming primary role in term",
        {"term": 42},
    )
    builder.emit(
        election_ms + 1_600,
        "I",
        "REPL",
        STATE_TRANSITION,
        "ReplCoord-3",
        "Member is now in state PRIMARY",
        {"newState": "PRIMARY", "oldState": "SECONDARY"},
    )


def _inject_index_build_abort(builder: Builder, truth: GroundTruth) -> None:
    """A build that competes for resources for thirteen minutes, then fails."""
    rng = builder.rng
    start_ms = truth.fault_start_min * 60_000
    end_ms = truth.fault_end_min * 60_000
    build_uuid = f"{_hex(rng, 8)}-{_hex(rng, 4)}-{_hex(rng, 4)}"

    builder.emit(
        start_ms,
        "I",
        "INDEX",
        INDEX_BUILD_STARTING,
        "IndexBuildsCoordinatorMongod-0",
        "Index build: starting",
        {
            "namespace": HOT_NS,
            "buildUUID": build_uuid,
            "properties": {
                "v": 2,
                "key": {"customerEmail": 1},
                "name": "customerEmail_1",
            },
        },
    )
    for offset in range(start_ms + 60_000, end_ms, 120_000):
        scanned = int(ORDERS_DOC_COUNT * (offset - start_ms) / (end_ms - start_ms))
        builder.emit(
            offset,
            "I",
            "INDEX",
            INDEX_BUILD_SCAN_DONE,
            "IndexBuildsCoordinatorMongod-0",
            "Index build: collection scan progress",
            {"namespace": HOT_NS, "buildUUID": build_uuid, "totalRecords": scanned},
        )
    builder.emit(
        end_ms,
        "E",
        "INDEX",
        INDEX_BUILD_FAILED,
        "IndexBuildsCoordinatorMongod-0",
        "Index build: failed",
        {
            "namespace": HOT_NS,
            "buildUUID": build_uuid,
            "error": {
                "code": 292,
                "codeName": "QueryExceededMemoryLimitNoDiskUseAllowed",
                "errmsg": "Index build failed: exceeded memory limit for external sort",
            },
        },
    )


_INJECTORS = {
    "baseline": lambda builder, truth: None,
    "slow-query-storm": _inject_slow_query_storm,
    "unselective-index": _inject_unselective_index,
    "connection-pool-exhaustion": _inject_connection_pool_exhaustion,
    "replica-stepdown": _inject_replica_stepdown,
    "index-build-abort": _inject_index_build_abort,
}


def generate(
    scenario: str = "baseline",
    *,
    seed: int = 1,
    minutes: int = 60,
    start: datetime | None = None,
) -> GeneratedLog:
    """Produce a synthetic corpus and its answer key.

    Raises KeyError for an unknown scenario rather than falling back to the
    baseline: a typo that silently produces a fault-free corpus would make every
    diagnostic look broken.
    """
    if scenario not in SCENARIOS:
        raise KeyError(
            f"unknown scenario {scenario!r}; known: {', '.join(sorted(SCENARIOS))}"
        )
    truth = SCENARIOS[scenario]
    builder = Builder(random.Random(seed), start or EPOCH)
    _baseline(builder, minutes)
    _INJECTORS[scenario](builder, truth)
    return GeneratedLog(
        lines=builder.lines(),
        truth=truth,
        start=start or EPOCH,
        minutes=minutes,
    )
