"""Drive the pipelines through a real aggregation engine and compare.

The in-memory store and the MongoDB one answer the same questions two different
ways, and a fake that quietly disagrees with the real thing is worse than no
fake: it makes tests pass that production fails. This module runs both against
the same corpus and asserts identical answers.

mongomock executes the $match/$group/$sort/$limit pipelines in-process, which is
enough to check the filter semantics that are easiest to get wrong -- absent
fields, empty $in, sort tiebreaks. What it cannot do is the operators added
after its coverage stops ($firstN, $dateTrunc) and $vectorSearch, which is not a
MongoDB operator at all but a mongot one.

Where mongomock cannot run a stage the test skips *by name*, saying which
operator was not exercised. A silent skip would leave the impression that a
pipeline had been verified when nothing ran, which is the failure mode this file
exists to prevent -- so the skips are loud and the list of them is short and
known.
"""

from __future__ import annotations

import pytest
from pymongo.errors import OperationFailure

from logtriage.fixtures.generator import generate
from logtriage.mongod.normalize import NormalizedRecord, normalize
from logtriage.mongod.records import LogRecord, parse_lines
from logtriage.store.base import LogFilter, from_document, to_document
from logtriage.store.memory import InMemoryLogStore
from logtriage.store.pipelines import (
    count_pipeline,
    facet_pipeline,
    fetch_by_ids_pipeline,
    sample_pipeline,
    timeline_pipeline,
)

mongomock = pytest.importorskip("mongomock")


def _records(scenario: str = "slow-query-storm") -> list[NormalizedRecord]:
    log = generate(scenario, minutes=30)
    return [normalize(r) for r in parse_lines(log.lines) if isinstance(r, LogRecord)]


@pytest.fixture(scope="module")
def corpus() -> list[NormalizedRecord]:
    return _records()


@pytest.fixture(scope="module")
def memory(corpus) -> InMemoryLogStore:
    return InMemoryLogStore(corpus)


@pytest.fixture(scope="module")
def collection(corpus):
    client = mongomock.MongoClient()
    coll = client["logtriage_test"]["records"]
    coll.insert_many([to_document(r) for r in corpus])
    return coll


# How mongomock reports an operator it does not have. It uses two different
# channels -- NotImplementedError for a group accumulator, OperationFailure for
# an unrecognised expression -- so both are matched, and only on these phrases.
# Anything else is a genuine pipeline error and must fail rather than skip: a
# broad `except OperationFailure: skip` would turn every malformed pipeline into
# a green run, which is the precise failure this module exists to catch.
_UNSUPPORTED_PHRASES = (
    "not supported by Mongomock",
    "Unrecognized expression",
    "is not a valid group operator",
)


def _run(collection, pipeline, operator: str):
    """Execute a pipeline, skipping by name when mongomock cannot.

    Named rather than silent: a skip that does not say what went unexercised is
    indistinguishable from a pass.
    """
    try:
        return list(collection.aggregate(pipeline))
    except (NotImplementedError, OperationFailure) as exc:
        if any(phrase in str(exc) for phrase in _UNSUPPORTED_PHRASES):
            pytest.skip(f"mongomock does not implement {operator}: {exc}")
        raise


FILTERS = [
    pytest.param(LogFilter(), id="unfiltered"),
    pytest.param(LogFilter(components=("NETWORK",)), id="one-component"),
    pytest.param(LogFilter(components=("NETWORK", "COMMAND")), id="two-components"),
    pytest.param(LogFilter(components=()), id="empty-tuple-is-unconstrained"),
    pytest.param(LogFilter(severities=("I",)), id="severity"),
    pytest.param(LogFilter(severities=("F",)), id="severity-matching-nothing"),
    pytest.param(LogFilter(log_ids=(51803,)), id="log-id"),
    pytest.param(LogFilter(namespaces=("shop.orders",)), id="namespace"),
    pytest.param(LogFilter(plan_summary="COLLSCAN"), id="plan-summary"),
    pytest.param(LogFilter(min_duration_ms=1000), id="duration-bound"),
    pytest.param(LogFilter(min_duration_ms=0), id="duration-bound-of-zero"),
    pytest.param(
        LogFilter(components=("COMMAND",), plan_summary="COLLSCAN"), id="conjunction"
    ),
    pytest.param(LogFilter(text="connection"), id="text"),
]


@pytest.mark.parametrize("flt", FILTERS)
def test_count_agrees(memory, collection, flt) -> None:
    """The filter semantics, which is where the two are likeliest to diverge.

    duration-bound-of-zero is the case worth naming: $gte does not match a
    missing field, so every connection message must stay out of a query about
    slow operations -- and a Python implementation treating None as 0 would
    include them all.
    """
    result = _run(collection, count_pipeline(flt), "$count")
    mongo_count = result[0]["count"] if result else 0
    assert mongo_count == memory.count(flt)


@pytest.mark.parametrize("flt", FILTERS)
def test_sample_returns_the_same_records_in_the_same_order(
    memory, collection, flt
) -> None:
    for order in ("time", "-time", "-duration"):
        docs = _run(collection, sample_pipeline(flt, limit=25, order=order), "$sort")
        mongo_ids = [d["_id"] for d in docs]
        memory_ids = [r.record_id for r in memory.sample(flt, limit=25, order=order)]
        assert mongo_ids == memory_ids, f"order={order} diverged"


@pytest.mark.parametrize("by", [["component"], ["severity"], ["ns"], ["ns", "op"]])
def test_facet_counts_agree(memory, collection, by) -> None:
    docs = _run(collection, facet_pipeline(LogFilter(), by, top=50), "$firstN")
    mongo = {tuple(sorted(d["_id"].items())): d["count"] for d in docs}
    expected = {
        tuple(sorted(b.key.items())): b.count
        for b in memory.facet(LogFilter(), by, top=50)
    }
    assert mongo == expected


def test_facet_aggregates_agree(memory, collection) -> None:
    docs = _run(collection, facet_pipeline(LogFilter(), ["ns"], top=50), "$firstN")
    mongo = {d["_id"]["ns"]: d for d in docs}
    for bucket in memory.facet(LogFilter(), ["ns"], top=50):
        found = mongo[bucket.key["ns"]]
        assert found["max_duration_ms"] == bucket.max_duration_ms
        assert found["total_docs_examined"] == bucket.total_docs_examined
        assert list(found["examples"]) == list(bucket.examples)


def test_timeline_agrees(memory, collection) -> None:
    docs = _run(collection, timeline_pipeline(LogFilter(), 300), "$dateTrunc")
    mongo = [(d["_id"], d["count"]) for d in docs]
    expected = [(b.start, b.count) for b in memory.timeline(LogFilter(), 300)]
    assert mongo == expected


def test_fetch_by_ids_agrees(memory, collection, corpus) -> None:
    """Citation resolution, the operation a report's honesty rests on."""
    wanted = [r.record_id for r in corpus[:5]] + ["L999999"]
    docs = _run(collection, fetch_by_ids_pipeline(wanted), "$match")
    mongo_ids = sorted(d["_id"] for d in docs)
    memory_ids = sorted(r.record_id for r in memory.fetch_by_ids(wanted))
    assert mongo_ids == memory_ids
    assert "L999999" not in mongo_ids, "an unknown id must resolve to nothing"


def test_stored_documents_rebuild_into_equal_records(collection, corpus) -> None:
    """A round trip through a real driver's BSON encoding, not just a dict."""
    by_id = {r.record_id: r for r in corpus}
    for doc in collection.find(limit=200):
        rebuilt = from_document(doc)
        original = by_id[rebuilt.record_id]
        assert rebuilt.ns == original.ns
        assert rebuilt.duration_ms == original.duration_ms
        assert rebuilt.plan_summary == original.plan_summary
        assert rebuilt.component == original.component
