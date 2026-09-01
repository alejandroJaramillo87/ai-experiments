"""Projection of slow-query records, and value-only redaction."""

from __future__ import annotations

import json

from logtriage.mongod.normalize import (
    SLOW_QUERY_ID,
    normalize,
    redact_command,
    redact_value,
)
from logtriage.mongod.records import LogRecord, parse_line

SLOW = json.dumps(
    {
        "t": {"$date": "2026-08-30T14:22:31.123Z"},
        "s": "I",
        "c": "COMMAND",
        "id": SLOW_QUERY_ID,
        "ctx": "conn1234",
        "msg": "Slow query",
        "attr": {
            "type": "command",
            "ns": "shop.orders",
            "command": {
                "find": "orders",
                "filter": {"customerEmail": "a@example.com"},
                "sort": {"createdAt": -1},
                "$db": "shop",
            },
            "planSummary": "COLLSCAN",
            "keysExamined": 0,
            "docsExamined": 1_240_000,
            "nreturned": 4,
            "numYields": 1210,
            "queryHash": "A1B2C3D4",
            "planCacheKey": "E5F6A7B8",
            "remote": "10.0.1.42:53422",
            "durationMillis": 4312,
            "workingMillis": 4180,
        },
    }
)


def _record(raw: str) -> LogRecord:
    record = parse_line(raw, 1)
    assert isinstance(record, LogRecord)
    return record


def test_slow_query_fields_are_lifted() -> None:
    out = normalize(_record(SLOW))
    assert out.ns == "shop.orders"
    assert out.op == "find"
    assert out.plan_summary == "COLLSCAN"
    assert out.duration_ms == 4312
    assert out.working_ms == 4180
    assert out.docs_examined == 1_240_000
    assert out.nreturned == 4
    assert out.query_hash == "A1B2C3D4"
    assert out.remote == "10.0.1.42:53422"


def test_non_slow_query_records_have_no_projection() -> None:
    """The projection is meaningless outside 51803 and must stay None there."""
    raw = json.dumps(
        {
            "t": {"$date": "2026-08-30T14:22:31.123Z"},
            "s": "I",
            "c": "NETWORK",
            "id": 22943,
            "ctx": "listener",
            "msg": "Connection accepted",
            "attr": {"remote": "10.0.1.9:5000", "connectionCount": 41},
        }
    )
    out = normalize(_record(raw))
    assert out.ns is None
    assert out.duration_ms is None
    assert out.plan_summary is None
    # The envelope still projects, and attr survives for citation.
    assert out.component == "NETWORK"
    assert out.attr["connectionCount"] == 41


def test_docs_per_returned_is_the_collscan_signal() -> None:
    out = normalize(_record(SLOW))
    assert out.docs_per_returned == 1_240_000 / 4


def test_ratios_are_none_when_nothing_was_returned() -> None:
    """A zero-result scan must not be given an invented ratio.

    Dividing by a floor would rank it against queries that did return rows; the
    signal for that case is docs_examined on its own.
    """
    doc = json.loads(SLOW)
    doc["attr"]["nreturned"] = 0
    out = normalize(_record(json.dumps(doc)))
    assert out.scan_ratio is None
    assert out.docs_per_returned is None
    assert out.docs_examined == 1_240_000


def test_namespace_falls_back_to_the_command_document() -> None:
    """Not every slow-query record carries attr.ns."""
    doc = json.loads(SLOW)
    del doc["attr"]["ns"]
    out = normalize(_record(json.dumps(doc)))
    assert out.ns == "shop.orders"


def test_redaction_keeps_field_names_and_drops_values() -> None:
    """Field names are the diagnostic content; values are the risk and the cost."""
    command = json.loads(SLOW)["attr"]["command"]
    out = redact_command(command)
    assert out["filter"] == {"customerEmail": "<string>"}
    assert "customerEmail" in json.dumps(out)
    assert "a@example.com" not in json.dumps(out)


def test_redaction_preserves_structural_subdocuments() -> None:
    """sort/projection/hint answer 'would an index help' and must survive."""
    command = json.loads(SLOW)["attr"]["command"]
    out = redact_command(command)
    assert out["sort"] == {"createdAt": -1}


def test_redaction_collapses_large_arrays_with_a_count() -> None:
    """A $in over ten thousand ids would otherwise consume the whole budget."""
    out = redact_value({"$in": list(range(10_000))})
    assert out == {"$in": ["<int>", "<+9999 more>"]}


def test_redaction_preserves_operator_keys_and_nesting() -> None:
    out = redact_value({"$and": [{"a": {"$gt": 5}}, {"b": "x"}]})
    assert out == {"$and": [{"a": {"$gt": "<int>"}}, "<+1 more>"]}


def test_redaction_is_bounded_on_pathological_nesting() -> None:
    deep: dict = {}
    node = deep
    for _ in range(60):
        node["n"] = {}
        node = node["n"]
    assert "<...>" in json.dumps(redact_value(deep))


def test_ingest_redaction_is_off_by_default_and_opt_in() -> None:
    record = _record(SLOW)
    assert normalize(record).attr["command"]["filter"] == {
        "customerEmail": "a@example.com"
    }
    redacted = normalize(record, redact=True)
    assert redacted.attr["command"]["filter"] == {"customerEmail": "<string>"}
    # The projection is unaffected either way: it holds counters, not values.
    assert redacted.docs_examined == 1_240_000
