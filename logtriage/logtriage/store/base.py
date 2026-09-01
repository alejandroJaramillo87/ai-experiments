"""The log store interface, and the document form both implementations share.

Two implementations exist: an in-memory one that needs nothing, and a MongoDB
one. They are held to the same interface so the whole tool runs end to end with
no services at all -- which is what makes the pipeline testable on a machine
with no database, and what will make it testable on the MacBook before mongod is
installed there.

The interface is deliberately small and concrete rather than a general query
language. Five operations cover every question the diagnostics and the model's
tools actually ask, and each maps to one MongoDB aggregation. A general
abstraction would have been harder to mirror faithfully, and a mirror that is
not faithful is worse than no mirror: the fake would pass tests the real one
fails.

fetch_by_ids is the one that looks like a convenience and is not. Every claim
this tool makes carries the ids of the records it came from, and a citation that
cannot be resolved is a citation nobody can check.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, fields
from datetime import UTC, datetime
from typing import Any, Literal, Protocol

from logtriage.mongod.normalize import NormalizedRecord

# How a sample is ordered. A closed set rather than a sort specification: these
# are the three orders anything asks for, and accepting arbitrary sorts would
# mean the fake and the real store could diverge on a sort nothing tested.
SampleOrder = Literal["time", "-time", "-duration"]

# Fields a facet may group by. Closed for the same reason, and because grouping
# by an unindexed field on a large collection is a performance trap that should
# not be reachable by passing a string.
FacetField = Literal[
    "component", "severity", "log_id", "ns", "plan_summary", "op", "ctx", "query_hash"
]

# Field names carried into the store document. Derived from the dataclass so a
# field added to NormalizedRecord cannot be silently dropped on the way to Mongo.
DOCUMENT_FIELDS = tuple(f.name for f in fields(NormalizedRecord))


@dataclass(frozen=True, slots=True)
class LogFilter:
    """A conjunctive filter over log records.

    Every field is optional and unset fields do not constrain. All set fields
    must match, and a tuple field matches if the record's value is any member --
    the shape MongoDB's $in gives directly, so the two implementations agree
    without either interpreting the other's semantics.
    """

    start: datetime | None = None
    end: datetime | None = None
    components: tuple[str, ...] = ()
    severities: tuple[str, ...] = ()
    log_ids: tuple[int, ...] = ()
    namespaces: tuple[str, ...] = ()
    plan_summary: str | None = None
    ctx: str | None = None
    min_duration_ms: int | None = None
    # Substring of msg, case-insensitive. Not a regex: a user-supplied regex
    # against a large collection is a denial-of-service on the server, and this
    # field exists for "find the connection messages", not for pattern matching.
    text: str | None = None


@dataclass(frozen=True, slots=True)
class Bucket:
    """One group of a faceted count, with the aggregates diagnostics need."""

    key: dict[str, Any]
    count: int
    max_duration_ms: int | None = None
    total_docs_examined: int | None = None
    examples: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class TimeBucket:
    """A count of records in one fixed-width slice of time."""

    start: datetime
    count: int


class LogStore(Protocol):
    """Structural queries over log records."""

    def insert(self, records: Iterable[NormalizedRecord]) -> int: ...

    def count(self, flt: LogFilter) -> int: ...

    def facet(
        self, flt: LogFilter, by: Sequence[FacetField], top: int = 20
    ) -> list[Bucket]: ...

    def timeline(self, flt: LogFilter, bucket_seconds: int) -> list[TimeBucket]: ...

    def sample(
        self, flt: LogFilter, limit: int = 10, order: SampleOrder = "time"
    ) -> list[NormalizedRecord]: ...

    def fetch_by_ids(self, ids: Sequence[str]) -> list[NormalizedRecord]: ...


def to_document(record: NormalizedRecord) -> dict[str, Any]:
    """Flatten a record into the document MongoDB stores.

    record_id becomes _id. It is already unique within a corpus and already the
    citation handle, so giving Mongo a second identifier would mean a citation
    and a primary key that could disagree.
    """
    doc: dict[str, Any] = {
        name: getattr(record, name) for name in DOCUMENT_FIELDS if name != "record_id"
    }
    doc["_id"] = record.record_id
    return doc


def from_document(doc: dict[str, Any]) -> NormalizedRecord:
    """Rebuild a record from its stored document."""
    values = {name: doc.get(name) for name in DOCUMENT_FIELDS if name != "record_id"}
    values["record_id"] = doc["_id"]
    values.setdefault("attr", {})
    if values.get("attr") is None:
        values["attr"] = {}
    values["ts"] = _aware(values.get("ts"))
    return NormalizedRecord(**values)  # type: ignore[arg-type]


def _aware(value: Any) -> Any:
    """Restore the UTC timezone BSON does not store.

    BSON dates are epoch milliseconds with no zone, and pymongo hands them back
    as *naive* datetimes unless the client was built with tz_aware=True. Every
    other path in this tool produces aware UTC -- the parser reads an ISO-8601
    timestamp with an offset -- so without this a record read back from MongoDB
    compares unequal to the same record parsed from the log, and comparing one
    against a filter bound raises TypeError.

    The clients this module hands out do set tz_aware, so this is the second of
    two guards. It is here as well because a caller may build their own client,
    and the failure is worth making impossible rather than documenting.
    """
    if isinstance(value, datetime) and value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value
