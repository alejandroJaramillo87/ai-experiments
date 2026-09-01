"""The MongoDB log store: the pipelines in pipelines.py, executed.

Deliberately thin. Every query this store answers was built as data by
`store/pipelines.py` and asserted there, and `tests/test_store_differential.py`
already executes those same pipelines through mongomock against the in-memory
store. So the interesting work -- what a filter means, how ties break, why
examples use $firstN -- happened in that module, and what is left here is
transport: hand the stages to the server, turn the documents back into records.

That split is the reason this file can be reviewed on a machine with no
database. A store that built its pipelines inline would put the semantics
somewhere only a real server could check.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Any

from logtriage.mongod.normalize import NormalizedRecord
from logtriage.store import pipelines
from logtriage.store.base import (
    Bucket,
    FacetField,
    LogFilter,
    SampleOrder,
    TimeBucket,
    from_document,
    to_document,
)

if TYPE_CHECKING:  # pragma: no cover - import shape only
    from pymongo.collection import Collection

# Indexes the pipelines need to not be a collection scan. Each one is the
# grouping or sort key of a query above, and the cost of their absence is
# invisible on a fixture and severe on a real corpus: the profile alone runs
# five facets, and a facet without its index reads every document.
#
# ts carries the timeline and every time-bounded filter. component and log_id
# are the two commonest facets. duration_ms serves the "-duration" sample order,
# which is the query behind "show me the slowest operations" and therefore the
# one a reader runs first.
INDEXES: tuple[tuple[tuple[str, int], ...], ...] = (
    (("ts", 1),),
    (("component", 1), ("ts", 1)),
    (("log_id", 1), ("ts", 1)),
    (("ns", 1), ("duration_ms", -1)),
    (("duration_ms", -1),),
)

# Records are keyed by record_id, which is a content handle rather than a
# sequence number, so re-ingesting the same log is a replace and not a
# duplicate-key error. Ingest is the step most likely to be run twice -- the
# first run is how you discover the file was truncated.
UPSERT_BATCH = 1000


class MongoLogStore:
    """LogStore over a MongoDB collection."""

    def __init__(self, collection: Collection[dict[str, Any]]) -> None:
        self._collection = collection

    def ensure_indexes(self) -> None:
        """Create the indexes the pipelines assume.

        Idempotent, and cheap when they already exist. Called explicitly rather
        than from __init__ because constructing a store should not write to the
        database -- a read-only user must be able to query one.
        """
        for keys in INDEXES:
            self._collection.create_index(list(keys))

    def insert(self, records: Iterable[NormalizedRecord]) -> int:
        from pymongo import ReplaceOne

        operations: list[Any] = []
        written = 0
        for record in records:
            document = to_document(record)
            operations.append(
                ReplaceOne({"_id": document["_id"]}, document, upsert=True)
            )
            if len(operations) >= UPSERT_BATCH:
                self._collection.bulk_write(operations, ordered=False)
                written += len(operations)
                operations = []
        if operations:
            self._collection.bulk_write(operations, ordered=False)
            written += len(operations)
        return written

    def count(self, flt: LogFilter) -> int:
        result = list(self._collection.aggregate(pipelines.count_pipeline(flt)))
        # $count emits no document for an empty match rather than a zero.
        return int(result[0]["count"]) if result else 0

    def facet(
        self, flt: LogFilter, by: Sequence[FacetField], top: int = 20
    ) -> list[Bucket]:
        documents = self._collection.aggregate(pipelines.facet_pipeline(flt, by, top))
        return [
            Bucket(
                key=dict(document["_id"]),
                count=int(document["count"]),
                max_duration_ms=document.get("max_duration_ms"),
                total_docs_examined=document.get("total_docs_examined"),
                examples=tuple(document.get("examples") or ()),
            )
            for document in documents
        ]

    def timeline(self, flt: LogFilter, bucket_seconds: int) -> list[TimeBucket]:
        documents = self._collection.aggregate(
            pipelines.timeline_pipeline(flt, bucket_seconds)
        )
        return [
            TimeBucket(start=document["_id"], count=int(document["count"]))
            for document in documents
        ]

    def sample(
        self, flt: LogFilter, limit: int = 10, order: SampleOrder = "time"
    ) -> list[NormalizedRecord]:
        documents = self._collection.aggregate(
            pipelines.sample_pipeline(flt, limit, order)
        )
        return [from_document(document) for document in documents]

    def fetch_by_ids(self, ids: Sequence[str]) -> list[NormalizedRecord]:
        if not ids:
            return []
        documents = self._collection.aggregate(pipelines.fetch_by_ids_pipeline(ids))
        return [from_document(document) for document in documents]


def log_collection(
    uri: str, database: str = "logtriage", collection: str = "records"
) -> Collection[dict[str, Any]]:
    """Open the collection this store reads.

    Separate from the store so a caller can hold one client for the log
    collection and the knowledge collection, which share a deployment and
    nothing else.

    tz_aware is not optional. BSON stores a date as epoch milliseconds with no
    zone, and pymongo's default is to return a naive datetime -- which every
    other path in this tool would then compare against an aware one, silently
    unequal or raising TypeError against a filter bound.
    """
    from pymongo import MongoClient

    return MongoClient(uri, tz_aware=True)[database][collection]
