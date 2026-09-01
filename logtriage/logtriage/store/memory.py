"""An in-memory LogStore, so the whole tool runs with no services.

This is not a stub. It is a second implementation held to the same interface as
the MongoDB one, and it is what lets triage run end to end on a laptop with
nothing installed, in CI, and on the MacBook before mongod is there. The
differential test drives both with the same queries and asserts identical
results, which is how a fake stays honest.

Where the two could disagree, the aggregation pipeline is the authority and this
file follows it -- see store/pipelines.py. The disagreements that actually
matter are all about absent values: MongoDB's comparison operators do not match
a missing field, `$in` over an empty list matches nothing, and sorting puts
missing fields before present ones. Each is mirrored explicitly below rather
than left to Python's defaults, which differ on every one of those points.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from datetime import datetime, timedelta

from logtriage.mongod.normalize import NormalizedRecord
from logtriage.store.base import (
    Bucket,
    FacetField,
    LogFilter,
    SampleOrder,
    TimeBucket,
)

# Examples retained per facet bucket. Matches the $firstN in the pipeline; the
# differential test fails if the two drift apart.
FACET_EXAMPLES = 3


def matches(record: NormalizedRecord, flt: LogFilter) -> bool:
    """Whether one record satisfies a filter.

    Mirrors match_stage. An unset filter field does not constrain, and an empty
    tuple counts as unset -- callers build filters from optional values, and
    MongoDB's `$in: []` semantics (match nothing) would turn an unconstrained
    filter into a silent empty result.
    """
    if flt.start is not None and record.ts < flt.start:
        return False
    if flt.end is not None and record.ts > flt.end:
        return False
    if flt.components and record.component not in flt.components:
        return False
    if flt.severities and record.severity not in flt.severities:
        return False
    if flt.log_ids and record.log_id not in flt.log_ids:
        return False
    if flt.namespaces and record.ns not in flt.namespaces:
        return False
    if flt.plan_summary is not None and record.plan_summary != flt.plan_summary:
        return False
    if flt.ctx is not None and record.ctx != flt.ctx:
        return False
    if flt.min_duration_ms is not None:
        # A record with no duration does not satisfy a duration bound. MongoDB's
        # $gte does not match a missing field, and treating None as zero here
        # would silently include every connection message in a query about slow
        # operations.
        if record.duration_ms is None or record.duration_ms < flt.min_duration_ms:
            return False
    if flt.text and flt.text.lower() not in record.msg.lower():
        return False
    return True


def _sort_value(record: NormalizedRecord, order: SampleOrder) -> tuple:
    """Sort keys mirroring the pipeline's, including its treatment of nulls.

    MongoDB sorts missing and null values before every present value in
    ascending order. Python cannot compare None to an int at all, so the
    ordering is made explicit with a leading presence flag rather than left to
    a comparison that would raise.

    Every key ends with record_id, which is what keeps the order total: mongod
    logs at millisecond resolution and ties are constant, so without it two runs
    of one query cite different records.
    """
    if order == "time":
        return (record.ts, record.record_id)
    if order == "-time":
        return (_Descending(record.ts), record.record_id)
    duration = record.duration_ms
    return (_Descending((duration is not None, duration or 0)), record.record_id)


class _Descending:
    """Reverses one component of a sort key.

    Needed because the sorts here are mixed -- descending on the value,
    ascending on record_id for the tiebreak -- and `reverse=True` would flip
    both, making the tiebreak non-deterministic in the opposite direction.
    """

    __slots__ = ("value",)

    def __init__(self, value: object) -> None:
        self.value = value

    def __lt__(self, other: _Descending) -> bool:
        return other.value < self.value  # type: ignore[operator]

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _Descending) and other.value == self.value

    def __hash__(self) -> int:
        return hash(self.value)


class InMemoryLogStore:
    """A LogStore backed by a list. Linear scans, and deliberately so.

    An index would be a second implementation of MongoDB's query planner, and a
    bug in it would look exactly like a bug in the tool. The corpora this runs
    against in testing are thousands of records, where a scan is instant.
    """

    def __init__(self, records: Iterable[NormalizedRecord] | None = None) -> None:
        self._records: list[NormalizedRecord] = []
        self._by_id: dict[str, NormalizedRecord] = {}
        if records:
            self.insert(records)

    def insert(self, records: Iterable[NormalizedRecord]) -> int:
        added = 0
        for record in records:
            # Last write wins, as an upsert on _id would. Re-ingesting a corpus
            # must not produce two copies of every record and double every count.
            if record.record_id in self._by_id:
                index = self._records.index(self._by_id[record.record_id])
                self._records[index] = record
            else:
                self._records.append(record)
                added += 1
            self._by_id[record.record_id] = record
        return added

    def __len__(self) -> int:
        return len(self._records)

    def _matching(self, flt: LogFilter) -> list[NormalizedRecord]:
        return [r for r in self._records if matches(r, flt)]

    def count(self, flt: LogFilter) -> int:
        return len(self._matching(flt))

    def facet(
        self, flt: LogFilter, by: Sequence[FacetField], top: int = 20
    ) -> list[Bucket]:
        groups: dict[tuple, list[NormalizedRecord]] = {}
        for record in sorted(self._matching(flt), key=lambda r: r.record_id):
            key = tuple(getattr(record, field) for field in by)
            groups.setdefault(key, []).append(record)

        buckets: list[Bucket] = []
        for key, members in groups.items():
            durations = [r.duration_ms for r in members if r.duration_ms is not None]
            examined = [r.docs_examined for r in members if r.docs_examined is not None]
            buckets.append(
                Bucket(
                    key=dict(zip(by, key, strict=True)),
                    count=len(members),
                    max_duration_ms=max(durations) if durations else None,
                    # $sum over a group where no document has the field yields 0,
                    # not null, so an empty list is 0 rather than None here.
                    total_docs_examined=sum(examined),
                    examples=tuple(r.record_id for r in members[:FACET_EXAMPLES]),
                )
            )
        buckets.sort(key=lambda b: (-b.count, _key_order(b.key)))
        return buckets[:top]

    def timeline(self, flt: LogFilter, bucket_seconds: int) -> list[TimeBucket]:
        if bucket_seconds <= 0:
            raise ValueError("bucket_seconds must be positive")
        counts: dict[datetime, int] = {}
        width = timedelta(seconds=bucket_seconds)
        for record in self._matching(flt):
            # $dateTrunc bins from the Unix epoch, not from the first record, so
            # two queries over overlapping windows produce aligned buckets. Doing
            # it any other way makes two timelines impossible to compare.
            epoch = datetime(1970, 1, 1, tzinfo=record.ts.tzinfo)
            elapsed = (record.ts - epoch) // width
            start = epoch + elapsed * width
            counts[start] = counts.get(start, 0) + 1
        return [
            TimeBucket(start=start, count=counts[start]) for start in sorted(counts)
        ]

    def sample(
        self, flt: LogFilter, limit: int = 10, order: SampleOrder = "time"
    ) -> list[NormalizedRecord]:
        ordered = sorted(self._matching(flt), key=lambda r: _sort_value(r, order))
        return ordered[:limit]

    def fetch_by_ids(self, ids: Sequence[str]) -> list[NormalizedRecord]:
        """Resolve citation handles.

        Unknown ids are omitted rather than raising: the caller checking a
        report's citations needs to know which ones failed to resolve, and
        comparing what it asked for against what came back says that, while an
        exception only says that something did.
        """
        found = [self._by_id[i] for i in set(ids) if i in self._by_id]
        return sorted(found, key=lambda r: r.record_id)


def _key_order(key: dict[str, object]) -> tuple[str, ...]:
    """A total, type-safe ordering for facet keys.

    Group keys mix strings, ints and None across fields, so the values cannot be
    compared directly. Rendering them keeps the sort total and matches the
    pipeline's `$sort: {_id: 1}` closely enough for the tiebreak, which is the
    only thing it is used for.
    """
    return tuple(f"{k}={v!r}" for k, v in sorted(key.items()))
