"""MongoDB aggregation pipelines, built as data.

Every function here returns a list of stages and touches no database. That is
what makes the MongoDB adapter reviewable on a machine that has no MongoDB: the
pipelines can be asserted directly, and mongomock can execute them in-process,
long before a real server is available.

The stages are also the specification the in-memory store is checked against.
Where the two could interpret a filter differently -- an empty tuple, a null
field, a missing field -- the pipeline is the authority, because it is the one
that has to be right against a real server.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

from logtriage.store.base import FacetField, LogFilter, SampleOrder


def match_stage(flt: LogFilter) -> dict[str, Any]:
    """The $match document for a filter.

    An unset field contributes nothing, so an empty filter matches everything.
    An empty tuple is treated as unset rather than as "match nothing": callers
    build filters by collecting optional values, and `$in: []` -- which matches
    no document at all -- would turn a filter nobody constrained into a silent
    empty result.
    """
    conditions: dict[str, Any] = {}

    if flt.start is not None or flt.end is not None:
        span: dict[str, Any] = {}
        if flt.start is not None:
            span["$gte"] = flt.start
        if flt.end is not None:
            # Inclusive: a window given as whole minutes should contain a record
            # landing exactly on its upper bound, and an exclusive bound makes
            # "the 14:00-14:15 window" quietly mean fourteen minutes and change.
            span["$lte"] = flt.end
        conditions["ts"] = span

    if flt.components:
        conditions["component"] = {"$in": list(flt.components)}
    if flt.severities:
        conditions["severity"] = {"$in": list(flt.severities)}
    if flt.log_ids:
        conditions["log_id"] = {"$in": list(flt.log_ids)}
    if flt.namespaces:
        conditions["ns"] = {"$in": list(flt.namespaces)}
    if flt.plan_summary is not None:
        conditions["plan_summary"] = flt.plan_summary
    if flt.ctx is not None:
        conditions["ctx"] = flt.ctx
    if flt.min_duration_ms is not None:
        conditions["duration_ms"] = {"$gte": flt.min_duration_ms}
    if flt.text:
        # Escaped and anchored to nothing: this is a substring search, and an
        # unescaped user string would be a regex whose cost is unbounded on the
        # server. "i" for case-insensitivity because a reader searching for
        # "election" should not have to know mongod's capitalisation.
        conditions["msg"] = {"$regex": re.escape(flt.text), "$options": "i"}

    return conditions


def count_pipeline(flt: LogFilter) -> list[dict[str, Any]]:
    return [{"$match": match_stage(flt)}, {"$count": "count"}]


def facet_pipeline(
    flt: LogFilter, by: Sequence[FacetField], top: int = 20
) -> list[dict[str, Any]]:
    """Group and count, carrying the aggregates the diagnostics need.

    Two details that look like style and are not.

    Examples use $firstN, not $push with a later $slice. $push accumulates every
    matching id into one array before anything trims it, so on a namespace with
    two hundred thousand slow queries the group document exceeds the 16MB BSON
    limit and the aggregation fails outright -- on exactly the corpus most worth
    profiling.

    The $sort before $group is what makes $firstN deterministic. Without it,
    "first" means whichever documents the server happened to feed the group, so
    two runs of the same facet cite different records and a report is not
    reproducible. Sorting by _id can be served by the primary key index.
    """
    group_id = {field: f"${field}" for field in by}
    return [
        {"$match": match_stage(flt)},
        {"$sort": {"_id": 1}},
        {
            "$group": {
                "_id": group_id,
                "count": {"$sum": 1},
                "max_duration_ms": {"$max": "$duration_ms"},
                "total_docs_examined": {"$sum": "$docs_examined"},
                "examples": {"$firstN": {"input": "$_id", "n": 3}},
            }
        },
        # Sorting on _id after count breaks ties totally, so a facet of the same
        # data is byte-identical between runs and between implementations.
        {"$sort": {"count": -1, "_id": 1}},
        {"$limit": top},
    ]


def timeline_pipeline(flt: LogFilter, bucket_seconds: int) -> list[dict[str, Any]]:
    """Count records per fixed-width time bucket.

    $dateTrunc rather than arithmetic on epoch milliseconds: it is what the
    server provides, it handles the boundaries correctly, and it keeps the
    bucket start a real date rather than a number the caller has to convert
    back.
    """
    return [
        {"$match": match_stage(flt)},
        {
            "$group": {
                "_id": {
                    "$dateTrunc": {
                        "date": "$ts",
                        "unit": "second",
                        "binSize": bucket_seconds,
                    }
                },
                "count": {"$sum": 1},
            }
        },
        {"$sort": {"_id": 1}},
    ]


_SORTS: dict[SampleOrder, dict[str, Any]] = {
    "time": {"ts": 1, "_id": 1},
    "-time": {"ts": -1, "_id": 1},
    "-duration": {"duration_ms": -1, "_id": 1},
}


def sample_pipeline(
    flt: LogFilter, limit: int = 10, order: SampleOrder = "time"
) -> list[dict[str, Any]]:
    """Return matching records in a defined order.

    Every sort carries _id as its final key. Without it, records sharing a
    timestamp -- which happens constantly, since mongod logs at millisecond
    resolution -- come back in whatever order the server happens to produce, and
    two runs of the same query cite different records.
    """
    return [
        {"$match": match_stage(flt)},
        {"$sort": _SORTS[order]},
        {"$limit": limit},
    ]


def fetch_by_ids_pipeline(ids: Sequence[str]) -> list[dict[str, Any]]:
    """Resolve citation handles back to records."""
    return [{"$match": {"_id": {"$in": list(ids)}}}, {"$sort": {"_id": 1}}]
