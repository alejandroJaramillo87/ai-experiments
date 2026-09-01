"""Turning a LogRecord into a flat document the stores can query.

`attr` is an open object whose shape is a function of `id`, so a store that
indexed it verbatim could not answer "which namespace was slowest" without
knowing every shape in advance. This module lifts the fields that diagnostics
and retrieval actually need onto the top level and leaves the rest in `attr`.

The slow-query record (COMMAND id 51803) is the one that matters most, because
it carries the counters that turn "this was slow" into "this was slow because it
read the whole collection": planSummary, keysExamined, docsExamined and
nreturned. MongoDB 8.0 added workingMillis alongside durationMillis -- the
former excludes time blocked on locks and flow control, so the pair separates a
query that is expensive from one that was merely waiting.

Redaction lives here too, and is deliberately value-only: a command document's
field names are the diagnostic content (they are what an index would have to
cover), while its literal values are both the privacy risk and, for a large $in,
most of the token cost.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Any

from logtriage.mongod.records import LogRecord

# COMMAND id 51803, "Slow query". Emitted for any operation exceeding
# slowOpThresholdMs (100ms by default) or sampled by slowOpSampleRate.
SLOW_QUERY_ID = 51803

# NETWORK connection lifecycle. Together these give the open-connection count
# over time without the server having to report it.
CONNECTION_ACCEPTED_ID = 22943
CONNECTION_ENDED_ID = 22944

# Placeholders substituted for literal values by redact_value. Chosen to stay
# valid JSON scalars so a redacted command document still renders as JSON.
_TYPE_NAMES = {
    bool: "<bool>",
    int: "<int>",
    float: "<double>",
    str: "<string>",
    type(None): "<null>",
}

# Keys inside a command document whose values are structural rather than data:
# they name fields and directions, so redacting them would destroy exactly the
# information a reader needs to decide whether an index would help.
_STRUCTURAL_KEYS = frozenset({"sort", "projection", "hint", "key", "keyPattern"})


def redact_value(value: Any, *, depth: int = 0, max_depth: int = 12) -> Any:
    """Replace literal values with type placeholders, preserving shape.

    Field names, operator keys ($gt, $in, $match) and nesting all survive; only
    the leaves change. An array keeps its first redacted element and a count of
    what followed, so a $in over ten thousand ids reads as one element plus
    "<+9999 more>" rather than consuming the entire context window.
    """
    if depth >= max_depth:
        # Defensive rather than expected: a command document nested this deeply
        # is malformed, and recursing without a floor turns that into a crash.
        return "<...>"
    if isinstance(value, dict):
        return {
            k: redact_value(v, depth=depth + 1, max_depth=max_depth)
            for k, v in value.items()
        }
    if isinstance(value, list):
        if not value:
            return []
        head = redact_value(value[0], depth=depth + 1, max_depth=max_depth)
        if len(value) == 1:
            return [head]
        return [head, f"<+{len(value) - 1} more>"]
    return _TYPE_NAMES.get(type(value), "<value>")


def redact_command(command: Any) -> Any:
    """Redact a command document, keeping structural sub-documents intact.

    `sort: {createdAt: -1}` and `projection: {_id: 0}` are answers to "would an
    index help", not user data, so their values are preserved verbatim.
    """
    if not isinstance(command, dict):
        return redact_value(command)
    out: dict[str, Any] = {}
    for key, value in command.items():
        out[key] = value if key in _STRUCTURAL_KEYS else redact_value(value)
    return out


@dataclass(frozen=True, slots=True)
class NormalizedRecord:
    """A flat, queryable projection of one log record.

    Field names are snake_case rather than mongod's camelCase: these are this
    tool's schema, queried by its own filters, and mixing the two conventions in
    one document is how a query ends up silently matching nothing.
    """

    record_id: str
    line_no: int
    ts: datetime
    severity: str
    component: str
    log_id: int
    ctx: str
    msg: str

    # Slow-query projection; None on every record that is not one.
    ns: str | None = None
    op: str | None = None
    duration_ms: int | None = None
    working_ms: int | None = None
    plan_summary: str | None = None
    keys_examined: int | None = None
    docs_examined: int | None = None
    nreturned: int | None = None
    num_yields: int | None = None
    query_hash: str | None = None
    plan_cache_key: str | None = None
    remote: str | None = None

    attr: dict[str, Any] = field(default_factory=dict)

    @property
    def scan_ratio(self) -> float | None:
        """Index keys examined per document returned.

        The canonical selectivity signal: 1.0 is a perfectly covering index, and
        a large value means the server walked a great deal of index to return
        very little. None when the record is not a slow query or returned
        nothing to divide by -- a zero-result scan is loud in docs_examined
        instead, and inventing a ratio for it would rank it against queries that
        did return rows.
        """
        if self.keys_examined is None or not self.nreturned:
            return None
        return self.keys_examined / self.nreturned

    @property
    def docs_per_returned(self) -> float | None:
        """Documents examined per document returned; the COLLSCAN cost signal."""
        if self.docs_examined is None or not self.nreturned:
            return None
        return self.docs_examined / self.nreturned


def _as_int(value: Any) -> int | None:
    """Coerce a counter, tolerating the float form some fields take."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return None


def _as_str(value: Any) -> str | None:
    return value if isinstance(value, str) else None


def _command_namespace(attr: dict[str, Any]) -> str | None:
    """The db.collection a slow query touched.

    `ns` is present on most slow-query records; where it is not, the command
    document names the collection under the command verb and the database under
    `$db`, which is the pair mongod itself joins to build ns.
    """
    ns = _as_str(attr.get("ns"))
    if ns:
        return ns
    command = attr.get("command")
    if not isinstance(command, dict):
        return None
    db = _as_str(command.get("$db"))
    for verb in (
        "find",
        "aggregate",
        "count",
        "distinct",
        "update",
        "delete",
        "insert",
        "findAndModify",
    ):
        collection = command.get(verb)
        if isinstance(collection, str) and db:
            return f"{db}.{collection}"
    return None


def _command_verb(attr: dict[str, Any]) -> str | None:
    """The operation name, e.g. find / aggregate / update.

    mongod puts it first in the command document, which is why the first key is
    the verb rather than something that needs searching for.
    """
    command = attr.get("command")
    if isinstance(command, dict):
        for key in command:
            if not key.startswith("$"):
                return key
    return _as_str(attr.get("type"))


def normalize(record: LogRecord, *, redact: bool = False) -> NormalizedRecord:
    """Project a parsed record into the store's document shape.

    `redact` is off by default. Redaction at ingest is for a corpus that must
    not hold literal values at rest; it is not the defence against spending the
    context window on them, which happens at render time where the budget is
    (see tools/budget.py) and so applies whatever was ingested.
    """
    attr = record.attr

    stored_attr = dict(attr)
    if redact and isinstance(stored_attr.get("command"), dict):
        stored_attr["command"] = redact_command(stored_attr["command"])

    base = NormalizedRecord(
        record_id=record.record_id,
        line_no=record.line_no,
        ts=record.ts,
        severity=record.severity,
        component=record.component,
        log_id=record.log_id,
        ctx=record.ctx,
        msg=record.msg,
        attr=stored_attr,
    )
    if record.log_id != SLOW_QUERY_ID:
        return base

    # replace() rather than a second full constructor call: the eight envelope
    # fields are identical either way, and duplicating them is how the two
    # branches drift apart when a field is added.
    return replace(
        base,
        ns=_command_namespace(attr),
        op=_command_verb(attr),
        duration_ms=_as_int(attr.get("durationMillis")),
        working_ms=_as_int(attr.get("workingMillis")),
        plan_summary=_as_str(attr.get("planSummary")),
        keys_examined=_as_int(attr.get("keysExamined")),
        docs_examined=_as_int(attr.get("docsExamined")),
        nreturned=_as_int(attr.get("nreturned")),
        num_yields=_as_int(attr.get("numYields")),
        query_hash=_as_str(attr.get("queryHash")),
        plan_cache_key=_as_str(attr.get("planCacheKey")),
        remote=_as_str(attr.get("remote")),
    )
