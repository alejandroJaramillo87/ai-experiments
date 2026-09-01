"""Parsing the mongod structured-log envelope.

Since 4.4 mongod writes one JSON object per line:

    {"t":{"$date":"2026-08-30T14:22:31.123+00:00"},"s":"I","c":"COMMAND",
     "id":51803,"ctx":"conn1234","msg":"Slow query","attr":{...}}

Two properties of that envelope drive the whole tool. The `id` is a stable
numeric identifier for the call site, so a mongod log is already templated at the
source -- there is nothing to infer. And `attr` is an open object whose shape
depends entirely on `id`, so it is parsed but not interpreted here; see
normalize.py.

Nothing in this module raises on bad input. Log rotation truncates a line
mid-write, a disk fills, a file gets concatenated with another -- and a triage
tool that dies on line 40,000 of 300,000 is useless exactly when it is needed.
parse_line returns a ParseError instead, and the caller counts them.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

# Severity letters mongod emits. D1..D5 are debug levels; a stock server logs at
# I and above, so a corpus full of D-lines means someone raised verbosity and the
# volume is expected rather than anomalous.
SEVERITIES = frozenset({"F", "E", "W", "I", "D1", "D2", "D3", "D4", "D5"})

# Components seen on a replica-set member. Not exhaustive and not enforced: a
# newer server introduces components freely, and rejecting an unknown one would
# turn a version skew into a parse failure. Kept for validating that a corpus
# looks like mongod output at all.
KNOWN_COMPONENTS = frozenset(
    {
        "ACCESS",
        "ASSERT",
        "COMMAND",
        "CONTROL",
        "ELECTION",
        "EXECUTOR",
        "FTDC",
        "GEO",
        "INDEX",
        "INITSYNC",
        "JOURNAL",
        "NETWORK",
        "QUERY",
        "RECOVERY",
        "REPL",
        "REPL_HB",
        "ROLLBACK",
        "SHARDING",
        "STORAGE",
        "TENANT_M",
        "TXN",
        "WRITE",
        "WT",
        "WTCHKPT",
        "-",
    }
)

# The maximum of a truncated line kept in a ParseError. Enough to recognise what
# broke without pulling a multi-megabyte concatenated blob into a report.
EXCERPT_CHARS = 200


@dataclass(frozen=True, slots=True)
class LogRecord:
    """One successfully parsed mongod log line."""

    line_no: int
    ts: datetime
    severity: str
    component: str
    log_id: int
    ctx: str
    msg: str
    attr: dict[str, Any] = field(default_factory=dict)
    # Which service wrote the line: "R" for a router (mongos), "S" for a shard
    # (mongod). Added in 8.0 and present on every line the 8.x manual shows.
    # Empty for anything older, and for a merged log it is the only thing that
    # says which process a record came from.
    svc: str = ""

    @property
    def record_id(self) -> str:
        """Stable citation handle.

        Line number rather than a hash or a generated key: a citation is only
        useful if a human can resolve it, and `sed -n '48213p' mongod.log` is
        the shortest path from a claim in a report back to the evidence.
        """
        return f"L{self.line_no}"


@dataclass(frozen=True, slots=True)
class ParseError:
    """One line that could not be read as a mongod log record."""

    line_no: int
    reason: str
    excerpt: str

    @property
    def record_id(self) -> str:
        return f"L{self.line_no}"


def _parse_timestamp(value: Any) -> datetime:
    """Read the {"$date": "..."} wrapper mongod uses for `t`.

    Accepts both offset forms the server emits across versions and platforms:
    an explicit +00:00 and a trailing Z. fromisoformat handles both from 3.11.
    """
    if not isinstance(value, dict) or "$date" not in value:
        raise ValueError("t is not a {$date: ...} object")
    raw = value["$date"]
    if not isinstance(raw, str):
        raise ValueError("t.$date is not a string")
    return datetime.fromisoformat(raw)


def parse_line(line: str, line_no: int) -> LogRecord | ParseError:
    """Parse one line. Never raises; a bad line becomes a ParseError."""
    text = line.strip()
    if not text:
        return ParseError(line_no, "blank line", "")
    try:
        doc = json.loads(text)
    except json.JSONDecodeError as exc:
        # The overwhelmingly common cause is a line truncated by rotation, which
        # looks like an unexpected end of input rather than corruption.
        return ParseError(line_no, f"invalid JSON: {exc.msg}", text[:EXCERPT_CHARS])
    if not isinstance(doc, dict):
        return ParseError(
            line_no, "top-level value is not an object", text[:EXCERPT_CHARS]
        )

    try:
        ts = _parse_timestamp(doc["t"])
    except (KeyError, ValueError) as exc:
        return ParseError(line_no, f"bad timestamp: {exc}", text[:EXCERPT_CHARS])

    missing = [k for k in ("s", "c", "id", "ctx", "msg") if k not in doc]
    if missing:
        return ParseError(
            line_no, f"missing field(s): {', '.join(missing)}", text[:EXCERPT_CHARS]
        )

    log_id = doc["id"]
    if not isinstance(log_id, int) or isinstance(log_id, bool):
        return ParseError(line_no, "id is not an integer", text[:EXCERPT_CHARS])

    attr = doc.get("attr", {})
    if not isinstance(attr, dict):
        return ParseError(line_no, "attr is not an object", text[:EXCERPT_CHARS])

    return LogRecord(
        line_no=line_no,
        ts=ts,
        severity=str(doc["s"]),
        component=str(doc["c"]),
        log_id=log_id,
        ctx=str(doc["ctx"]),
        msg=str(doc["msg"]),
        attr=attr,
        svc=str(doc.get("svc", "")),
    )


def parse_lines(
    lines: Iterable[str], start: int = 1
) -> Iterator[LogRecord | ParseError]:
    """Parse an iterable of lines, numbering from `start` (1-based, like sed)."""
    for offset, line in enumerate(lines):
        yield parse_line(line, start + offset)


def read_log(path: str | Path) -> Iterator[LogRecord | ParseError]:
    """Stream a log file.

    A generator rather than a list: the design target is a 100MB corpus, and
    materialising it costs several times its size in Python objects for no gain
    -- every consumer here either aggregates or filters.
    """
    with open(path, encoding="utf-8", errors="replace") as handle:
        yield from parse_lines(handle)
