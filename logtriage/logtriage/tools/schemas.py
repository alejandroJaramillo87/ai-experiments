"""The tools the model is given, and the dispatcher behind them.

Five tools, ordered by how much they are worth. log_profile summarises the whole
corpus and is the one that makes the rest usable. query_logs answers counted,
grouped, time-bounded questions -- the ones a search index answers badly and an
aggregation answers exactly. search_logs and search_docs find text. fetch_records
resolves a citation back to the record behind it, which is what lets the model
check itself and the report check the model.

Semantic search is deliberately last in that ordering. It is the tool that feels
most like the point of a retrieval system and is the weakest of the five here,
because the questions that matter about a log are counting questions: how many,
when did it start, which namespace, did it happen before or after. Similarity
cannot answer any of those, and a model that reaches for it first produces
confident answers built on ten records that resembled the question.

Nothing here raises on bad arguments. A model will emit a malformed call --
wrong type, missing field, invented enum member -- and the loop has to survive
it and, better, tell the model what was wrong so the retry succeeds. An
exception ends the run; a structured error costs one turn.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from logtriage.embed.base import Embedder
from logtriage.store.base import FacetField, LogFilter, LogStore
from logtriage.store.knowledge import VECTOR_WEIGHT, KnowledgeStore
from logtriage.tools.budget import DEFAULT_RESULT_TOKENS, truncate_rows

# Facet fields exposed to the model. A closed list in the schema is worth more
# than validation after the fact: it is in the prompt, so the model sees what is
# groupable rather than guessing and being corrected.
GROUPABLE = ("component", "severity", "log_id", "ns", "plan_summary", "op", "ctx")

# Ceiling on rows any single tool call returns, before the token budget applies.
# Bounds the work as well as the output: without it a model can ask for every
# record in a corpus and get a timeout instead of an answer.
MAX_ROWS = 50


@dataclass(frozen=True, slots=True)
class ToolResult:
    """What a tool call produced, and what it cited."""

    name: str
    text: str
    ok: bool = True
    cited_ids: tuple[str, ...] = field(default_factory=tuple)


def _error(name: str, message: str, hint: str = "") -> ToolResult:
    """A failure the model can act on rather than an exception that ends the run."""
    text = f"error: {message}"
    if hint:
        text += f"\nhint: {hint}"
    return ToolResult(name=name, text=text, ok=False)


SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "log_profile",
            "description": (
                "Summarise the whole log: time span, record and error counts, "
                "components, distinct call sites, and slow queries by namespace. "
                "Call this first. It is the only tool that can tell you what the "
                "log does NOT contain, which is often the answer."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_logs",
            "description": (
                "Count and group log records by exact criteria. Use this for any "
                "question of the form how many, when, or which -- it is exact, "
                "where search is approximate."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "group_by": {
                        "type": "array",
                        "items": {"type": "string", "enum": list(GROUPABLE)},
                        "description": "Fields to group by. Omit for a plain count.",
                    },
                    "component": {
                        "type": "string",
                        "description": "e.g. COMMAND, NETWORK, REPL, ELECTION, INDEX",
                    },
                    "severity": {"type": "string", "enum": ["F", "E", "W", "I"]},
                    "log_id": {"type": "integer", "description": "mongod message id"},
                    "ns": {"type": "string", "description": "namespace, db.collection"},
                    "plan_summary": {"type": "string", "description": "e.g. COLLSCAN"},
                    "min_duration_ms": {"type": "integer"},
                    "start": {"type": "string", "description": "ISO-8601 lower bound"},
                    "end": {"type": "string", "description": "ISO-8601 upper bound"},
                    "limit": {
                        "type": "integer",
                        "description": f"max rows, <= {MAX_ROWS}",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_logs",
            "description": (
                "Find log records whose message contains a substring. Returns the "
                "matching records themselves, newest-slowest first. Use it to "
                "locate a message you can quote; use query_logs to count."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "substring of the message",
                    },
                    "limit": {"type": "integer"},
                    "order": {
                        "type": "string",
                        "enum": ["time", "-time", "-duration"],
                    },
                },
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_docs",
            "description": (
                "Search the MongoDB manual. Use it to check what a log message "
                "means, what a parameter does, or how a fix is applied -- not to "
                "find out what happened, which only the log can say."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "limit": {"type": "integer"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_records",
            "description": (
                "Retrieve log records by id (e.g. L4821) to read them in full. "
                "Use this to verify a record says what you are about to claim it "
                "says before citing it."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "ids": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["ids"],
            },
        },
    },
]


def _as_int(value: Any) -> int | None:
    """Coerce a model-supplied integer.

    Models emit "50" for an integer field routinely. Rejecting that is
    technically correct and wastes a turn on something unambiguous; rejecting a
    float that is not whole is not, since silently truncating 1.5 changes the
    question that was asked.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            return None
    return None


def _as_time(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        # A bare "Z" and an explicit offset are both accepted; a naive timestamp
        # is not, because comparing it against tz-aware record timestamps raises
        # and there is no correct zone to assume on the model's behalf.
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else None


def facet_fields(names: Sequence[str]) -> list[FacetField]:
    """Narrow validated strings to the facet literal type."""
    return [n for n in names if n in GROUPABLE]  # type: ignore[misc]


class ToolBox:
    """Binds the tool names to the stores behind them."""

    def __init__(
        self,
        logs: LogStore,
        knowledge: KnowledgeStore | None = None,
        embedder: Embedder | None = None,
        profile_text: str = "",
        budget_tokens: int = DEFAULT_RESULT_TOKENS,
    ) -> None:
        self.logs = logs
        self.knowledge = knowledge
        self.embedder = embedder
        self.profile_text = profile_text
        self.budget_tokens = budget_tokens

    def schemas(self) -> list[dict[str, Any]]:
        """The tool definitions, minus any whose backing store is absent.

        Offering a tool that cannot work is worse than offering fewer: the model
        spends turns on it and reads the failures as evidence about the log.
        """
        available = []
        for schema in SCHEMAS:
            name = schema["function"]["name"]
            if name == "search_docs" and not (self.knowledge and self.embedder):
                continue
            available.append(schema)
        return available

    def call(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        handler = {
            "log_profile": self._log_profile,
            "query_logs": self._query_logs,
            "search_logs": self._search_logs,
            "search_docs": self._search_docs,
            "fetch_records": self._fetch_records,
        }.get(name)
        if handler is None:
            return _error(
                name,
                f"no such tool: {name}",
                f"available: {', '.join(s['function']['name'] for s in self.schemas())}",
            )
        if not isinstance(arguments, dict):
            return _error(name, "arguments must be an object")
        return handler(arguments)

    def _limit(self, arguments: dict[str, Any]) -> int:
        limit = _as_int(arguments.get("limit")) or 10
        return max(1, min(limit, MAX_ROWS))

    def _build_filter(self, arguments: dict[str, Any]) -> LogFilter | ToolResult:
        for field_name in ("start", "end"):
            if field_name in arguments and _as_time(arguments[field_name]) is None:
                return _error(
                    "query_logs",
                    f"{field_name} is not an ISO-8601 timestamp with a timezone",
                    "e.g. 2026-08-30T14:00:00Z",
                )
        return LogFilter(
            start=_as_time(arguments.get("start")),
            end=_as_time(arguments.get("end")),
            components=(arguments["component"],) if arguments.get("component") else (),
            severities=(arguments["severity"],) if arguments.get("severity") else (),
            log_ids=(
                (_as_int(arguments["log_id"]),)  # type: ignore[arg-type]
                if _as_int(arguments.get("log_id")) is not None
                else ()
            ),
            namespaces=(arguments["ns"],) if arguments.get("ns") else (),
            plan_summary=arguments.get("plan_summary") or None,
            min_duration_ms=_as_int(arguments.get("min_duration_ms")),
        )

    def _log_profile(self, arguments: dict[str, Any]) -> ToolResult:
        if not self.profile_text:
            return _error("log_profile", "no profile was computed for this run")
        return ToolResult(name="log_profile", text=self.profile_text)

    def _query_logs(self, arguments: dict[str, Any]) -> ToolResult:
        built = self._build_filter(arguments)
        if isinstance(built, ToolResult):
            return built

        group_by = arguments.get("group_by") or []
        if isinstance(group_by, str):
            group_by = [group_by]
        if not isinstance(group_by, list):
            return _error("query_logs", "group_by must be an array of field names")
        unknown = [g for g in group_by if g not in GROUPABLE]
        if unknown:
            return _error(
                "query_logs",
                f"cannot group by {', '.join(map(str, unknown))}",
                f"groupable fields: {', '.join(GROUPABLE)}",
            )

        if not group_by:
            count = self.logs.count(built)
            return ToolResult(
                name="query_logs", text=f"{count} records match that filter"
            )

        buckets = self.logs.facet(built, facet_fields(group_by), self._limit(arguments))
        if not buckets:
            # Stated rather than returned empty. "No records match" is a finding;
            # a blank result reads as a broken tool.
            return ToolResult(
                name="query_logs", text="no records match that filter", cited_ids=()
            )
        rows = [
            "  "
            + ", ".join(f"{k}={v}" for k, v in b.key.items())
            + f": {b.count}"
            + (f", max {b.max_duration_ms}ms" if b.max_duration_ms else "")
            + f" (e.g. {', '.join(b.examples)})"
            for b in buckets
        ]
        cited = tuple(i for b in buckets for i in b.examples)
        return ToolResult(
            name="query_logs",
            text=truncate_rows(
                rows,
                self.budget_tokens,
                header=f"{len(buckets)} groups by {', '.join(group_by)}:",
                footer_hint="more groups omitted",
            ),
            cited_ids=cited,
        )

    def _search_logs(self, arguments: dict[str, Any]) -> ToolResult:
        text = arguments.get("text")
        if not isinstance(text, str) or not text.strip():
            return _error("search_logs", "text is required and must be a string")
        order = arguments.get("order", "time")
        if order not in ("time", "-time", "-duration"):
            return _error(
                "search_logs",
                f"unknown order: {order}",
                "one of time, -time, -duration",
            )
        built = self._build_filter({**arguments, "text": text})
        if isinstance(built, ToolResult):
            return built
        from dataclasses import replace

        records = self.logs.sample(
            replace(built, text=text), limit=self._limit(arguments), order=order
        )
        if not records:
            return ToolResult(name="search_logs", text=f"no records contain {text!r}")
        return ToolResult(
            name="search_logs",
            text=truncate_rows(
                [_render_record(r) for r in records],
                self.budget_tokens,
                header=f"{len(records)} records containing {text!r}:",
                footer_hint="more records omitted",
            ),
            cited_ids=tuple(r.record_id for r in records),
        )

    def _search_docs(self, arguments: dict[str, Any]) -> ToolResult:
        if not (self.knowledge and self.embedder):
            return _error("search_docs", "no documentation index is loaded")
        query = arguments.get("query")
        if not isinstance(query, str) or not query.strip():
            return _error("search_docs", "query is required and must be a string")
        (vector,) = self.embedder.embed([query])
        # An embedder with no semantics contributes a ranking that is noise, and
        # RRF has no way to tell noise from a second opinion. Weighting it to
        # zero makes the offline path honest text search rather than hybrid
        # search that looks like it is doing more.
        weight = VECTOR_WEIGHT if self.embedder.semantic else 0.0
        hits = self.knowledge.hybrid_search(
            query, vector, k=self._limit(arguments), vector_weight=weight
        )
        if not hits:
            return ToolResult(
                name="search_docs", text=f"the manual has nothing matching {query!r}"
            )
        return ToolResult(
            name="search_docs",
            text=truncate_rows(
                [
                    f"  [{h.chunk_id}] {h.chunk.breadcrumb}\n{h.chunk.text}"
                    for h in hits
                ],
                self.budget_tokens,
                header=f"{len(hits)} documentation passages for {query!r}:",
                footer_hint="more passages omitted",
            ),
            cited_ids=tuple(h.chunk_id for h in hits),
        )

    def _fetch_records(self, arguments: dict[str, Any]) -> ToolResult:
        ids = arguments.get("ids")
        if isinstance(ids, str):
            ids = [ids]
        if not isinstance(ids, list) or not ids:
            return _error(
                "fetch_records", "ids is required and must be a non-empty array"
            )
        if not all(isinstance(i, str) for i in ids):
            return _error("fetch_records", "every id must be a string, e.g. L4821")
        wanted = [str(i) for i in ids][:MAX_ROWS]
        records = self.logs.fetch_by_ids(wanted)
        missing = sorted(set(wanted) - {r.record_id for r in records})
        rows = [_render_record(r) for r in records]
        if missing:
            # Named individually. The model needs to know which of its citations
            # does not exist, not merely that one did not.
            rows.append(f"  not found: {', '.join(missing)}")
        return ToolResult(
            name="fetch_records",
            text=truncate_rows(
                rows,
                self.budget_tokens,
                header=f"{len(records)} of {len(wanted)} records:",
                footer_hint="more records omitted",
            ),
            cited_ids=tuple(r.record_id for r in records),
        )


def _render_record(record: Any) -> str:
    """One log record as a single readable line.

    Command documents are redacted here rather than at ingest. The literal
    values in a large $in are most of the token cost of a record, and redacting
    at render means it applies however the corpus was stored.
    """
    from logtriage.mongod.normalize import redact_command

    parts = [
        f"  {record.record_id} {record.ts.isoformat()} [{record.severity}]",
        f"{record.component}:{record.log_id}",
        record.msg,
    ]
    if record.ns:
        parts.append(f"ns={record.ns}")
    if record.duration_ms is not None:
        parts.append(f"{record.duration_ms}ms")
    if record.plan_summary:
        parts.append(f"plan={record.plan_summary}")
    if record.docs_examined is not None and record.nreturned is not None:
        parts.append(f"docs={record.docs_examined}/returned={record.nreturned}")
    command = record.attr.get("command")
    if isinstance(command, dict):
        parts.append(f"command={redact_command(command)}")
    return " ".join(str(p) for p in parts)
