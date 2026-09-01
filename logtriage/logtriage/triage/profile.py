"""The whole-corpus summary, in a few thousand tokens.

This is the single highest-value thing the tool produces. A 100MB corpus is
roughly 300,000 lines and cannot be read by a model with any context window that
runs at an interactive speed -- but it is a few hundred distinct call sites, a
handful of components, one time span, and a short list of things that went
wrong. That fits easily, and it is what turns "answer questions about a log you
cannot see" into "answer questions about a log you have already been shown the
shape of".

The profile is deliberately computed rather than retrieved. Search, semantic or
otherwise, can only return what a query asked for, so it cannot tell a reader
what is absent -- and "the corpus contains no election at all" is frequently the
answer that closes an investigation. A summary that counts everything can.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime

from logtriage.mongod.normalize import NormalizedRecord
from logtriage.templates.extract import (
    FreeTextCluster,
    TemplateIndex,
    TemplateStat,
    cluster_free_text,
)
from logtriage.tools.budget import DEFAULT_PROFILE_TOKENS, truncate_rows

# How many of each list the rendered profile carries. Sized so a profile stays
# inside DEFAULT_PROFILE_TOKENS on a realistic corpus without truncation doing
# the work -- truncation is the floor, not the plan.
TOP_TEMPLATES = 25
TOP_NAMESPACES = 10
TOP_ERROR_TEMPLATES = 10
TOP_FREE_TEXT = 8

# Severities that are reported individually rather than only counted.
PROBLEM_SEVERITIES = ("F", "E", "W")


@dataclass(frozen=True, slots=True)
class NamespaceStat:
    """Slow-query behaviour for one namespace."""

    ns: str
    slow_count: int
    collscan_count: int
    max_duration_ms: int
    median_duration_ms: int
    worst_docs_per_returned: float | None
    examples: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CorpusProfile:
    """A computed description of an entire corpus."""

    source: str
    records_parsed: int
    parse_errors: int
    span_start: datetime | None
    span_end: datetime | None
    severity_counts: dict[str, int]
    component_counts: dict[str, int]
    templates: tuple[TemplateStat, ...]
    distinct_templates: int
    namespaces: tuple[NamespaceStat, ...]
    problem_templates: tuple[TemplateStat, ...]
    free_text: tuple[FreeTextCluster, ...] = ()

    @property
    def duration_seconds(self) -> float | None:
        if self.span_start is None or self.span_end is None:
            return None
        return (self.span_end - self.span_start).total_seconds()


def _median(values: list[int]) -> int:
    """Median without importing statistics for one call.

    Lower median on an even count, deliberately: these are latency figures a
    reader compares against thresholds, and interpolating between two observed
    durations invents a number no operation actually took.
    """
    if not values:
        return 0
    ordered = sorted(values)
    return ordered[(len(ordered) - 1) // 2]


def _namespace_stats(records: list[NormalizedRecord]) -> list[NamespaceStat]:
    by_ns: dict[str, list[NormalizedRecord]] = {}
    for record in records:
        if record.ns and record.duration_ms is not None:
            by_ns.setdefault(record.ns, []).append(record)

    stats: list[NamespaceStat] = []
    for ns, group in by_ns.items():
        durations = [r.duration_ms for r in group if r.duration_ms is not None]
        ratios = [r.docs_per_returned for r in group if r.docs_per_returned is not None]
        collscans = [r for r in group if r.plan_summary == "COLLSCAN"]
        # Cite the collection scans preferentially: on a namespace that has any,
        # they are what a reader needs to see, and a citation list of three
        # healthy queries would be actively misleading.
        cited = (collscans or sorted(group, key=lambda r: -(r.duration_ms or 0)))[:3]
        stats.append(
            NamespaceStat(
                ns=ns,
                slow_count=len(group),
                collscan_count=len(collscans),
                max_duration_ms=max(durations, default=0),
                median_duration_ms=_median(durations),
                worst_docs_per_returned=max(ratios) if ratios else None,
                examples=tuple(r.record_id for r in cited),
            )
        )
    # Collection scans first, then raw volume: a namespace with one scan is more
    # actionable than one with a hundred well-indexed slow queries.
    stats.sort(key=lambda s: (-s.collscan_count, -s.slow_count, s.ns))
    return stats


def build_profile(
    records: Iterable[NormalizedRecord],
    *,
    source: str = "<stream>",
    parse_errors: int = 0,
) -> CorpusProfile:
    """Compute a profile in one pass over the records."""
    index = TemplateIndex()
    severity_counts: dict[str, int] = {}
    component_counts: dict[str, int] = {}
    slow_records: list[NormalizedRecord] = []
    free_text_messages: list[str] = []
    span_start: datetime | None = None
    span_end: datetime | None = None
    total = 0

    for record in records:
        total += 1
        index.add(record)
        severity_counts[record.severity] = severity_counts.get(record.severity, 0) + 1
        component_counts[record.component] = (
            component_counts.get(record.component, 0) + 1
        )
        if span_start is None or record.ts < span_start:
            span_start = record.ts
        if span_end is None or record.ts > span_end:
            span_end = record.ts
        if record.ns is not None:
            slow_records.append(record)
        error = record.attr.get("error")
        if isinstance(error, dict) and isinstance(error.get("errmsg"), str):
            free_text_messages.append(error["errmsg"])
        elif isinstance(error, str):
            free_text_messages.append(error)
        errmsg = record.attr.get("errmsg")
        if isinstance(errmsg, str):
            free_text_messages.append(errmsg)

    stats = index.stats()
    problems = [s for s in stats if s.severity in PROBLEM_SEVERITIES]

    return CorpusProfile(
        source=source,
        records_parsed=total,
        parse_errors=parse_errors,
        span_start=span_start,
        span_end=span_end,
        severity_counts=dict(sorted(severity_counts.items())),
        component_counts=dict(
            sorted(component_counts.items(), key=lambda kv: (-kv[1], kv[0]))
        ),
        templates=tuple(stats[:TOP_TEMPLATES]),
        distinct_templates=len(stats),
        namespaces=tuple(_namespace_stats(slow_records)[:TOP_NAMESPACES]),
        problem_templates=tuple(problems[:TOP_ERROR_TEMPLATES]),
        free_text=tuple(cluster_free_text(free_text_messages)[:TOP_FREE_TEXT]),
    )


def render_profile(
    profile: CorpusProfile, *, budget_tokens: int = DEFAULT_PROFILE_TOKENS
) -> str:
    """Render a profile as the text the model is shown.

    Plain text rather than JSON: this is read, not parsed, and the JSON envelope
    would spend a fifth of the budget on punctuation. The per-section budgets
    are fixed shares rather than a single pool so that a corpus with thousands of
    namespaces cannot crowd out the error list, which is usually the section
    that matters most.
    """
    lines: list[str] = ["=== corpus profile ==="]
    lines.append(f"source: {profile.source}")
    lines.append(
        f"records: {profile.records_parsed} parsed, {profile.parse_errors} unparseable"
    )
    if profile.span_start and profile.span_end:
        duration = profile.duration_seconds or 0
        lines.append(
            f"span: {profile.span_start.isoformat()} .. {profile.span_end.isoformat()}"
            f" ({duration / 60:.1f} minutes)"
        )
    else:
        lines.append("span: empty corpus")
    lines.append(
        "severities: "
        + ", ".join(f"{k}={v}" for k, v in profile.severity_counts.items())
    )
    lines.append(
        "components: "
        + ", ".join(f"{k}={v}" for k, v in profile.component_counts.items())
    )
    lines.append(f"distinct call sites: {profile.distinct_templates}")

    if profile.problem_templates:
        lines.append("")
        lines.append(
            truncate_rows(
                [
                    f"  [{s.severity}] {s.template} x{s.count} {s.msg} (e.g. {', '.join(s.examples)})"
                    for s in profile.problem_templates
                ],
                budget_tokens // 4,
                header="--- warnings and errors ---",
                footer_hint="more problem call sites omitted",
            )
        )
    else:
        lines.append("")
        # Stated rather than left absent. Absence of evidence is a finding, and a
        # reader who does not see this line cannot tell whether the corpus was
        # clean or the section was dropped.
        lines.append("--- warnings and errors ---")
        lines.append("  none: no records above severity I")

    if profile.namespaces:
        lines.append("")
        lines.append(
            truncate_rows(
                [
                    f"  {n.ns}: {n.slow_count} slow, {n.collscan_count} COLLSCAN, "
                    f"median {n.median_duration_ms}ms, max {n.max_duration_ms}ms, "
                    f"worst docs/returned "
                    f"{'n/a' if n.worst_docs_per_returned is None else format(n.worst_docs_per_returned, '.0f')}"
                    f" (e.g. {', '.join(n.examples)})"
                    for n in profile.namespaces
                ],
                budget_tokens // 4,
                header="--- slow queries by namespace ---",
                footer_hint="more namespaces omitted",
            )
        )

    lines.append("")
    lines.append(
        truncate_rows(
            [f"  {s.template} x{s.count} {s.msg}" for s in profile.templates],
            budget_tokens // 4,
            header="--- most frequent call sites ---",
            footer_hint="more call sites omitted",
        )
    )

    if profile.free_text:
        lines.append("")
        lines.append(
            truncate_rows(
                [f"  x{c.count} {c.template}" for c in profile.free_text],
                budget_tokens // 8,
                header="--- clustered error text ---",
                footer_hint="more clusters omitted",
            )
        )

    return "\n".join(lines)
