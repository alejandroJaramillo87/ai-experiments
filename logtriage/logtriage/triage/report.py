"""Rendering a triage report, and checking that it says only what it can support.

Two mechanisms here do the real work, and both exist because a language model
writing about logs fails in specific, predictable ways.

The first is citation validation. Every record id and documentation chunk id the
model writes is extracted and resolved against the stores. One that does not
resolve is marked in the output rather than removed, because a reader needs to
see that a claim was unsupported -- deleting the citation would leave the claim
standing and looking better than it is.

The second is the timeline. A model shown a collection-scan finding and an
election finding in the same corpus will reach for the election as the cause,
because it is the more dramatic event and a causal story is cheaper to generate
than a careful one. Nothing in the prose can prevent that. What can is printing,
above the narrative, the first occurrence of every finding in order -- so the
ordering is stated as data before anything has a chance to assume it, and a
reader can see in one line that the queries were already slow six minutes before
the first heartbeat failed.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime

from logtriage.mongod.diagnostics import Finding
from logtriage.store.base import LogStore
from logtriage.store.knowledge import KnowledgeStore
from logtriage.triage.profile import CorpusProfile

# A record citation: L followed by a line number. Word-bounded so "L1" inside an
# identifier is not mistaken for one.
RECORD_CITATION_RE = re.compile(r"\bL\d+\b")

# A documentation citation: the chunk id, a document path and an ordinal.
CHUNK_CITATION_RE = re.compile(r"\b[\w./-]+\.txt#\d+\b")

# What an unresolvable citation is marked with. Loud and greppable: it has to
# survive being pasted into a ticket.
UNVERIFIED = "[unverified]"


@dataclass(frozen=True, slots=True)
class CitationAudit:
    """Which of a narrative's citations resolve, and which do not."""

    records_cited: tuple[str, ...] = field(default_factory=tuple)
    records_resolved: tuple[str, ...] = field(default_factory=tuple)
    chunks_cited: tuple[str, ...] = field(default_factory=tuple)
    chunks_resolved: tuple[str, ...] = field(default_factory=tuple)

    @property
    def unresolved_records(self) -> tuple[str, ...]:
        return tuple(sorted(set(self.records_cited) - set(self.records_resolved)))

    @property
    def unresolved_chunks(self) -> tuple[str, ...]:
        return tuple(sorted(set(self.chunks_cited) - set(self.chunks_resolved)))

    @property
    def clean(self) -> bool:
        return not self.unresolved_records and not self.unresolved_chunks

    @property
    def total_cited(self) -> int:
        return len(set(self.records_cited)) + len(set(self.chunks_cited))


def audit_citations(
    narrative: str,
    logs: LogStore,
    knowledge: KnowledgeStore | None = None,
) -> CitationAudit:
    """Resolve every citation a narrative makes."""
    records = tuple(dict.fromkeys(RECORD_CITATION_RE.findall(narrative)))
    chunks = tuple(dict.fromkeys(CHUNK_CITATION_RE.findall(narrative)))

    resolved_records = (
        tuple(r.record_id for r in logs.fetch_by_ids(records)) if records else ()
    )
    resolved_chunks = (
        tuple(c.chunk_id for c in knowledge.fetch_by_ids(chunks))
        if chunks and knowledge
        else ()
    )
    return CitationAudit(
        records_cited=records,
        records_resolved=resolved_records,
        chunks_cited=chunks,
        chunks_resolved=resolved_chunks,
    )


def mark_unverified(narrative: str, audit: CitationAudit) -> str:
    """Annotate citations that do not resolve, in place.

    Marked rather than removed. Deleting a bad citation leaves the sentence
    standing with nothing behind it, which reads as *better* supported than a
    sentence whose citation is visibly broken.
    """
    marked = narrative
    for citation in (*audit.unresolved_records, *audit.unresolved_chunks):
        marked = re.sub(
            rf"(?<!\w){re.escape(citation)}(?!\w)",
            f"{citation} {UNVERIFIED}",
            marked,
        )
    return marked


@dataclass(frozen=True, slots=True)
class TriageReport:
    """Everything one triage run produced."""

    source: str
    profile: CorpusProfile
    findings: tuple[Finding, ...]
    narrative: str
    audit: CitationAudit
    tool_calls: int = 0
    truth_notes: tuple[str, ...] = field(default_factory=tuple)


def _timeline_rows(findings: Sequence[Finding]) -> list[str]:
    dated = [f for f in findings if f.first_seen is not None]
    dated.sort(key=lambda f: (f.first_seen or datetime.max, f.kind))
    rows: list[str] = []
    previous: datetime | None = None
    for finding in dated:
        assert finding.first_seen is not None
        gap = ""
        if previous is not None:
            minutes = (finding.first_seen - previous).total_seconds() / 60
            # The gap is printed because it is the number that settles a causal
            # question, and a reader should not have to subtract two timestamps
            # to find it.
            gap = f"  (+{minutes:.1f} min)"
        rows.append(
            f"  {finding.first_seen.isoformat()}  {finding.kind}"
            + (f" [{finding.ns}]" if finding.ns else "")
            + gap
        )
        previous = finding.first_seen
    return rows


def render_report(report: TriageReport) -> str:
    """The report a human reads."""
    lines: list[str] = [f"=== triage: {report.source} ===", ""]

    if report.findings:
        lines.append("--- first occurrence, in order ---")
        lines.extend(_timeline_rows(report.findings))
        lines.append("")
        # Stated outright rather than left for the reader to infer, because the
        # inference the evidence invites is frequently backwards.
        lines.append(
            "  Ordering is evidence. A finding that began earlier cannot have "
            "been caused by one that began later."
        )
        lines.append("")

    lines.append("--- findings ---")
    if not report.findings:
        # An explicit statement, not an empty section: "nothing was found" and
        # "the detectors did not run" look identical otherwise.
        lines.append("  none: no detector matched this corpus")
    for finding in report.findings:
        lines.append(f"  [{finding.severity}] {finding.kind}: {finding.summary}")
        lines.append(f"    evidence: {', '.join(finding.evidence_ids)}")
    lines.append("")

    lines.append("--- assessment ---")
    lines.append(mark_unverified(report.narrative, report.audit) or "  (none)")
    lines.append("")

    lines.append("--- citations ---")
    audit = report.audit
    lines.append(
        f"  {len(set(audit.records_resolved))} of {len(set(audit.records_cited))} "
        f"record citations resolve"
        + (
            f"; {len(set(audit.chunks_resolved))} of {len(set(audit.chunks_cited))} "
            "documentation citations resolve"
            if audit.chunks_cited
            else ""
        )
    )
    if not audit.clean:
        # A warning rather than a footnote. An unresolvable citation means the
        # model wrote an id that does not exist, which is the failure mode this
        # whole mechanism exists to surface.
        unresolved = (*audit.unresolved_records, *audit.unresolved_chunks)
        lines.append(
            f"  WARNING: {len(unresolved)} citation(s) do not resolve: "
            f"{', '.join(unresolved)}"
        )
        lines.append("  A claim resting on one of these is unsupported by this corpus.")
    lines.append(f"  tool calls: {report.tool_calls}")

    return "\n".join(lines)
