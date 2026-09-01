"""The triage loop: profile, detect, then ask the model to explain.

The ordering is the whole design. Deterministic detectors run first and their
output goes into the prompt, so the model arrives at a corpus that has already
been counted. Its job is to explain, rank and connect -- not to discover, which
is the task it is worst at and the one where its failures are most confident.

The loop itself is small on purpose. A model with five tools and a bounded turn
count is easier to reason about than an agent framework, and every turn spent is
a prefill of the whole conversation, which on the target hardware is the
dominant cost.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from logtriage.llm.base import Completion, LlmClient, Message
from logtriage.mongod.diagnostics import Finding, run_all
from logtriage.mongod.normalize import NormalizedRecord
from logtriage.store.base import LogStore
from logtriage.store.knowledge import KnowledgeStore
from logtriage.tools.schemas import ToolBox
from logtriage.triage.profile import CorpusProfile, build_profile, render_profile
from logtriage.triage.report import TriageReport, audit_citations

# Turns the model may take before the loop stops asking. Each turn re-prefills
# the whole conversation, and on the target hardware prefill dominates, so this
# is a latency budget rather than a safety limit.
MAX_TURNS = 8

SYSTEM_PROMPT = """\
You are triaging mongod logs for an engineer who will act on what you say.

You have been given a computed profile of the whole log and a list of findings
produced by deterministic detectors. Those numbers are already correct. Do not
recount them, and do not contradict them without evidence you can cite.

Your job is to explain what happened, in what order, and what to do about it.

Rules, in order of importance:

1. Cite everything. Every factual claim carries the record ids it rests on, like
   L4821. A claim with no citation will be read as unsupported, and a citation
   that does not resolve will be marked as such in the output.

2. Do not assert a cause you cannot order in time. The log gives you first
   occurrence times for every finding. If A began before B, B did not cause A.
   Elections, stepdowns and failovers are consequences at least as often as
   causes: a busy primary that cannot answer heartbeats gets voted out, and the
   election that follows explains nothing about why it was busy.

3. Say what you do not know. If the log does not contain the evidence needed to
   settle a question, say that, and say what would settle it. An admitted gap is
   worth more than a confident guess, because the reader can act on the first.

4. Prefer query_logs over search. Counting questions have exact answers; search
   returns what resembles the question, which is not the same thing.

5. Be brief. An engineer reads the first three sentences.
"""


@dataclass(frozen=True, slots=True)
class TriageOutcome:
    """The report, plus the conversation that produced it."""

    report: TriageReport
    messages: tuple[Message, ...]


def _findings_block(findings: Sequence[Finding]) -> str:
    if not findings:
        return (
            "No detector matched this corpus. That is a result: the automated "
            "checks for collection scans, unselective indexes, connection "
            "exhaustion, elections, failed index builds and repeated errors all "
            "found nothing."
        )
    lines = ["Deterministic findings, worst first:"]
    for finding in findings:
        first = finding.first_seen.isoformat() if finding.first_seen else "unknown"
        lines.append(
            f"- [{finding.severity}] {finding.kind}"
            + (f" on {finding.ns}" if finding.ns else "")
            + f", first seen {first}: {finding.summary}"
        )
        lines.append(f"  evidence: {', '.join(finding.evidence_ids)}")
        if finding.metrics:
            lines.append(
                "  metrics: "
                + ", ".join(f"{k}={v}" for k, v in finding.metrics.items())
            )
    return "\n".join(lines)


def build_prompt(profile_text: str, findings: Sequence[Finding], question: str) -> str:
    return "\n\n".join(
        [
            profile_text,
            _findings_block(findings),
            f"Task: {question}",
        ]
    )


DEFAULT_QUESTION = (
    "What happened here, in what order, and what should be done about it? "
    "If one finding explains another, say which way round and cite the times."
)


def run_triage(
    records: Sequence[NormalizedRecord] | None,
    logs: LogStore,
    llm: LlmClient,
    *,
    knowledge: KnowledgeStore | None = None,
    embedder=None,
    source: str = "<stream>",
    question: str = DEFAULT_QUESTION,
    parse_errors: int = 0,
    max_turns: int = MAX_TURNS,
    profile: CorpusProfile | None = None,
) -> TriageOutcome:
    """Profile, detect, converse, and render.

    `records` is optional: a caller that already has a profile passes it
    instead, which is what a Mongo-backed run does -- the records live in the
    database and pulling them all back to compute a profile would defeat the
    point of having one.
    """
    if profile is None:
        if records is None:
            raise ValueError("run_triage needs either records or a precomputed profile")
        profile = build_profile(records, source=source, parse_errors=parse_errors)
    profile_text = render_profile(profile)

    findings = run_all(logs)
    toolbox = ToolBox(
        logs=logs,
        knowledge=knowledge,
        embedder=embedder,
        profile_text=profile_text,
    )

    messages: list[Message] = [
        Message(role="system", content=SYSTEM_PROMPT),
        Message(role="user", content=build_prompt(profile_text, findings, question)),
    ]

    tool_calls = 0
    completion = Completion()
    for _ in range(max_turns):
        completion = llm.complete(messages, tools=toolbox.schemas())
        if not completion.wants_tools:
            break
        messages.append(
            Message(
                role="assistant",
                content=completion.content,
                tool_calls=completion.tool_calls,
            )
        )
        for call in completion.tool_calls:
            result = toolbox.call(call.name, call.arguments)
            tool_calls += 1
            messages.append(
                Message(
                    role="tool",
                    content=result.text,
                    tool_call_id=call.id,
                    name=call.name,
                )
            )
    else:
        # The loop ran out of turns with the model still asking for tools. Said
        # plainly in the narrative rather than silently returning what there is,
        # because a truncated investigation and a finished one look identical
        # from the outside.
        completion = Completion(
            content=(
                (completion.content + "\n\n" if completion.content else "")
                + f"(Investigation stopped after {max_turns} turns with the model "
                "still requesting tools. This assessment is incomplete.)"
            )
        )

    narrative = completion.content.strip()
    messages.append(Message(role="assistant", content=narrative))

    report = TriageReport(
        source=source,
        profile=profile,
        findings=tuple(findings),
        narrative=narrative,
        audit=audit_citations(narrative, logs, knowledge),
        tool_calls=tool_calls,
    )
    return TriageOutcome(report=report, messages=tuple(messages))
