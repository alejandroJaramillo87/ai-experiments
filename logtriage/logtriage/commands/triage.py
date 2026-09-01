"""Triage a mongod log: profile it, run the detectors, and explain the result.

The default model backend is `scripted`, which replays a fixture and needs
nothing. That is not a placeholder -- it is how the pipeline is exercised
without a served model, and it makes `logtriage triage` runnable in a checkout
with no GPU. Point --llm at a served model to get an actual assessment.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict


def configure(sub) -> None:
    p = sub.add_parser(
        "triage",
        help="profile a mongod log, run the detectors, and explain the findings",
        description=__doc__,
    )
    p.add_argument("log", help="path to a mongod JSON log file")
    p.add_argument(
        "--llm",
        default="scripted",
        choices=["scripted", "none"],
        help="model backend (default: scripted, which replays a fixture offline)",
    )
    p.add_argument(
        "--question",
        default=None,
        help="what to ask about the log (default: what happened and what to do)",
    )
    p.add_argument(
        "--redact",
        action="store_true",
        help="replace literal command values with type placeholders at ingest",
    )
    p.add_argument(
        "--kb",
        nargs="?",
        const="",
        default=None,
        help="load a documentation index so the model can search the manual "
        "(default location: <kb dir>/index; see `logtriage index-kb`)",
    )
    p.add_argument("--json", action="store_true", help="emit the report as JSON")
    p.set_defaults(func=cmd)


def _load_kb(path: str | None, out):
    """Open a documentation index, or explain why there is none.

    Returns (knowledge, embedder). A missing index is a hard error rather than a
    quiet fall-through: --kb was asked for explicitly, and a run that silently
    proceeds without documentation looks identical to one that used it.
    """
    if path is None:
        return None, None

    from pathlib import Path

    from logtriage.cli import LogtriageError
    from logtriage.embed.hashing import DEFAULT_SEED, DIMENSIONS, HashingEmbedder
    from logtriage.kb.fetch import default_kb_dir
    from logtriage.store.knowledge import InMemoryKnowledgeStore

    directory = Path(path) if path else default_kb_dir() / "index"
    if not directory.is_dir():
        raise LogtriageError(
            f"No documentation index at {directory}\n"
            "  Solution: fetch the manual and build an index\n"
            "  Command: python -m logtriage fetch-kb && python -m logtriage index-kb"
        )
    try:
        knowledge, meta = InMemoryKnowledgeStore.load(
            directory,
            expect={
                "embedder_seed": DEFAULT_SEED,
                "embedder_dimensions": DIMENSIONS,
            },
        )
    except (OSError, ValueError) as exc:
        raise LogtriageError(str(exc)) from exc

    out.info(
        f"Manual {meta.get('manual_version', '?')} at "
        f"{str(meta.get('corpus_commit', '?'))[:12]}: {len(knowledge)} passages"
    )
    return knowledge, HashingEmbedder()


def _canned_client(findings):
    """A scripted model that summarises the findings it was handed.

    Deliberately mechanical. It cites only ids the detectors produced, so a run
    with this backend exercises the whole loop -- dispatch, audit, render --
    without pretending to demonstrate anything about a real model's judgement.
    """
    from logtriage.llm.scripted import ScriptedLlmClient, says

    if not findings:
        return ScriptedLlmClient.of(
            says(
                "No detector matched this corpus, and the profile shows no records "
                "above severity I. Nothing here needs action. If a problem was "
                "reported for this window, the evidence for it is not in this log."
            )
        )

    worst = findings[0]
    ordered = [f for f in findings if f.first_seen]
    ordered.sort(key=lambda f: f.first_seen)
    lines = [
        f"The most severe finding is {worst.kind}"
        + (f" on {worst.ns}" if worst.ns else "")
        + f": {worst.summary} (see {', '.join(worst.evidence_ids)})."
    ]
    if len(ordered) > 1:
        first, last = ordered[0], ordered[-1]
        lines.append(
            f"In time order, {first.kind} appears first at "
            f"{first.first_seen.isoformat()} and {last.kind} last at "
            f"{last.first_seen.isoformat()}; on that ordering {last.kind} cannot "
            f"have caused {first.kind}."
        )
    lines.append(
        "This assessment was produced by the scripted backend, which restates "
        "the detectors rather than reasoning about them."
    )
    return ScriptedLlmClient.of(says(" ".join(lines)))


def cmd(args: argparse.Namespace) -> int:
    from logtriage.cli import LogtriageError
    from logtriage.mongod.diagnostics import run_all
    from logtriage.mongod.normalize import normalize
    from logtriage.mongod.records import LogRecord, ParseError, read_log
    from logtriage.output import Printer
    from logtriage.store.memory import InMemoryLogStore
    from logtriage.triage.pipeline import DEFAULT_QUESTION, run_triage
    from logtriage.triage.report import render_report

    out = Printer(quiet=args.json)

    try:
        parsed = list(read_log(args.log))
    except OSError as exc:
        raise LogtriageError(
            f"Cannot read log: {exc}\n"
            f"  Solution: check the path exists and is readable\n"
            f"  Command: ls -l {args.log}"
        ) from exc

    records = [
        normalize(r, redact=args.redact) for r in parsed if isinstance(r, LogRecord)
    ]
    errors = [p for p in parsed if isinstance(p, ParseError)]
    if not records:
        raise LogtriageError(
            f"No mongod log records found in {args.log} "
            f"({len(errors)} unparseable lines)\n"
            "  Solution: confirm this is a mongod 4.4+ structured (JSON) log\n"
            f"  Command: head -1 {args.log}"
        )

    store = InMemoryLogStore(records)
    knowledge, embedder = _load_kb(args.kb, out)

    if args.llm == "none":
        # Detectors only. Worth having on its own: it is the half of the output
        # that carries no model judgement at all, and some readers want exactly
        # that.
        from logtriage.triage.profile import build_profile
        from logtriage.triage.report import CitationAudit, TriageReport

        report = TriageReport(
            source=args.log,
            profile=build_profile(records, source=args.log, parse_errors=len(errors)),
            findings=tuple(run_all(store)),
            narrative="(no model backend; detector output only)",
            audit=CitationAudit(),
        )
    else:
        outcome = run_triage(
            records,
            store,
            _canned_client(run_all(store)),
            knowledge=knowledge,
            embedder=embedder,
            source=args.log,
            question=args.question or DEFAULT_QUESTION,
            parse_errors=len(errors),
        )
        report = outcome.report

    if args.json:
        payload = {
            "source": report.source,
            "findings": [asdict(f) for f in report.findings],
            "narrative": report.narrative,
            "citations": asdict(report.audit),
            "tool_calls": report.tool_calls,
            "parse_errors": len(errors),
        }
        print(json.dumps(payload, indent=2, default=str))
        return 0

    out.line(render_report(report))
    out.line()
    if errors:
        out.warn(
            f"Unparseable lines: {len(errors)} (first at line {errors[0].line_no})"
        )
    if not report.audit.clean:
        # Exit 1: a report whose citations do not resolve is a failed run, not a
        # successful one with a caveat. A caller in a pipeline needs to be able
        # to tell those apart without parsing the prose.
        out.error("Report contains citations that do not resolve")
        return 1
    out.ok(f"Findings: {len(report.findings)}")
    return 0
