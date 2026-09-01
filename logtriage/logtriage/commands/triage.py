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

from logtriage.store.factory import BACKENDS as STORE_BACKENDS


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
        choices=["scripted", "llama", "none"],
        help="model backend (default: scripted, which replays a fixture offline; "
        "llama needs LOGTRIAGE_LLAMA_URL; none runs the detectors only)",
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
    p.add_argument(
        "--store",
        default="memory",
        choices=list(STORE_BACKENDS),
        help="where the records are queried from (default: memory, built from "
        "the log just parsed; mongo ingests into the deployment and queries it)",
    )
    p.add_argument(
        "--uri",
        default=None,
        help="MongoDB connection string for --store mongo (default: LOGTRIAGE_MONGO_URI)",
    )
    p.add_argument(
        "--no-ingest",
        action="store_true",
        help="with --store mongo, query records already present instead of "
        "writing them first",
    )
    p.add_argument("--json", action="store_true", help="emit the report as JSON")
    p.set_defaults(func=cmd)


def _open_store(args, records, out):
    """The store the detectors and the model's tools query.

    Ingest is idempotent -- `record_id` is a content handle, so a re-run
    replaces rather than duplicating -- which is what makes writing on every
    triage run safe rather than merely convenient. `--no-ingest` is for the case
    where the data is already there and the write is the expensive part.
    """
    from logtriage.store.factory import log_store

    store = log_store(args.store, args.uri)
    if args.store == "memory":
        store.insert(records)
        return store

    if args.no_ingest:
        out.skip("Ingest skipped; querying records already in the deployment")
    else:
        out.run(f"Ingesting {len(records):,} records into MongoDB")
        store.insert(records)

    # A citation that cannot be resolved is a citation nobody can check, so an
    # empty store is a failed run rather than a report with no evidence.
    if not store.count(_everything()):
        from logtriage.cli import LogtriageError

        raise LogtriageError(
            "No records in the deployment to triage\n"
            "  Solution: ingest the log first, or drop --no-ingest\n"
            f"  Command: python -m logtriage ingest {args.log} --store mongo"
        )
    return store


def _everything():
    from logtriage.store.base import LogFilter

    return LogFilter()


def _load_kb_from_mongo(uri: str | None, out):
    """Open the documentation index held by MongoDB and mongot.

    The embedder comes from the index's own metadata, exactly as it does for an
    on-disk index -- `index-kb --to mongo` writes it to a sibling collection for
    this reason. Without it a caller would have to name an embedder on the
    command line, and naming the wrong one produces a plausible ranking over
    meaningless similarities with nothing to catch it.
    """
    from logtriage.cli import LogtriageError
    from logtriage.embed.factory import from_index_meta
    from logtriage.store.factory import mongo_knowledge_store

    knowledge = mongo_knowledge_store(uri)
    meta = knowledge.load_meta()
    if not meta:
        raise LogtriageError(
            "The chunk collection does not record which embedder built it\n"
            "  Querying it would rank against a similarity that means nothing.\n"
            "  Solution: rebuild the index, which writes its own description\n"
            "  Command: python -m logtriage index-kb --to mongo --refresh"
        )

    embedder, expect = from_index_meta(meta)
    width = int(meta.get("embedder_dimensions", 0))
    if width and width != embedder.dimensions:
        raise LogtriageError(
            f"The index was built at {width} dimensions and this embedder "
            f"produces {embedder.dimensions}\n"
            "  Solution: point LOGTRIAGE_EMBED_URL at the model that built it, "
            "or rebuild\n"
            "  Command: python -m logtriage index-kb --to mongo --refresh --embedder llama"
        )

    out.info(
        f"Manual {meta.get('manual_version', '?')} at "
        f"{str(meta.get('corpus_commit', '?'))[:12]}: {len(knowledge)} passages "
        f"({expect.get('embedder', '?')} embeddings, via mongot)"
    )
    return knowledge, embedder


def _load_kb(path: str | None, out, backend: str = "memory", uri: str | None = None):
    """Open a documentation index, or explain why there is none.

    Returns (knowledge, embedder). A missing index is a hard error rather than a
    quiet fall-through: --kb was asked for explicitly, and a run that silently
    proceeds without documentation looks identical to one that used it.
    """
    if path is None:
        return None, None

    if backend == "mongo":
        return _load_kb_from_mongo(uri, out)

    import json
    from pathlib import Path

    from logtriage.cli import LogtriageError
    from logtriage.embed.factory import from_index_meta
    from logtriage.kb.fetch import default_kb_dir
    from logtriage.store.knowledge import META_FILE, InMemoryKnowledgeStore

    directory = Path(path) if path else default_kb_dir() / "index"
    if not directory.is_dir():
        raise LogtriageError(
            f"No documentation index at {directory}\n"
            "  Solution: fetch the manual and build an index\n"
            "  Command: python -m logtriage fetch-kb && python -m logtriage index-kb"
        )

    # The embedder is reconstructed from the index rather than chosen by a flag.
    # An index searched with an embedder other than the one that built it still
    # returns its k best chunks, ranked by a similarity that means nothing --
    # so the index names its own embedder and this rebuilds that one.
    try:
        stored = json.loads((directory / META_FILE).read_text(encoding="utf-8"))
        embedder, expect = from_index_meta(stored)
        knowledge, meta = InMemoryKnowledgeStore.load(directory, expect=expect)
    except (OSError, ValueError) as exc:
        raise LogtriageError(str(exc)) from exc

    out.info(
        f"Manual {meta.get('manual_version', '?')} at "
        f"{str(meta.get('corpus_commit', '?'))[:12]}: {len(knowledge)} passages "
        f"({meta.get('embedder', '?')} embeddings)"
    )
    return knowledge, embedder


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


def _client(backend: str, findings):
    """The model backend named on the command line.

    `llama` is the only one that reaches a model. The other two exist so the
    loop can be exercised and so detector output can be had without one, and
    neither should ever be mistaken for an assessment -- the scripted client's
    own narrative says so in its last sentence.
    """
    if backend == "llama":
        from logtriage import config
        from logtriage.llm.llama import LlamaClient

        return LlamaClient(config.require(config.LLAMA_URL))
    return _canned_client(findings)


def cmd(args: argparse.Namespace) -> int:
    from logtriage.cli import LogtriageError
    from logtriage.mongod.diagnostics import run_all
    from logtriage.mongod.normalize import normalize
    from logtriage.mongod.records import LogRecord, ParseError, read_log
    from logtriage.output import Printer
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

    store = _open_store(args, records, out)
    knowledge, embedder = _load_kb(args.kb, out, args.store, args.uri)

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
            _client(args.llm, run_all(store)),
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
