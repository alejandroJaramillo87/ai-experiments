"""Score retrieval against the committed ground-truth set.

The first command in this project that produces a quality number rather than a
count. What makes it worth trusting is not here -- it is in
`logtriage/evals/README.md`, which records how the questions were chosen and
what the set cannot tell you. Read that before quoting a row.

Three legs against one index, reported over three slices. The comparison that
matters is a semantic embedder against the hashing baseline on the `no-overlap`
slice: items whose wording shares no distinctive term with their answer page,
where meaning can beat vocabulary. A win on the `overlap` slice alone is a
retriever doing string matching, and printing both is what makes that visible.

`--pool` writes the union of every leg's top-k for adjudication. Recall against
an incomplete label set understates every retriever, so the pool is how the set
improves: anything retrieved but unlabelled is a candidate label a human rules
on. See docs/verification/desktop-model.md D12.
"""

from __future__ import annotations

import argparse
import json

# Legs in report order: the two that can be measured alone, then the fusion.
LEGS = ("text", "vector", "hybrid")


def configure(sub) -> None:
    p = sub.add_parser(
        "eval-retrieval",
        help="score retrieval against the committed ground-truth question set",
        description=__doc__,
    )
    p.add_argument(
        "--index",
        default=None,
        help="on-disk index to search (default: <kb dir>/index)",
    )
    p.add_argument(
        "--store",
        default="file",
        choices=["file", "mongo"],
        help="where the index lives (default: file; mongo queries mongot)",
    )
    p.add_argument(
        "--uri",
        default=None,
        help="MongoDB connection string for --store mongo (default: LOGTRIAGE_MONGO_URI)",
    )
    p.add_argument(
        "--k",
        type=int,
        default=10,
        help="how many hits to retrieve per question (default: 10)",
    )
    p.add_argument(
        "--pool",
        default=None,
        help="write the union of every leg's hits to this file, for adjudication",
    )
    p.add_argument("--json", action="store_true", help="emit the scores as JSON")
    p.set_defaults(func=cmd)


def _open_index(args, out):
    """The store to search, and the embedder that built it.

    Both paths reconstruct the embedder from the index rather than taking it
    from a flag, for the reason `triage --kb` does: an index searched with a
    different embedder returns k chunks ranked by a similarity that means
    nothing, and nothing errors.
    """
    if args.store == "mongo":
        from logtriage.commands.triage import _load_kb_from_mongo

        return _load_kb_from_mongo(args.uri, out)

    from logtriage.commands.triage import _load_kb

    return _load_kb(args.index or "", out)


def _ranks(store, embedder, questions, leg: str, k: int, pool: dict[str, set[str]]):
    """Retrieve for every question on one leg and record where the answer landed."""
    from logtriage.store.knowledge import VECTOR_WEIGHT

    found: dict[str, int | None] = {}
    for question in questions:
        if leg == "text":
            hits = store.text_search(question.question, k=k)
        elif leg == "vector":
            (vector,) = embedder.embed([question.question])
            hits = store.vector_search(vector, k=k)
        else:
            (vector,) = embedder.embed([question.question])
            # The production weight, not zero: this leg is being measured, and
            # measuring it at the weight `search_docs` would use for a
            # non-semantic embedder would measure the text leg twice.
            hits = store.hybrid_search(
                question.question, vector, k=k, vector_weight=VECTOR_WEIGHT
            )

        documents = [hit.chunk.document for hit in hits]
        pool.setdefault(question.id, set()).update(documents)

        from logtriage.evals.metrics import rank_of

        found[question.id] = rank_of(question, documents)
    return found


def _render(table: dict[str, dict[str, object]], semantic: bool) -> str:
    from logtriage.evals.metrics import CUTOFFS

    header = (
        f"{'leg':8} {'slice':11} {'n':>4}"
        + "".join(f"{'R@' + str(c):>8}" for c in CUTOFFS)
        + f"{'MRR':>8}"
    )
    lines = [header, "-" * len(header)]
    for leg in LEGS:
        for name, scores in table[leg].items():  # type: ignore[union-attr]
            recall = "".join(f"{scores.recall[c]:>8.2f}" for c in CUTOFFS)  # type: ignore[attr-defined]
            lines.append(
                f"{leg:8} {name:11} {scores.questions:>4}{recall}{scores.mrr:>8.3f}"  # type: ignore[attr-defined]
            )
    if not semantic:
        lines += [
            "",
            "  These are the lexical baseline. The embedder hashes tokens, so the",
            "  vector and hybrid rows measure word overlap projected into a vector",
            "  space, not meaning. They are the number a real embedder must beat.",
        ]
    return "\n".join(lines)


def cmd(args: argparse.Namespace) -> int:
    from logtriage.cli import LogtriageError
    from logtriage.evals.metrics import load, score, slices
    from logtriage.output import Printer

    out = Printer(quiet=args.json)

    try:
        questions = load()
    except (OSError, ValueError) as exc:
        raise LogtriageError(f"Cannot read the ground-truth set: {exc}") from exc

    store, embedder = _open_index(args, out)
    if store is None or embedder is None:  # pragma: no cover - _load_kb raises first
        raise LogtriageError("No documentation index to evaluate against")

    out.run(f"Scoring {len(questions)} questions at k={args.k}")

    pool: dict[str, set[str]] = {}
    table: dict[str, dict[str, object]] = {}
    for leg in LEGS:
        ranks = _ranks(store, embedder, questions, leg, args.k, pool)
        table[leg] = {
            name: score({q.id: ranks[q.id] for q in group})
            for name, group in slices(questions).items()
        }

    if args.pool:
        _write_pool(args.pool, questions, pool, out)

    payload = {
        "questions": len(questions),
        "k": args.k,
        "embedder": type(embedder).__name__,
        "semantic": embedder.semantic,
        "results": {
            leg: {name: s.as_dict() for name, s in slices_.items()}  # type: ignore[union-attr,attr-defined]
            for leg, slices_ in table.items()
        },
    }

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    out.line()
    out.line(_render(table, embedder.semantic))
    out.line()
    _report_misses(table, questions, out)
    return 0


def _report_misses(table, questions, out) -> None:
    """Name the questions the hybrid leg did not answer.

    A score that moved is worth less than the item that moved it, and a list of
    ids is what a reader needs to decide whether the miss is a retrieval failure
    or a label that is wrong.
    """
    hybrid = table["hybrid"]["all"]
    missed = [q for q in questions if hybrid.ranks.get(q.id) is None]
    if not missed:
        out.ok("Every question found an acceptable page in the top k on the hybrid leg")
        return
    out.warn(f"{len(missed)} question(s) found no acceptable page on the hybrid leg:")
    for question in missed:
        marker = "" if question.lexical_overlap else "  [no-overlap]"
        out.line(f"    {question.id}: {question.question}{marker}")


def _write_pool(path: str, questions, pool: dict[str, set[str]], out) -> None:
    """Write every page any leg returned, marked against the current labels.

    The output is the adjudication worksheet: `labelled` is what the set already
    accepts, `candidate` is what a retriever thought was relevant and nobody has
    ruled on. Only the second needs a human.
    """
    from pathlib import Path

    by_id = {q.id: q for q in questions}
    rows = []
    for question_id, documents in sorted(pool.items()):
        question = by_id[question_id]
        for document in sorted(documents):
            rows.append(
                {
                    "id": question_id,
                    "question": question.question,
                    "page": document,
                    "status": "labelled" if document in question.pages else "candidate",
                }
            )
    Path(path).write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )
    candidates = sum(1 for row in rows if row["status"] == "candidate")
    out.ok(
        f"Pool written to {path}: {len(rows)} rows, {candidates} awaiting a decision"
    )
