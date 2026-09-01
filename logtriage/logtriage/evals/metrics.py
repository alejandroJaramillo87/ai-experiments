"""Loading the labelled set, and scoring a ranking against it.

Binary relevance, because the labels are binary: a page either answers the
question or it does not, and inventing graded relevance would mean inventing the
grades. `pages` is a set of acceptable answers rather than a ranking -- several
manual pages legitimately answer "why is this query slow", and insisting on one
would measure agreement with an arbitrary choice rather than retrieval.

Two metrics, and they answer different questions. Recall@k asks whether the
answer is on the page the reader will look at; MRR asks how far down it was.
A retriever that puts the right page tenth scores the same recall@10 as one that
puts it first, and the model reading the results has a token budget.

The slice is the point. `lexical_overlap: false` items share no distinctive term
with their answer page, so they are where an embedder with semantics can beat
word overlap. The average over both halves lets easy items hide a retriever that
only does string matching, which is exactly the failure this set exists to
detect.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Where the committed set lives. Beside this module so it travels with the
# package rather than depending on a working directory.
GROUND_TRUTH = Path(__file__).parent / "retrieval.jsonl"

# Cut-offs reported. 1 is "the reader sees it first", 10 is "it is on the page
# of results at all"; 5 is the one that moves when a ranking improves without
# becoming perfect.
CUTOFFS = (1, 5, 10)

# MRR is computed to the deepest cut-off rather than unbounded: a hit at rank
# 200 contributes 0.005 and is not a result anybody acts on.
MRR_AT = 10


@dataclass(frozen=True, slots=True)
class Question:
    """One labelled question."""

    id: str
    question: str
    pages: tuple[str, ...]
    source: str
    lexical_overlap: bool
    note: str = ""


@dataclass(slots=True)
class Scores:
    """What one retriever scored over one set of questions."""

    questions: int = 0
    recall: dict[int, float] = field(default_factory=dict)
    mrr: float = 0.0
    # Per question, the rank the first acceptable page landed at, or None. Kept
    # so a caller can show which questions failed rather than only how many --
    # a number that moved is worth less than the item that moved it.
    ranks: dict[str, int | None] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "questions": self.questions,
            "recall": {f"@{k}": round(v, 4) for k, v in sorted(self.recall.items())},
            "mrr": round(self.mrr, 4),
        }


def load(path: Path = GROUND_TRUTH) -> list[Question]:
    """Read the committed set, refusing a malformed one.

    Strict about its own file: a typo in a label is a silently wrong measurement
    later, and this is the only place that can catch it cheaply.
    """
    items: list[Question] = []
    seen: set[str] = set()
    for number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{path}:{number}: not valid JSON: {error}") from error
        missing = {"id", "question", "pages", "source", "lexical_overlap"} - set(raw)
        if missing:
            raise ValueError(f"{path}:{number}: missing {sorted(missing)}")
        if not raw["pages"]:
            raise ValueError(f"{path}:{number}: {raw['id']} lists no acceptable page")
        if raw["id"] in seen:
            raise ValueError(f"{path}:{number}: duplicate id {raw['id']}")
        seen.add(raw["id"])
        items.append(
            Question(
                id=str(raw["id"]),
                question=str(raw["question"]),
                pages=tuple(raw["pages"]),
                source=str(raw["source"]),
                lexical_overlap=bool(raw["lexical_overlap"]),
                note=str(raw.get("note", "")),
            )
        )
    if not items:
        raise ValueError(f"{path}: no questions")
    return items


def rank_of(question: Question, documents: Sequence[str]) -> int | None:
    """1-based rank of the first acceptable page, or None if it is absent.

    `documents` is the retrieved pages in order, one per hit. Several hits may
    come from the same page -- a page is many chunks -- and the first is what
    counts: a reader opening the top result gets the page, not the chunk.
    """
    acceptable = set(question.pages)
    for position, document in enumerate(documents, start=1):
        if document in acceptable:
            return position
    return None


def score(ranks: dict[str, int | None]) -> Scores:
    """Aggregate per-question ranks into recall and MRR."""
    total = len(ranks)
    result = Scores(questions=total, ranks=dict(ranks))
    if not total:
        return result
    for cutoff in CUTOFFS:
        hits = sum(1 for rank in ranks.values() if rank is not None and rank <= cutoff)
        result.recall[cutoff] = hits / total
    result.mrr = (
        sum(
            1.0 / rank for rank in ranks.values() if rank is not None and rank <= MRR_AT
        )
        / total
    )
    return result


def slices(questions: Iterable[Question]) -> dict[str, list[Question]]:
    """The three views a result is reported over.

    `no-overlap` is the headline. `overlap` is reported beside it not because it
    is interesting on its own but because a retriever that wins there and loses
    on the other half is a retriever doing string matching, and seeing both
    numbers is what makes that visible.
    """
    items = list(questions)
    return {
        "all": items,
        "no-overlap": [q for q in items if not q.lexical_overlap],
        "overlap": [q for q in items if q.lexical_overlap],
    }
