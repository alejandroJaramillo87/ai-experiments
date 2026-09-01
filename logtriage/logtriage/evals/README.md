# Retrieval evaluation

`retrieval.jsonl` is the labelled question set `logtriage eval-retrieval` measures against. This
file records how it was built, because the method is what decides whether the number means
anything.

## Table of Contents

- [The problem this is built against](#the-problem-this-is-built-against)
- [How the questions were chosen](#how-the-questions-were-chosen)
- [The fields](#the-fields)
- [What the metrics are](#what-the-metrics-are)
- [Known weaknesses](#known-weaknesses)

## The problem this is built against

The set that existed before this one was eight questions in `tests/test_kb_corpus.py`, each
written to contain its answer page's own vocabulary — `planSummary COLLSCAN currentOp output
field` for the `currentOp` page. That is a fine smoke test for whether lexical retrieval works
at all. It is not a quality measure, and scoring a semantic embedder against it would flatter
word overlap and understate the model, because every question is one word overlap can already
answer.

The deeper risk is that a set written by the same person who wrote the retriever tends to be a
set the retriever passes. Four choices are made against that, and each is a decision rather than
a detail.

## How the questions were chosen

**Questions come from the task, not from the corpus.** The situations are the eight kinds of
finding `mongod/diagnostics.py` produces and the three log identifiers the manual documents —
what an operator is actually holding when they reach for the manual. A question written by
reading a page is a question that page can already answer.

**Pages were chosen from what the manual says each page is about** — its title and its own
`:ref:` anchor — not from what any retriever returned for the question. `source` records which
of the three provenances each item came from.

That method failed once during authoring, and the correction is recorded rather than quietly
made. Seven items were first labelled `core/query-plans.txt`, chosen from its title, "Query
Plans". Reading the page shows it is about the **plan cache** — it contains no occurrence of
`COLLSCAN` and none of "collection scan". The page that actually explains a collection scan is
`tutorial/analyze-query-plan.txt`, "Interpret Explain Plan Results". The seven labels were
repointed before any measurement was taken, on the evidence of reading both pages, not on the
evidence of what retrieved better. Choosing a page by its title is the same shortcut this
document warns against one paragraph earlier, and it was made anyway.

**`lexical_overlap` marks the discriminating half.** An item is `false` when its wording shares
no distinctive term with the answer page: an operator saying "the database stopped answering"
rather than "connection pool exhaustion". Those are the items where meaning can beat vocabulary,
and they are the headline slice — the average over both halves would let easy items hide a
retriever that only does string matching.

**The set is committed before anything is measured against it.** Writing labels after seeing
what retrieves is how a ground truth becomes a description of the retriever.

## The fields

```json
{"id": "conn-refused-02",
 "question": "the server stopped accepting new connections",
 "pages": ["administration/connection-pool-overview.txt", "reference/ulimit.txt"],
 "source": "detector:connection_refused",
 "lexical_overlap": false,
 "note": "why these pages answer it"}
```

`pages` is a set of acceptable answers, not a ranking. Relevance is binary and a hit counts if
any listed page is retrieved: several manual pages legitimately answer "why is this query slow",
and insisting on one would measure agreement with an arbitrary choice.

`source` is one of `detector:<kind>`, `log-id:<id>`, or `manual-ref:<topic>`.

## What the metrics are

`metrics.py` computes recall@1, recall@5, recall@10 and MRR@10, reported per retrieval leg
(text, vector, hybrid) and per embedder, and separately for the `lexical_overlap: false` slice.

The `HashingEmbedder` row is the lexical baseline. That embedder has no semantics — it hashes
tokens, so nearness is vocabulary overlap — and its own module docstring already names this as
the one thing it is good for beyond plumbing: *if a real embedding model cannot beat word
overlap on a retrieval set, that is worth knowing.*

Measured over the corpus at commit `88ec8192b924`, at k=10:

| leg | slice | n | R@1 | R@5 | R@10 | MRR |
|-----|-------|---|-----|-----|------|-----|
| text | all | 40 | 0.23 | 0.42 | 0.55 | 0.305 |
| text | no-overlap | 18 | 0.06 | 0.11 | 0.33 | 0.094 |
| text | overlap | 22 | 0.36 | 0.68 | 0.73 | 0.478 |
| vector | all | 40 | 0.23 | 0.33 | 0.50 | 0.284 |
| vector | no-overlap | 18 | 0.00 | 0.06 | 0.28 | 0.044 |
| vector | overlap | 22 | 0.41 | 0.55 | 0.68 | 0.480 |
| hybrid | all | 40 | 0.33 | 0.45 | 0.57 | 0.388 |
| hybrid | no-overlap | 18 | 0.06 | 0.17 | 0.33 | 0.105 |
| hybrid | overlap | 22 | 0.55 | 0.68 | 0.77 | 0.619 |

The gap between the two slices is the thing to read: 0.77 recall@10 where the question shares
vocabulary with its answer, 0.33 where it does not. That is what word overlap can and cannot do,
and it is the shape a real embedder has to change. A model that improves the `overlap` row and
leaves `no-overlap` where it is has not added meaning to retrieval; it has added a second way of
matching strings.

## Known weaknesses

Stated because a labelled set always has them and an unstated one is a claim.

**The labels are incomplete.** Only pages someone thought of are listed, so a retriever that
returns a genuinely good page nobody labelled is scored as wrong. Recall against an incomplete
set understates every retriever, and unequally. The fix is pooling, and it is a runbook entry
rather than a promise: `eval-retrieval --pool` writes the union of the top-10 from every leg and
embedder, and `docs/verification/desktop-model.md` D12 adjudicates what comes back.

**One person wrote the questions and chose the pages.** The controls above reduce that; they do
not remove it. A second reader disagreeing with a label is a finding, and the file is meant to
be edited when that happens — with the change recorded, since a label edited after a measurement
invalidates the measurement.

**Forty items is small.** Differences of a point or two between two rows are noise. The
comparison worth making is a clear one on the no-overlap slice, not a ranking of near-ties.
