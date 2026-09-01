# Phase D: the model

**Operational runbook. Executed against the Linux workstation, not read for understanding: the
operator's pasted output is the evidence. Keep the commands runnable — the documentation gate
checks every path, make target and CLI verb named here.**

**Entries: 0 verified, 0 blocked, 4 pending, of 4.**

Real embeddings, the first retrieval-quality number this project will have, and the first
evidence about how Gemma actually behaves. Everything before this phase used a hashing embedder
with no semantics and a scripted client that replays fixtures.

Blocked on [desktop-search.md](desktop-search.md).

## Table of Contents

- [D10 — real embeddings over the real corpus](#d10--real-embeddings-over-the-real-corpus)
- [D11 — retrieval quality against the ground truth](#d11--retrieval-quality-against-the-ground-truth)
- [D12 — pooled adjudication of the label set](#d12--pooled-adjudication-of-the-label-set)
- [D13 — a real Gemma 4 tool round trip](#d13--a-real-gemma-4-tool-round-trip)

## D10 — real embeddings over the real corpus

Implementation: `logtriage/embed/llama.py`, `logtriage/embed/factory.py`.

**Criteria:** `index-kb --embedder llama` completes over all 15,025 chunks; the index metadata
records `embedder: llama` and the model's true width, probed rather than configured.

Two things to watch, because both produce a well-formed index that means nothing:

- **The served model must be an embedding model.** A chat model started with `--embedding`
  answers these requests with well-formed vectors that carry very little, and nothing in the
  client can tell the difference. Confirm from `.env.logtriage` which checkpoint is loaded.
- **Pooling must match what the model was trained with.** `mean` is the default; the wrong
  choice gives unit vectors that rank badly and error nowhere.

Record the embed wall-clock. The hashing baseline embeds the corpus in 2.7s; a served model is
the difference between an index build being a coffee break and an afternoon, and D16 needs the
figure.

**Command:**

```bash
cd logtriage
export LOGTRIAGE_EMBED_URL=http://127.0.0.1:8007
time uv run python -m logtriage index-kb --embedder llama --refresh --json
```

Observed:

_(pending)_

Status: PENDING

## D11 — retrieval quality against the ground truth

Implementation: `logtriage/commands/eval_retrieval.py`, `logtriage/evals/metrics.py`,
`logtriage/evals/retrieval.jsonl`. Method and caveats: `logtriage/evals/README.md`.

**This is the first retrieval-quality number this project will have.** Read
`logtriage/evals/README.md` before quoting it — the set is 40 questions written by one person,
and what makes the number worth anything is the method rather than the arithmetic.

**Criteria:** the comparison is against the lexical baseline, which was measured in the sandbox
with the hashing embedder over the corpus at commit `88ec8192b924` and is the row to beat:

| leg | slice | n | R@1 | R@5 | R@10 | MRR |
|-----|-------|---|-----|-----|------|-----|
| hybrid | all | 40 | 0.33 | 0.45 | 0.57 | 0.388 |
| hybrid | no-overlap | 18 | 0.06 | 0.17 | 0.33 | 0.105 |
| hybrid | overlap | 22 | 0.55 | 0.68 | 0.77 | 0.619 |

**The `no-overlap` row is the criterion.** Those 18 questions share no distinctive vocabulary
with their answer page, so they are the ones word overlap cannot reach — 0.33 recall@10 against
0.77 where the wording does match. A real embedder should close that gap.

A model that improves the `overlap` row and leaves `no-overlap` at 0.33 has not added meaning to
retrieval; it has added a second way of matching strings. That result is worth recording as
plainly as a win, and the hashing embedder exists in this tree precisely so it can be recognised.

Do not adjust the question set to improve the number. If a label looks wrong, that is D12.

**Command:**

```bash
cd logtriage
export LOGTRIAGE_EMBED_URL=http://127.0.0.1:8007
uv run python -m logtriage eval-retrieval --json
uv run python -m logtriage eval-retrieval --store mongo --json   # through mongot
```

Observed:

_(pending)_

Status: PENDING

## D12 — pooled adjudication of the label set

Implementation: `logtriage/commands/eval_retrieval.py` (`--pool`), `logtriage/evals/README.md`.

**Criteria:** every `candidate` row in the pool is ruled on, and the ones judged relevant are
added to `retrieval.jsonl` with their reason. Then D11 is re-run and **both** numbers are kept.

Why this exists: recall against an incomplete label set understates every retriever, and
unequally — a retriever that finds a genuinely good page nobody labelled scores as wrong. Only
pages someone thought of are listed, and pooling is the standard fix. The pool is the union of
the top-10 from every leg and embedder, so anything any retriever thought was relevant gets
looked at once.

The adjudication is a human's. The rule is whether the page answers the question, not whether
including it improves a score — and a label added after seeing that it helps is a label that has
stopped being evidence.

Record which items changed. The before-and-after pair is what a later reader needs to know how
much of any improvement was retrieval and how much was labelling.

**Command:**

```bash
cd logtriage
uv run python -m logtriage eval-retrieval --pool /tmp/pool.jsonl --json
grep '"status": "candidate"' /tmp/pool.jsonl | head -40
# adjudicate, edit logtriage/evals/retrieval.jsonl, then:
uv run pytest tests/test_evals.py -q
uv run python -m logtriage eval-retrieval --json
```

Observed:

_(pending)_

Status: PENDING

## D13 — a real Gemma 4 tool round trip

Implementation: `logtriage/llm/llama.py`, `logtriage/triage/pipeline.py`,
`logtriage/triage/report.py`.

**Criteria:** three separable observations. Record all three even when the first fails, because
they fail independently and each is a different fact.

1. **Tool calls happen at all.** The report's `tool calls:` line must be non-zero. Zero with
   `JINJA_ENABLED=true` means the chat template is not being applied and the model's call
   arrived as prose in `content` — a run that looks successful and consulted nothing.
2. **The thinking channel does not leak into the narrative.** If the report carries reasoning
   text, `LOGTRIAGE_REASONING_FORMAT` is the knob; record which value works, because nothing in
   this tree currently knows.
3. **The causality direction.** `replica-stepdown` is built to invite the error: its slow
   queries begin six minutes before the first heartbeat failure and continue after the new
   primary is elected, so the dramatic event is the consequence and the available reading is
   backwards.

On the third, failing is as informative as passing. The report prints first-occurrence times
with their gaps above the narrative specifically so the disproof is visible before the story
starts; whether that is enough for this model is exactly what is unknown. A narrative that gets
it wrong with the ordering printed above it is a finding about the guard rail, not just about the
model.

Every citation must still resolve — exit 0 — or the run failed regardless of what the prose said.

**Command:**

```bash
cd logtriage
export LOGTRIAGE_LLAMA_URL=http://127.0.0.1:8006
export LOGTRIAGE_EMBED_URL=http://127.0.0.1:8007
uv run python -m logtriage triage /tmp/rs.log --llm llama --store mongo --kb
echo "exit=$?"
```

Observed:

_(pending)_

Status: PENDING

## Related

- [desktop-scale.md](desktop-scale.md) — D14 to D16, the measurements
- [../design/citations.md](../design/citations.md) — what the causality guard is for
- [../design/prompting.md](../design/prompting.md) — why the chat template is load-bearing
