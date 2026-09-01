# Verification campaign

**Operational — executed against the repository or a host, not read for understanding. Keep the
commands runnable.**

What has been established, what has not, and the protocol for closing the gap. The standing list
below is the honest inventory: it is maintained as capabilities are added, and adding one without
its caveat is a defect.

## Table of Contents

- [Protocol](#protocol)
- [Phase order](#phase-order)
- [What is not verified](#what-is-not-verified)
- [Entry format](#entry-format)

## Protocol

1. **State the acceptance criteria before running anything.** What output would count as a pass,
   and what would count as a failure. Deciding afterwards is how an ambiguous result becomes a
   pass.
2. **Run the command on the machine that can answer it.** See
   [../operations/three-machines.md](../operations/three-machines.md).
3. **Paste the command and its output verbatim** into the entry's Observed block. Not a summary
   — a summary is where a failure becomes a caveat.
4. **Mark the entry.** `VERIFIED` needs evidence of one of two kinds: pasted output, or the name
   of a test function that asserts the signal. An entry whose evidence is "the gate was green"
   without a named check is not verified.

An agent may not mark an entry VERIFIED on the strength of a command it could not run. Print the
command, stop, and wait for the human's output.

## Phase order

| Phase | Machine | Runbook |
|-------|---------|---------|
| A, B, C | Cloud sandbox | Covered by `make quality` and the corpus tier; no runbook |
| D | Linux desktop | [desktop.md](desktop.md) |
| E | MacBook M4 | [macbook.md](macbook.md) |

Phase D blocks Phase E: there is no point verifying that an unproven pipeline was extracted
correctly.

## What is not verified

Stated because the alternative is a reader assuming coverage that does not exist.

**`$vectorSearch` and `$search` have never executed.** They are `mongot` operators rather than
MongoDB ones, `mongomock` implements neither and could not, and that path stays unexercised until
it meets a real deployment. What *is* covered offline is everything around it: the document round
trip, `upsert`, `fetch_by_ids`, and the shape of both index definitions.

**`$firstN` and `$dateTrunc` are unexercised.** `mongomock` lacks both, so the facet and timeline
differential cases skip. They skip **by name**, saying which operator went untested, because a
silent skip is indistinguishable from a pass.

**No bulk write has been executed.** `mongomock` 4.3.0 — the current release — cannot execute any
pymongo 4.17 bulk update; both `ReplaceOne` and `UpdateOne` now carry a `sort` argument its
builder does not accept. Rewriting ingest as a loop of single calls would satisfy the fake and
cost a round trip per record on a real ingest, so the operations `insert` builds are asserted
through a recording double instead, and their execution is left to pymongo and the server.

**No real embeddings have been produced, so no retrieval quality has been measured.** The corpus
is real and the chunking is verified against it, but the vectors are hashes. Every ranking
assertion in the corpus suite runs on the text leg, which is honest lexical matching; the vector
leg is asserted only to be well-formed. Questions that need meaning rather than words are listed
in `tests/test_kb_corpus.py` as `SEMANTIC_QUESTIONS` and deliberately not asserted.

**Nothing establishes how a model behaves.** Every test uses the scripted client, so a narrative
passing the causality check is evidence about the guard rails and not about the model.

**Nine of the twelve fixture log ids are unverified, and cannot be verified here.** MongoDB
publishes no catalogue. See [../mongodb/log-ids.md](../mongodb/log-ids.md).

**No container in this stack has ever started.** The compose file, both MongoDB configurations,
and the bootstrap script are written against the manual's own containerised deployment procedure
and parse cleanly, and that is all. `shellcheck` is not installed in the sandbox, so the two
shell scripts are checked only by `bash -n`.

## Entry format

```markdown
### D3 — mongot answers a $search query

**Criteria:** a `$search` over the indexed manual returns at least one chunk whose document is
`reference/log-messages.txt`.

**Command:**
    make logtriage-shell-mongo

**Observed:** _(paste output here)_

**Status:** pending
```

Status is one of `pending`, `verified`, `blocked`. A blocked entry names what blocks it.
