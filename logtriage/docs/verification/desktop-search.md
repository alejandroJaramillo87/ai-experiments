# Phase D: ingest and search

**Operational runbook. Executed against the Linux workstation, not read for understanding: the
operator's pasted output is the evidence. Keep the commands runnable — the documentation gate
checks every path, make target and CLI verb named here.**

**Entries: 0 verified, 0 blocked, 4 pending, of 4.**

The four entries that put data into the deployment and take it out again. D8 is the one that
matters most: `$search` and `$vectorSearch` are `mongot` operators, not MongoDB ones, `mongomock`
cannot stand in for them, and nothing in this project has ever executed either.

Blocked on [desktop-deployment.md](desktop-deployment.md).

## Table of Contents

- [D6 — ingest a log into MongoDB](#d6--ingest-a-log-into-mongodb)
- [D7 — index the manual into mongot and wait for Ready](#d7--index-the-manual-into-mongot-and-wait-for-ready)
- [D8 — $search and $vectorSearch return ranked results](#d8--search-and-vectorsearch-return-ranked-results)
- [D9 — triage end to end against the deployment](#d9--triage-end-to-end-against-the-deployment)

## D6 — ingest a log into MongoDB

Implementation: `logtriage/commands/ingest.py`, `logtriage/store/mongo.py`.

**Criteria:** the record count written matches the count parsed; a second run of the same
command writes the same number and leaves the collection the same size. `record_id` is a content
handle, so re-ingesting must replace rather than duplicate — and ingest is the step most likely
to be run twice, because the first run is how you discover the file was truncated.

The five indexes are created and the second `--ensure-indexes` is a no-op.

**Command:**

```bash
cd logtriage
uv run python -m logtriage gen-logs --scenario replica-stepdown --seed 1 --out /tmp/rs.log
uv run python -m logtriage ingest /tmp/rs.log --store mongo --ensure-indexes --json
uv run python -m logtriage ingest /tmp/rs.log --store mongo --json   # idempotence
```

Observed:

_(pending)_

Status: PENDING

## D7 — index the manual into mongot and wait for Ready

Implementation: `logtriage/commands/index_kb.py`, `logtriage/store/knowledge_mongo.py`.

**Criteria:** all 15,025 chunks reach the collection; both search indexes report `READY`; the
command blocks until they do rather than returning early.

That wait is the entry's point. Search index creation is asynchronous and a query against an
index still building returns an empty result rather than an error, so a command that returned
here would hand the operator a deployment that looks broken for the next few minutes and a
retrieval bug that is not one.

Record how long the wait took. It scales with the corpus rather than being constant, and D16 has
no other source for that figure.

The index also writes its own description to a sibling collection — which embedder built it, at
what width. D9 and D11 both read that back rather than being told on the command line.

**Command:**

```bash
cd logtriage
time uv run python -m logtriage index-kb --to mongo --refresh --json
```

Observed:

_(pending)_

Status: PENDING

## D8 — $search and $vectorSearch return ranked results

Implementation: `logtriage/store/knowledge_mongo.py`.

**Criteria:** three separate observations, and they fail independently.

1. Both indexes report `READY` in `getSearchIndexes()`. Query before this and an empty result
   means nothing.
2. A `$search` for a log message returns at least one chunk whose `document` is
   `reference/log-messages.txt`.
3. A `$vectorSearch` returns `k` hits with descending `vectorSearchScore`.

This is the first execution of either operator anywhere in this project. Both index definitions
were written against the manual's reference and have been checked only as data — that the shapes
are right is what this settles.

**Command:**

```bash
make logtriage-shell-mongo
```

Then, in the shell:

```javascript
use logtriage
db.chunks.getSearchIndexes()
db.chunks.aggregate([
  {$search: {index: "chunk_text",
             text: {query: "slow query", path: ["text", "heading_path", "document"]}}},
  {$limit: 3},
  {$project: {document: 1, score: {$meta: "searchScore"}}}
])
```

Observed:

_(pending)_

Status: PENDING

## D9 — triage end to end against the deployment

Implementation: `logtriage/commands/triage.py`, `logtriage/store/factory.py`.

**Criteria:** exit 0 with every citation resolving, against records and documentation both held
by MongoDB rather than in memory. The report's citation audit is the check — a record id that
does not resolve exits 1, and that is the failure this entry is looking for.

`--no-ingest` on the second run must query what D6 already wrote rather than writing it again.

Note what is still in memory: the profile is built from the parsed records, so the corpus is held
once even with `--store mongo`. That is a stated non-goal of this phase rather than an oversight,
and D14 measures what it costs.

**Command:**

```bash
cd logtriage
uv run python -m logtriage triage /tmp/rs.log --store mongo --kb
echo "exit=$?"
uv run python -m logtriage triage /tmp/rs.log --store mongo --no-ingest --kb
echo "exit=$?"
```

Observed:

_(pending)_

Status: PENDING

## Related

- [desktop-model.md](desktop-model.md) — D10 to D13, which this phase unblocks
- [../mongodb/search.md](../mongodb/search.md) — why D8 is the interesting one
- [README.md](README.md) — the protocol and the entry format
