# Phase D: scale

**Operational runbook. Executed against the Linux workstation, not read for understanding: the
operator's pasted output is the evidence. Keep the commands runnable — the documentation gate
checks every path, make target and CLI verb named here.**

**Entries: 0 verified, 0 blocked, 3 pending, of 3.**

Numbers rather than pass/fail. Everything so far has run against fixtures of a few thousand
lines; a real `mongod` log is millions, and nothing in this project has met one.

Every entry here follows `.claude/skills/benchmarking/SKILL.md`: discard the warm-up, take at
least three samples, report **median and spread**, and record the fixed conditions beside the
number. `logtriage bench` does all four, which is why these entries call it rather than `time`.

There is no regression criterion yet. The 5% rule needs a recorded baseline for this workload on
this machine, and these entries are what create one.

Blocked on [desktop-model.md](desktop-model.md).

## Table of Contents

- [D14 — ingest throughput on a real log](#d14--ingest-throughput-on-a-real-log)
- [D15 — retrieval latency](#d15--retrieval-latency)
- [D16 — end-to-end triage latency](#d16--end-to-end-triage-latency)

## D14 — ingest throughput on a real log

Implementation: `logtriage/commands/ingest.py`, `logtriage/commands/bench.py`,
`logtriage/store/mongo.py`.

**Criteria:** a rate, for a log of at least 100MB, into both backends. Record median and spread
for each, and the memory the process reached.

Three things this is actually measuring, and they are worth separating in the notes:

- **The write path.** Ingest streams — `read_log` yields, normalization is a generator, and the
  store batches at 1,000 — so the corpus is never materialised as a list. The `--store memory`
  row is the parse-and-normalize cost with no database; the `mongo` row adds the round trips.
- **The bulk write, at scale.** `mongomock` cannot execute a pymongo 4.17 bulk update at all, so
  every `ReplaceOne` in this project has been checked as an operation and never as a write until
  D5. This is the first one at a size where batching matters.
- **The profile's memory bound.** `triage` builds its profile from the parsed records, so the
  corpus is held once even with `--store mongo`. That is a stated non-goal of this phase; this
  is where the cost of it becomes a number. If a 100MB log cannot be profiled on this machine,
  that is the finding, and a store-side profile over the existing aggregations becomes Phase F
  work rather than a hypothetical.

**Command:**

```bash
cd logtriage
export LOGTRIAGE_MONGO_URI='mongodb://logtriage:<password>@127.0.0.1:27017/?authSource=admin&replicaSet=rs0&directConnection=true'
uv run python -m logtriage bench ingest --log /path/to/real-mongod.log --store memory --json
uv run python -m logtriage bench ingest --log /path/to/real-mongod.log --store mongo --json
/usr/bin/time -v uv run python -m logtriage ingest /path/to/real-mongod.log --store mongo 2>&1 | tail -20
```

Observed:

_(pending)_

Status: PENDING

## D15 — retrieval latency

Implementation: `logtriage/commands/bench.py`, `logtriage/store/knowledge.py`,
`logtriage/store/knowledge_mongo.py`.

**Criteria:** median and spread for the 40-question set, on both stores, with the served
embedder.

The comparison is the interesting part. The in-memory store scores every chunk with an exact
dot product — deliberately, since an approximate index would be a second implementation of the
thing being tested — and over 15,025 chunks that measured **3 questions/second** in the sandbox
with the hashing embedder. `mongot`'s `$vectorSearch` is approximate and should be far faster,
and the served embedder adds a round trip per question that the hashing one did not.

So three costs are mixed together and the notes should separate them: the embedding round trip,
the search itself, and the fusion. `--k` changes the third; `numCandidates` is twenty times `k`
and is the recall/latency knob on the `mongot` side.

**Command:**

```bash
cd logtriage
export LOGTRIAGE_EMBED_URL=http://127.0.0.1:8007
uv run python -m logtriage bench retrieval --json
uv run python -m logtriage eval-retrieval --store mongo --json    # for the timing in context
```

Observed:

_(pending)_

Status: PENDING

## D16 — end-to-end triage latency

Implementation: `logtriage/triage/pipeline.py`, `logtriage/commands/triage.py`.

**Criteria:** wall-clock for a complete `triage --llm llama --store mongo --kb` run, plus the
same run with `--llm none` so the model's share is separable from everything else.

`bench triage` deliberately excludes the model: a served 31B model's latency swamps the profile
and the detectors, and averaging them together would hide both. Use it for the non-model half
and time the full run directly for the total.

Record alongside: peak VRAM with both servers resident (D4 has the idle figure), the number of
tool calls the run made, and the context size in `.env.logtriage`. A latency without its turn
count is not comparable to the next one — the loop takes up to eight turns and most runs take
one or two.

**Command:**

```bash
cd logtriage
export LOGTRIAGE_LLAMA_URL=http://127.0.0.1:8006
export LOGTRIAGE_EMBED_URL=http://127.0.0.1:8007
uv run python -m logtriage bench triage --log /tmp/rs.log --store mongo --json
time uv run python -m logtriage triage /tmp/rs.log --llm llama --store mongo --kb
time uv run python -m logtriage triage /tmp/rs.log --llm none --store mongo
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
```

Observed:

_(pending)_

Status: PENDING

## Related

- [README.md](README.md) — the protocol and the entry format
- [macbook.md](macbook.md) — Phase E, which compares against these figures
- [../operations/three-machines.md](../operations/three-machines.md) — what each machine proves
