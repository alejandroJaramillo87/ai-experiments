# Phase D: the Linux desktop

**Operational — executed against the workstation. Keep the commands runnable.**

Everything that needs Docker, a GPU, or a served model. Nine entries, all pending: no container
in this stack has started yet.

Run them in order. D1 through D4 establish the deployment; D5 through D9 establish that the tool
works against it.

## Table of Contents

- [Deployment](#deployment)
- [Search](#search)
- [The model](#the-model)
- [Scale](#scale)

## Deployment

### D1 — the stack starts and every container reports healthy

**Criteria:** four containers running; `logtriage-mongo` and `logtriage-mongot` both healthy.
`read_only: true` is applied to all four and is the most likely thing to fail — if a container
cannot write somewhere it needs to, it exits at startup.

**Command:**

```bash
cp env.logtriage.example .env.logtriage    # fill in the two model paths first
make logtriage-secrets
make logtriage-up && sleep 60 && make logtriage-status
```

**Observed:** _(pending)_

**Status:** pending

### D2 — the bootstrap initiates the replica set and creates the mongot user

**Criteria:** `rs.status().ok` is 1; a user `mongotUser` exists in `admin` with the
`searchCoordinator` role. Running the script twice must be a no-op, not an error.

**Command:**

```bash
make logtriage-bootstrap
make logtriage-bootstrap     # idempotence
```

**Observed:** _(pending)_

**Status:** pending

### D3 — mongot reaches SERVING and syncs from the deployment

**Criteria:** the health endpoint answers `SERVING`. Note that this proves the process is up and
nothing more — it does not check whether any index is `Ready`.

**Command:**

```bash
curl -s http://127.0.0.1:8008/health
make logtriage-logs SERVICE=logtriage-mongot
```

**Observed:** _(pending)_

**Status:** pending

### D4 — the two llama-servers load their models and answer

**Criteria:** both `/v1/health` endpoints return 200. `logtriage-embed` must return a vector from
`/v1/embeddings`; `logtriage-llm` must **not** — if it does, `EMBEDDING` leaked into the wrong
service.

**Command:**

```bash
curl -sf http://127.0.0.1:8006/v1/health && echo llm-ok
curl -sf http://127.0.0.1:8007/v1/health && echo embed-ok
curl -s http://127.0.0.1:8007/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"input":["planSummary COLLSCAN"]}' | head -c 200
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
```

**Observed:** _(pending)_

**Status:** pending

## Search

### D5 — the differential suite against a real MongoDB

**Criteria:** every case that skips under `mongomock` — the `$firstN` facets, the `$dateTrunc`
timeline — passes here, and the `MongoLogStore` results still match `InMemoryLogStore` exactly.
This is also the first execution of a bulk `ReplaceOne`.

**Command:**

```bash
cd logtriage
LOGTRIAGE_MONGO_URI='mongodb://logtriage:<password>@127.0.0.1:27017/?authSource=admin&replicaSet=rs0&directConnection=true' \
  uv run pytest -m 'differential or services' -v
```

**Observed:** _(pending)_

**Status:** pending

### D6 — $search and $vectorSearch return results

**Criteria:** both search indexes reach `Ready`; a `$search` for a log message returns at least
one chunk from `reference/log-messages.txt`; a `$vectorSearch` returns `k` hits with descending
scores. **Wait for `Ready` before querying** — a query against a building index returns nothing
rather than erroring, which looks exactly like a bad query.

This is the first execution of either operator anywhere in this project.

**Command:**

```bash
make logtriage-shell-mongo
# db.chunks.getSearchIndexes()          -> both status: READY
# db.chunks.aggregate([{$search: {index: "chunk_text",
#     text: {query: "slow query", path: ["text","heading_path","document"]}}}, {$limit: 3}])
```

**Observed:** _(pending)_

**Status:** pending

## The model

### D7 — real embeddings, and the first retrieval quality measurement

**Criteria:** `index-kb --embedder llama` completes over all 15,025 chunks; the index metadata
records `embedder: llama` and the model's true width. Then the corpus suite's
`SEMANTIC_QUESTIONS` — the ones deliberately not asserted under the hashing embedder — are run
by hand and the answers recorded.

**This is the first retrieval-quality number this project will have.** Record it against the
lexical baseline: if a real embedding model cannot beat word overlap on these questions, that is
worth knowing and is the reason the hashing embedder exists.

**Command:**

```bash
cd logtriage
export LOGTRIAGE_EMBED_URL=http://127.0.0.1:8007
uv run python -m logtriage index-kb --embedder llama --refresh --json
uv run pytest -m corpus -v
```

**Observed:** _(pending)_

**Status:** pending

### D8 — a real Gemma 4 tool round trip

**Criteria:** the model calls at least one tool, the loop dispatches it, and the report's
citations all resolve (exit 0). Two specific things to look for:

- **Tool calls actually happen.** `tool calls: 0` in the report with `JINJA_ENABLED=true` means
  the template is not being applied and the call arrived as prose.
- **The thinking channel does not leak into the narrative.** If it does,
  `LOGTRIAGE_REASONING_FORMAT` is the knob; record which value works.

Then the causality check: run `replica-stepdown`, whose slow queries begin six minutes before the
first heartbeat failure, and see whether the narrative gets the direction right. Passing is
interesting; failing is more interesting, and either way it is a fact about the model rather than
about the guard rails.

**Command:**

```bash
cd logtriage
export LOGTRIAGE_LLAMA_URL=http://127.0.0.1:8006
uv run python -m logtriage gen-logs --scenario replica-stepdown --seed 1 --out /tmp/rs.log
uv run python -m logtriage triage /tmp/rs.log --llm llama --kb
echo "exit=$?"
```

**Observed:** _(pending)_

**Status:** pending

## Scale

### D9 — ingest throughput and end-to-end latency on a real log

**Criteria:** numbers, not a pass/fail. Record ingest rate for a log of at least 100MB, the
wall-clock of a `triage` run against the served model, and peak VRAM with both servers loaded.
The fixtures are thousands of lines; a real log is millions, and nothing so far has run at that
size.

**Command:**

```bash
cd logtriage
time uv run python -m logtriage profile /path/to/real-mongod.log
time uv run python -m logtriage triage /path/to/real-mongod.log --llm llama --kb
```

**Observed:** _(pending)_

**Status:** pending

## Related

- [../operations/stack.md](../operations/stack.md) — what these commands are driving
- [../mongodb/search.md](../mongodb/search.md) — why D6 is the interesting one
- [macbook.md](macbook.md) — the phase this one blocks
