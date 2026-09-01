# Phase D: deployment

**Operational runbook. Executed against the Linux workstation, not read for understanding: the
operator's pasted output is the evidence. Keep the commands runnable — the documentation gate
checks every path, make target and CLI verb named here.**

**Entries: 0 verified, 0 blocked, 5 pending, of 5.**

The stack up, and the first execution of anything in it. No container in this stack has ever
started, so D1 is the first time the compose file, both MongoDB configurations, and the
bootstrap script meet a Docker daemon.

Run in order. D1 through D4 establish the deployment; D5 is the first real-server check of code
that until now has only met `mongomock`.

## Table of Contents

- [D1 — the stack starts and every container reports healthy](#d1--the-stack-starts-and-every-container-reports-healthy)
- [D2 — the bootstrap initiates the replica set and creates the mongot user](#d2--the-bootstrap-initiates-the-replica-set-and-creates-the-mongot-user)
- [D3 — mongot reaches SERVING](#d3--mongot-reaches-serving)
- [D4 — both llama-servers load and answer](#d4--both-llama-servers-load-and-answer)
- [D5 — the differential suite against a real MongoDB](#d5--the-differential-suite-against-a-real-mongodb)

## D1 — the stack starts and every container reports healthy

Implementation: `docker-compose.logtriage.yaml`, `docker/logtriage/mongod.conf`,
`docker/logtriage/mongot.conf`, `scripts/logtriage/generate-secrets.sh`.

**Criteria:** four containers running; `logtriage-mongo` and `logtriage-mongot` both report
healthy. The two inference services may still be loading — D4 covers those.

The likely failure is `read_only: true`, which is applied to all four services and has never
been exercised. A container that cannot write somewhere it needs to exits at startup rather
than reporting unhealthy, so an exited container is the signal to read the logs for a path.
The second likely failure is keyfile ownership: `mongod` refuses a keyfile it does not own, and
exits without an obvious reason.

**Command:**

```bash
cp env.logtriage.example .env.logtriage   # then fill in the two model paths
make logtriage-secrets
make logtriage-up
sleep 60
make logtriage-status
docker ps -a --filter name=logtriage --format '{{.Names}}\t{{.Status}}'
```

Observed:

_(pending)_

Status: PENDING

## D2 — the bootstrap initiates the replica set and creates the mongot user

Implementation: `scripts/logtriage/bootstrap.sh`.

**Criteria:** `rs.status().ok` is 1; a user `mongotUser` exists in `admin` carrying the
`searchCoordinator` role. Running the script a second time must be a no-op and exit 0 — the
natural thing to do after a failed start is to run it again, and a bootstrap that is not
idempotent turns that into a second failure.

MongoDB Search follows the collection through a change stream, and change streams do not exist
on a standalone server. That is why there is a replica set at all, and why this step cannot be
expressed in the compose file.

**Command:**

```bash
make logtriage-bootstrap
make logtriage-bootstrap     # idempotence: must be a no-op, exit 0
```

Observed:

_(pending)_

Status: PENDING

## D3 — mongot reaches SERVING

Implementation: `docker/logtriage/mongot.conf`, `docker-compose.logtriage.yaml`.

**Criteria:** the health endpoint answers `SERVING`.

Read that answer narrowly. It reports that the process is up; it does not check whether any
index has reached `Ready`, and a query against an index still building returns nothing rather
than erroring. A healthy `mongot` container is not a working search index, and D7 is where that
distinction is actually tested.

Until D2 completes, `NOT_SERVING` is the correct answer — without its user, `mongot` cannot open
the change stream it syncs through.

**Command:**

```bash
curl -s http://127.0.0.1:8008/health
make logtriage-logs SERVICE=logtriage-mongot
```

Observed:

_(pending)_

Status: PENDING

## D4 — both llama-servers load and answer

Implementation: `docker-compose.logtriage.yaml`, `docker/llama-gpu/entrypoint.sh`.

**Criteria:** both `/v1/health` endpoints return 200. `logtriage-embed` returns a vector from
`/v1/embeddings`; `logtriage-llm` must **not** — if it does, `EMBEDDING=true` has reached the
wrong service, and the triage model is serving embeddings instead of chat.

Also record peak VRAM with both models resident. Two servers share one card and the budget is
real; D16 compares against this figure.

**Command:**

```bash
curl -sf http://127.0.0.1:8006/v1/health && echo llm-ok
curl -sf http://127.0.0.1:8007/v1/health && echo embed-ok
curl -s http://127.0.0.1:8007/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"input":["planSummary COLLSCAN"]}' | head -c 200
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
```

Observed:

_(pending)_

Status: PENDING

## D5 — the differential suite against a real MongoDB

Implementation: `logtriage/store/mongo.py`, `logtriage/store/pipelines.py`,
`tests/test_store_differential.py`, `tests/test_store_mongo.py`.

**Criteria:** every case that skips under `mongomock` passes here — the `$firstN` facets and the
`$dateTrunc` timeline — and `MongoLogStore` still returns exactly what `InMemoryLogStore`
returns.

This is also the first execution of a bulk `ReplaceOne` anywhere in this project. `mongomock`
4.3.0 cannot execute any pymongo 4.17 bulk update, so `insert` has been verified only as the
operations it builds, through a recording double. Whether the server executes them is what this
entry settles.

A skip here is a failure, not a pass. The point of running on a real server is that nothing
should skip.

**Command:**

```bash
cd logtriage
export LOGTRIAGE_MONGO_URI='mongodb://logtriage:<password>@127.0.0.1:27017/?authSource=admin&replicaSet=rs0&directConnection=true'
uv run pytest -m 'differential or services' -v
uv run pytest tests/test_store_differential.py tests/test_store_mongo.py -v -rs
```

Observed:

_(pending)_

Status: PENDING

## Related

- [desktop-search.md](desktop-search.md) — D6 to D9, which this phase unblocks
- [../operations/stack.md](../operations/stack.md) — what these commands are driving
- [README.md](README.md) — the protocol and the entry format
