# MongoDB Search: mongot, $search, and $vectorSearch

What the search half of the deployment is, why it is a second process, and what this project has
and has not established about it.

## Table of Contents

- [mongot is not MongoDB](#mongot-is-not-mongodb)
- [What that requires of the deployment](#what-that-requires-of-the-deployment)
- [The two index definitions](#the-two-index-definitions)
- [Querying, and how the legs are fused](#querying-and-how-the-legs-are-fused)
- [This path has never executed](#this-path-has-never-executed)

## mongot is not MongoDB

`$search` and `$vectorSearch` are **not** MongoDB aggregation operators. They are resolved by
`mongot`, a separate process that MongoDB Community does not ship. `mongod` is told where to find
it and forwards the stage; `mongot` maintains its own Lucene indexes by following the collection
through a change stream.

Two things follow, and both are load-bearing for this project:

- **It is a second container.** MongoDB Search for a self-managed deployment ships as
  `mongodb/mongodb-community-search`, versioned independently of the server. Community 8.2.0 is
  the floor for using it at all.
- **No in-process fake can stand in for it.** `mongomock` implements neither operator and could
  not: they are not MongoDB semantics. That is why `MongoKnowledgeStore` is the one adapter in
  this tree with no offline proof.

## What that requires of the deployment

`mongot` follows the collection through a change stream, and change streams need a replica set.
A standalone `mongod` cannot be searched, however small the data.

So the stack runs a single-member replica set, keyfile authentication between the two processes,
and a dedicated user for `mongot`:

| Requirement | Where |
|-------------|-------|
| `replSetName`, and the set initiated | `docker/logtriage/mongod.conf`, `scripts/logtriage/bootstrap.sh` |
| `mongotHost` and `searchIndexManagementHostAndPort`, equal | `docker/logtriage/mongod.conf` |
| `syncSource` pointing back at the server | `docker/logtriage/mongot.conf` |
| A user with the `searchCoordinator` role | `scripts/logtriage/bootstrap.sh` |
| The shared keyfile and password file | `make logtriage-secrets` |

`searchCoordinator` carries `readAnyDatabase` and write access to the internal
`__mdb_internal_search` database. Nothing else should hold it, and nothing should write to that
database by hand.

[../operations/stack.md](../operations/stack.md) owns the running of it.

## The two index definitions

Held as data in `logtriage/store/knowledge_mongo.py`, for the same reason the aggregations in
`store/pipelines.py` are: a definition that can be printed and read is one a human can check
against the manual without a server.

**`chunk_vectors`** (`type: vectorSearch`) indexes the `embedding` field with `similarity:
cosine` at the embedder's own width. Both embedders in this tree return unit vectors, for which
cosine and `dotProduct` rank identically; cosine is named because `embed/base.py:cosine` is what
a reader will compare it against. The width is not configured — it is probed from the embedder,
because a dimension count in a config file is a number that can be wrong and whose failure
surfaces much later as an error about the index.

**`chunk_text`** (`type: search`) uses a **static** mapping over three fields: `text`,
`heading_path`, `document`. A dynamic mapping would index every field including the embedding
array and the numeric token estimate — index size spent on fields no query searches.

## Querying, and how the legs are fused

`vector_search` issues `$vectorSearch` with `numCandidates` at twenty times the limit. The search
is approximate, and that multiplier is the recall/latency knob; too small a value silently
returns worse results rather than erroring.

`hybrid_search` runs both legs and fuses them client-side by weighted reciprocal rank, calling
the same `fuse` function the in-memory store calls. That costs a round trip against a server-side
`$rankFusion` and buys the guarantee that the two implementations rank identically — worth more
here, because retrieval quality is the thing this project cannot yet measure, so a difference
between the fake and the server would be attributed to the corpus rather than to the store.

`fetch_by_ids` is deliberately **not** a search. Citations resolve by identity through a plain
`$match`, so a report's verifiability never depends on a search index being `Ready`.

## This path has never executed

Stated plainly because the code reads as though it works.

- **No `$search` or `$vectorSearch` query has ever run.** `mongomock` cannot execute them. What
  is verified offline is the surrounding behaviour: the document round trip, `upsert`,
  `fetch_by_ids`, and the shape of both index definitions — see `tests/test_store_mongo.py`.
- **Search index creation is asynchronous.** The call returns before the index is queryable, and
  querying one still building returns nothing rather than an error. A caller that indexes and
  immediately retrieves must wait for `Ready`.
- **A healthy `mongot` container is not a working search index.** Its health endpoint answers
  `SERVING` as soon as the process is up; it does not check whether any index has reached
  `Ready`.

[../verification/desktop-search.md](../verification/desktop-search.md) carries the checks that would close
these.

## Related

- [../design/architecture.md](../design/architecture.md) — where this store sits
- [../operations/stack.md](../operations/stack.md) — running mongod and mongot
- [../corpus/README.md](../corpus/README.md) — what gets indexed
