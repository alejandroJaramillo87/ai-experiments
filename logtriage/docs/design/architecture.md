# Architecture: four ports, two implementations each

How the tool is put together, and why the shape is a set of narrow interfaces rather than a
layer over a database.

## Table of Contents

- [The four ports](#the-four-ports)
- [Why two implementations of everything](#why-two-implementations-of-everything)
- [The interfaces are small and closed](#the-interfaces-are-small-and-closed)
- [Where the semantics live](#where-the-semantics-live)
- [Selecting an implementation](#selecting-an-implementation)

## The four ports

| Port | Interface | Served | Offline |
|------|-----------|--------|---------|
| `LogStore` | `logtriage/store/base.py` | `MongoLogStore` — MongoDB aggregations | `InMemoryLogStore` |
| `KnowledgeStore` | `logtriage/store/knowledge.py` | `MongoKnowledgeStore` — `$vectorSearch` and `$search` via `mongot` | `InMemoryKnowledgeStore` |
| `Embedder` | `logtriage/embed/base.py` | `LlamaEmbedder` — llama-server embeddings | `HashingEmbedder` |
| `LlmClient` | `logtriage/llm/base.py` | `LlamaClient` — llama-server chat completions | `ScriptedLlmClient` |

The served implementations live in `store/mongo.py`, `store/knowledge_mongo.py`,
`embed/llama.py` and `llm/llama.py`; the offline ones in `store/memory.py`,
`store/knowledge.py`, `embed/hashing.py` and `llm/scripted.py`.

`LogStore` and `KnowledgeStore` are held apart deliberately. Logs are queried structurally —
counted, grouped, bounded by time — and documentation is queried by meaning. They share a
deployment and nothing else, and the separation prevents the common mistake: reaching for the
vector index to answer "how many collection scans were there", which returns the ten chunks most
similar to the question rather than the answer.

## Why two implementations of everything

The offline set is what makes the whole pipeline — parse, profile, detect, retrieve, cite,
report — runnable in a checkout with nothing installed, on a machine with no GPU and no
database. That is not a testing convenience. It is what let every rule in the chunker be
written and checked before a server existed, and it is what will let the tool be verified on the
MacBook before `mongod` is installed there.

The cost is the risk that a fake and the real thing quietly disagree, which is worse than having
no fake at all: it makes tests pass that production fails. Three things hold them together.

**The pipelines are shared, not reimplemented.** `store/pipelines.py` builds every MongoDB
aggregation as data and touches no database. `MongoLogStore` executes them;
`tests/test_store_differential.py` executes the same stages through `mongomock` and compares the
answers to `InMemoryLogStore` across 13 filters. Where the two could interpret a filter
differently — an empty tuple, a null field, a missing field — the pipeline is the authority,
because it is the one that has to be right against a real server.

**The fusion is shared.** `store/knowledge.py:fuse` is called by both knowledge stores. Ranking
is the one place they could diverge with nothing able to notice, because a differently-ordered
list of plausible chunks is indistinguishable from an ambiguous query. One implementation makes
them agree by construction, leaving only the legs themselves to differ — which is the difference
that is supposed to exist.

**The fakes announce their limits.** `HashingEmbedder.semantic` returns `False`, and that
property is read at runtime rather than being a comment: `tools/schemas.py` weights the vector
leg to zero for a non-semantic embedder, because fusing a meaningless ranking with a lexical one
measurably pushed the correct page out of the top three. `ScriptedLlmClient` raises when its
script runs out rather than answering with an empty string, because a fake that returns nothing
makes a broken loop look like a working one.

## The interfaces are small and closed

`LogStore` is five operations, and each maps to one aggregation. It is deliberately not a query
language. A general abstraction would have been harder to mirror faithfully, and a mirror that
is not faithful is worse than no mirror.

Two of its types are closed sets rather than free strings:

- `SampleOrder` is `time`, `-time`, `-duration`. Accepting arbitrary sorts would let the two
  implementations diverge on a sort nothing tested.
- `FacetField` is eight named fields. Grouping by an arbitrary field on a large collection is a
  performance trap that should not be reachable by passing a string.

`fetch_by_ids` looks like a convenience and is not. Every claim the tool makes carries the ids
of the records behind it, and a citation that cannot be resolved is a citation nobody can check.
See [citations.md](citations.md).

## Where the semantics live

The rule is that anything a real server could disagree with is expressed as data, so it can be
read and checked without one:

| Decision | Expressed as | Read it in |
|----------|--------------|------------|
| What a log filter means | `$match` document | `store/pipelines.py` |
| How facets group, sort, and break ties | aggregation stages | `store/pipelines.py` |
| Which indexes the queries assume | `INDEXES` | `store/mongo.py` |
| What `mongot` indexes, and how | index definitions | `store/knowledge_mongo.py` |
| How two ranked lists combine | `fuse` | `store/knowledge.py` |

## Selecting an implementation

Nothing branches on hostname, platform, or which machine the code thinks it is on. Four
environment variables are the whole of the deployment configuration, read in
`logtriage/config.py`; unset means "use the offline implementation", and only a command that was
explicitly asked for a served backend turns that into a failure.

```bash
$ python -m logtriage triage mongod.log                  # scripted model, no services
$ python -m logtriage triage mongod.log --llm llama      # needs LOGTRIAGE_LLAMA_URL
$ python -m logtriage index-kb --embedder llama          # needs LOGTRIAGE_EMBED_URL
```

The embedder is the one choice that is *not* re-selectable at query time. An index is only
searchable by the embedder that built it, so `index-kb` writes that embedder's identity into the
index metadata and `triage --kb` rebuilds the one named there. Searched with a different
embedder, an index still returns its k best chunks, ranked by a similarity that means nothing,
and nothing anywhere would error. `embed/factory.py` owns that pairing.

## Related

- [citations.md](citations.md) — the evidence contract every claim rests on
- [prompting.md](prompting.md) — the tool surface the model is given
- [../mongodb/search.md](../mongodb/search.md) — what `mongot` adds and what it costs
- [../operations/stack.md](../operations/stack.md) — the services these adapters talk to
