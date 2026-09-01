"""The MongoDB knowledge store: $vectorSearch and $search through mongot.

This is the one adapter with no offline proof at all. `$search` and
`$vectorSearch` are not MongoDB operators -- they are mongot operators, resolved
by a separate process that MongoDB Community does not ship -- so mongomock does
not implement them, cannot implement them, and no amount of local testing will
tell you whether the stages below are right. They are written against the
manual's own reference and stay unverified until a real deployment runs them.

Two things are therefore done deliberately here rather than left to a later fix.

The index definitions are data, in SEARCH_INDEXES and VECTOR_INDEX, for the same
reason the aggregations in pipelines.py are: a definition that can be printed
and read is one a human can check against the manual without a server.

The fusion is not reimplemented. `hybrid_search` collects both legs and calls
`knowledge.fuse`, the same function the in-memory store calls, so the two cannot
rank differently. Reimplementing RRF here -- or reaching for $rankFusion, which
would put a second ranking implementation on the server -- would create exactly
the divergence that nothing in this tree could detect.
"""

from __future__ import annotations

import time
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Any

from logtriage.kb.chunk import Chunk
from logtriage.store.knowledge import VECTOR_WEIGHT, Hit, fuse

if TYPE_CHECKING:  # pragma: no cover - import shape only
    from pymongo.collection import Collection

# Index names. They are part of the query -- a $vectorSearch names the index it
# runs against -- so a rename here is a rename in three places, and the compiler
# will not tell you.
VECTOR_INDEX_NAME = "chunk_vectors"
TEXT_INDEX_NAME = "chunk_text"

# Field holding the embedding. Kept out of the projected chunk fields below so a
# retrieved chunk does not carry 768 floats back to the client on every hit.
VECTOR_FIELD = "embedding"

# How many candidates $vectorSearch considers before returning `limit`. The
# search is approximate, and this is the knob that trades recall for latency;
# MongoDB's guidance is a multiple of the limit, and too small a value silently
# returns fewer or worse results rather than erroring.
CANDIDATE_MULTIPLIER = 20

# How long to wait for a search index to become queryable, and how often to ask.
# Generous: mongot builds its Lucene index by reading the collection through a
# change stream, so the wait scales with the corpus rather than being constant,
# and 15,025 chunks is not instant. A timeout that fires early would send the
# reader looking for a fault that is not there.
READY_TIMEOUT = 600.0
READY_POLL = 2.0

# Where the index's own description lives: which embedder built it, at what
# width, over which corpus commit. A sibling collection, so it is never a
# document a query can return.
META_COLLECTION = "index_meta"
META_ID = "logtriage-index"

# Chunk fields, derived from the dataclass so a field added to Chunk cannot be
# silently dropped on the way back from the server.
CHUNK_FIELDS = tuple(name for name in Chunk.__dataclass_fields__ if name != "chunk_id")


def vector_index_definition(dimensions: int) -> dict[str, Any]:
    """The mongot search index backing $vectorSearch.

    `similarity` must match what the embedder produces. Both embedders in this
    tree return unit vectors, for which cosine and dotProduct rank identically;
    cosine is named because it is what `embed.base.cosine` computes, and a
    reader comparing the two should find the same word.
    """
    return {
        "fields": [
            {
                "type": "vector",
                "path": VECTOR_FIELD,
                "numDimensions": dimensions,
                "similarity": "cosine",
            }
        ]
    }


def text_index_definition() -> dict[str, Any]:
    """The mongot search index backing $search.

    Static rather than dynamic. A dynamic mapping would index every field,
    including the embedding array and the numeric token estimate, which costs
    index size for fields no query searches. The three named here are what a
    question is actually matched against: the passage, the breadcrumb that says
    where it sits, and the page it came from.
    """
    return {
        "mappings": {
            "dynamic": False,
            "fields": {
                "text": {"type": "string"},
                "heading_path": {"type": "string"},
                "document": {"type": "string"},
            },
        }
    }


def to_document(chunk: Chunk, vector: Sequence[float]) -> dict[str, Any]:
    """Flatten a chunk and its embedding into the stored document.

    chunk_id becomes _id: it is already unique and already the citation handle,
    so a second identifier would be one that could disagree with the citation.
    """
    document: dict[str, Any] = {name: getattr(chunk, name) for name in CHUNK_FIELDS}
    # Tuples are stored as BSON arrays and come back as lists; converting on the
    # way out rather than trusting the round trip keeps the dataclass honest.
    document["heading_path"] = list(chunk.heading_path)
    document["anchors"] = list(chunk.anchors)
    document["references"] = list(chunk.references)
    document["_id"] = chunk.chunk_id
    document[VECTOR_FIELD] = list(vector)
    return document


def from_document(document: dict[str, Any]) -> Chunk:
    """Rebuild a chunk from its stored document."""
    return Chunk(
        chunk_id=document["_id"],
        document=document["document"],
        heading_path=tuple(document.get("heading_path") or ()),
        text=document["text"],
        token_estimate=int(document.get("token_estimate", 0)),
        has_code=bool(document.get("has_code", False)),
        anchors=tuple(document.get("anchors") or ()),
        references=tuple(document.get("references") or ()),
        line_no=int(document.get("line_no", 0)),
    )


# Everything except the embedding. Projected explicitly so a hit costs the text
# it is made of rather than the text plus its vector.
_PROJECTION: dict[str, Any] = {VECTOR_FIELD: 0}


class MongoKnowledgeStore:
    """KnowledgeStore over a MongoDB collection with mongot search indexes."""

    def __init__(self, collection: Collection[dict[str, Any]]) -> None:
        self._collection = collection

    def __len__(self) -> int:
        return int(self._collection.count_documents({}))

    def ensure_search_indexes(self, dimensions: int) -> None:
        """Create the two search indexes, if they are not already there.

        Search index creation is asynchronous: the call returns before the index
        is queryable, and querying one that is still building returns nothing
        rather than an error. Callers that index and immediately retrieve must
        wait for the index to report Ready -- see
        logtriage/docs/mongodb/search.md.
        """
        from pymongo.operations import SearchIndexModel

        existing = {index["name"] for index in self._collection.list_search_indexes()}
        models = []
        if VECTOR_INDEX_NAME not in existing:
            models.append(
                SearchIndexModel(
                    definition=vector_index_definition(dimensions),
                    name=VECTOR_INDEX_NAME,
                    type="vectorSearch",
                )
            )
        if TEXT_INDEX_NAME not in existing:
            models.append(
                SearchIndexModel(
                    definition=text_index_definition(),
                    name=TEXT_INDEX_NAME,
                    type="search",
                )
            )
        if models:
            self._collection.create_search_indexes(models)

    def save_meta(self, meta: dict[str, Any]) -> None:
        """Record what built this index, beside it.

        The on-disk index carries its embedder's identity in index.json and
        `triage --kb` rebuilds exactly that embedder. The MongoDB index needs
        the same, for the same reason: querying an index with an embedder other
        than the one that built it still returns k chunks, ranked by a
        similarity that means nothing, and nothing anywhere errors.

        A sibling collection rather than a document in the chunk collection. A
        metadata document living among the chunks would be counted by __len__,
        returned by a $search that happened to match it, and would need
        excluding from every query written later.
        """
        self._meta_collection().replace_one(
            {"_id": META_ID}, {**meta, "_id": META_ID}, upsert=True
        )

    def load_meta(self) -> dict[str, Any]:
        """What built this index, or {} if nothing recorded it."""
        document = self._meta_collection().find_one({"_id": META_ID})
        if not document:
            return {}
        return {key: value for key, value in document.items() if key != "_id"}

    def _meta_collection(self) -> Collection[dict[str, Any]]:
        return self._collection.database[META_COLLECTION]

    def index_status(self) -> dict[str, str]:
        """Each search index's build status, by name.

        Separate from waiting so a caller can report the state without blocking
        -- which is what an operator wants when the wait has already timed out
        and the question is whether anything is happening at all.
        """
        return {
            str(index["name"]): str(index.get("status", "UNKNOWN"))
            for index in self._collection.list_search_indexes()
        }

    def wait_until_ready(
        self, timeout: float = READY_TIMEOUT, interval: float = READY_POLL
    ) -> dict[str, str]:
        """Block until both search indexes are queryable, or raise.

        Index creation is asynchronous and a query against an index still
        building returns an empty result rather than an error -- so without this
        the first $vectorSearch after an index build returns nothing and looks
        exactly like a bad query. Every minute spent debugging that is a minute
        spent on a race.

        Returns the final status map so a caller can record it. Raises on
        timeout naming what each index was doing, because "it did not become
        ready" and "one of them FAILED" need different responses.
        """
        from logtriage.cli import LogtriageError

        deadline = time.monotonic() + timeout
        status: dict[str, str] = {}
        while True:
            status = self.index_status()
            wanted = {VECTOR_INDEX_NAME, TEXT_INDEX_NAME}
            if wanted <= status.keys() and all(
                status[name] == "READY" for name in wanted
            ):
                return status
            failed = {n: s for n, s in status.items() if s in ("FAILED", "DELETING")}
            if failed:
                raise LogtriageError(
                    f"A search index will not become ready: {failed}\n"
                    "  Solution: read the mongot log for the reason it rejected the definition\n"
                    "  Command: make logtriage-logs SERVICE=logtriage-mongot"
                )
            if time.monotonic() >= deadline:
                raise LogtriageError(
                    f"Search indexes were not ready within {timeout:.0f}s: {status}\n"
                    "  A query against a building index returns nothing rather than erroring,\n"
                    "  so proceeding now would look like a retrieval bug.\n"
                    "  Solution: check that mongot is running and syncing\n"
                    "  Command: curl -s http://127.0.0.1:8008/health"
                )
            time.sleep(interval)

    def upsert(
        self, chunks: Iterable[Chunk], vectors: Iterable[Sequence[float]]
    ) -> int:
        from pymongo import ReplaceOne

        operations = [
            ReplaceOne({"_id": chunk.chunk_id}, to_document(chunk, vector), upsert=True)
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
        if not operations:
            return 0
        self._collection.bulk_write(operations, ordered=False)
        return len(operations)

    def vector_search(self, embedding: Sequence[float], k: int = 5) -> list[Hit]:
        pipeline: list[dict[str, Any]] = [
            {
                "$vectorSearch": {
                    "index": VECTOR_INDEX_NAME,
                    "path": VECTOR_FIELD,
                    "queryVector": list(embedding),
                    "numCandidates": k * CANDIDATE_MULTIPLIER,
                    "limit": k,
                }
            },
            {"$project": {**_PROJECTION, "score": {"$meta": "vectorSearchScore"}}},
        ]
        return [
            Hit(
                chunk=from_document(document),
                score=float(document["score"]),
                leg="vector",
            )
            for document in self._collection.aggregate(pipeline)
        ]

    def text_search(self, query: str, k: int = 5) -> list[Hit]:
        if not query.strip():
            return []
        pipeline: list[dict[str, Any]] = [
            {
                "$search": {
                    "index": TEXT_INDEX_NAME,
                    "text": {
                        "query": query,
                        "path": ["text", "heading_path", "document"],
                    },
                }
            },
            {"$limit": k},
            {"$project": {**_PROJECTION, "score": {"$meta": "searchScore"}}},
        ]
        return [
            Hit(
                chunk=from_document(document),
                score=float(document["score"]),
                leg="text",
            )
            for document in self._collection.aggregate(pipeline)
        ]

    def hybrid_search(
        self,
        query: str,
        embedding: Sequence[float],
        k: int = 5,
        vector_weight: float = VECTOR_WEIGHT,
    ) -> list[Hit]:
        """Fuse the two legs, using the same function the in-memory store uses.

        Two round trips rather than one server-side $rankFusion. That costs a
        request and buys the guarantee that the two implementations rank
        identically, which is worth more here: retrieval quality is the thing
        this project cannot yet measure, so a difference between the fake and
        the server would be attributed to the corpus rather than to the store.
        """
        vector_hits = self.vector_search(embedding, k=k * 4) if vector_weight else []
        text_hits = self.text_search(query, k=k * 4)
        return fuse(vector_hits, text_hits, vector_weight=vector_weight, k=k)

    def fetch_by_ids(self, ids: Sequence[str]) -> list[Chunk]:
        """Resolve chunk citations, omitting ids that are not present.

        A plain $match, not a search: citations resolve by identity, and routing
        them through mongot would make a report's verifiability depend on
        whether a search index happened to be Ready.
        """
        if not ids:
            return []
        documents = self._collection.find(
            {"_id": {"$in": list(set(ids))}}, _PROJECTION
        ).sort("_id", 1)
        return [from_document(document) for document in documents]


def knowledge_collection(
    uri: str, database: str = "logtriage", collection: str = "chunks"
) -> Collection[dict[str, Any]]:
    """Open the collection this store reads.

    tz_aware for the same reason as its counterpart in store/mongo.py: one
    client setting decides whether dates come back with a zone, and the two
    collections share a deployment.
    """
    from pymongo import MongoClient

    return MongoClient(uri, tz_aware=True)[database][collection]
