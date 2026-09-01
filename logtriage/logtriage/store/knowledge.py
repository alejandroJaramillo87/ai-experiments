"""The documentation store: vector, text, and hybrid retrieval over chunks.

Held apart from LogStore deliberately. Logs are queried structurally -- counted,
grouped, bounded by time -- and documentation is queried by meaning. They share
a database and nothing else, and the one mistake this separation prevents is the
common one: reaching for the vector index to answer "how many collection scans
were there", which returns the ten chunks most similar to the question and not
the answer.

Two implementations, as with LogStore. The in-memory one scores every chunk with
a dot product; the MongoDB one issues $vectorSearch and $search through mongot.
The second cannot be exercised here at all -- mongomock has no implementation of
either, and they are not MongoDB operators but mongot ones -- so that path is
genuinely unverified until it runs against a real server, and the tests say so
rather than implying coverage that does not exist.
"""

from __future__ import annotations

import json
from array import array
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

from logtriage.embed.base import cosine
from logtriage.kb.chunk import Chunk

# Files an on-disk index is made of. The vectors are one packed block rather
# than a field inside each chunk record: a float per line of JSON costs about
# nine times the bytes and parses two orders of magnitude slower, and the whole
# reason for persisting is that rebuilding is the expensive path.
CHUNKS_FILE = "chunks.jsonl"
VECTORS_FILE = "vectors.f32"
META_FILE = "index.json"

# Weight given to the vector leg when fusing with the text leg. Vector
# retrieval finds a passage that explains a concept in different words; text
# retrieval finds the passage containing the exact identifier or error string.
# For log diagnosis the exact string is often the whole question -- someone
# pastes a message and wants the page about it -- so the text leg is weighted
# nearly as heavily as the vector one.
VECTOR_WEIGHT = 0.6

# Reciprocal-rank-fusion constant. Dampens the contribution of top ranks so one
# leg cannot dominate purely by being confident; 60 is the value the original
# RRF work used and there is no reason here to differ.
RRF_K = 60


@dataclass(frozen=True, slots=True)
class Hit:
    """One retrieved chunk and why it was retrieved."""

    chunk: Chunk
    score: float
    leg: str = "vector"

    @property
    def chunk_id(self) -> str:
        return self.chunk.chunk_id


class KnowledgeStore(Protocol):
    """Semantic and lexical retrieval over documentation chunks."""

    def upsert(
        self, chunks: Iterable[Chunk], vectors: Iterable[Sequence[float]]
    ) -> int: ...

    def vector_search(self, embedding: Sequence[float], k: int = 5) -> list[Hit]: ...

    def text_search(self, query: str, k: int = 5) -> list[Hit]: ...

    def hybrid_search(
        self,
        query: str,
        embedding: Sequence[float],
        k: int = 5,
        vector_weight: float = VECTOR_WEIGHT,
    ) -> list[Hit]: ...

    def fetch_by_ids(self, ids: Sequence[str]) -> list[Chunk]: ...

    def __len__(self) -> int: ...


class InMemoryKnowledgeStore:
    """Brute-force cosine over every chunk.

    Exact rather than approximate, which is the point: an approximate index
    would be a second implementation of the thing being tested, and a recall
    difference between it and the real index would be indistinguishable from a
    bug in the retrieval logic. A whole manual is tens of thousands of chunks,
    and a dot product over 256 dimensions is fast enough that exact search costs
    nothing worth optimising.
    """

    def __init__(self) -> None:
        self._chunks: list[Chunk] = []
        self._vectors: list[Sequence[float]] = []
        self._tokens: list[frozenset[str]] = []
        self._index: dict[str, int] = {}

    def upsert(
        self, chunks: Iterable[Chunk], vectors: Iterable[Sequence[float]]
    ) -> int:
        from logtriage.embed.hashing import tokenize

        added = 0
        for chunk, vector in zip(chunks, vectors, strict=True):
            # Two representations chosen for the size the real corpus turned out
            # to be. Tokenizing here rather than in text_search moves work that
            # was being repeated on every query -- measured at 1.3s per query
            # over the manual -- into a one-off cost at build time. Packing the
            # vector into an array of floats takes the same corpus from about
            # 330MB of boxed Python floats to 40MB, with identical scores.
            packed = array("f", vector)
            tokens = frozenset(tokenize(chunk.render()))
            position = self._index.get(chunk.chunk_id)
            if position is None:
                self._index[chunk.chunk_id] = len(self._chunks)
                self._chunks.append(chunk)
                self._vectors.append(packed)
                self._tokens.append(tokens)
                added += 1
            else:
                self._chunks[position] = chunk
                self._vectors[position] = packed
                self._tokens[position] = tokens
        return added

    def __len__(self) -> int:
        return len(self._chunks)

    def vector_search(self, embedding: Sequence[float], k: int = 5) -> list[Hit]:
        scored = [
            Hit(chunk=chunk, score=cosine(embedding, vector), leg="vector")
            for chunk, vector in zip(self._chunks, self._vectors, strict=True)
        ]
        # chunk_id breaks ties so two searches over one index rank identically;
        # without it, chunks with equal scores come back in list order, which
        # changes whenever the index is rebuilt.
        scored.sort(key=lambda hit: (-hit.score, hit.chunk_id))
        return scored[:k]

    def text_search(self, query: str, k: int = 5) -> list[Hit]:
        """Token-overlap search, standing in for MongoDB's $search.

        Not an attempt to reimplement Lucene scoring. It ranks by how much of
        the query's vocabulary a chunk contains, which is enough to check that
        the fusion below combines two differently-ordered lists correctly --
        the thing that can actually be got wrong here.
        """
        from logtriage.embed.hashing import tokenize

        terms = frozenset(tokenize(query))
        if not terms:
            return []
        hits: list[Hit] = []
        for chunk, tokens in zip(self._chunks, self._tokens, strict=True):
            overlap = len(terms & tokens)
            if overlap:
                hits.append(Hit(chunk=chunk, score=overlap / len(terms), leg="text"))
        hits.sort(key=lambda hit: (-hit.score, hit.chunk_id))
        return hits[:k]

    def hybrid_search(
        self,
        query: str,
        embedding: Sequence[float],
        k: int = 5,
        vector_weight: float = VECTOR_WEIGHT,
    ) -> list[Hit]:
        """Fuse the two legs by reciprocal rank.

        Rank rather than raw score, because the two scores are not comparable:
        a cosine and a term-overlap fraction live on different scales, and
        adding them weights whichever happens to be larger. Fusing on position
        is what $rankFusion does on the server, so the two implementations agree
        on method as well as interface.

        `vector_weight` is a parameter rather than only a constant because the
        right value depends on whether the vectors mean anything. At 0 this is
        exactly text search, which is the correct behaviour for an embedder that
        declares itself non-semantic: fusing a meaningless ranking in does not
        dilute the noise, it promotes it.
        """
        vector_hits = self.vector_search(embedding, k=k * 4) if vector_weight else []
        text_hits = self.text_search(query, k=k * 4)

        scores: dict[str, float] = {}
        chunks: dict[str, Chunk] = {}
        legs: dict[str, set[str]] = {}
        for hits, weight in (
            (vector_hits, vector_weight),
            (text_hits, 1 - vector_weight),
        ):
            for rank, hit in enumerate(hits, start=1):
                scores[hit.chunk_id] = scores.get(hit.chunk_id, 0.0) + weight / (
                    RRF_K + rank
                )
                chunks[hit.chunk_id] = hit.chunk
                legs.setdefault(hit.chunk_id, set()).add(hit.leg)

        fused = [
            Hit(
                chunk=chunks[chunk_id],
                score=score,
                leg="+".join(sorted(legs[chunk_id])),
            )
            for chunk_id, score in scores.items()
        ]
        fused.sort(key=lambda hit: (-hit.score, hit.chunk_id))
        return fused[:k]

    def fetch_by_ids(self, ids: Sequence[str]) -> list[Chunk]:
        """Resolve chunk citations, omitting ids that are not present.

        Same contract as LogStore.fetch_by_ids and for the same reason: a report
        cites documentation as well as records, and the checker needs to learn
        which citations failed rather than that one did.
        """
        found = [self._chunks[self._index[i]] for i in set(ids) if i in self._index]
        return sorted(found, key=lambda chunk: chunk.chunk_id)

    def save(self, directory: Path, meta: dict[str, Any]) -> None:
        """Write the index so a triage run does not rebuild it.

        `meta` describes what produced the index -- the corpus commit, the
        embedder and its seed. It is written verbatim and checked on load,
        because an index and the thing that reads it disagreeing about either is
        a failure with no symptom: retrieval still returns its k best chunks,
        scored against vectors that mean something else.
        """
        directory.mkdir(parents=True, exist_ok=True)
        dimensions = len(self._vectors[0]) if self._vectors else 0

        with open(directory / CHUNKS_FILE, "w", encoding="utf-8") as handle:
            for chunk in self._chunks:
                handle.write(json.dumps(asdict(chunk), sort_keys=True) + "\n")

        packed = array("f")
        for vector in self._vectors:
            packed.extend(vector)
        (directory / VECTORS_FILE).write_bytes(packed.tobytes())

        (directory / META_FILE).write_text(
            json.dumps(
                {**meta, "chunks": len(self._chunks), "dimensions": dimensions},
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(
        cls, directory: Path, expect: dict[str, Any] | None = None
    ) -> tuple[InMemoryKnowledgeStore, dict[str, Any]]:
        """Read an index back, refusing one built by something else.

        A mismatch raises rather than warns. The whole failure mode this guards
        is silent: a stale index answers every query, plausibly, from a corpus
        that is not the one on disk.
        """
        meta = json.loads((directory / META_FILE).read_text(encoding="utf-8"))
        for key, value in (expect or {}).items():
            if meta.get(key) != value:
                raise ValueError(
                    f"Index at {directory} was built with {key}={meta.get(key)!r}, "
                    f"but this run expects {value!r}. Rebuild it with "
                    f"`python -m logtriage index-kb --refresh`."
                )

        dimensions = int(meta["dimensions"])
        packed = array("f")
        packed.frombytes((directory / VECTORS_FILE).read_bytes())

        store = cls()
        with open(directory / CHUNKS_FILE, encoding="utf-8") as handle:
            for position, line in enumerate(handle):
                record = json.loads(line)
                chunk = Chunk(
                    **{
                        **record,
                        "heading_path": tuple(record["heading_path"]),
                        "anchors": tuple(record["anchors"]),
                        "references": tuple(record["references"]),
                    }
                )
                start = position * dimensions
                store._adopt(chunk, packed[start : start + dimensions])
        if len(store) != int(meta["chunks"]):
            raise ValueError(
                f"Index at {directory} declares {meta['chunks']} chunks but "
                f"carries {len(store)}; it is truncated or was written by a "
                f"different version."
            )
        return store, meta

    def _adopt(self, chunk: Chunk, vector: array[float]) -> None:
        """Insert an already-packed vector, skipping the array copy in upsert."""
        from logtriage.embed.hashing import tokenize

        self._index[chunk.chunk_id] = len(self._chunks)
        self._chunks.append(chunk)
        self._vectors.append(vector)
        self._tokens.append(frozenset(tokenize(chunk.render())))
