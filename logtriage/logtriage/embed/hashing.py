"""A deterministic embedder with no semantics whatsoever.

This exists to exercise plumbing: chunking, indexing, storage, retrieval,
citation. It hashes tokens into a fixed-width vector, so two texts sharing
vocabulary land near each other and two texts sharing meaning do not.

Say that plainly wherever a number produced with it is reported. A retrieval
score from this embedder measures word overlap, and reading it as a quality
result would be a mistake -- the kind that survives a long time, because the
number looks exactly like the real one. Real embeddings come from a served model.

It is genuinely useful for one thing beyond plumbing: it is a lexical baseline.
If a real embedding model cannot beat word overlap on a retrieval set, that is
worth knowing.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Sequence

# Vector width. Small enough that a brute-force cosine over a whole manual is
# instant, wide enough that unrelated vocabulary rarely collides.
DIMENSIONS = 256

# Seed folded into every token hash, so a different seed gives an independent
# projection. Fixed by default: an index built today must be searchable
# tomorrow, and an embedder seeded from the clock would silently invalidate one.
DEFAULT_SEED = 20260830

TOKEN_RE = re.compile(r"[a-z0-9_$.]+")


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens, dotted identifiers kept both whole and split.

    Dots, dollars and underscores stay inside tokens because the vocabulary that
    matters here is `db.collection`, `$vectorSearch` and `planSummary`, and
    splitting those into fragments loses exactly the terms worth matching on.

    But keeping them *only* whole loses the other half. Someone pasting
    `setProfilingLevel` into a search finds nothing, because the manual writes
    `db.setProfilingLevel` and that is one token which does not equal the query.
    So a dotted token yields itself and its segments: `db.collection` matches a
    search for either `db.collection` or `collection`, which is what a reader
    means by both.
    """
    tokens: list[str] = []
    for token in TOKEN_RE.findall(text.lower()):
        tokens.append(token)
        if "." in token:
            tokens.extend(part for part in token.split(".") if part)
    return tokens


class HashingEmbedder:
    """Hashes tokens into a unit vector. Deterministic, offline, meaningless."""

    def __init__(self, dimensions: int = DIMENSIONS, seed: int = DEFAULT_SEED) -> None:
        if dimensions < 1:
            raise ValueError("dimensions must be positive")
        self._dimensions = dimensions
        self._seed = seed

    @property
    def dimensions(self) -> int:
        return self._dimensions

    @property
    def semantic(self) -> bool:
        """No. Token hashing preserves vocabulary overlap and nothing else."""
        return False

    def _bucket_and_sign(self, token: str) -> tuple[int, float]:
        digest = hashlib.sha256(f"{self._seed}:{token}".encode()).digest()
        bucket = int.from_bytes(digest[:4], "big") % self._dimensions
        # A signed hash rather than counts alone: without it every vector sits in
        # the positive orthant, cosine similarity has a high floor, unrelated
        # documents score alike, and ranking loses its resolution.
        sign = 1.0 if digest[4] & 1 else -1.0
        return bucket, sign

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for text in texts:
            vector = [0.0] * self._dimensions
            for token in tokenize(text):
                bucket, sign = self._bucket_and_sign(token)
                vector[bucket] += sign
            norm = math.sqrt(sum(v * v for v in vector))
            if norm:
                vector = [v / norm for v in vector]
            # An empty or fully-cancelling text yields the zero vector, which has
            # no direction. Left as zeros rather than faked: it scores 0 against
            # everything, which is the honest answer.
            out.append(vector)
        return out
