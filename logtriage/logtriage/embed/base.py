"""The embedding interface.

Two implementations: a deterministic hashing one that needs nothing, and a real
one backed by llama-server's /v1/embeddings. The split exists because the whole
pipeline -- chunk, index, retrieve, cite -- can be exercised without a model,
and only the *quality* of retrieval needs one.

Keeping that boundary explicit is what stops a sandbox retrieval score from
being mistaken for a real one. The hashing embedder has no semantics at all; see
its module docstring.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol


class Embedder(Protocol):
    """Turns text into vectors."""

    @property
    def dimensions(self) -> int: ...

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed a batch.

        Batched rather than one at a time because the real implementation is an
        HTTP round trip, and per-chunk requests over a manual with thousands of
        chunks is the difference between a minute and an hour.
        """
        ...


def cosine(left: Sequence[float], right: Sequence[float]) -> float:
    """Cosine similarity, assuming neither vector is zero-length.

    No normalisation here: both implementations return unit vectors, so this is
    a dot product, and dividing anyway would hide a bug in either one behind a
    correct-looking score.
    """
    return sum(a * b for a, b in zip(left, right, strict=True))
