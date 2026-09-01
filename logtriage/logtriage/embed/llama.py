"""Embeddings from llama-server's /v1/embeddings.

The real half of the Embedder pair. Its counterpart, HashingEmbedder, has no
semantics at all; this one is why the vector leg of retrieval is worth running.
`semantic` returns True, and `tools/schemas.py` reads exactly that property to
decide whether to weight the vector leg above zero -- so swapping this in is the
whole of what turns hybrid search on.

Batched because it is an HTTP round trip. Over the real manual that is 15,025
chunks: per-chunk requests turn a few minutes into an afternoon, most of it
spent on TCP.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from logtriage.embed.base import EmbeddingError
from logtriage.transport import TransportError, post_json

# How many texts go in one request. Large enough that the per-request overhead
# disappears, small enough that one oversized batch cannot exceed the server's
# request limit and fail the whole indexing run at chunk 12,000.
DEFAULT_BATCH = 64

# Embedding a batch should not take the five minutes a 31B generation might.
# A stall here means the server is wedged, and finding that out at chunk 100 is
# better than at chunk 15,000.
DEFAULT_TIMEOUT = 120.0

# Probe text for discovering the dimension count. Content is irrelevant -- only
# the length of the returned vector is read -- but it must be non-empty, since
# some servers reject an empty input.
PROBE = "dimension probe"


class LlamaEmbedder:
    """Embedder backed by a llama.cpp server in embedding mode."""

    def __init__(
        self,
        base_url: str,
        model: str = "default",
        batch_size: int = DEFAULT_BATCH,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        self._url = base_url.rstrip("/") + "/v1/embeddings"
        self._model = model
        self._batch_size = batch_size
        self._timeout = timeout
        self._dimensions: int | None = None

    @property
    def semantic(self) -> bool:
        """True: nearness in this space is nearness in meaning.

        The claim rests on the served model actually being an embedding model.
        A chat model started with --embedding will answer these requests with
        well-formed vectors that mean very little, and nothing here can tell the
        difference -- which is why the compose file names a dedicated embedding
        model and says so.
        """
        return True

    @property
    def dimensions(self) -> int:
        """The width of this model's vectors, discovered once and cached.

        Probed rather than configured. A dimension count in a config file is a
        number that can be wrong, and the failure it produces -- a search index
        built for the wrong width -- is rejected by the server much later, with
        an error about the index rather than about the setting.
        """
        if self._dimensions is None:
            self._dimensions = len(self._request([PROBE])[0])
        return self._dimensions

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            vectors.extend(self._request(texts[start : start + self._batch_size]))
        if self._dimensions is None and vectors:
            self._dimensions = len(vectors[0])
        return vectors

    def _request(self, batch: Sequence[str]) -> list[list[float]]:
        try:
            payload = post_json(
                self._url,
                {"model": self._model, "input": list(batch)},
                timeout=self._timeout,
            )
        except TransportError as error:
            raise EmbeddingError(str(error)) from error

        data = payload.get("data")
        if not isinstance(data, list) or len(data) != len(batch):
            raise EmbeddingError(
                f"expected {len(batch)} embeddings, got "
                f"{len(data) if isinstance(data, list) else type(data).__name__}"
            )

        # Ordering is by the index field, not by position. The OpenAI shape
        # permits a server to return them out of order, and a silently permuted
        # batch would pair every chunk with another chunk's vector -- an index
        # that looks entirely normal and retrieves nonsense.
        ordered = sorted(data, key=lambda entry: int(entry.get("index", 0)))
        return [_unit(entry) for entry in ordered]


def _unit(entry: dict[str, Any]) -> list[float]:
    """Extract one embedding and normalise it to unit length.

    `embed.base.cosine` is a bare dot product: it documents that both embedders
    return unit vectors and deliberately does not divide, so that a bug in
    either one shows up as a wrong score rather than being hidden by the
    normalisation. That contract has to be kept here. llama.cpp normalises for
    some pooling types and not others, so doing it unconditionally is the only
    way the contract holds whatever POOLING is set to.
    """
    raw = entry.get("embedding")
    if not isinstance(raw, list) or not raw:
        raise EmbeddingError(f"embedding is not a non-empty array: {raw!r}")

    # llama-server returns [[...]] rather than [...] for some pooling
    # configurations; unwrap one level rather than storing a vector of lists.
    if isinstance(raw[0], list):
        raw = raw[0]

    vector = [float(value) for value in raw]
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0.0:
        raise EmbeddingError("embedding is all zeros; the server returned no signal")
    return [value / norm for value in vector]
