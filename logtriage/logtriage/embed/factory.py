"""Choosing an embedder, and recording which one built an index.

The two halves belong together. An index is only searchable by the embedder that
built it -- query vectors from a different model, or the same model at a
different width, land nowhere near the stored ones -- and the failure has no
symptom: retrieval still returns its k best chunks, ranked by a similarity that
means nothing. So the embedder's identity is written into the index metadata
here, and checked against a freshly constructed one on the way back in.

`descriptor` is what makes that check possible, and it is why this is a module
rather than an if-statement in two commands.
"""

from __future__ import annotations

from typing import Any

from logtriage.embed.base import Embedder

# Backends `index-kb --embedder` and `triage --kb` accept. A closed set: the
# name is written into an index and read back, so it is a stored format rather
# than a convenience.
BACKENDS = ("hashing", "llama")


def make_embedder(name: str) -> Embedder:
    """Construct an embedder by name."""
    if name == "hashing":
        from logtriage.embed.hashing import HashingEmbedder

        return HashingEmbedder()
    if name == "llama":
        from logtriage import config
        from logtriage.embed.llama import LlamaEmbedder

        return LlamaEmbedder(config.require(config.EMBED_URL))

    from logtriage.cli import LogtriageError

    raise LogtriageError(
        f"Unknown embedder: {name}\n"
        f"  Accepted: {', '.join(BACKENDS)}\n"
        "  Command: python -m logtriage index-kb --embedder hashing"
    )


def descriptor(name: str, embedder: Embedder) -> dict[str, Any]:
    """The identity fields stored beside an index and checked when it is read.

    The hashing embedder carries its seed, because a different seed is a
    different projection of the same vocabulary and nothing else would
    distinguish the two. A served model carries nothing but its width: which
    checkpoint llama-server has loaded is not something the client can see, so
    that mismatch is a deployment question the index cannot guard against, and
    docs/verification/desktop.md records it rather than this pretending
    otherwise.
    """
    fields: dict[str, Any] = {
        "embedder": name,
        "embedder_dimensions": embedder.dimensions,
    }
    if name == "hashing":
        from logtriage.embed.hashing import DEFAULT_SEED

        fields["embedder_seed"] = DEFAULT_SEED
    return fields


def from_index_meta(meta: dict[str, Any]) -> tuple[Embedder, dict[str, Any]]:
    """Rebuild the embedder an index was built with, and what to check.

    Returns the embedder and the `expect` mapping for
    `InMemoryKnowledgeStore.load`. Reconstructing from the index rather than
    from a flag is the point: a reader who indexed with one embedder and
    searched with another would get a plausible ranking over meaningless
    similarities, and no error anywhere.
    """
    name = str(meta.get("embedder", "hashing"))
    embedder = make_embedder(name)
    return embedder, descriptor(name, embedder)
