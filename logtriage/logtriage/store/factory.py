"""Choosing a store backend by name.

The counterpart of `embed/factory.py`, and it exists for the same reason: two
commands now need to make the same choice, and a second copy of the decision is
a second place for it to be made differently.

`memory` is not a placeholder. It is how `ingest` and `triage` run in a checkout
with no database -- the whole pipeline exercised end to end, with the same
interface, so the command is developed and tested against something before it
meets a server. What it is not is durable: an in-memory store is discarded when
the process exits, so ingesting into it and then querying from a second command
finds nothing. The commands say so rather than leaving a reader to discover it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NoReturn

from logtriage.store.base import LogStore

if TYPE_CHECKING:  # pragma: no cover - import shape only
    from logtriage.store.knowledge_mongo import MongoKnowledgeStore

# Backends the `--store` flag accepts. A closed set: the name reaches the user
# in an error message and in `--json` output, so it is an interface rather than
# an implementation detail.
BACKENDS = ("memory", "mongo")


def log_store(backend: str, uri: str | None = None) -> LogStore:
    """Open a log store by backend name.

    `uri` overrides LOGTRIAGE_MONGO_URI. Unset and un-overridden, the mongo
    backend fails by name rather than connecting to a default localhost that
    may be somebody else's database.
    """
    if backend == "memory":
        from logtriage.store.memory import InMemoryLogStore

        return InMemoryLogStore()

    if backend == "mongo":
        from logtriage import config
        from logtriage.store.mongo import MongoLogStore, log_collection

        return MongoLogStore(log_collection(uri or config.require(config.MONGO_URI)))

    _unknown(backend)


def mongo_knowledge_store(uri: str | None = None) -> MongoKnowledgeStore:
    """Open the documentation store backed by MongoDB and mongot.

    Named for the backend rather than parameterised, because there is no
    symmetry to exploit: the offline knowledge store is loaded from an on-disk
    index built by `index-kb`, which is a different operation with a different
    failure mode, and `commands/triage.py` owns that path.

    Returns the concrete class rather than the KnowledgeStore protocol. The
    protocol covers retrieval; `wait_until_ready` and `ensure_search_indexes`
    are deployment operations that only mongot has, and widening the protocol
    to carry them would put two methods on the in-memory store that could only
    raise.
    """
    from logtriage import config
    from logtriage.store.knowledge_mongo import (
        MongoKnowledgeStore,
        knowledge_collection,
    )

    return MongoKnowledgeStore(
        knowledge_collection(uri or config.require(config.MONGO_URI))
    )


def _unknown(backend: str) -> NoReturn:
    from logtriage.cli import LogtriageError

    raise LogtriageError(
        f"Unknown store backend: {backend}\n"
        f"  Accepted: {', '.join(BACKENDS)}\n"
        "  Command: python -m logtriage ingest mongod.log --store memory"
    )
