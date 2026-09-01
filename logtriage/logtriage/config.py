"""Where the served backends live, read from the environment.

Four variables are the whole of this tool's deployment configuration. That is
deliberate and it is what makes the same code run in three places: nothing
branches on hostname, on platform, or on which machine it thinks it is. The
sandbox sets none of them and gets the offline implementations; the workstation
points them at containers; the MacBook points the two llama URLs at a native
Metal server and the Mongo one at a container.

Unset is a legitimate state, not an error, which is why the accessors return
None rather than raising. A missing variable means "use the offline
implementation"; only a command that was explicitly asked for a served backend
turns that into a failure, and then `require` names the variable.
"""

from __future__ import annotations

import os

# The chat model, OpenAI-compatible, as served by llama-server.
LLAMA_URL = "LOGTRIAGE_LLAMA_URL"

# The embedding model. A second server rather than a second endpoint, because
# llama.cpp's --embedding mode replaces the chat endpoints.
EMBED_URL = "LOGTRIAGE_EMBED_URL"

# The MongoDB deployment holding both the log records and the documentation
# chunks. They share a database and nothing else.
MONGO_URI = "LOGTRIAGE_MONGO_URI"

# Where the documentation corpus is fetched from and kept. Read by kb/fetch.py.
KB_SOURCE = "LOGTRIAGE_KB_SOURCE"

# What a correct value looks like, per variable, for the error message below.
# The ports match docker-compose.logtriage.yaml; a message that suggested the
# chat port for the embedding server would be worse than no example, because it
# fails much later as an unusable response rather than as a missing variable.
EXAMPLES = {
    LLAMA_URL: "http://127.0.0.1:8006",
    EMBED_URL: "http://127.0.0.1:8007",
    MONGO_URI: "mongodb://127.0.0.1:27017/?replicaSet=rs0&directConnection=true",
    KB_SOURCE: "/path/to/mongodb-docs",
}


def _read(name: str) -> str | None:
    """An empty variable is unset.

    Shell scripts and compose files export empty strings for values nobody
    filled in, and an empty URL fails much later, as a connection error against
    an address that is not one.
    """
    value = os.environ.get(name, "").strip()
    return value or None


def llama_url() -> str | None:
    return _read(LLAMA_URL)


def embed_url() -> str | None:
    return _read(EMBED_URL)


def mongo_uri() -> str | None:
    return _read(MONGO_URI)


def require(name: str) -> str:
    """The value of a variable a served backend cannot run without.

    Raises LogtriageError so the message reaches the user as a named failure
    with a remedy, rather than as a traceback about a URL of None.
    """
    from logtriage.cli import LogtriageError

    value = _read(name)
    if value is None:
        raise LogtriageError(
            f"{name} is not set, and this backend needs it\n"
            "  Solution: point it at the served endpoint\n"
            f"  Command: export {name}={EXAMPLES.get(name, '<value>')}"
        )
    return value
