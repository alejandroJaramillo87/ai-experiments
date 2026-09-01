"""One JSON-over-HTTP round trip, shared by the two llama-server adapters.

Standard library only. Adding an HTTP client would be the third dependency in a
project that has two, for a single POST with a timeout -- and every dependency
is one more thing that has to install cleanly on the MacBook before this tool
runs there at all.

The error handling is the substance of this module rather than an afterthought.
A served model fails in ways a fake never does: the server is not up yet, the
model is still loading, the request exceeded the context window, the connection
dropped mid-generation. Each of those has to arrive at the caller as something
that says which one it was, because "the model did not answer" is the least
useful sentence a log-triage tool could print.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any

# A 31B model generating a long answer takes a while, and a timeout that fires
# mid-generation looks exactly like a hung server. Generous by default, and an
# argument because indexing a whole manual wants a shorter one -- there, a stall
# should surface immediately rather than after five minutes per batch.
DEFAULT_TIMEOUT = 300.0

# Retries cover the connection failing, not the request being wrong. A 4xx is
# the caller's fault and will fail identically on every attempt, so retrying one
# turns a clear error into a slow clear error.
DEFAULT_RETRIES = 2
RETRY_BACKOFF = 2.0

# Server error bodies are quoted into the exception because llama-server says
# something useful in them. Bounded so a stack trace stays readable when the
# body is an HTML error page.
MAX_ERROR_BODY = 2000


class TransportError(RuntimeError):
    """The server could not be reached, or answered with something unusable."""


def post_json(
    url: str,
    payload: dict[str, Any],
    timeout: float = DEFAULT_TIMEOUT,
    retries: int = DEFAULT_RETRIES,
) -> dict[str, Any]:
    """POST a JSON document and return the JSON response.

    Retries only on transport failures and 5xx, with a doubling backoff. A 4xx
    is raised on the first attempt.
    """
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )

    last: Exception | None = None
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                raw = response.read()
            break
        except urllib.error.HTTPError as error:
            detail = _error_body(error)
            if error.code < 500:
                raise TransportError(
                    f"{url} returned HTTP {error.code}: {detail}"
                ) from error
            last = TransportError(f"{url} returned HTTP {error.code}: {detail}")
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            # URLError wraps a refused connection, an unresolvable host, and a
            # dropped socket alike; the reason carries which.
            last = TransportError(f"{url} unreachable: {error}")

        if attempt < retries:
            time.sleep(RETRY_BACKOFF**attempt)
    else:
        raise last if last is not None else TransportError(f"{url} failed")

    try:
        decoded = json.loads(raw)
    except json.JSONDecodeError as error:
        # A body that is not JSON usually means something in front of the server
        # answered instead of it -- a proxy, or a different service on the port.
        raise TransportError(
            f"{url} returned a body that is not JSON: {raw[:MAX_ERROR_BODY]!r}"
        ) from error

    if not isinstance(decoded, dict):
        raise TransportError(
            f"{url} returned {type(decoded).__name__}, expected object"
        )
    return decoded


def _error_body(error: urllib.error.HTTPError) -> str:
    try:
        return error.read().decode("utf-8", "replace")[:MAX_ERROR_BODY]
    except Exception:  # pragma: no cover - the body is already lost
        return "<unreadable>"
