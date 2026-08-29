"""Streaming SSE client: the one place a wall-clock stamp is taken.

Stdlib urllib only, so the suite runs on the host without resolving the uv environment,
the same constraint common/serving.py works under.

Two decisions here decide whether the numbers mean anything.

First, the arrival stamp is taken the moment a `data:` line is read, before it is JSON
parsed. Parsing inside the timed region charges the client's own deserialization to the
server's inter-token latency.

Second, a chunk is a token arrival only if it carries text. The retired load generator
counted every `data:` line and then removed one or two by heuristic
(`token_arrival_times[:-1]`, plus a second trim if the penultimate line mentioned
timings) to compensate for the `[DONE]` sentinel and the usage frame. That is a guess
about a wire format. Testing `choices[0].text` is a fact about the chunk in hand, and it
stays correct when an engine adds or removes a trailing frame.
"""

from __future__ import annotations

import dataclasses
import json
import time
import urllib.error
import urllib.request

from bench.adapters import Adapter, ServerTimings, Usage
from common.serving import ServingError, ServingTransientError

# The specification makes the space after the colon optional (WHATWG HTML, server-sent
# events: a single leading space is stripped if present). Both pinned engines write it,
# so requiring it matched every frame this suite has seen and would have silently
# yielded zero arrivals against one that does not. The payload is stripped below, which
# is what removes the space when it is there.
SSE_PREFIX = b"data:"
SSE_DONE = b"[DONE]"


@dataclasses.dataclass
class StreamResult:
    """One completed streaming request as observed by the client."""

    t_submit: float
    arrivals: list[float]
    usage: Usage | None
    timings: ServerTimings | None
    chunks_seen: int


def _chunk_text(payload: dict) -> str:
    choices = payload.get("choices") or []
    if not choices:
        return ""
    first = choices[0]
    # /v1/completions puts it in "text"; a chat-shaped delta would use "delta.content".
    # Both are read so that an engine served through the chat surface decodes here too,
    # which costs one lookup and removes a whole class of silent zero-arrival result.
    text = first.get("text")
    if text is None:
        text = (first.get("delta") or {}).get("content")
    return text or ""


def stream(adapter: Adapter, body: dict, timeout: float) -> StreamResult:
    """POST a streaming completion and capture per-chunk arrival times."""
    request = urllib.request.Request(
        adapter.url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        method="POST",
    )

    arrivals: list[float] = []
    usage: Usage | None = None
    timings: ServerTimings | None = None
    chunks_seen = 0

    t_submit = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            for raw in response:
                if not raw.startswith(SSE_PREFIX):
                    continue
                # Stamp first: anything after this line is the client's own cost.
                now = time.perf_counter()
                payload_bytes = raw[len(SSE_PREFIX) :].strip()
                if payload_bytes == SSE_DONE:
                    continue
                chunks_seen += 1
                try:
                    payload = json.loads(payload_bytes)
                except json.JSONDecodeError:
                    continue
                if _chunk_text(payload):
                    arrivals.append(now)
                # usage and timings ride on the terminal frame, which carries no text.
                usage = adapter.parse_usage(payload) or usage
                timings = adapter.parse_timings(payload) or timings
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:400]
        if exc.code >= 500:
            raise ServingTransientError(
                f"{adapter.service.name}: HTTP {exc.code} from {adapter.url}: {detail}"
            ) from exc
        raise ServingError(
            f"{adapter.service.name}: HTTP {exc.code} from {adapter.url}: {detail}"
        ) from exc
    except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
        raise ServingTransientError(
            f"{adapter.service.name}: {adapter.url} unreachable: {exc}"
        ) from exc

    if not arrivals:
        raise ServingError(
            f"{adapter.service.name}: the response carried no text chunks "
            f"({chunks_seen} frames seen). A benchmark cannot time a generation that "
            "did not happen"
        )
    return StreamResult(t_submit, arrivals, usage, timings, chunks_seen)
