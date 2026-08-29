"""Scripted stand-ins for a streaming inference server.

Two fakes, because two different questions need answering and one fake cannot
answer both.

`ScriptedResponse` replaces the object `urllib.request.urlopen` returns. It carries
its own delays, so a frame can be made to arrive late without a socket, a port, or a
thread. Deterministic and fast, which is what a test asserting that a 40 ms gap on the
wire becomes a 40 ms inter-token latency needs.

`ScriptedServer` is a real HTTP server on a real port. It costs a thread and a few
milliseconds of setup, and it buys the things a substituted response object cannot
reach: the genuine urllib request path, HTTP status codes as urllib raises them, read
timeouts, and whatever the kernel does to frames written close together. The existing
`FakeResponse` in test_adapters.py proves decoding; only a socket can prove timing.

Both take the same frame script, so one scripted sequence drives either.
"""

from __future__ import annotations

import contextlib
import dataclasses
import http.server
import json
import threading
import time

# Written by every engine this suite targets. The prefix is llama.cpp's and vLLM's;
# the specification permits omitting the space, which is its own recorded finding.
SSE_PREFIX = b"data: "
SSE_DONE = b"data: [DONE]\n\n"


@dataclasses.dataclass(frozen=True)
class Frame:
    """One server-sent event, and how long the server waits before sending it."""

    delay_s: float
    line: bytes

    @classmethod
    def of(cls, payload: dict | bytes | str, delay_s: float = 0.0) -> Frame:
        if isinstance(payload, dict):
            body = json.dumps(payload).encode("utf-8")
        elif isinstance(payload, str):
            body = payload.encode("utf-8")
        else:
            body = payload
        return cls(delay_s, SSE_PREFIX + body + b"\n\n")


def text_chunk(text: str) -> dict:
    """A content-bearing completion chunk, in the shape both engines emit."""
    return {
        "choices": [{"text": text, "index": 0, "finish_reason": None}],
        "object": "text_completion",
    }


def usage_frame(prompt_tokens: int, completion_tokens: int, **extra) -> dict:
    """A terminal frame carrying usage and no text.

    vLLM sends an empty choices list; llama.cpp sends one choice whose text is empty,
    because the text already went out in the last partial chunk. Both are covered by
    reading the text field structurally rather than by dropping the final element.
    """
    body: dict = {
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
        "object": "text_completion",
    }
    body.update(extra)
    return body


def script(texts: list[str], gap_s: float, ttft_s: float, usage: dict) -> list[Frame]:
    """A whole generation: one frame per text, then the terminal usage frame."""
    frames = [Frame.of(text_chunk(texts[0]), ttft_s)]
    frames += [Frame.of(text_chunk(t), gap_s) for t in texts[1:]]
    frames.append(Frame.of(usage, gap_s))
    return frames


class ScriptedResponse:
    """What urlopen returns, with the script's delays built in.

    Iterating yields one line at a time, sleeping first. `bench.client.stream` stamps
    each line as it is read, so the sleep lands where the network delay would.
    """

    def __init__(self, frames: list[Frame], done: bool = True):
        self._frames = frames
        self._done = done

    def __enter__(self) -> ScriptedResponse:
        return self

    def __exit__(self, *exc_info) -> bool:
        return False

    def __iter__(self):
        for frame in self._frames:
            if frame.delay_s:
                time.sleep(frame.delay_s)
            yield frame.line
        if self._done:
            yield SSE_DONE


def _handler_for(frames: list[Frame], status: int, body: bytes):
    class Handler(http.server.BaseHTTPRequestHandler):
        protocol_version = (
            "HTTP/1.0"  # close per response; no keep-alive to reason about
        )

        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length") or 0)
            if length:
                self.rfile.read(length)
            if status != 200:
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for frame in frames:
                if frame.delay_s:
                    time.sleep(frame.delay_s)
                # Flush per frame: without it the frames coalesce in the write buffer
                # and every arrival stamp collapses onto the last one.
                self.wfile.write(frame.line)
                self.wfile.flush()
            self.wfile.write(SSE_DONE)
            self.wfile.flush()

        def log_message(self, *args) -> None:
            """Silence the default stderr access log."""

    return Handler


@contextlib.contextmanager
def ScriptedServer(  # noqa: N802 - a context manager used as a constructor
    frames: list[Frame],
    status: int = 200,
    error_body: bytes = b'{"error": "scripted"}',
):
    """A real server on an ephemeral port, yielding its base URL.

    Follows the pattern tests/common/test_obs_serve.py uses for the dashboard: port 0
    so the kernel assigns, a daemon thread, and shutdown before close.
    """
    server = http.server.ThreadingHTTPServer(
        ("127.0.0.1", 0), _handler_for(frames, status, error_body)
    )
    server.daemon_threads = True
    # The default half-second poll interval is what shutdown() waits out, and it
    # dominates the runtime of every test that starts a server.
    thread = threading.Thread(
        target=server.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True
    )
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
