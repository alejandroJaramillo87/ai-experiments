"""The two llama-server adapters, driven against a real HTTP server.

There is no model here and no GPU, and these tests are still worth running. What
they check is the wire contract -- the exact JSON sent, and what is made of the
JSON that comes back -- and that is where an HTTP adapter is actually wrong.
Serving the requests from a stdlib http.server rather than a mocked urlopen is
the point: a mock asserts that the code called the function the test expected,
which is a restatement of the code, while a socket asserts that a server
receives a document it can act on.

What this does not establish, and no test in this tree can: whether Gemma
answers usefully, whether its tool calls are sensible, whether the embedding
model's vectors mean anything. Those need a served model; see
logtriage/docs/verification/desktop-model.md.
"""

from __future__ import annotations

import json
import math
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest

from logtriage import transport
from logtriage.embed.base import EmbeddingError
from logtriage.embed.llama import LlamaEmbedder
from logtriage.llm.base import LlmError, Message, ToolCall
from logtriage.llm.llama import LlamaClient


class _Handler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler's spelling
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length)
        server: Any = self.server
        server.requests.append((self.path, json.loads(raw) if raw else None))

        status, payload = server.responses.pop(0) if server.responses else (200, {})
        body = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args: Any) -> None:
        """Silence the per-request line; pytest output is the report here."""


class FakeServer:
    """A queue of canned responses and a record of what was asked for."""

    def __init__(self) -> None:
        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self._httpd.requests = []  # type: ignore[attr-defined]
        self._httpd.responses = []  # type: ignore[attr-defined]
        # serve_forever polls for the shutdown flag; the default half-second
        # interval is paid on every teardown, which over this file was thirteen
        # seconds of a fourteen-second run.
        self._thread = threading.Thread(
            target=self._httpd.serve_forever,
            kwargs={"poll_interval": 0.01},
            daemon=True,
        )
        self._thread.start()

    @property
    def url(self) -> str:
        host, port = self._httpd.server_address[:2]
        return f"http://{host}:{port}"

    @property
    def requests(self) -> list[tuple[str, Any]]:
        return self._httpd.requests  # type: ignore[attr-defined,no-any-return]

    def reply(self, payload: Any, status: int = 200) -> None:
        self._httpd.responses.append((status, payload))  # type: ignore[attr-defined]

    def close(self) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()


@pytest.fixture
def server() -> Iterator[FakeServer]:
    fake = FakeServer()
    yield fake
    fake.close()


@pytest.fixture(autouse=True)
def no_backoff(monkeypatch) -> None:
    """Keep the retry tests fast without disabling the retries themselves."""
    monkeypatch.setattr(transport.time, "sleep", lambda _seconds: None)


def _embeddings(*vectors: list[float]) -> dict[str, Any]:
    return {
        "data": [{"index": i, "embedding": vector} for i, vector in enumerate(vectors)]
    }


def _completion(
    content: str = "", tool_calls: list[dict[str, Any]] | None = None, **extra: Any
) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
    return {"choices": [{"message": message, **extra}]}


class TestEmbeddingRequests:
    def test_the_request_carries_the_batch_as_a_list(self, server) -> None:
        server.reply(_embeddings([1.0, 0.0], [0.0, 1.0]))
        LlamaEmbedder(server.url).embed(["alpha", "beta"])

        path, body = server.requests[0]
        assert path == "/v1/embeddings"
        assert body["input"] == ["alpha", "beta"]
        assert body["model"] == "default"

    def test_a_batch_larger_than_the_limit_is_split(self, server) -> None:
        """Indexing the real manual is 15,025 chunks. One request per chunk is
        an afternoon of TCP; one request for all of them exceeds what the server
        will accept."""
        server.reply(_embeddings([1.0, 0.0], [0.0, 1.0]))
        server.reply(_embeddings([1.0, 1.0]))

        vectors = LlamaEmbedder(server.url, batch_size=2).embed(["a", "b", "c"])

        assert len(server.requests) == 2
        assert server.requests[0][1]["input"] == ["a", "b"]
        assert server.requests[1][1]["input"] == ["c"]
        assert len(vectors) == 3

    def test_an_empty_batch_makes_no_request(self, server) -> None:
        assert LlamaEmbedder(server.url).embed([]) == []
        assert server.requests == []


class TestEmbeddingResponses:
    def test_vectors_come_back_in_index_order_not_arrival_order(self, server) -> None:
        """The OpenAI shape permits a server to answer out of order. Trusting
        position would pair every chunk with another chunk's vector -- an index
        that looks entirely normal and retrieves nonsense."""
        server.reply(
            {
                "data": [
                    {"index": 1, "embedding": [0.0, 1.0]},
                    {"index": 0, "embedding": [1.0, 0.0]},
                ]
            }
        )
        vectors = LlamaEmbedder(server.url).embed(["first", "second"])
        assert vectors == [[1.0, 0.0], [0.0, 1.0]]

    def test_vectors_are_normalised_to_unit_length(self, server) -> None:
        """embed.base.cosine is a bare dot product and documents that both
        embedders return unit vectors. llama.cpp normalises for some pooling
        types and not others, so the contract is kept here."""
        server.reply(_embeddings([3.0, 4.0]))
        (vector,) = LlamaEmbedder(server.url).embed(["x"])
        assert math.isclose(sum(v * v for v in vector), 1.0, rel_tol=1e-6)
        assert math.isclose(vector[0], 0.6, rel_tol=1e-6)

    def test_a_nested_vector_is_unwrapped(self, server) -> None:
        server.reply({"data": [{"index": 0, "embedding": [[1.0, 0.0]]}]})
        assert LlamaEmbedder(server.url).embed(["x"]) == [[1.0, 0.0]]

    def test_a_short_batch_is_an_error_not_a_silent_gap(self, server) -> None:
        server.reply(_embeddings([1.0, 0.0]))
        with pytest.raises(EmbeddingError, match="expected 2 embeddings"):
            LlamaEmbedder(server.url).embed(["a", "b"])

    def test_an_all_zero_vector_is_rejected(self, server) -> None:
        """It normalises to a division by zero, and stored as-is it would score
        identically against every query."""
        server.reply(_embeddings([0.0, 0.0]))
        with pytest.raises(EmbeddingError, match="all zeros"):
            LlamaEmbedder(server.url).embed(["x"])

    def test_dimensions_are_probed_once_and_cached(self, server) -> None:
        server.reply(_embeddings([1.0, 0.0, 0.0, 0.0]))
        embedder = LlamaEmbedder(server.url)
        assert embedder.dimensions == 4
        assert embedder.dimensions == 4
        assert len(server.requests) == 1

    def test_embedding_declares_itself_semantic(self, server) -> None:
        """search_docs reads this property to decide whether to weight the
        vector leg above zero, so it is the switch that turns hybrid search on."""
        assert LlamaEmbedder(server.url).semantic is True


class TestChatRequests:
    def test_tools_are_offered_with_tool_choice_auto(self, server) -> None:
        """'required' would be wrong: the last turn of the loop is the one that
        writes the report, and a model forced to call a tool can never take it."""
        server.reply(_completion("done"))
        tools = [{"type": "function", "function": {"name": "search_docs"}}]
        LlamaClient(server.url).complete([Message(role="user", content="hi")], tools)

        path, body = server.requests[0]
        assert path == "/v1/chat/completions"
        assert body["tools"] == tools
        assert body["tool_choice"] == "auto"
        assert body["stream"] is False
        assert body["temperature"] == 0.0

    def test_no_tool_choice_is_sent_when_no_tools_are_offered(self, server) -> None:
        server.reply(_completion("done"))
        LlamaClient(server.url).complete([Message(role="user", content="hi")])
        _, body = server.requests[0]
        assert "tools" not in body
        assert "tool_choice" not in body

    def test_a_prior_tool_call_is_sent_with_arguments_as_a_json_string(
        self, server
    ) -> None:
        """The OpenAI shape llama-server implements. Sending the parsed object
        instead is accepted by some servers and silently drops the call on
        others, which reads as a model that stopped using tools."""
        server.reply(_completion("done"))
        history = [
            Message(
                role="assistant",
                tool_calls=(ToolCall(id="c1", name="count", arguments={"ns": "db.c"}),),
            ),
            Message(role="tool", content="12", tool_call_id="c1", name="count"),
        ]
        LlamaClient(server.url).complete(history)

        _, body = server.requests[0]
        call = body["messages"][0]["tool_calls"][0]
        assert isinstance(call["function"]["arguments"], str)
        assert json.loads(call["function"]["arguments"]) == {"ns": "db.c"}
        assert body["messages"][1]["tool_call_id"] == "c1"


class TestChatResponses:
    def test_prose_comes_back_as_content(self, server) -> None:
        server.reply(_completion("The slow queries began first."))
        answer = LlamaClient(server.url).complete([Message(role="user", content="?")])
        assert answer.content == "The slow queries began first."
        assert answer.wants_tools is False

    def test_a_tool_call_is_parsed_from_its_json_string_arguments(self, server) -> None:
        server.reply(
            _completion(
                tool_calls=[
                    {
                        "id": "call_7",
                        "type": "function",
                        "function": {
                            "name": "search_docs",
                            "arguments": '{"query": "COLLSCAN", "k": 3}',
                        },
                    }
                ]
            )
        )
        answer = LlamaClient(server.url).complete([Message(role="user", content="?")])
        assert answer.wants_tools
        (call,) = answer.tool_calls
        assert call.id == "call_7"
        assert call.name == "search_docs"
        assert call.arguments == {"query": "COLLSCAN", "k": 3}

    def test_object_arguments_are_accepted_too(self, server) -> None:
        """Off-spec, but a template can emit it, and failing a whole run over a
        difference the caller cannot act on would be the wrong trade."""
        server.reply(
            _completion(
                tool_calls=[
                    {"function": {"name": "count", "arguments": {"ns": "db.c"}}}
                ]
            )
        )
        answer = LlamaClient(server.url).complete([Message(role="user", content="?")])
        assert answer.tool_calls[0].arguments == {"ns": "db.c"}

    def test_a_missing_call_id_is_synthesised_rather_than_dropping_the_call(
        self, server
    ) -> None:
        server.reply(
            _completion(tool_calls=[{"function": {"name": "count", "arguments": "{}"}}])
        )
        answer = LlamaClient(server.url).complete([Message(role="user", content="?")])
        assert answer.tool_calls[0].id == "call_0"

    def test_malformed_arguments_raise_rather_than_dispatching_an_empty_call(
        self, server
    ) -> None:
        """An empty dict would run the wrong query and report its answer as
        evidence, which is worse than failing."""
        server.reply(
            _completion(
                tool_calls=[{"function": {"name": "count", "arguments": "{not json"}}]
            )
        )
        with pytest.raises(LlmError, match="not valid JSON"):
            LlamaClient(server.url).complete([Message(role="user", content="?")])

    def test_truncation_is_raised_rather_than_returned(self, server) -> None:
        """A report cut off mid-sentence still parses, still renders, and its
        missing half is invisible to the citation check."""
        server.reply(_completion("The cause was", finish_reason="length"))
        with pytest.raises(LlmError, match="truncated"):
            LlamaClient(server.url).complete([Message(role="user", content="?")])

    def test_an_error_payload_returned_with_status_200_is_still_an_error(
        self, server
    ) -> None:
        server.reply({"error": {"message": "failed to apply chat template"}})
        with pytest.raises(LlmError, match="no choices"):
            LlamaClient(server.url).complete([Message(role="user", content="?")])


class TestTransportFailures:
    def test_a_server_error_is_retried_and_then_raised(self, server) -> None:
        server.reply({"error": "loading model"}, status=503)
        server.reply({"error": "loading model"}, status=503)
        server.reply({"error": "loading model"}, status=503)

        with pytest.raises(LlmError, match="HTTP 503"):
            LlamaClient(server.url).complete([Message(role="user", content="?")])
        # Two retries after the first attempt, per DEFAULT_RETRIES.
        assert len(server.requests) == 3

    def test_a_client_error_is_not_retried(self, server) -> None:
        """It will fail identically every time; retrying turns a clear error
        into a slow clear error."""
        server.reply({"error": "context window exceeded"}, status=400)
        with pytest.raises(LlmError, match="HTTP 400"):
            LlamaClient(server.url).complete([Message(role="user", content="?")])
        assert len(server.requests) == 1

    def test_a_retry_that_succeeds_returns_the_answer(self, server) -> None:
        server.reply({"error": "loading model"}, status=503)
        server.reply(_completion("ready now"))
        answer = LlamaClient(server.url).complete([Message(role="user", content="?")])
        assert answer.content == "ready now"

    def test_a_non_json_body_names_the_problem(self, server) -> None:
        """Usually something in front of the server answered instead of it."""
        server.reply(b"<html>502 Bad Gateway</html>")
        with pytest.raises(LlmError, match="not JSON"):
            LlamaClient(server.url).complete([Message(role="user", content="?")])

    def test_an_unreachable_server_is_reported_as_such(self) -> None:
        # Port 1 is reserved and nothing listens there.
        with pytest.raises(LlmError, match="unreachable"):
            LlamaClient("http://127.0.0.1:1").complete(
                [Message(role="user", content="?")]
            )

    def test_the_embedder_reports_transport_failure_as_an_embedding_error(self) -> None:
        with pytest.raises(EmbeddingError, match="unreachable"):
            LlamaEmbedder("http://127.0.0.1:1").embed(["x"])
