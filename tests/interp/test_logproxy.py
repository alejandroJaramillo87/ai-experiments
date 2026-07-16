"""W3 logproxy unit tests for the pure record-shaping helpers: SSE
reassembly, non-streaming extraction, and partial records for malformed
bodies, plus the FastAPI proxy route driven end to end with a monkeypatched
upstream. The full byte-equivalence acceptance check against a live service is
host-side; these tests cover the logic that can regress silently."""

from __future__ import annotations

import importlib.util
import json
import os
import sys

import httpx
import pytest

pytest.importorskip("fastapi", reason="fastapi required for logproxy import")
pytest.importorskip("httpx", reason="httpx required for logproxy import")

from fastapi.testclient import TestClient  # noqa: E402  (after importorskip)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(scope="module")
def logproxy():
    spec = importlib.util.spec_from_file_location(
        "logproxy",
        os.path.join(REPO_ROOT, "docker", "interp-logproxy", "logproxy.py"),
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["logproxy"] = module
    spec.loader.exec_module(module)
    return module


class TestExtraction:
    def test_non_streaming_chat(self, logproxy):
        body = {
            "choices": [{"message": {"content": "hello"}}],
            "usage": {"completion_tokens": 2},
        }
        text, usage = logproxy.extract_response(body)
        assert text == "hello"
        assert usage == {"completion_tokens": 2}

    def test_non_streaming_completions(self, logproxy):
        text, _ = logproxy.extract_response({"choices": [{"text": "raw"}]})
        assert text == "raw"

    def test_malformed_body(self, logproxy):
        assert logproxy.extract_response({"error": "boom"}) == (None, None)

    def test_streamed_sse_reassembly(self, logproxy):
        chunks = [
            b'data: {"choices": [{"delta": {"content": "hel"}}]}\n\n',
            b'data: {"choices": [{"delta": {"content": "lo"}}], '
            b'"usage": {"completion_tokens": 2}}\n\n',
            b"data: [DONE]\n\n",
        ]
        text, usage = logproxy.extract_streamed(chunks)
        assert text == "hello"
        assert usage == {"completion_tokens": 2}

    def test_streamed_partial_garbage(self, logproxy):
        # an interrupted stream logs what was seen, never raises
        chunks = [
            b'data: {"choices": [{"delta": {"content": "par"}}]}\n\n',
            b"data: {truncated",
        ]
        text, _ = logproxy.extract_streamed(chunks)
        assert text == "par"


class TestRecords:
    def test_base_record_fields(self, logproxy):
        record = logproxy.base_record(
            {"model": "m", "messages": [{"role": "user", "content": "q"}]}, 0.5
        )
        assert record["model"] == "m"
        assert record["messages"][0]["content"] == "q"
        assert record["latency_s"] == 0.5
        assert record["error"] is None

    def test_append_record_is_jsonl(self, logproxy, tmp_path, monkeypatch):
        monkeypatch.setattr(logproxy, "LOG_DIR", str(tmp_path))
        logproxy.append_record({"timestamp": "t", "model": "m"})
        files = os.listdir(tmp_path / logproxy.TARGET_SERVICE)
        assert len(files) == 1
        line = (tmp_path / logproxy.TARGET_SERVICE / files[0]).read_text()
        assert json.loads(line) == {"timestamp": "t", "model": "m"}


class FakeUpstream:
    """A stand-in for httpx's streaming response: serves the same bytes to the
    non-streaming (aread) and streaming (aiter_raw) relay paths so a test can
    assert the proxy relays them faithfully. Tracks whether it was closed."""

    def __init__(self, status_code=200, chunks=None, headers=None):
        self.status_code = status_code
        self._chunks = list(chunks or [])
        self.headers = httpx.Headers(headers or {"content-type": "application/json"})
        self.aclosed = False

    async def aread(self):
        return b"".join(self._chunks)

    async def aiter_raw(self):
        for chunk in self._chunks:
            yield chunk

    async def aclose(self):
        self.aclosed = True


def read_records(logproxy, log_dir):
    """Every JSONL record the proxy appended under log_dir this test."""
    directory = os.path.join(log_dir, logproxy.TARGET_SERVICE)
    if not os.path.isdir(directory):
        return []
    records = []
    for name in sorted(os.listdir(directory)):
        with open(os.path.join(directory, name)) as f:
            records.extend(json.loads(line) for line in f if line.strip())
    return records


class TestProxyRoute:
    """The proxy route end to end: upstream is the module's httpx client, its
    send() monkeypatched so no service is needed."""

    def test_non_streaming_passthrough_relays_and_records(
        self, logproxy, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(logproxy, "LOG_DIR", str(tmp_path))
        body = json.dumps(
            {
                "choices": [{"message": {"content": "hi there"}}],
                "usage": {"completion_tokens": 3},
            }
        ).encode()
        # A non-200 status proves the code is relayed from upstream, not defaulted.
        upstream = FakeUpstream(status_code=201, chunks=[body])

        async def fake_send(request, stream=False):
            return upstream

        monkeypatch.setattr(logproxy.client, "send", fake_send)

        with TestClient(logproxy.app) as client:
            resp = client.post(
                "/v1/chat/completions",
                json={"model": "m", "messages": [{"role": "user", "content": "q"}]},
            )
        assert resp.status_code == 201
        assert resp.content == body  # byte-for-byte passthrough
        assert upstream.aclosed  # the response was drained and closed

        records = read_records(logproxy, str(tmp_path))
        assert len(records) == 1
        assert records[0]["status_code"] == 201
        assert records[0]["response_text"] == "hi there"
        assert records[0]["usage"] == {"completion_tokens": 3}
        assert records[0]["model"] == "m"
        assert records[0]["error"] is None

    def test_upstream_connection_failure_returns_502_and_records(
        self, logproxy, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(logproxy, "LOG_DIR", str(tmp_path))

        async def fake_send(request, stream=False):
            raise httpx.ConnectError("connection refused")

        monkeypatch.setattr(logproxy.client, "send", fake_send)

        with TestClient(logproxy.app) as client:
            resp = client.post(
                "/v1/chat/completions",
                json={"model": "m", "messages": [{"role": "user", "content": "q"}]},
            )
        assert resp.status_code == 502
        assert "upstream unreachable" in resp.json()["error"]

        records = read_records(logproxy, str(tmp_path))
        assert len(records) == 1
        assert "upstream request failed" in records[0]["error"]
        assert records[0]["response_text"] is None
        assert records[0]["model"] == "m"

    def test_streamed_sse_relays_bytes_and_reassembles_record(
        self, logproxy, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(logproxy, "LOG_DIR", str(tmp_path))
        chunks = [
            b'data: {"choices": [{"delta": {"content": "hel"}}]}\n\n',
            b'data: {"choices": [{"delta": {"content": "lo"}}], '
            b'"usage": {"completion_tokens": 2}}\n\n',
            b"data: [DONE]\n\n",
        ]
        upstream = FakeUpstream(
            status_code=200,
            chunks=chunks,
            headers={"content-type": "text/event-stream"},
        )

        async def fake_send(request, stream=False):
            return upstream

        monkeypatch.setattr(logproxy.client, "send", fake_send)

        with TestClient(logproxy.app) as client:
            resp = client.post(
                "/v1/chat/completions",
                json={
                    "model": "m",
                    "stream": True,
                    "messages": [{"role": "user", "content": "q"}],
                },
            )
        assert resp.status_code == 200
        assert resp.content == b"".join(chunks)  # relayed byte-for-byte
        assert upstream.aclosed

        records = read_records(logproxy, str(tmp_path))
        assert len(records) == 1
        assert records[0]["response_text"] == "hello"  # reassembled from the SSE
        assert records[0]["usage"] == {"completion_tokens": 2}
        assert records[0]["status_code"] == 200
        assert records[0]["error"] is None
