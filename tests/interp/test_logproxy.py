"""W3 logproxy unit tests for the pure record-shaping helpers: SSE
reassembly, non-streaming extraction, and partial records for malformed
bodies. The byte-equivalence acceptance check is host-side (needs the
services); these tests cover the logic that can regress silently."""

from __future__ import annotations

import importlib.util
import json
import os
import sys

import pytest

pytest.importorskip("fastapi", reason="fastapi required for logproxy import")
pytest.importorskip("httpx", reason="httpx required for logproxy import")

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
