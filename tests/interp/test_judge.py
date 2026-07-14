"""Unit tests for interp.miniprojects.common.judge (mocked endpoints, CPU only)."""

from __future__ import annotations

import json

import pytest
import requests

from interp.miniprojects.common import judge


class FakeResponse:
    def __init__(self, status_code=200, content="", json_body=None):
        self.status_code = status_code
        self._content = content
        self._json = json_body

    def json(self):
        if self._json is not None:
            return self._json
        return {"choices": [{"message": {"content": self._content}}]}

    def raise_for_status(self):
        if self.status_code >= 400:
            error = requests.HTTPError(f"{self.status_code}")
            error.response = self
            raise error


def _mock_health(monkeypatch, healthy=True):
    monkeypatch.setattr(
        judge.requests,
        "get",
        lambda url, timeout: FakeResponse(200 if healthy else 503),
    )


def test_parse_score_extracts_json():
    parsed = judge._parse_score('noise {"score": 7, "reason": "ok"} trailing')
    assert parsed["score"] == 7


def test_parse_score_rejects_scoreless():
    with pytest.raises(ValueError, match="lacks 'score'"):
        judge._parse_score('{"reason": "no score here"}')


def test_pick_endpoint_prefers_vllm(monkeypatch):
    _mock_health(monkeypatch, healthy=True)
    assert judge.pick_endpoint() == "http://localhost:8003/v1"


def test_pick_endpoint_raises_when_all_down(monkeypatch):
    _mock_health(monkeypatch, healthy=False)
    with pytest.raises(RuntimeError, match="no judge service is up"):
        judge.pick_endpoint()


def test_explicit_endpoint_healthcheck_failure(monkeypatch):
    _mock_health(monkeypatch, healthy=False)
    with pytest.raises(RuntimeError, match="failed its healthcheck"):
        judge.pick_endpoint("http://localhost:9999/v1")


def test_score_happy_path(monkeypatch):
    _mock_health(monkeypatch)
    replies = iter(['{"score": 8, "reason": "good"}', '{"score": 2, "reason": "bad"}'])
    monkeypatch.setattr(
        judge.requests,
        "post",
        lambda url, json, timeout: FakeResponse(content=next(replies)),
    )
    results = judge.score(["item a", "item b"], rubric="happiness 0-10")
    assert [r["score"] for r in results] == [8, 2]


def test_score_falls_back_on_response_format_rejection(monkeypatch):
    _mock_health(monkeypatch)
    calls = []

    def fake_post(url, json, timeout):
        calls.append(json)
        if "response_format" in json:
            return FakeResponse(status_code=400)
        return FakeResponse(content='{"score": 5}')

    monkeypatch.setattr(judge.requests, "post", fake_post)
    results = judge.score(["item"], rubric="r")
    assert results[0]["score"] == 5
    assert "response_format" not in calls[-1]


def test_score_survives_malformed_replies(monkeypatch):
    _mock_health(monkeypatch)
    monkeypatch.setattr(
        judge.requests,
        "post",
        lambda url, json, timeout: FakeResponse(content="not json at all"),
    )
    results = judge.score(["item"], rubric="r", retries=1)
    assert results[0]["score"] is None
    assert "error" in results[0]


def test_score_schema_is_valid_json():
    json.dumps(judge.SCORE_SCHEMA)
