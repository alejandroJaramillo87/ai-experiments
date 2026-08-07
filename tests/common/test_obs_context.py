"""Unit tests for run-context capture (roadmap OBS-7)."""

from __future__ import annotations

import argparse
import json
import logging
import platform

import pytest

from common.obs import context, core
from common.obs.dispatch import dispatch


@pytest.fixture(autouse=True)
def clean_obs_state(monkeypatch, tmp_path):
    """Isolate env, context, and handler state so tests never leak."""
    monkeypatch.delenv("OBS_LOG_LEVEL", raising=False)
    monkeypatch.setenv("OBS_LOG_DIR", str(tmp_path / "obs-logs"))
    token = core._context.set(core._RunContext())
    yield
    core._context.reset(token)
    logger = logging.getLogger("obs")
    for handler in list(logger.handlers):
        if not isinstance(handler, logging.NullHandler):
            logger.removeHandler(handler)
            handler.close()
    logger.setLevel(logging.NOTSET)


def test_context_contains_git_sha_python_and_platform():
    ctx = context.run_context()
    assert ctx["git_sha"]
    assert ctx["python"] == platform.python_version()
    assert ctx["platform"] == platform.platform()
    assert isinstance(ctx["packages"], dict)


def test_missing_packages_are_omitted(monkeypatch):
    def fake_version(name):
        if name == "torch":
            return "2.9.1+cu130"
        raise context.PackageNotFoundError(name)

    monkeypatch.setattr(context, "version", fake_version)
    assert context.run_context()["packages"] == {"torch": "2.9.1+cu130"}


def test_gpu_failure_omits_key_without_exception(monkeypatch):
    monkeypatch.setattr(context, "_gpu_info", lambda: None)
    assert "gpu" not in context.run_context()


def test_gpu_query_never_raises():
    # With no GPU or pynvml (this sandbox) the query returns None; with a
    # real GPU it returns the name/driver pair. Either way it cannot raise.
    info = context._gpu_info()
    assert info is None or set(info) == {"name", "driver"}


def test_environment_values_are_never_captured(monkeypatch):
    monkeypatch.setenv("FAKE_API_TOKEN", "supersecretvalue123")
    assert "supersecretvalue123" not in json.dumps(context.run_context())


def _ok_handler(args):
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="toy")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("ok").set_defaults(func=_ok_handler)
    return parser


def test_dispatch_emits_run_context_after_cmd_start(tmp_path):
    assert dispatch("toycli", build_parser, ["ok"]) == 0
    files = list((tmp_path / "obs-logs" / "toycli").glob("R-*.jsonl"))
    assert len(files) == 1
    records = [json.loads(line) for line in files[0].read_text().splitlines()]
    assert [r["event"] for r in records] == ["cmd.start", "run.context", "cmd.end"]
    record = records[1]
    assert record["level"] == "info"
    assert record["git_sha"]
    assert record["python"] == platform.python_version()
    assert isinstance(record["packages"], dict)
    assert record["run_id"] == records[0]["run_id"]
