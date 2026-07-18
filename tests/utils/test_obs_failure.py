"""Unit tests for structured failure records (roadmap OBS-5)."""

from __future__ import annotations

import argparse
import json
import logging

import pytest

from common.obs import core
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


def _raise_handler(args):
    raise RuntimeError("checkpoint dir missing")


def _interrupt_handler(args):
    raise KeyboardInterrupt


def _sigterm_handler(args):
    raise SystemExit(1)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="toy")
    sub = parser.add_subparsers(dest="command", required=True)
    for name, func in (
        ("raise", _raise_handler),
        ("interrupt", _interrupt_handler),
        ("sigterm", _sigterm_handler),
    ):
        sub.add_parser(name).set_defaults(func=func)
    return parser


def read_records(tmp_path):
    files = list((tmp_path / "obs-logs" / "toycli").glob("R-*.jsonl"))
    assert len(files) == 1
    return [json.loads(line) for line in files[0].read_text().splitlines()]


def _deep_exception(depth: int) -> BaseException:
    """Build an exception whose traceback has ``depth`` distinct frames.

    Each generated function sits on its own line, so Python's recursive-
    frame collapsing never kicks in and the frame count is exact.
    """
    lines = []
    for i in range(depth - 1):
        lines.append(f"def f{i}():")
        lines.append(f"    f{i + 1}()")
    lines.append(f"def f{depth - 1}():")
    lines.append("    raise ValueError('deep')")
    namespace: dict = {}
    exec("\n".join(lines), namespace)
    try:
        namespace["f0"]()
    except ValueError as exc:
        return exc
    raise AssertionError("generated chain did not raise")


def test_uncaught_exception_writes_one_fail_record(tmp_path, capsys):
    with pytest.raises(RuntimeError):
        dispatch("toycli", build_parser, ["raise"])
    records = read_records(tmp_path)
    fails = [r for r in records if r["event"] == "fail"]
    assert len(fails) == 1
    fail = fails[0]
    assert fail["level"] == "error"
    assert fail["exc_type"] == "RuntimeError"
    assert fail["exc_msg"] == "checkpoint dir missing"
    assert fail["exit_code"] == 1
    assert "RuntimeError: checkpoint dir missing" in fail["traceback"]
    assert "_raise_handler" in fail["traceback"]
    # one run id ties cmd.start, fail, and cmd.end together
    assert {r["run_id"] for r in records} == {fail["run_id"]}
    assert [r["event"] for r in records] == ["cmd.start", "fail", "cmd.end"]
    # file-only: the interpreter prints the traceback after the re-raise,
    # so the record itself adds nothing to the console
    assert capsys.readouterr().err == ""


def test_record_failure_truncates_deep_traceback(tmp_path):
    core.configure("toycli", run_id="R-20260101-000000-abc123")
    core.bind_run("toycli", "raise", "R-20260101-000000-abc123")
    core.record_failure(_deep_exception(core.TRACEBACK_FRAME_LIMIT + 20), 1)
    records = read_records(tmp_path)
    assert len(records) == 1
    assert records[0]["traceback"].count('File "') == core.TRACEBACK_FRAME_LIMIT
    assert records[0]["traceback"].rstrip().endswith("ValueError: deep")


@pytest.mark.parametrize(
    ("command", "expected"),
    [("interrupt", KeyboardInterrupt), ("sigterm", SystemExit)],
)
def test_cancellation_produces_no_fail_record(tmp_path, command, expected):
    with pytest.raises(expected):
        dispatch("toycli", build_parser, [command])
    records = read_records(tmp_path)
    assert [r["event"] for r in records] == ["cmd.start", "cmd.end"]
