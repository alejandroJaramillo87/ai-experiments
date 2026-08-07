"""Unit tests for the traced CLI dispatch (roadmap OBS-4)."""

from __future__ import annotations

import argparse
import json
import logging
import time

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


def _ok_handler(args):
    return 0


def _usage_handler(args):
    return 2


def _none_handler(args):
    return None


def _raise_handler(args):
    raise RuntimeError("boom")


def _interrupt_handler(args):
    raise KeyboardInterrupt


def _sigterm_handler(args):
    raise SystemExit(1)


def _phase_handler(args):
    with core.phase("load"):
        time.sleep(0.01)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="toy")
    sub = parser.add_subparsers(dest="command", required=True)
    for name, func in (
        ("ok", _ok_handler),
        ("usage", _usage_handler),
        ("none", _none_handler),
        ("raise", _raise_handler),
        ("interrupt", _interrupt_handler),
        ("sigterm", _sigterm_handler),
        ("phased", _phase_handler),
    ):
        sub.add_parser(name).set_defaults(func=func)
    return parser


def read_records(tmp_path):
    files = list((tmp_path / "obs-logs" / "toycli").glob("R-*.jsonl"))
    assert len(files) == 1
    return [json.loads(line) for line in files[0].read_text().splitlines()]


def test_success_writes_start_and_end_with_shared_run_id(tmp_path):
    assert dispatch("toycli", build_parser, ["ok"]) == 0
    records = read_records(tmp_path)
    assert [r["event"] for r in records] == ["cmd.start", "run.context", "cmd.end"]
    start, _, end = records
    assert start["argv"] == ["ok"]
    assert start["cli"] == "toycli"
    assert start["command"] == "ok"
    assert end["exit_code"] == 0
    assert end["duration_s"] >= 0
    assert start["run_id"] == end["run_id"] != ""


def test_integer_return_passes_through(tmp_path):
    assert dispatch("toycli", build_parser, ["usage"]) == 2
    assert read_records(tmp_path)[-1]["exit_code"] == 2


def test_none_return_maps_to_zero(tmp_path):
    assert dispatch("toycli", build_parser, ["none"]) == 0
    assert read_records(tmp_path)[-1]["exit_code"] == 0


def test_uncaught_exception_records_exit_1_and_reraises(tmp_path):
    with pytest.raises(RuntimeError, match="boom"):
        dispatch("toycli", build_parser, ["raise"])
    end = read_records(tmp_path)[-1]
    assert end["event"] == "cmd.end"
    assert end["exit_code"] == 1


def test_keyboard_interrupt_records_130_and_reraises(tmp_path):
    with pytest.raises(KeyboardInterrupt):
        dispatch("toycli", build_parser, ["interrupt"])
    assert read_records(tmp_path)[-1]["exit_code"] == 130


def test_mid_run_system_exit_records_own_code(tmp_path):
    with pytest.raises(SystemExit):
        dispatch("toycli", build_parser, ["sigterm"])
    assert read_records(tmp_path)[-1]["exit_code"] == 1


def test_usage_error_exits_before_any_record(tmp_path):
    with pytest.raises(SystemExit):
        dispatch("toycli", build_parser, ["no-such-command"])
    assert not (tmp_path / "obs-logs").exists()


def test_uncreatable_log_root_degrades_to_console_only(tmp_path, monkeypatch):
    blocker = tmp_path / "not-a-dir"
    blocker.write_text("")
    monkeypatch.setenv("OBS_LOG_DIR", str(blocker / "logs"))
    assert dispatch("toycli", build_parser, ["ok"]) == 0


def test_phase_records_span_with_duration(tmp_path, capsys):
    assert dispatch("toycli", build_parser, ["phased"]) == 0
    records = read_records(tmp_path)
    assert [r["event"] for r in records] == [
        "cmd.start",
        "run.context",
        "phase.start",
        "phase.end",
        "cmd.end",
    ]
    assert records[2]["phase"] == "load"
    assert records[3]["phase"] == "load"
    assert records[3]["duration_s"] > 0
    # file-only spans: nothing is added to the console
    assert capsys.readouterr().err == ""
