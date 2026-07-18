"""Unit tests for the layer 2 resource audit hook (roadmap OBS-6)."""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys

import pytest

from common.obs import audit, core
from common.obs.dispatch import dispatch

# a stdlib module no test dependency imports, so the import event fires
SPARE_MODULE = "colorsys"


@pytest.fixture(autouse=True)
def clean_obs_state(monkeypatch, tmp_path):
    """Isolate env, context, handler, and audit-flag state per test."""
    monkeypatch.delenv("OBS_LOG_LEVEL", raising=False)
    monkeypatch.delenv("OBS_TRACE", raising=False)
    monkeypatch.setenv("OBS_LOG_DIR", str(tmp_path / "obs-logs"))
    token = core._context.set(core._RunContext())
    yield
    audit._disable()
    core._context.reset(token)
    logger = logging.getLogger("obs")
    for handler in list(logger.handlers):
        if not isinstance(handler, logging.NullHandler):
            logger.removeHandler(handler)
            handler.close()
    logger.setLevel(logging.NOTSET)


def _touch_resources(args):
    with open(args.marker, "w") as f:
        f.write("audited")
    sys.modules.pop(SPARE_MODULE, None)
    # the import audit event fires only for the import statement, not for
    # importlib.import_module
    import colorsys  # noqa: F401

    subprocess.run([sys.executable, "-c", "pass"], check=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="toy")
    sub = parser.add_subparsers(dest="command", required=True)
    touch = sub.add_parser("touch")
    touch.add_argument("--marker")
    touch.set_defaults(func=_touch_resources)
    return parser


def read_records(tmp_path):
    files = list((tmp_path / "obs-logs" / "toycli").glob("R-*.jsonl"))
    assert len(files) == 1
    return [json.loads(line) for line in files[0].read_text().splitlines()]


def run_touch(tmp_path) -> tuple[str, list[dict]]:
    marker = str(tmp_path / "marker-audit.txt")
    assert dispatch("toycli", build_parser, ["touch", "--marker", marker]) == 0
    return marker, read_records(tmp_path)


def test_trace_records_open_import_and_spawn(tmp_path, monkeypatch):
    monkeypatch.setenv("OBS_TRACE", "1")
    marker, records = run_touch(tmp_path)

    opens = [r for r in records if r["event"] == "open"]
    assert any(r["path"] == marker and "w" in r["mode"] for r in opens)

    imports = [r for r in records if r["event"] == "import"]
    assert any(r["module"] == SPARE_MODULE for r in imports)

    spawns = [r for r in records if r["event"] == "spawn"]
    assert any("-c" in r["argv"] for r in spawns)

    for record in opens + imports + spawns:
        assert record["level"] == "debug"
        assert record["run_id"] == records[0]["run_id"]


def test_trace_off_records_no_audit_events(tmp_path):
    _, records = run_touch(tmp_path)
    assert [r["event"] for r in records] == ["cmd.start", "cmd.end"]


def test_explicit_log_level_wins_over_trace(tmp_path, monkeypatch):
    monkeypatch.setenv("OBS_TRACE", "1")
    monkeypatch.setenv("OBS_LOG_LEVEL", "info")
    _, records = run_touch(tmp_path)
    assert [r["event"] for r in records] == ["cmd.start", "cmd.end"]


def test_hook_swallows_internal_errors(tmp_path, monkeypatch):
    def broken_emit(*args, **kwargs):
        raise RuntimeError("instrumentation bug")

    monkeypatch.setenv("OBS_TRACE", "1")
    monkeypatch.setattr(audit, "emit", broken_emit)
    marker, _ = run_touch(tmp_path)
    # the audited open succeeded even though every emit raised
    assert open(marker).read() == "audited"
