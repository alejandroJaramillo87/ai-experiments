"""Unit tests for the observability logging core (roadmap OBS-1)."""

from __future__ import annotations

import json
import logging
import os
import re

import pytest

from common.obs import core

RUN_ID = "R-20260101-000000-abc123"


@pytest.fixture(autouse=True)
def clean_obs_state(monkeypatch):
    """Isolate env, context, and handler state so tests never leak."""
    monkeypatch.delenv("OBS_LOG_LEVEL", raising=False)
    monkeypatch.delenv("OBS_LOG_DIR", raising=False)
    token = core._context.set(core._RunContext())
    yield
    core._context.reset(token)
    logger = logging.getLogger("obs")
    for handler in list(logger.handlers):
        if not isinstance(handler, logging.NullHandler):
            logger.removeHandler(handler)
            handler.close()
    logger.setLevel(logging.NOTSET)


def _configure(tmp_path, **kwargs):
    core.configure("training", run_id=RUN_ID, log_dir=str(tmp_path), **kwargs)
    core.bind_run("training", "train-sft", RUN_ID)
    return tmp_path / "training" / f"{RUN_ID}.jsonl"


def test_info_writes_one_json_line_with_base_fields(tmp_path):
    log_file = _configure(tmp_path)
    core.get_logger("test").info("dataset ready", examples=3)
    lines = log_file.read_text().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["level"] == "info"
    assert record["run_id"] == RUN_ID
    assert record["cli"] == "training"
    assert record["command"] == "train-sft"
    assert record["pid"] == os.getpid()
    assert record["event"] == "log"
    assert record["msg"] == "dataset ready"
    assert record["examples"] == 3
    assert re.fullmatch(
        r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}[+-]\d{2}:\d{2}", record["ts"]
    )


def test_human_line_on_stderr_without_timestamp(tmp_path, capsys):
    _configure(tmp_path)
    core.get_logger("test").info("hello")
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "[INFO] hello\n"


def test_warning_renders_warn_tag(tmp_path, capsys):
    _configure(tmp_path)
    core.get_logger("test").warning("disk almost full")
    assert capsys.readouterr().err == "[WARN] disk almost full\n"


def test_obs_log_level_env_suppresses_info(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("OBS_LOG_LEVEL", "warning")
    log_file = _configure(tmp_path)
    log = core.get_logger("test")
    log.info("quiet")
    log.warning("loud")
    records = [json.loads(line) for line in log_file.read_text().splitlines()]
    assert [r["msg"] for r in records] == ["loud"]
    assert "quiet" not in capsys.readouterr().err


def test_record_carries_bound_run_id(tmp_path):
    log_file = _configure(tmp_path)
    core.bind_run("interp", "probe-train", "R-20260102-000000-def456")
    core.get_logger("test").info("bound")
    record = json.loads(log_file.read_text())
    assert record["run_id"] == "R-20260102-000000-def456"
    assert record["cli"] == "interp"
    assert record["command"] == "probe-train"
    assert core.current_run_id() == "R-20260102-000000-def456"


def test_new_run_id_format():
    assert re.fullmatch(r"R-\d{8}-\d{6}-[0-9a-f]{6}", core.new_run_id())
    assert core.new_run_id() != core.new_run_id()


def test_unserializable_field_coerced_with_repr(tmp_path):
    log_file = _configure(tmp_path)
    core.get_logger("test").info("x", obj=object())
    record = json.loads(log_file.read_text())
    assert "object object at" in record["obj"]


def test_set_run_dir_records_event_without_console_noise(tmp_path, capsys):
    log_file = _configure(tmp_path)
    run_dir = tmp_path / "run"
    core.set_run_dir(run_dir)
    assert core.current_run_dir() == str(run_dir)
    record = json.loads(log_file.read_text())
    assert record["event"] == "run.dir"
    assert record["path"] == str(run_dir)
    assert record["level"] == "info"
    assert capsys.readouterr().err == ""


def test_reconfigure_does_not_stack_handlers(tmp_path, capsys):
    _configure(tmp_path)
    log_file = _configure(tmp_path)
    core.get_logger("test").info("once")
    assert capsys.readouterr().err == "[INFO] once\n"
    assert len(log_file.read_text().splitlines()) == 1


def test_event_method_names_record(tmp_path):
    log_file = _configure(tmp_path)
    core.get_logger("test").event("dataset_loaded", examples=512)
    record = json.loads(log_file.read_text())
    assert record["event"] == "dataset_loaded"
    assert record["examples"] == 512
    assert record["msg"] == ""


def test_emit_console_false_writes_file_only(tmp_path, capsys):
    log_file = _configure(tmp_path)
    core.emit("error", "rendered elsewhere", console=False, status="ERROR")
    assert capsys.readouterr().err == ""
    record = json.loads(log_file.read_text())
    assert record["level"] == "error"
    assert record["status"] == "ERROR"


def test_unconfigured_core_is_silent(capsys):
    core.emit("error", "nobody listening")
    core.get_logger("test").warning("still nobody")
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_package_reexports_public_api():
    import common.obs as obs

    for name in obs.__all__:
        assert getattr(obs, name) is getattr(core, name)
