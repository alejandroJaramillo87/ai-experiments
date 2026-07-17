"""Tests for common/obs/core.py — the OBS-1 structured-logging core.

Covers the acceptance bullets (one parseable JSON line with the base fields; the
human line on stderr not stdout; AIX_LOG_LEVEL=warning suppresses info; the record
carries the bound run_id) plus the identity sanitizer seam and the console seam.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from common.obs import core
from common.obs.core import (
    configure,
    current_run_id,
    get_logger,
    new_run_id,
)

BASE_FIELDS = {"ts", "level", "run_id", "cli", "command", "pid", "event", "msg"}


def _read_records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def test_info_writes_one_parseable_json_line_with_base_fields(tmp_path):
    run_id = configure(
        "train", run_id="R-20260101-000000-abcdef", log_dir=str(tmp_path)
    )
    get_logger("x").info("hello", foo=1)
    records = _read_records(tmp_path / "train" / f"{run_id}.jsonl")
    assert len(records) == 1
    record = records[0]
    assert BASE_FIELDS <= set(record)
    assert record["msg"] == "hello"
    assert record["level"] == "INFO"
    assert record["foo"] == 1


def test_human_line_on_stderr_not_stdout(tmp_path, capsys):
    configure("train", log_dir=str(tmp_path))
    get_logger("x").info("hello")
    captured = capsys.readouterr()
    assert "[INFO]" in captured.err
    assert "hello" in captured.err
    assert captured.out == ""


def test_warning_level_suppresses_info(tmp_path, monkeypatch):
    monkeypatch.setenv("AIX_LOG_LEVEL", "warning")
    run_id = configure("train", log_dir=str(tmp_path))
    logger = get_logger("x")
    logger.info("dropped")
    logger.warning("kept")
    records = _read_records(tmp_path / "train" / f"{run_id}.jsonl")
    assert len(records) == 1
    assert records[0]["msg"] == "kept"
    assert records[0]["level"] == "WARNING"


def test_record_carries_bound_run_id(tmp_path):
    run_id = configure(
        "interp", run_id="R-20260101-000000-abc123", log_dir=str(tmp_path)
    )
    assert current_run_id() == "R-20260101-000000-abc123"
    get_logger("x").info("m")
    records = _read_records(tmp_path / "interp" / f"{run_id}.jsonl")
    assert records[0]["run_id"] == "R-20260101-000000-abc123"
    assert records[0]["cli"] == "interp"


def test_sanitizer_seam_is_identity(tmp_path):
    assert core._sanitize("s3cr3t") == "s3cr3t"
    assert core._sanitize({"k": "v"}) == {"k": "v"}
    assert core._sanitize([1, 2, 3]) == [1, 2, 3]
    run_id = configure("train", log_dir=str(tmp_path))
    get_logger("x").info("m", secret="s3cr3t")
    records = _read_records(tmp_path / "train" / f"{run_id}.jsonl")
    assert records[0]["secret"] == "s3cr3t"


def test_console_flag_suppresses_stderr_but_writes_file(tmp_path, capsys):
    run_id = configure("train", log_dir=str(tmp_path))
    get_logger("x").info("file-only", console=False)
    captured = capsys.readouterr()
    assert captured.err == ""
    assert captured.out == ""
    records = _read_records(tmp_path / "train" / f"{run_id}.jsonl")
    assert records[0]["msg"] == "file-only"


def test_non_serializable_field_coerced_with_repr(tmp_path):
    run_id = configure("train", log_dir=str(tmp_path))
    get_logger("x").info("m", obj=object())
    records = _read_records(tmp_path / "train" / f"{run_id}.jsonl")
    assert records[0]["obj"].startswith("<object")


def test_new_run_id_format():
    assert re.fullmatch(r"R-\d{8}-\d{6}-[0-9a-f]{6}", new_run_id())
