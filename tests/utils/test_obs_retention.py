"""Unit tests for event-file retention (roadmap OBS-8)."""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path

import pytest

from common.obs import core
from common.obs.retention import prune

# comfortably past the default 30-day window
OLD_AGE_S = 40 * 86400


@pytest.fixture(autouse=True)
def clean_obs_state(monkeypatch, tmp_path):
    """Isolate env, context, and handler state so tests never leak."""
    monkeypatch.delenv("OBS_LOG_LEVEL", raising=False)
    monkeypatch.delenv("OBS_RETENTION_DAYS", raising=False)
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


def make_file(log_dir: Path, name: str, age_s: int = 0) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / name
    path.write_text("{}\n")
    if age_s:
        stamp = time.time() - age_s
        os.utime(path, (stamp, stamp))
    return path


def test_old_event_file_is_removed(tmp_path):
    old = make_file(tmp_path, "R-20250101-000000-aaaaaa.jsonl", OLD_AGE_S)
    prune(tmp_path, 30)
    assert not old.exists()


def test_recent_event_file_is_kept(tmp_path):
    recent = make_file(tmp_path, "R-20260718-000000-bbbbbb.jsonl")
    prune(tmp_path, 30)
    assert recent.exists()


def test_catalog_survives_regardless_of_age(tmp_path):
    catalog = make_file(tmp_path, "runs.jsonl", OLD_AGE_S)
    prune(tmp_path, 30)
    assert catalog.exists()


def test_foreign_file_is_never_a_candidate(tmp_path):
    foreign = make_file(tmp_path, "notes.jsonl", OLD_AGE_S)
    prune(tmp_path, 30)
    assert foreign.exists()


def test_unremovable_file_is_skipped_not_fatal(tmp_path, monkeypatch):
    make_file(tmp_path, "R-20250101-000000-cccccc.jsonl", OLD_AGE_S)

    def deny(self, missing_ok=False):
        raise OSError("permission denied")

    monkeypatch.setattr(Path, "unlink", deny)
    prune(tmp_path, 30)


def test_nonpositive_window_disables_pruning(tmp_path):
    old = make_file(tmp_path, "R-20250101-000000-dddddd.jsonl", OLD_AGE_S)
    prune(tmp_path, 0)
    assert old.exists()


def test_missing_directory_is_not_fatal(tmp_path):
    prune(tmp_path / "absent", 30)


def test_configure_prunes_the_cli_directory(tmp_path):
    log_dir = tmp_path / "obs-logs" / "toycli"
    old = make_file(log_dir, "R-20250101-000000-eeeeee.jsonl", OLD_AGE_S)
    run_id = core.configure("toycli")
    assert not old.exists()
    assert (log_dir / f"{run_id}.jsonl").exists()


def test_invalid_retention_value_disables_pruning(tmp_path, monkeypatch):
    monkeypatch.setenv("OBS_RETENTION_DAYS", "soon")
    log_dir = tmp_path / "obs-logs" / "toycli"
    old = make_file(log_dir, "R-20250101-000000-ffffff.jsonl", OLD_AGE_S)
    core.configure("toycli")
    assert old.exists()
