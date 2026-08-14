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


def test_live_locked_file_survives_regardless_of_age(tmp_path):
    import fcntl

    log_dir = tmp_path / "training"
    old = make_file(log_dir, "R-old-live.jsonl", age_s=90 * 86400)
    holder = open(old, "a")
    try:
        fcntl.flock(holder.fileno(), fcntl.LOCK_SH)
        prune(log_dir, 30)
        assert old.exists()
    finally:
        holder.close()
    # once the writer is gone the same file prunes normally
    prune(log_dir, 30)
    assert not old.exists()


def test_configure_holds_a_lock_on_its_event_file(tmp_path):
    import fcntl

    run_id = "R-20260718-000000-cccccc"
    core.configure("training", run_id=run_id, log_dir=str(tmp_path))
    path = tmp_path / "training" / f"{run_id}.jsonl"
    with open(path, "rb") as probe:
        with pytest.raises(OSError):
            fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)


def test_viewer_commands_do_not_prune(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("OBS_RETENTION_DAYS", "30")
    log_dir = tmp_path / "obs-logs" / "training"
    old = make_file(log_dir, "R-old-viewed.jsonl", age_s=90 * 86400)

    from training.cli import main as training_main

    assert training_main(["runs"]) == 0
    capsys.readouterr()
    assert old.exists(), "runs must not prune the files it lists"

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    assert training_main(["report", "--run", str(run_dir)]) == 1
    capsys.readouterr()
    assert old.exists(), "report must not prune the files it renders"

    # a non-viewer command still prunes
    core.configure("training", log_dir=str(tmp_path / "obs-logs"))
    assert not old.exists()


# The tests above hold the lock in the process that prunes. flock conflicts
# between two open file descriptions even within one process, so those are
# genuine conflicts rather than accidents -- but no test in this suite runs two
# processes, and two processes is the case retention exists for: one command
# pruning at startup while another run is still writing. The test below is the
# only one that crosses a process boundary.

LIVE_RUN = """
import fcntl, os, sys, time
from common.obs import core
core.configure("training", run_id=sys.argv[2], log_dir=sys.argv[1],
               run_retention=False)
path = os.path.join(sys.argv[1], "training", sys.argv[2] + ".jsonl")
# age the file past any window while still holding it open, which is exactly
# the shape of a long-idle run: old mtime, live writer.
old = time.time() - 90 * 86400
os.utime(path, (old, old))
print("ready", flush=True)
sys.stdin.readline()
"""


def test_a_live_run_in_another_process_survives_a_startup_prune(tmp_path):
    """The scenario retention is written for, across a real process boundary.

    A long-idle run's mtime ages past the window while its writer still holds
    the file. If a concurrently starting command pruned it, the rest of that
    run would be written to a deleted inode.
    """
    import subprocess
    import sys

    log_dir = tmp_path / "obs-logs"
    run_id = "R-20250101-000000-abcdef"
    live = subprocess.Popen(
        [sys.executable, "-c", LIVE_RUN, str(log_dir), run_id],
        cwd=Path(__file__).parents[2],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert live.stdout.readline().strip() == "ready"
        path = log_dir / "training" / f"{run_id}.jsonl"
        assert path.exists()
        assert path.stat().st_mtime < time.time() - 30 * 86400

        prune(log_dir / "training", 30)
        assert path.exists(), "a live run's file must survive another process's prune"
    finally:
        live.stdin.close()
        live.wait(timeout=10)

    # the writer is gone, so the same file is now an ordinary aged candidate
    prune(log_dir / "training", 30)
    assert not path.exists()
