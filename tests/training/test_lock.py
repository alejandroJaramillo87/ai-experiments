"""D9 run lock: acquire/release, contention, staleness, and the serve_eval
preflight path. No GPU or torch required."""

import json
import os

import pytest

from training.runtime.lock import TrainLockError, read_lock, run_lock


def test_acquire_writes_and_release_removes(tmp_path):
    path = str(tmp_path / ".train.lock")
    with run_lock("tiny-e2e", path=path) as held:
        assert held == path
        contents = read_lock(path)
        assert contents["pid"] == os.getpid()
        assert contents["run"] == "tiny-e2e"
    assert not os.path.exists(path)


def test_release_happens_on_exception(tmp_path):
    path = str(tmp_path / ".train.lock")
    with pytest.raises(RuntimeError, match="boom"):
        with run_lock("tiny-e2e", path=path):
            raise RuntimeError("boom")
    assert not os.path.exists(path)


def test_second_acquire_refuses_while_held(tmp_path):
    path = str(tmp_path / ".train.lock")
    with run_lock("first", path=path):
        with pytest.raises(TrainLockError, match="one GPU consumer"):
            with run_lock("second", path=path):
                pass
    assert not os.path.exists(path)


def test_stale_lock_reports_manual_removal(tmp_path):
    path = str(tmp_path / ".train.lock")
    # A pid that cannot be alive: max_pid is bounded well below 2**22 default
    with open(path, "w") as f:
        json.dump({"pid": 2**22 + 1, "run": "crashed", "created": "x"}, f)
    with pytest.raises(TrainLockError, match=f"rm {path}"):
        with run_lock("next", path=path):
            pass
    assert os.path.exists(path)  # never auto-removed


def test_corrupt_lock_is_treated_as_stale(tmp_path):
    # A lock file with invalid JSON cannot name a live holder, so read_lock
    # maps it to an unknown, dead pid and run_lock reports it as stale. Like
    # any stale lock it is reported for manual removal, never auto-replaced.
    path = str(tmp_path / ".train.lock")
    with open(path, "w") as f:
        f.write("{ not valid json ")
    assert read_lock(path) == {"pid": None, "run": "unknown", "created": "unknown"}
    with pytest.raises(TrainLockError, match="Stale training lock"):
        with run_lock("next", path=path):
            pass
    assert os.path.exists(path)  # corrupt lock preserved, not auto-removed


def test_release_leaves_foreign_lock_alone(tmp_path):
    # If our lock file was replaced by another holder mid-run (operator error),
    # release must not delete the successor's lock.
    path = str(tmp_path / ".train.lock")
    with run_lock("mine", path=path):
        os.remove(path)
        with open(path, "w") as f:
            json.dump({"pid": os.getpid() + 1, "run": "other", "created": "x"}, f)
    assert os.path.exists(path)


def test_env_var_overrides_path(tmp_path, monkeypatch):
    path = str(tmp_path / "custom.lock")
    monkeypatch.setenv("TRAIN_LOCK_PATH", path)
    with run_lock("env-run"):
        assert os.path.exists(path)


def test_missing_parent_directory_is_a_clear_error(tmp_path):
    with pytest.raises(TrainLockError, match="does not exist"):
        with run_lock("x", path=str(tmp_path / "nope" / ".train.lock")):
            pass
