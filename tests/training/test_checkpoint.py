"""Unit tests for atomic checkpoint save, retention, and resume resolution (Phase 0 D6).

The save/retention mechanics are exercised with stub model/optimizer/scheduler
objects so these tests stay fast; the full five-component round trip through a
real HF model is covered by tests/training/test_loop.py.
"""

import json
import os

import pytest

torch = pytest.importorskip("torch")

from training import checkpoint as ckpt  # noqa: E402


class StubModel:
    """Minimal save_pretrained-compatible stand-in for an HF model."""

    def save_pretrained(self, path: str, safe_serialization: bool = True) -> None:
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "model.safetensors"), "wb") as f:
            f.write(b"stub-weights")


class StubStateful:
    def state_dict(self) -> dict:
        return {"stub": 1}


def save_step(root, step: int) -> str:
    return ckpt.save_checkpoint(
        root, step, StubModel(), StubStateful(), StubStateful(),
        {"shard": 0, "offset": step, "window": 128},
        {"tokens_seen": step * 256, "loss": 1.0})


def test_save_creates_complete_checkpoint_and_latest(tmp_path):
    root = str(tmp_path / "checkpoints")
    path = save_step(root, 5)
    assert os.path.basename(path) == "step_000000005"
    for name in ("model/model.safetensors", "optimizer.pt", "scheduler.pt",
                 "dataloader.json", "rng.pt", "meta.json"):
        assert os.path.exists(os.path.join(path, name))
    with open(os.path.join(path, "meta.json")) as f:
        assert f.read() and json.load(open(os.path.join(path, "meta.json")))["step"] == 5
    assert ckpt.latest_checkpoint(root) == path
    assert ckpt.list_checkpoints(root) == [5]


def test_latest_repoints_on_each_save(tmp_path):
    root = str(tmp_path / "checkpoints")
    save_step(root, 5)
    path10 = save_step(root, 10)
    assert ckpt.latest_checkpoint(root) == path10
    assert ckpt.list_checkpoints(root) == [5, 10]


def test_kill_between_tmp_write_and_rename_preserves_previous_latest(
        tmp_path, monkeypatch):
    """Simulated SIGKILL after the tmp tree is written but before the rename
    publishes it: the previous latest stays intact and resumable, and the
    debris never appears as a checkpoint."""
    root = str(tmp_path / "checkpoints")
    path5 = save_step(root, 5)

    real_rename = os.rename

    def killed_rename(src, dst):
        if src.endswith(".tmp") and "step_" in os.path.basename(dst):
            raise KeyboardInterrupt("simulated SIGKILL")
        return real_rename(src, dst)

    monkeypatch.setattr(os, "rename", killed_rename)
    with pytest.raises(KeyboardInterrupt):
        save_step(root, 10)
    monkeypatch.undo()

    assert os.path.isdir(os.path.join(root, "step_000000010.tmp"))
    assert ckpt.list_checkpoints(root) == [5]
    assert ckpt.latest_checkpoint(root) == path5
    assert ckpt.resolve_resume(root, "latest") == path5

    # Retention sweeps the debris without touching the good checkpoint
    ckpt.apply_retention(root, retain_last=3, retain_every=100)
    assert not os.path.exists(os.path.join(root, "step_000000010.tmp"))
    assert ckpt.list_checkpoints(root) == [5]


def test_republish_replaces_orphan_step_dir(tmp_path, monkeypatch):
    """Crash between the publish rename and the `latest` repoint leaves an
    orphan step dir; re-reaching that step after resuming from the older
    latest must replace it rather than fail the rename with ENOTEMPTY."""
    root = str(tmp_path / "checkpoints")
    save_step(root, 5)

    real_symlink = os.symlink

    def killed_symlink(src, dst):
        raise KeyboardInterrupt("simulated SIGKILL")

    monkeypatch.setattr(os, "symlink", killed_symlink)
    with pytest.raises(KeyboardInterrupt):
        save_step(root, 10)
    monkeypatch.setattr(os, "symlink", real_symlink)

    # step_10 exists but latest still points at step_5 (the crash window)
    assert os.path.isdir(os.path.join(root, "step_000000010"))
    assert ckpt.latest_checkpoint(root).endswith("step_000000005")

    path10 = save_step(root, 10)  # resume re-reaches step 10
    assert ckpt.latest_checkpoint(root) == path10


def test_retention_keeps_last_n_plus_every_mth(tmp_path):
    root = str(tmp_path / "checkpoints")
    for step in (5, 10, 15, 20, 25):
        save_step(root, step)
    deleted = ckpt.apply_retention(root, retain_last=2, retain_every=10)
    # last two (20, 25) plus multiples of 10 (10, 20); 5 and 15 go, oldest first
    assert deleted == [5, 15]
    assert ckpt.list_checkpoints(root) == [10, 20, 25]


def test_retention_never_deletes_latest_target(tmp_path):
    root = str(tmp_path / "checkpoints")
    for step in (3, 7, 9):
        save_step(root, step)
    # Repoint latest at the oldest checkpoint, then apply a policy that would
    # otherwise delete it
    link = os.path.join(root, ckpt.LATEST_NAME)
    os.remove(link)
    os.symlink(ckpt.step_dir_name(3), link)
    ckpt.apply_retention(root, retain_last=1, retain_every=1000)
    assert 3 in ckpt.list_checkpoints(root)
    assert ckpt.latest_checkpoint(root) is not None


def test_resolve_resume_latest_requires_a_checkpoint(tmp_path):
    root = str(tmp_path / "checkpoints")
    with pytest.raises(FileNotFoundError):
        ckpt.resolve_resume(root, "latest")


def test_resolve_resume_explicit_path(tmp_path):
    root = str(tmp_path / "checkpoints")
    path = save_step(root, 5)
    assert ckpt.resolve_resume(root, path) == path
    with pytest.raises(FileNotFoundError):
        ckpt.resolve_resume(root, str(tmp_path / "not-a-checkpoint"))
