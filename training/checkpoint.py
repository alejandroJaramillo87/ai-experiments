"""Resumable checkpointing (Phase 0 D6).

One directory per step under <run_dir>/checkpoints/ (resolved D6 design
decision 2: checkpoints live inside the D5 run directory):

    step_000000123/
      model/            HF save_pretrained sharded safetensors (D7 consumes this)
      optimizer.pt      torch.save state_dict
      scheduler.pt      torch.save state_dict
      dataloader.json   ShardedTokenDataset cursor (shard, offset, window)
      rng.pt            python, numpy, torch CPU, torch CUDA states
      meta.json         step, tokens seen, loss, snapshot hash, git SHA,
                        library versions, wall-clock

ZeRO-Offload runs (P2-D5, meta checkpoint_format "deepspeed") replace
optimizer.pt with engine state written by DeepSpeed's checkpoint API into a
deepspeed/ subdirectory (fp32 master weights and Adam moments); model/,
scheduler.pt (the scheduler stays loop-owned), dataloader.json, rng.pt, and
meta.json are unchanged, so retention, resume resolution, and D7 export
treat both formats identically.

Atomicity: everything is written into step_X.tmp/, every file and the parent
directory are fsynced, then one os.rename publishes the checkpoint and a
`latest` symlink is updated (also via rename). A crash at any point leaves
either the previous complete state or an ignorable *.tmp directory — never a
half checkpoint that resume could pick up.

torch is imported lazily so the CPU sandbox can import this module.
"""

from __future__ import annotations

import json
import os
import random
import re
import shutil

import numpy as np

STEP_DIR_RE = re.compile(r"^step_(\d{9})$")
LATEST_NAME = "latest"


def step_dir_name(step: int) -> str:
    return f"step_{step:09d}"


def _fsync_path(path: str) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _fsync_tree(root: str) -> None:
    """fsync every file and directory under root, bottom-up."""
    for dirpath, _dirnames, filenames in os.walk(root, topdown=False):
        for name in filenames:
            _fsync_path(os.path.join(dirpath, name))
        _fsync_path(dirpath)


def _collect_rng_state() -> dict:
    import torch

    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": None,
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: dict) -> None:
    import torch

    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if state["torch_cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def list_checkpoints(root: str) -> list[int]:
    """Complete checkpoint steps under root, ascending. Ignores *.tmp debris."""
    if not os.path.isdir(root):
        return []
    steps = []
    for name in os.listdir(root):
        m = STEP_DIR_RE.match(name)
        if m and os.path.isdir(os.path.join(root, name)):
            steps.append(int(m.group(1)))
    return sorted(steps)


def latest_checkpoint(root: str) -> str | None:
    """Path the `latest` symlink resolves to, or None when absent/dangling."""
    link = os.path.join(root, LATEST_NAME)
    if not os.path.islink(link):
        return None
    target = os.path.join(root, os.readlink(link))
    return target if os.path.isdir(target) else None


def save_checkpoint(root: str, step: int, model, optimizer, scheduler,
                    dataset_state: dict, meta: dict) -> str:
    """Atomically write the step's checkpoint and repoint `latest`. Returns its path."""
    import torch

    os.makedirs(root, exist_ok=True)
    final = os.path.join(root, step_dir_name(step))
    tmp = final + ".tmp"
    if os.path.exists(tmp):
        shutil.rmtree(tmp)

    model.save_pretrained(os.path.join(tmp, "model"), safe_serialization=True)
    torch.save(optimizer.state_dict(), os.path.join(tmp, "optimizer.pt"))
    torch.save(scheduler.state_dict(), os.path.join(tmp, "scheduler.pt"))
    with open(os.path.join(tmp, "dataloader.json"), "w") as f:
        json.dump(dataset_state, f, indent=2)
    torch.save(_collect_rng_state(), os.path.join(tmp, "rng.pt"))
    with open(os.path.join(tmp, "meta.json"), "w") as f:
        json.dump({"step": step, **meta}, f, indent=2)

    _fsync_tree(tmp)
    # A crash between the publish rename and the `latest` repoint leaves an
    # orphan step dir; resuming from the older `latest` and re-reaching this
    # step would fail the rename with ENOTEMPTY. Replace the orphan.
    if os.path.exists(final):
        shutil.rmtree(final)
    os.rename(tmp, final)
    _fsync_path(root)

    # Repoint `latest` atomically: create a temp symlink, rename over the old one.
    link = os.path.join(root, LATEST_NAME)
    tmp_link = link + ".tmp"
    if os.path.lexists(tmp_link):
        os.remove(tmp_link)
    os.symlink(step_dir_name(step), tmp_link)
    os.rename(tmp_link, link)
    _fsync_path(root)
    return final


DEEPSPEED_TAG = "engine"  # DeepSpeed checkpoint tag under <step>/deepspeed/


def save_checkpoint_deepspeed(root: str, step: int, engine, scheduler,
                              dataset_state: dict, meta: dict) -> str:
    """Atomically write a ZeRO-Offload checkpoint and repoint `latest`.

    Same tmp-dir + fsync + rename protocol as save_checkpoint; the engine
    writes its partitioned optimizer state (fp32 masters, Adam moments)
    under deepspeed/, and the HF model/ export comes from engine.module so
    D7 export works from any retained checkpoint.
    """
    import torch

    os.makedirs(root, exist_ok=True)
    final = os.path.join(root, step_dir_name(step))
    tmp = final + ".tmp"
    if os.path.exists(tmp):
        shutil.rmtree(tmp)

    engine.module.save_pretrained(os.path.join(tmp, "model"), safe_serialization=True)
    engine.save_checkpoint(os.path.join(tmp, "deepspeed"), tag=DEEPSPEED_TAG)
    torch.save(scheduler.state_dict(), os.path.join(tmp, "scheduler.pt"))
    with open(os.path.join(tmp, "dataloader.json"), "w") as f:
        json.dump(dataset_state, f, indent=2)
    torch.save(_collect_rng_state(), os.path.join(tmp, "rng.pt"))
    with open(os.path.join(tmp, "meta.json"), "w") as f:
        json.dump({"step": step, **meta}, f, indent=2)

    _fsync_tree(tmp)
    # Same orphan window as save_checkpoint: replace a step dir left by a
    # crash between publish and the `latest` repoint.
    if os.path.exists(final):
        shutil.rmtree(final)
    os.rename(tmp, final)
    _fsync_path(root)

    link = os.path.join(root, LATEST_NAME)
    tmp_link = link + ".tmp"
    if os.path.lexists(tmp_link):
        os.remove(tmp_link)
    os.symlink(step_dir_name(step), tmp_link)
    os.rename(tmp_link, link)
    _fsync_path(root)
    return final


def load_checkpoint_deepspeed(path: str, engine, scheduler, dataset) -> dict:
    """Restore a ZeRO-Offload checkpoint in place; returns the meta dict.

    The engine restores module weights and offloaded optimizer state from
    deepspeed/; scheduler, dataloader, and RNG restoration match
    load_checkpoint so resume behavior is uniform across formats.
    """
    with open(os.path.join(path, "meta.json")) as f:
        meta = json.load(f)
    if meta.get("checkpoint_format") != "deepspeed":
        raise ValueError(
            f"Checkpoint at {path} has checkpoint_format="
            f"{meta.get('checkpoint_format')!r}; resuming it under ZeRO-Offload "
            "would silently reinitialize optimizer state. Match the recipe's "
            "memory.offload to the checkpoint's format.")

    import torch

    load_path, _client_state = engine.load_checkpoint(
        os.path.join(path, "deepspeed"), tag=DEEPSPEED_TAG,
        load_lr_scheduler_states=False)
    if load_path is None:
        raise FileNotFoundError(f"No DeepSpeed engine state under {path}/deepspeed")
    scheduler.load_state_dict(
        torch.load(os.path.join(path, "scheduler.pt"), weights_only=True))
    with open(os.path.join(path, "dataloader.json")) as f:
        dataset.load_state_dict(json.load(f))
    _restore_rng_state(torch.load(os.path.join(path, "rng.pt"), weights_only=False))
    return meta


def _is_emergency(root: str, step: int) -> bool:
    """True when the step's meta.json carries emergency=true (a D10 watchdog
    abort save). Unreadable or absent meta counts as not-emergency."""
    try:
        with open(os.path.join(root, step_dir_name(step), "meta.json")) as f:
            return bool(json.load(f).get("emergency"))
    except (OSError, ValueError):
        return False


def apply_retention(root: str, retain_last: int, retain_every: int) -> list[int]:
    """Delete checkpoints beyond the retention policy; returns deleted steps.

    Keeps the last retain_last checkpoints plus every step divisible by
    retain_every; never deletes the checkpoint `latest` points to, nor any
    checkpoint marked emergency=true (D10 abort saves are post-mortem
    evidence and the resume point after a crash). Deletion is oldest first.
    Stray *.tmp directories are always removed.
    """
    steps = list_checkpoints(root)
    if os.path.isdir(root):
        for name in os.listdir(root):
            if name.endswith(".tmp") and os.path.isdir(os.path.join(root, name)):
                shutil.rmtree(os.path.join(root, name))

    keep = set(steps[-retain_last:]) if retain_last > 0 else set()
    keep.update(s for s in steps if s % retain_every == 0)
    latest = latest_checkpoint(root)
    if latest is not None:
        m = STEP_DIR_RE.match(os.path.basename(latest))
        if m:
            keep.add(int(m.group(1)))

    deleted = []
    for s in steps:
        if s not in keep and not _is_emergency(root, s):
            shutil.rmtree(os.path.join(root, step_dir_name(s)))
            deleted.append(s)
    return deleted


def resolve_resume(root: str, resume: str) -> str:
    """Map checkpoint.resume ('latest' or a path) to a checkpoint directory."""
    if resume == "latest":
        path = latest_checkpoint(root)
        if path is None:
            raise FileNotFoundError(
                f"checkpoint.resume=latest but no checkpoint exists under {root}")
        return path
    if not os.path.isdir(resume) or not os.path.isfile(
            os.path.join(resume, "meta.json")):
        raise FileNotFoundError(f"Not a checkpoint directory: {resume}")
    return resume


def load_checkpoint(path: str, model, optimizer, scheduler, dataset) -> dict:
    """Restore all five components in place; returns the checkpoint's meta dict."""
    import torch
    from safetensors.torch import load_file

    model_dir = os.path.join(path, "model")
    shards = [f for f in os.listdir(model_dir) if f.endswith(".safetensors")]
    state = {}
    for shard in shards:
        state.update(load_file(os.path.join(model_dir, shard)))
    model.load_state_dict(state, strict=False)  # tied weights are re-derived
    model.tie_weights()
    optimizer.load_state_dict(
        torch.load(os.path.join(path, "optimizer.pt"), weights_only=True))
    scheduler.load_state_dict(
        torch.load(os.path.join(path, "scheduler.pt"), weights_only=True))
    with open(os.path.join(path, "dataloader.json")) as f:
        dataset.load_state_dict(json.load(f))
    _restore_rng_state(torch.load(os.path.join(path, "rng.pt"), weights_only=False))
    with open(os.path.join(path, "meta.json")) as f:
        return json.load(f)
