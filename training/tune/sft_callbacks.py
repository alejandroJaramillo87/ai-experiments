"""TRL/transformers callback bridge for the F3 SFT entry point.

Bridges TRL's TrainerCallback API to the Phase 0 machinery instead of
duplicating it: metrics fan out through the D6 Tracker, step-level watchdog
signals go through the D10 Watchdog (a WatchdogAbort raised here propagates
out of trainer.train(); training/tune/sft.py takes the emergency checkpoint), and
D6-style retention runs after every save.

Checkpoint layout (resolved F3 decision 1): transformers owns the
checkpoint-<step>/ directories and their resume state. After each save a
callback writes a D6-style meta.json into the directory — the part F6 export
reads — applies retention, and updates a `latest` symlink via atomic rename.
transformers' writes are not atomic, so completeness is checked first: a
crash mid-save leaves a directory without meta.json, which resume, retention,
and export all ignore.

The pure helpers at module level are unit-testable without torch or
transformers; build_callbacks() imports transformers lazily and returns the
callback instances the trainer registers.
"""

from __future__ import annotations

import json
import os
import re
import time

from common.provenance import git_sha as _git_sha

CHECKPOINT_PREFIX = "checkpoint-"
_CHECKPOINT_RE = re.compile(r"^checkpoint-(\d+)$")
# transformers writes trainer_state.json last in a checkpoint save; its
# presence marks the directory as complete enough to stamp with meta.json.
TRAINER_STATE_FILE = "trainer_state.json"
META_NAME = "meta.json"
LATEST_LINK = "latest"


def checkpoint_dir_name(step: int) -> str:
    return f"{CHECKPOINT_PREFIX}{step}"


def list_trl_checkpoints(root: str) -> list[int]:
    """Steps of the complete (meta.json-stamped) checkpoints under root."""
    if not os.path.isdir(root):
        return []
    steps = []
    for name in os.listdir(root):
        m = _CHECKPOINT_RE.match(name)
        if m and os.path.isfile(os.path.join(root, name, META_NAME)):
            steps.append(int(m.group(1)))
    return sorted(steps)


def checkpoint_complete(ckpt_dir: str) -> bool:
    return os.path.isfile(os.path.join(ckpt_dir, TRAINER_STATE_FILE))


def _library_versions() -> dict:
    versions = {}
    for name in ("torch", "transformers", "trl", "peft"):
        try:
            versions[name] = __import__(name).__version__
        except Exception:
            versions[name] = "unavailable"
    return versions


def _attention_implementation() -> str:
    """Resolved attention path, or 'unavailable' where torch is absent."""
    try:
        from training.models.model import attention_implementation

        return attention_implementation()
    except Exception:
        return "unavailable"


def write_checkpoint_meta(
    ckpt_dir: str,
    step: int,
    *,
    adapter_only: bool,
    base_model: str,
    backend: str,
    recipe_hash: str,
    base_revision: str | None = None,
    base_config_sha256: str | None = None,
    emergency: bool = False,
    distill: dict | None = None,
) -> str:
    """Stamp a TRL checkpoint directory with the D6-style meta.json F6 reads.

    adapter_only tells export whether the directory holds a PEFT adapter to
    merge into base_model or full weights; base_revision and
    base_config_sha256 record the base identity resolved at load time so the
    F6 export preflight can detect base drift. distill carries the F5
    provenance fields (tier, teacher identity/revision or F4 manifest
    reference) so a distilled checkpoint cites its teacher from the meta
    alone. Written atomically (tmp + rename) so a partial meta.json can never
    mark an incomplete checkpoint complete.
    """
    meta = {
        "step": step,
        "checkpoint_format": "trl",
        "adapter_only": adapter_only,
        "base_model": base_model,
        "base_revision": base_revision,
        "base_config_sha256": base_config_sha256,
        "backend": backend,
        "recipe_hash": recipe_hash,
        "git_sha": _git_sha(),
        "library_versions": _library_versions(),
        "wall_clock": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        # Resolved attention path at save time (flash_attention_2 or the
        # sdpa fallback): provenance for the loaded-model configuration.
        "attention_implementation": _attention_implementation(),
    }
    if distill is not None:
        meta["distill"] = distill
    if emergency:
        meta["emergency"] = True  # exempt from retention deletion
    path = os.path.join(ckpt_dir, META_NAME)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(meta, f, indent=2)
    os.rename(tmp, path)
    return path


def _is_emergency(root: str, step: int) -> bool:
    path = os.path.join(root, checkpoint_dir_name(step), META_NAME)
    try:
        with open(path) as f:
            return bool(json.load(f).get("emergency"))
    except (OSError, ValueError):
        return False


def apply_trl_retention(root: str, retain_last: int, retain_every: int) -> list[int]:
    """D6 retention over checkpoint-<step>/ directories; returns deleted steps.

    Keeps the newest retain_last checkpoints plus every retain_every-th step,
    and never deletes emergency checkpoints or directories without meta.json
    (an in-progress or crashed save is not this callback's to remove).
    """
    import shutil

    steps = list_trl_checkpoints(root)
    keep = set(steps[-retain_last:]) if retain_last > 0 else set()
    if retain_every > 0:
        keep.update(s for s in steps if s % retain_every == 0)
    deleted = []
    for step in steps:
        if step in keep or _is_emergency(root, step):
            continue
        shutil.rmtree(os.path.join(root, checkpoint_dir_name(step)))
        deleted.append(step)
    return deleted


def update_latest(root: str, step: int) -> None:
    """Point root/latest at checkpoint-<step> via atomic rename."""
    target = checkpoint_dir_name(step)
    tmp = os.path.join(root, f".{LATEST_LINK}.tmp")
    if os.path.lexists(tmp):
        os.remove(tmp)
    os.symlink(target, tmp)
    os.replace(tmp, os.path.join(root, LATEST_LINK))


def latest_trl_checkpoint(root: str) -> str | None:
    """Newest complete checkpoint directory, or None."""
    steps = list_trl_checkpoints(root)
    if not steps:
        return None
    return os.path.join(root, checkpoint_dir_name(steps[-1]))


def tokens_per_second(tokens_per_step: int, step_delta: int, elapsed_s: float) -> float:
    if elapsed_s <= 0 or step_delta <= 0:
        return 0.0
    return tokens_per_step * step_delta / elapsed_s


def build_callbacks(
    recipe,
    ckpt_root: str,
    tracker,
    watchdog,
    recipe_hash: str,
    trainable_report=None,
    distill_meta: dict | None = None,
    base_identity=None,
) -> list:
    """Instantiate the trainer callbacks (imports transformers lazily).

    Returns [tracking, checkpoint] or [tracking, watchdog, checkpoint];
    ordering keeps metric fan-out ahead of a possible watchdog abort so the
    aborting step's metrics are recorded.
    """
    from transformers import TrainerCallback

    tokens_per_step = (
        recipe.data.micro_batch * recipe.data.grad_accum * recipe.data.seq_len
    )
    backend = recipe.sft.backend
    adapter_only = bool(recipe.peft.enabled)
    base_model = recipe.model.hf.id

    class TrackingCallback(TrainerCallback):
        """Fan trainer logs out through the D6 Tracker (D10 metrics list)."""

        def __init__(self) -> None:
            self._last_step = 0
            self._last_time = time.monotonic()
            self._reported = trainable_report is None

        def on_train_begin(self, args, state, control, **kwargs):
            # A resumed run starts at the checkpoint's global_step; measuring
            # the first window from 0 would report thousands of steps over
            # seconds and an absurd tokens_per_s.
            self._last_step = state.global_step
            self._last_time = time.monotonic()

        def on_log(self, args, state, control, logs=None, **kwargs):
            if not logs or tracker is None:
                return
            metrics = {}
            if not self._reported:
                metrics["train/trainable_params"] = trainable_report.trainable
                metrics["train/trainable_percent"] = trainable_report.percent
                self._reported = True
            for src, dst in (
                ("loss", "train/loss"),
                ("grad_norm", "train/grad_norm"),
                ("learning_rate", "train/lr"),
                ("eval_loss", "eval/holdout_loss"),
                ("ce_loss", "train/ce_loss"),
                ("kl_loss", "train/kl_loss"),
            ):
                if src in logs:
                    metrics[dst] = logs[src]
            now = time.monotonic()
            if "loss" in logs and state.global_step > self._last_step:
                metrics["train/tokens_per_s"] = tokens_per_second(
                    tokens_per_step,
                    state.global_step - self._last_step,
                    now - self._last_time,
                )
                self._last_step, self._last_time = state.global_step, now
            if metrics:
                metrics["train/backend"] = 1.0 if backend == "unsloth" else 0.0
                tracker.log(metrics, state.global_step)

    class WatchdogCallback(TrainerCallback):
        """Feed loss/grad_norm to the D10 watchdog; a WatchdogAbort raised
        by on_step propagates out of trainer.train() and training/tune/sft.py
        takes the emergency checkpoint."""

        def on_log(self, args, state, control, logs=None, **kwargs):
            if not logs or "loss" not in logs:
                return
            watchdog.on_step(
                state.global_step,
                float(logs["loss"]),
                float(logs.get("grad_norm", 0.0)),
            )

    class CheckpointCallback(TrainerCallback):
        """Stamp meta.json, apply D6 retention, update latest (decision 1)."""

        def on_save(self, args, state, control, **kwargs):
            ckpt_dir = os.path.join(ckpt_root, checkpoint_dir_name(state.global_step))
            if not checkpoint_complete(ckpt_dir):
                return  # partial save; leave unstamped so consumers ignore it
            write_checkpoint_meta(
                ckpt_dir,
                state.global_step,
                adapter_only=adapter_only,
                base_model=base_model,
                backend=backend,
                recipe_hash=recipe_hash,
                base_revision=getattr(base_identity, "revision", None),
                base_config_sha256=getattr(base_identity, "config_sha256", None),
                distill=distill_meta,
            )
            apply_trl_retention(
                ckpt_root, recipe.checkpoint.retain_last, recipe.checkpoint.retain_every
            )
            update_latest(ckpt_root, state.global_step)

    callbacks = [TrackingCallback()]
    if watchdog is not None:
        callbacks.append(WatchdogCallback())
    callbacks.append(CheckpointCallback())
    return callbacks
