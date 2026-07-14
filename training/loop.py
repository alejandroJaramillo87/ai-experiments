"""Reference training loop (Phase 0 D6).

Deliberately small: a dense HF model (hub id or scratch spec), standard
forward/backward/step with gradient accumulation, driven entirely by a recipe.
It exists so Phase 0's checkpoint/resume/tracking facilities are testable and
later phases have a known-good skeleton; it is not the Phase 1 SFT trainer
(TRL is) nor the Phase 3 production loop.

Layout per run name (resolved D6 design decision 2):

    ${run.output_root}/${run.name}/
      <timestamp>/            per-invocation run dir: config snapshot, sinks
      checkpoints/            shared across invocations so resume=latest works

Data flows through ShardedTokenDataset.next_window() directly — the dataset
cursor is the authoritative resume state (dataloader.json). The D9 prefetch
DataLoader wraps this later. When the corpus is exhausted the cursor wraps to
the start (sequential multi-epoch reading; no sampler exists in Phase 0).

Determinism: the loop seeds python/numpy/torch from run.seed on a fresh start
and restores full RNG state on resume. The resume acceptance test additionally
sets torch.use_deterministic_algorithms(True); the production default remains
nondeterministic for speed.
"""

from __future__ import annotations

import datetime
import hashlib
import random
import subprocess
import sys
import time
from dataclasses import dataclass, field

import numpy as np
from omegaconf import DictConfig

from training import checkpoint as ckpt
from training.data.shards import ShardedTokenDataset
from training.model import build_model, build_optimizer, build_scheduler
from training.recipes.loader import create_run_dir, resolved_yaml
from training.tracking import Tracker


@dataclass
class TrainResult:
    """Summary the CLI prints and the acceptance tests assert against."""

    run_dir: str
    checkpoint_root: str
    final_step: int
    tokens_seen: int
    losses: list[tuple[int, float]] = field(default_factory=list)  # (step, loss)


def snapshot_hash(recipe: DictConfig) -> str:
    """Hash of the resolved recipe with checkpoint.resume nulled: resuming a
    run necessarily changes that one field and must not trip the drift warning."""
    from omegaconf import OmegaConf

    clone = OmegaConf.merge(recipe, OmegaConf.create({"checkpoint": {"resume": None}}))
    return hashlib.sha256(resolved_yaml(clone).encode()).hexdigest()


def _git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
            timeout=5, check=True).stdout.strip()
    except Exception:
        return "unknown"


def _library_versions() -> dict:
    import torch
    import transformers

    return {"torch": torch.__version__, "transformers": transformers.__version__}


def _seed_everything(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _next_batch(dataset: ShardedTokenDataset, micro_batch: int):
    """Stack micro_batch windows, wrapping the cursor at corpus end."""
    windows = []
    while len(windows) < micro_batch:
        window = dataset.next_window()
        if window is None:
            dataset.load_state_dict(
                {"shard": 0, "offset": 0, "window": dataset.window})
            continue
        windows.append(window.astype(np.int64))
    return np.stack(windows)


def train(recipe: DictConfig, stop_after: int | None = None) -> TrainResult:
    """Run the recipe to schedule.total_steps, checkpointing and tracking.

    stop_after halts the loop after that step without touching the schedule;
    the resume acceptance test uses it to interrupt a run at a checkpoint
    boundary while keeping total_steps (and thus the LR schedule) identical
    to the uninterrupted comparison run.
    """
    import torch

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    resuming = recipe.checkpoint.resume is not None
    if not resuming:
        _seed_everything(recipe.run.seed)

    dataset = ShardedTokenDataset(recipe.data.source, window=recipe.data.seq_len)
    model = build_model(recipe).to(device)
    optimizer = build_optimizer(model, recipe)
    scheduler = build_scheduler(optimizer, recipe)

    # A resume within the same wall-clock second as the interrupted run is
    # legitimate (the acceptance test does exactly that); nudge the timestamp
    # forward instead of failing on the collision.
    stamp = datetime.datetime.now()
    for _ in range(60):
        try:
            run_name_dir = create_run_dir(recipe, now=stamp)
            break
        except FileExistsError:
            stamp += datetime.timedelta(seconds=1)
    else:
        raise FileExistsError(
            f"Could not allocate a run directory under "
            f"{recipe.run.output_root}/{recipe.run.name}")
    ckpt_root = f"{recipe.run.output_root}/{recipe.run.name}/checkpoints"

    start_step, tokens_seen = 0, 0
    recipe_hash = snapshot_hash(recipe)
    if resuming:
        path = ckpt.resolve_resume(ckpt_root, recipe.checkpoint.resume)
        meta = ckpt.load_checkpoint(path, model, optimizer, scheduler, dataset)
        start_step, tokens_seen = meta["step"], meta.get("tokens_seen", 0)
        if meta.get("recipe_hash") not in (None, recipe_hash):
            print(f"[loop] WARN: resuming with a recipe that differs from the "
                  f"checkpoint's (hash {meta['recipe_hash'][:12]} vs "
                  f"{recipe_hash[:12]}); the LR schedule may shift", file=sys.stderr)

    tracker = Tracker(recipe.tracking, recipe.run.name, run_name_dir)
    tokens_per_step = (recipe.data.micro_batch * recipe.data.grad_accum
                       * recipe.data.seq_len)
    result = TrainResult(run_dir=run_name_dir, checkpoint_root=ckpt_root,
                         final_step=start_step, tokens_seen=tokens_seen)

    def save(step: int, loss: float) -> None:
        meta = {
            "tokens_seen": tokens_seen, "loss": loss,
            "recipe_hash": recipe_hash, "git_sha": _git_sha(),
            "library_versions": _library_versions(),
            "wall_clock": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "checkpoint_format": "torch",  # DeepSpeed runs record their own
        }
        ckpt.save_checkpoint(ckpt_root, step, model, optimizer, scheduler,
                             dataset.state_dict(), meta)
        ckpt.apply_retention(ckpt_root, recipe.checkpoint.retain_last,
                             recipe.checkpoint.retain_every)

    model.train()
    try:
        for step in range(start_step + 1, recipe.schedule.total_steps + 1):
            step_start = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            loss_sum = 0.0
            for _ in range(recipe.data.grad_accum):
                batch = torch.from_numpy(
                    _next_batch(dataset, recipe.data.micro_batch)).to(device)
                loss = model(input_ids=batch, labels=batch).loss
                loss_sum += loss.item()
                (loss / recipe.data.grad_accum).backward()
            # Infinite max_norm measures the total gradient norm without clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), float("inf")).item()
            optimizer.step()
            scheduler.step()

            tokens_seen += tokens_per_step
            step_loss = loss_sum / recipe.data.grad_accum
            step_time = time.perf_counter() - step_start
            metrics = {
                "loss": step_loss, "lr": scheduler.get_last_lr()[0],
                "grad_norm": grad_norm, "tokens_per_s": tokens_per_step / step_time,
                "step_time_s": step_time,
            }
            if device.type == "cuda":
                metrics["gpu_mem_allocated_gb"] = (
                    torch.cuda.memory_allocated(device) / 2**30)
            tracker.log(metrics, step)
            result.losses.append((step, step_loss))
            result.final_step, result.tokens_seen = step, tokens_seen

            is_last = step == recipe.schedule.total_steps
            if step % recipe.checkpoint.interval_steps == 0 or is_last:
                save(step, step_loss)
            if stop_after is not None and step >= stop_after:
                break
    finally:
        tracker.close()
    return result
