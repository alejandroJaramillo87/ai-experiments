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

Data flows through the D9 prefetch DataLoader (training/data/loader.py):
sequential windows from a flat index, wrapping at corpus end (multi-epoch).
The authoritative resume state (dataloader.json) is the index of the next
window to consume — counted here as batches are trained on, never by the
loader's read-ahead — so multi-worker prefetch cannot desynchronize resume.
Pre-D9 cursor checkpoints convert on load. The loop measures data_wait_ms per
step (time blocked on the next batch), the D9 overlap acceptance metric, and
holds the D9 run lock for its duration.

ZeRO-Offload (P2-D5): memory.offload: zero_offload routes model, optimizer,
and checkpointing through a DeepSpeed engine (training/runtime/offload.py) — bf16
weights and gradients in VRAM, fp32 master weights and Adam moments in host
RAM, the optimizer step on CPUAdam. The LR scheduler stays loop-owned and
checkpoints keep the D6 layout with a deepspeed/ engine-state subdirectory.

Determinism: the loop seeds python/numpy/torch from run.seed on a fresh start
and restores full RNG state on resume. The resume acceptance test additionally
sets torch.use_deterministic_algorithms(True); the production default remains
nondeterministic for speed.
"""

from __future__ import annotations

import datetime
import hashlib
import os
import random
import sys
import time
from dataclasses import dataclass, field
from typing import cast

import numpy as np
from omegaconf import DictConfig

from common.obs import set_run_dir
from common.provenance import git_sha as _git_sha
from training.data.loader import CudaPrefetcher, build_dataloader, resume_index
from training.data.mixture import MixtureDataset
from training.data.shards import ShardedTokenDataset
from training.data.staging import stage_shards
from training.models.model import (
    attention_implementation,
    build_model,
    build_optimizer,
)
from training.recipes.loader import create_run_dir, resolved_yaml
from training.runtime import affinity as aff
from training.runtime import checkpoint as ckpt
from training.runtime.lock import run_lock
from training.runtime.schedules import build_scheduler
from training.runtime.tracking import Tracker
from training.runtime.watchdog import Watchdog, WatchdogAbort


@dataclass
class TrainResult:
    """Summary the CLI prints and the acceptance tests assert against."""

    run_dir: str
    checkpoint_root: str
    final_step: int
    tokens_seen: int
    losses: list[tuple[int, float]] = field(default_factory=list)  # (step, loss)
    lrs: list[tuple[int, float]] = field(
        default_factory=list
    )  # (step, lr) — P2-D3 acceptance
    data_wait_ms_mean: float = 0.0  # D9 overlap acceptance metric
    tokens_per_source: dict[str, int] = field(default_factory=dict)  # P2-D2 accounting


def snapshot_hash(recipe: DictConfig) -> str:
    """Hash of the resolved recipe with checkpoint.resume nulled: resuming a
    run necessarily changes that one field and must not trip the drift warning."""
    from omegaconf import OmegaConf

    clone = cast(
        DictConfig,
        OmegaConf.merge(recipe, OmegaConf.create({"checkpoint": {"resume": None}})),
    )
    return hashlib.sha256(resolved_yaml(clone).encode()).hexdigest()


def _library_versions() -> dict:
    import torch
    import transformers

    versions = {"torch": torch.__version__, "transformers": transformers.__version__}
    try:
        import deepspeed

        versions["deepspeed"] = deepspeed.__version__
    except ImportError:
        pass
    return versions


def _seed_everything(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class _DataStateSlot:
    """Captures dataloader.json on checkpoint load; the loop resumes from a
    flat window index instead of mutating the dataset's cursor."""

    def __init__(self) -> None:
        self.state: dict | None = None

    def load_state_dict(self, state: dict) -> None:
        self.state = state


def _sources_with_replay(
    recipe: DictConfig, domain: list[tuple], run_dir: str
) -> list[tuple]:
    """Expand the replay block into mixture sources (P2-D4).

    Domain weights are scaled to 1 - fraction of the total draw and each
    replay source enters at fraction * its relative share, served from a
    ReplayBuffer loaded into host RAM. The total byte budget (explicit
    replay.max_gb or the calculator-derived headroom) splits across replay
    sources by the same shares.
    """
    from training.continual.replay import ReplayBuffer, resolve_max_bytes

    fraction = recipe.replay.fraction
    domain_total = sum(w for _, _, w in domain)
    sources = [
        (name, path, w / domain_total * (1.0 - fraction)) for name, path, w in domain
    ]
    total_bytes = resolve_max_bytes(recipe)
    replay_total = sum(s.weight for s in recipe.replay.sources)
    for s in recipe.replay.sources:
        share = s.weight / replay_total
        buffer = ReplayBuffer(
            s.path,
            window=recipe.data.seq_len,
            max_bytes=int(total_bytes * share),
            seed=recipe.run.seed,
            run_dir=run_dir,
            name=s.name,
            ram_floor_gb=recipe.watchdog.ram_floor_gb,
        )
        sources.append((s.name, buffer, fraction * share))
    return sources


def train(recipe: DictConfig, stop_after: int | None = None) -> TrainResult:
    """Run the recipe to schedule.total_steps, checkpointing and tracking.

    stop_after halts the loop after that step without touching the schedule;
    the resume acceptance test uses it to interrupt a run at a checkpoint
    boundary while keeping total_steps (and thus the LR schedule) identical
    to the uninterrupted comparison run.
    """
    with run_lock(recipe.run.name):
        return _train_locked(recipe, stop_after)


def _train_locked(recipe: DictConfig, stop_after: int | None) -> TrainResult:
    import torch

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    resuming = recipe.checkpoint.resume is not None
    if not resuming:
        _seed_everything(recipe.run.seed)

    workers_mask = None
    workers_allowance = None
    if recipe.affinity.enabled:
        main_mask, workers_mask = aff.resolve_masks(recipe.affinity)
        # Captured before the main pin: forked workers must validate their
        # (disjoint) mask against the container allowance, not the main mask.
        workers_allowance = os.sched_getaffinity(0)
        aff.apply_mask(main_mask)

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
            f"{recipe.run.output_root}/{recipe.run.name}"
        )
    set_run_dir(run_name_dir)
    ckpt_root = f"{recipe.run.output_root}/{recipe.run.name}/checkpoints"

    # Built after the run dir exists: replay buffers write their loaded-set
    # manifests there (P2-D4).
    mixing = recipe.mixture.enabled
    dataset: MixtureDataset | ShardedTokenDataset
    if mixing:
        sources = [(s.name, s.path, s.weight) for s in recipe.mixture.sources]
        if recipe.replay.enabled:
            sources = _sources_with_replay(recipe, sources, run_name_dir)
        dataset = MixtureDataset(
            sources,
            window=recipe.data.seq_len,
            seed=recipe.run.seed,
            strict_single_epoch=recipe.mixture.strict_single_epoch,
        )
    else:
        source = recipe.data.source
        if recipe.staging.enabled:
            source = stage_shards(source, recipe.staging)
        dataset = ShardedTokenDataset(source, window=recipe.data.seq_len)
    offload = recipe.memory.offload == "zero_offload"
    effective_optimizer = recipe.optimizer.name
    engine = None
    if offload:
        # The chain degrades cpu_adam -> paged_adamw8bit -> adamw when the
        # pre-built DeepSpeed extension is unavailable; only cpu_adam runs
        # through the engine. The resolved name is stamped into checkpoint
        # meta below so a degraded run stays attributable.
        from training.runtime.offload import resolve_offload_optimizer

        effective_optimizer = resolve_offload_optimizer(recipe)
        offload = effective_optimizer == "cpu_adam"
    if offload:
        # ZeRO-Offload path (P2-D5): the engine owns device placement,
        # gradient accumulation, and the CPUAdam optimizer step; frozen
        # parameters never enter its optimizer groups. The LR scheduler
        # stays loop-owned (resolved P2-D5 decision 2) and steps the client
        # optimizer's param groups directly.
        from training.runtime.offload import initialize_engine

        model = build_model(recipe)
        engine = initialize_engine(model, recipe)
        device = engine.device
        client_optimizer = getattr(engine.optimizer, "optimizer", engine.optimizer)
        scheduler = build_scheduler(client_optimizer, recipe)
    else:
        model = build_model(recipe).to(device)
        params = None
        if recipe.memory.offload == "zero_offload":
            # Degraded from the engine path: keep the recipe's freeze policy
            # so the fallback optimizer trains the same parameter set.
            from training.runtime.offload import apply_trainable_filter

            params = apply_trainable_filter(model, recipe)
        optimizer = build_optimizer(
            model, recipe, name=effective_optimizer, params=params
        )
        scheduler = build_scheduler(optimizer, recipe)
    attn_implementation = attention_implementation()

    start_step, tokens_seen, next_index = 0, 0, 0
    # Cumulative tokens drawn per mixture source (P2-D2 accounting); carried
    # through checkpoint meta so a resumed run's final table stays cumulative.
    tokens_per_source = (
        np.zeros(len(dataset.names), dtype=np.int64)
        if isinstance(dataset, MixtureDataset)
        else None
    )
    recipe_hash = snapshot_hash(recipe)
    if resuming:
        path = ckpt.resolve_resume(ckpt_root, recipe.checkpoint.resume)
        slot = _DataStateSlot()
        if offload:
            meta = ckpt.load_checkpoint_deepspeed(path, engine, scheduler, slot)
        else:
            meta = ckpt.load_checkpoint(path, model, optimizer, scheduler, slot)
        next_index = resume_index(dataset, slot.state)
        start_step, tokens_seen = meta["step"], meta.get("tokens_seen", 0)
        if mixing:
            assert isinstance(dataset, MixtureDataset)
            saved = meta.get("tokens_per_source", {})
            tokens_per_source = np.array(
                [saved.get(name, 0) for name in dataset.names], dtype=np.int64
            )
        if meta.get("recipe_hash") not in (None, recipe_hash):
            print(
                f"[loop] WARN: resuming with a recipe that differs from the "
                f"checkpoint's (hash {meta['recipe_hash'][:12]} vs "
                f"{recipe_hash[:12]}); the LR schedule may shift",
                file=sys.stderr,
            )

    tracker = Tracker(
        recipe.tracking,
        recipe.run.name,
        run_name_dir,
        header={
            "tier": "pretrain",
            "model_id": (
                recipe.model.hf.id
                if recipe.model.hf
                else f"scratch-{recipe.model.scratch.layers}L"
                f"-{recipe.model.scratch.hidden}d"
            ),
            "recipe_hash": recipe_hash,
            "schedule": {
                "type": recipe.schedule.type,
                "warmup_steps": recipe.schedule.warmup_steps,
                "total_steps": recipe.schedule.total_steps,
                "decay_start_frac": recipe.schedule.decay_start_frac,
                "final_lr_frac": recipe.schedule.final_lr_frac,
                "decay_shape": recipe.schedule.decay_shape,
            },
        },
    )
    tokens_per_step = (
        recipe.data.micro_batch * recipe.data.grad_accum * recipe.data.seq_len
    )
    result = TrainResult(
        run_dir=run_name_dir,
        checkpoint_root=ckpt_root,
        final_step=start_step,
        tokens_seen=tokens_seen,
    )

    def save(step: int, loss: float, emergency: bool = False) -> None:
        meta = {
            "tokens_seen": tokens_seen,
            "loss": loss,
            "recipe_hash": recipe_hash,
            "git_sha": _git_sha(),
            "library_versions": _library_versions(),
            "wall_clock": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "checkpoint_format": "deepspeed" if offload else "torch",
            # Resolved (not requested) choices: fallbacks fire silently
            # otherwise and provenance would lie.
            "optimizer": effective_optimizer,
            "attention_implementation": attn_implementation,
        }
        if emergency:
            meta["emergency"] = True  # exempt from retention deletion
        # Consumed-window (or draw) index, not the loader's read-ahead
        # position: exact resume regardless of num_workers/prefetch depth.
        if mixing:
            assert isinstance(dataset, MixtureDataset)
            assert tokens_per_source is not None
            meta["tokens_per_source"] = {
                name: int(n)
                for name, n in zip(dataset.names, tokens_per_source, strict=True)
            }
            data_state = {
                "format": "mixture",
                "next_draw": next_index,
                "window": dataset.window,
            }
        else:
            data_state = {
                "format": "index",
                "next_index": next_index % len(dataset),
                "window": dataset.window,
            }
        if offload:
            ckpt.save_checkpoint_deepspeed(
                ckpt_root, step, engine, scheduler, data_state, meta
            )
        else:
            ckpt.save_checkpoint(
                ckpt_root, step, model, optimizer, scheduler, data_state, meta
            )
        ckpt.apply_retention(
            ckpt_root, recipe.checkpoint.retain_last, recipe.checkpoint.retain_every
        )

    watchdog = None
    if recipe.watchdog.enabled:
        os.makedirs(ckpt_root, exist_ok=True)  # disk-floor check needs the path
        watchdog = Watchdog(
            recipe.watchdog,
            run_name_dir,
            tracker=tracker,
            disk_path=ckpt_root,
            start_step=start_step,
        )
        watchdog.start()

    model.train()
    wait_total = 0.0
    batches = None
    # The watchdog thread and tracker are live from here on: dataloader or
    # prefetcher construction failures must still stop the thread and close
    # the tracker, or the process hangs with an orphaned sampler.
    try:
        worker_init = (
            aff.worker_init_fn(workers_mask, allowed=workers_allowance)
            if workers_mask
            else None
        )
        batches = CudaPrefetcher(
            build_dataloader(
                dataset,
                recipe.data,
                start_index=next_index,
                worker_init_fn=worker_init,
            ),
            device,
        )
        for step in range(start_step + 1, recipe.schedule.total_steps + 1):
            step_start = time.perf_counter()
            if not offload:
                optimizer.zero_grad(set_to_none=True)
            loss_sum, data_wait_s, opt_wait_s = 0.0, 0.0, 0.0
            for _ in range(recipe.data.grad_accum):
                batch = batches.next()
                data_wait_s += batches.last_wait_s
                next_index += recipe.data.micro_batch
                if mixing:
                    assert isinstance(dataset, MixtureDataset)
                    batch, tags = batch
                    tokens_per_source += recipe.data.seq_len * np.bincount(
                        tags.numpy(), minlength=len(dataset.names)
                    )
                if offload:
                    assert engine is not None
                    # The engine owns loss scaling, accumulation boundaries,
                    # and zero_grad: engine.step() is called every micro-batch
                    # and performs the CPUAdam step only on the last one
                    # (gradient_accumulation_steps in the generated config).
                    loss = engine(input_ids=batch, labels=batch).loss
                    loss_sum += loss.item()
                    engine.backward(loss)
                    opt_start = time.perf_counter()
                    engine.step()
                    opt_wait_s += time.perf_counter() - opt_start
                else:
                    loss = model(input_ids=batch, labels=batch).loss
                    loss_sum += loss.item()
                    (loss / recipe.data.grad_accum).backward()
            if offload:
                assert engine is not None
                # DeepSpeed computes the global norm during its step; None
                # (not yet available) reports as 0.0 rather than crashing.
                norm = engine.get_global_grad_norm()
                grad_norm = float(norm) if norm is not None else 0.0
            else:
                # Infinite max_norm measures the total gradient norm without clipping
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float("inf")
                ).item()
                optimizer.step()
            scheduler.step()

            tokens_seen += tokens_per_step
            step_loss = loss_sum / recipe.data.grad_accum
            step_time = time.perf_counter() - step_start
            metrics = {
                "loss": step_loss,
                "lr": scheduler.get_last_lr()[0],
                "grad_norm": grad_norm,
                "tokens_per_s": tokens_per_step / step_time,
                "step_time_s": step_time,
                "data_wait_ms": data_wait_s * 1000.0,  # D9 overlap metric
            }
            if offload:
                # CPUAdam-bound diagnosis (P2-D5): when this dominates
                # step_time_s, raise grad_accum before touching anything else.
                metrics["optimizer_step_s"] = opt_wait_s
            if mixing:
                assert isinstance(dataset, MixtureDataset)
                assert tokens_per_source is not None
                for name, n in zip(dataset.names, tokens_per_source, strict=True):
                    metrics[f"tokens_source_{name}"] = int(n)
                result.tokens_per_source = {
                    name: int(n)
                    for name, n in zip(dataset.names, tokens_per_source, strict=True)
                }
            if device.type == "cuda":
                metrics["gpu_mem_allocated_gb"] = (
                    torch.cuda.memory_allocated(device) / 2**30
                )
            tracker.log(metrics, step)
            result.losses.append((step, step_loss))
            result.lrs.append((step, scheduler.get_last_lr()[0]))
            result.final_step, result.tokens_seen = step, tokens_seen
            wait_total += data_wait_s * 1000.0
            result.data_wait_ms_mean = wait_total / len(result.losses)

            if watchdog is not None:
                try:
                    watchdog.on_step(step, step_loss, grad_norm)
                except WatchdogAbort:
                    # Best-effort final state for post-mortem and resume; the
                    # abort must surface even if the save itself fails.
                    try:
                        save(step, step_loss, emergency=True)
                    except Exception as exc:
                        print(
                            f"[loop] WARN: emergency checkpoint failed: {exc}",
                            file=sys.stderr,
                        )
                    raise

            is_last = step == recipe.schedule.total_steps
            if step % recipe.checkpoint.interval_steps == 0 or is_last:
                if watchdog is not None and watchdog.checkpoint_writes_blocked:
                    print(
                        f"[loop] WARN: skipping checkpoint at step {step}: "
                        f"disk free below watchdog floor",
                        file=sys.stderr,
                    )
                else:
                    save(step, step_loss)
                    # Dashboard marker (OBS-10); the emergency save above is
                    # not marked — the watchdog alert already records it.
                    tracker.log({"checkpoint_saved": 1}, step)
            if stop_after is not None and step >= stop_after:
                break
    finally:
        if watchdog is not None:
            watchdog.stop()
        if batches is not None:
            try:
                batches.close()  # release dataloader workers promptly
            except Exception as exc:
                # worker teardown failure must not skip tracker.close
                print(f"[loop] WARN: dataloader close failed: {exc}", file=sys.stderr)
        tracker.close()
    return result
