"""Cross-field recipe validation (Phase 0 D5).

validate(recipe) checks the invariants the type system cannot express and is
called by every entry point before any GPU allocation. It collects every
violation and raises one RecipeValidationError listing all of them, so a bad
recipe is fixed in one pass rather than one error at a time.

Invariants whose consumers are not built yet are stubbed with the field
contract in place:
- fp8 dimension support (D8): _check_fp8_dims is a no-op until the benchmark
  harness defines the supported-dims table.
- watchdog thresholds (D10): only positivity is checked here; semantics land
  with the watchdog.
"""

from __future__ import annotations

import os

from omegaconf import DictConfig, OmegaConf

from training.recipes.schema import (
    FP8_BACKENDS,
    OFFLOAD_MODES,
    OPTIMIZER_NAMES,
    PRECISION_DTYPES,
    SCHEDULE_TYPES,
)


class RecipeValidationError(ValueError):
    """One or more cross-field invariants failed; message lists all of them."""

    def __init__(self, problems: list[str]) -> None:
        self.problems = problems
        lines = "\n".join(f"  - {p}" for p in problems)
        super().__init__(f"Recipe validation failed ({len(problems)} problem(s)):\n{lines}")


def _check_missing(recipe: DictConfig, problems: list[str]) -> None:
    # OmegaConf raises on access of MISSING fields; surface them as one list.
    for key in OmegaConf.missing_keys(recipe):
        problems.append(f"{key}: required field is not set")


def _check_model(recipe: DictConfig, problems: list[str]) -> None:
    has_hf = recipe.model.hf is not None
    has_scratch = recipe.model.scratch is not None
    if has_hf == has_scratch:
        problems.append("model: exactly one of model.hf or model.scratch must be set")
    if has_scratch and not has_hf:
        s = recipe.model.scratch
        if not OmegaConf.is_missing(s, "heads") and not OmegaConf.is_missing(s, "kv_heads"):
            if s.kv_heads > s.heads or (s.kv_heads > 0 and s.heads % s.kv_heads != 0):
                problems.append(
                    f"model.scratch: heads ({s.heads}) must be a positive multiple of "
                    f"kv_heads ({s.kv_heads})")


def _check_optimizer(recipe: DictConfig, problems: list[str]) -> None:
    opt = recipe.optimizer
    if opt.name not in OPTIMIZER_NAMES:
        problems.append(f"optimizer.name: {opt.name!r} not in {OPTIMIZER_NAMES}")
    if not OmegaConf.is_missing(opt, "lr") and opt.lr <= 0:
        problems.append(f"optimizer.lr: must be positive, got {opt.lr}")
    if opt.name == "cpu_adam" and recipe.memory.offload == "none":
        problems.append("optimizer: cpu_adam requires memory.offload=zero_offload")


def _check_schedule(recipe: DictConfig, problems: list[str]) -> None:
    sched = recipe.schedule
    if sched.type not in SCHEDULE_TYPES:
        problems.append(f"schedule.type: {sched.type!r} not in {SCHEDULE_TYPES}")
    if sched.warmup_steps < 0:
        problems.append(f"schedule.warmup_steps: must be >= 0, got {sched.warmup_steps}")
    if not OmegaConf.is_missing(sched, "total_steps"):
        if sched.total_steps <= 0:
            problems.append(f"schedule.total_steps: must be positive, got {sched.total_steps}")
        elif sched.warmup_steps > sched.total_steps:
            problems.append(
                f"schedule: warmup_steps ({sched.warmup_steps}) exceeds "
                f"total_steps ({sched.total_steps})")


def _check_fp8_dims(recipe: DictConfig, problems: list[str]) -> None:
    # STUB (D8): warn-level dimension coverage checks land with the benchmark
    # harness's supported-dims table. Field contract is stable already.
    return


def _check_precision(recipe: DictConfig, problems: list[str]) -> None:
    prec = recipe.precision
    if prec.dtype not in PRECISION_DTYPES:
        problems.append(f"precision.dtype: {prec.dtype!r} not in {PRECISION_DTYPES}")
    if prec.dtype == "fp8":
        if prec.fp8_backend not in FP8_BACKENDS:
            problems.append(f"precision.fp8_backend: {prec.fp8_backend!r} not in {FP8_BACKENDS}")
        _check_fp8_dims(recipe, problems)


def _check_memory(recipe: DictConfig, problems: list[str]) -> None:
    if recipe.memory.offload not in OFFLOAD_MODES:
        problems.append(f"memory.offload: {recipe.memory.offload!r} not in {OFFLOAD_MODES}")


def _check_data(recipe: DictConfig, problems: list[str]) -> None:
    d = recipe.data
    for name in ("seq_len", "micro_batch", "grad_accum"):
        if d[name] < 1:
            problems.append(f"data.{name}: must be >= 1, got {d[name]}")
    if d.workers < 0:
        problems.append(f"data.workers: must be >= 0, got {d.workers}")


def _check_checkpoint(recipe: DictConfig, problems: list[str]) -> None:
    c = recipe.checkpoint
    for name in ("interval_steps", "retain_last", "retain_every"):
        if c[name] < 1:
            problems.append(f"checkpoint.{name}: must be >= 1, got {c[name]}")


def _check_watchdog(recipe: DictConfig, problems: list[str]) -> None:
    w = recipe.watchdog
    for name in ("loss_spike_factor", "gpu_temp_limit_c", "disk_free_floor_gb"):
        if w[name] <= 0:
            problems.append(f"watchdog.{name}: must be positive, got {w[name]}")


def _check_tokenizer_matches_shards(recipe: DictConfig, problems: list[str]) -> None:
    """When data.source is a prepared D2 shard directory, the tokenizer must
    hash-match the manifest. Skipped when no manifest is present (HF dataset id
    or shards not yet prepared on this machine)."""
    if OmegaConf.is_missing(recipe.data, "source") or OmegaConf.is_missing(
            recipe.tokenizer, "path"):
        return
    source = recipe.data.source
    manifest_path = os.path.join(source, "manifest.json")
    if not (os.path.isdir(source) and os.path.isfile(manifest_path)):
        return
    from training.data.prepare import load_tokenizer
    from training.data.shards import load_manifest

    manifest = load_manifest(source)
    manifest_hash = manifest.get("tokenizer_hash")
    try:
        tokenizer_hash = load_tokenizer(recipe.tokenizer.path).content_hash
    except Exception as exc:
        problems.append(f"tokenizer.path: cannot load {recipe.tokenizer.path}: {exc}")
        return
    if manifest_hash != tokenizer_hash:
        problems.append(
            f"tokenizer: hash {tokenizer_hash} does not match shard manifest "
            f"hash {manifest_hash} in {source} (retokenize or fix tokenizer.path)")


def validate(recipe: DictConfig) -> None:
    """Check every cross-field invariant; raise RecipeValidationError listing
    all violations. Entry points call this before any GPU allocation."""
    problems: list[str] = []
    _check_missing(recipe, problems)
    _check_model(recipe, problems)
    _check_optimizer(recipe, problems)
    _check_schedule(recipe, problems)
    _check_precision(recipe, problems)
    _check_memory(recipe, problems)
    _check_data(recipe, problems)
    _check_checkpoint(recipe, problems)
    _check_watchdog(recipe, problems)
    _check_tokenizer_matches_shards(recipe, problems)
    if problems:
        raise RecipeValidationError(problems)
