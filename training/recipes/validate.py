"""Cross-field recipe validation (Phase 0 D5).

validate(recipe) checks the invariants the type system cannot express and is
called by every entry point before any GPU allocation. It collects every
violation and raises one RecipeValidationError listing all of them, so a bad
recipe is fixed in one pass rather than one error at a time.

Invariants whose consumers are not built yet are stubbed with the field
contract in place:
- fp8 dimension support (D8): _check_fp8_dims is a no-op until the benchmark
  harness defines the supported-dims table.
"""

from __future__ import annotations

import math
import os

from omegaconf import DictConfig, ListConfig, OmegaConf

from training.recipes.schema import (
    COMPUTE_DTYPES,
    DISTILL_MODES,
    DISTILL_TIERS,
    FP8_BACKENDS,
    KL_DIRECTIONS,
    OFFLOAD_MODES,
    OPTIMIZER_NAMES,
    ORGANISM_KINDS,
    PEFT_METHODS,
    PRECISION_DTYPES,
    QUANTIZE_BASE_MODES,
    SCHEDULE_TYPES,
    SFT_BACKENDS,
    TARGET_MODULES_AUTO,
    WSD_DECAY_SHAPES,
)
from training.schedules import (
    WSD_DECAY_SHAPE_DEFAULT,
    WSD_DECAY_START_FRAC_DEFAULT,
    WSD_FINAL_LR_FRAC_DEFAULT,
)


class RecipeValidationError(ValueError):
    """One or more cross-field invariants failed; message lists all of them."""

    def __init__(self, problems: list[str]) -> None:
        self.problems = problems
        lines = "\n".join(f"  - {p}" for p in problems)
        super().__init__(f"Recipe validation failed ({len(problems)} problem(s)):\n{lines}")


def _check_missing(recipe: DictConfig, problems: list[str]) -> None:
    # OmegaConf raises on access of MISSING fields; surface them as one list.
    # data.source is intentionally unset when the mixture block replaces it
    # (P2-D2); _check_mixture enforces the mutual exclusion.
    skip = {"data.source"} if recipe.mixture.enabled else set()
    for key in OmegaConf.missing_keys(recipe):
        if key in skip:
            continue
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
    if sched.type == "wsd":
        _check_wsd(recipe, problems)


def _check_wsd(recipe: DictConfig, problems: list[str]) -> None:
    """P2-D3 invariants: wsd needs total_steps, warmup must end before the
    decay window opens, frac knobs must be in range, and non-default knobs
    are pretrain-loop-only (the sft path maps wsd onto transformers'
    warmup_stable_decay, which has no decay shape or LR floor)."""
    sched = recipe.schedule
    if OmegaConf.is_missing(sched, "total_steps"):
        problems.append("schedule.total_steps: required when type is wsd")
    if not 0.0 < sched.decay_start_frac <= 1.0:
        problems.append(f"schedule.decay_start_frac: must be in (0, 1], "
                        f"got {sched.decay_start_frac}")
    if not 0.0 <= sched.final_lr_frac < 1.0:
        problems.append(f"schedule.final_lr_frac: must be in [0, 1), "
                        f"got {sched.final_lr_frac}")
    if sched.decay_shape not in WSD_DECAY_SHAPES:
        problems.append(f"schedule.decay_shape: {sched.decay_shape!r} "
                        f"not in {WSD_DECAY_SHAPES}")
    if (not OmegaConf.is_missing(sched, "total_steps")
            and 0.0 < sched.decay_start_frac <= 1.0
            and sched.warmup_steps >= int(sched.total_steps * sched.decay_start_frac)):
        problems.append(
            f"schedule: warmup_steps ({sched.warmup_steps}) reaches the wsd "
            f"decay start ({int(sched.total_steps * sched.decay_start_frac)}); "
            "no stable plateau remains")
    non_default = (
        sched.decay_start_frac != WSD_DECAY_START_FRAC_DEFAULT
        or sched.final_lr_frac != WSD_FINAL_LR_FRAC_DEFAULT
        or sched.decay_shape != WSD_DECAY_SHAPE_DEFAULT)
    if recipe.sft.enabled and non_default:
        problems.append(
            "schedule: wsd knobs (decay_start_frac, final_lr_frac, "
            "decay_shape) are not applied by the sft path; remove them or "
            "run through the pretrain loop")


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


def _check_peft(recipe: DictConfig, problems: list[str]) -> None:
    """F2 invariants: field domains plus the QLoRA combination rules
    (nf4 requires an adapter — no full fine-tuning of a 4-bit base — and
    fp8 precision is mutually exclusive with nf4 base quantization)."""
    p = recipe.peft
    if p.method not in PEFT_METHODS:
        problems.append(f"peft.method: {p.method!r} not in {PEFT_METHODS}")
    if p.quantize_base not in QUANTIZE_BASE_MODES:
        problems.append(
            f"peft.quantize_base: {p.quantize_base!r} not in {QUANTIZE_BASE_MODES}")
    if p.compute_dtype not in COMPUTE_DTYPES:
        problems.append(
            f"peft.compute_dtype: {p.compute_dtype!r} not in {COMPUTE_DTYPES}")
    if p.rank < 1:
        problems.append(f"peft.rank: must be >= 1, got {p.rank}")
    if p.alpha <= 0:
        problems.append(f"peft.alpha: must be positive, got {p.alpha}")
    if not 0.0 <= p.dropout < 1.0:
        problems.append(f"peft.dropout: must be in [0, 1), got {p.dropout}")
    modules = p.target_modules
    if modules != TARGET_MODULES_AUTO:
        if not isinstance(modules, (list, ListConfig)) or len(modules) == 0 or not all(
                isinstance(m, str) and m for m in modules):
            problems.append(
                "peft.target_modules: must be 'auto' or a non-empty list of "
                f"module names, got {modules!r}")
    if p.quantize_base == "nf4":
        if not p.enabled:
            problems.append(
                "peft.quantize_base: nf4 requires peft.enabled=true "
                "(no full fine-tuning of a 4-bit base)")
        if recipe.precision.dtype == "fp8":
            problems.append(
                "peft: quantize_base=nf4 and precision.dtype=fp8 are mutually "
                "exclusive (pick one weight-compression scheme)")


def _check_sft(recipe: DictConfig, problems: list[str]) -> None:
    """F1/F3 invariants: an enabled sft group needs a prepared dataset and an
    HF base model (the chat template belongs to a pretrained model; scratch
    models have none). Manifest and tokenizer hashes are enforced at load by
    training/data/sft.load_sft_dataset."""
    s = recipe.sft
    if not s.enabled:
        return
    if s.backend not in SFT_BACKENDS:
        problems.append(f"sft.backend: must be one of {SFT_BACKENDS}, "
                        f"got {s.backend!r}")
    if s.eval_steps < 0:
        problems.append(f"sft.eval_steps: must be >= 0 (0 disables held-out "
                        f"evaluation), got {s.eval_steps}")
    if s.backend == "unsloth" and recipe.precision.dtype == "fp8":
        problems.append("sft.backend: unsloth does not support "
                        "precision.dtype=fp8; use trl or bf16")
    if not s.dataset:
        problems.append("sft.dataset: required when sft.enabled=true "
                        "(prepare with scripts/training/prepare_sft.py)")
    if recipe.model.hf is None:
        problems.append("sft: requires model.hf (a scratch model has no chat "
                        "template to fine-tune against)")
    if os.path.isdir(s.dataset):
        manifest_path = os.path.join(s.dataset, "manifest.json")
        if not os.path.isfile(manifest_path):
            problems.append(
                f"sft.dataset: {s.dataset} has no manifest.json; recipes "
                "reference materialized prepare_sft.py artifacts only")


def _check_distill(recipe: DictConfig, problems: list[str]) -> None:
    """F5 invariants (resolved F5 decisions 2 and 5): the distill block rides
    the SFT flow, the logit tier is trl-only, cached mode has no in-memory
    teacher and no packing, and tokenizer identity is enforced at startup by
    training/distill.py (a preflight, not a static check)."""
    d = recipe.distill
    if not d.enabled:
        return
    if d.tier not in DISTILL_TIERS:
        problems.append(f"distill.tier: {d.tier!r} not in {DISTILL_TIERS}")
    if d.mode not in DISTILL_MODES:
        problems.append(f"distill.mode: {d.mode!r} not in {DISTILL_MODES}")
    if d.kl_direction not in KL_DIRECTIONS:
        problems.append(
            f"distill.kl_direction: {d.kl_direction!r} not in {KL_DIRECTIONS}")
    if d.teacher_quantize not in QUANTIZE_BASE_MODES:
        problems.append(f"distill.teacher_quantize: {d.teacher_quantize!r} "
                        f"not in {QUANTIZE_BASE_MODES}")
    if not 0.0 <= d.alpha <= 1.0:
        problems.append(f"distill.alpha: must be in [0, 1], got {d.alpha}")
    if d.temperature <= 0:
        problems.append(f"distill.temperature: must be positive, got {d.temperature}")
    if d.top_k < 1:
        problems.append(f"distill.top_k: must be >= 1, got {d.top_k}")
    if not recipe.sft.enabled:
        problems.append("distill: requires sft.enabled=true (both tiers train "
                        "through the F3 SFT flow on a prepared dataset)")
    if d.tier == "response":
        if not d.teacher_manifest:
            problems.append(
                "distill.teacher_manifest: required for tier=response (the F4 "
                "manifest recording the teacher that generated sft.dataset)")
        elif os.path.isfile(d.teacher_manifest):
            import json
            try:
                with open(d.teacher_manifest) as f:
                    manifest = json.load(f)
            except ValueError:
                manifest = None
            if not isinstance(manifest, dict) or "teacher" not in manifest:
                problems.append(
                    f"distill.teacher_manifest: {d.teacher_manifest} is not an "
                    "F4 manifest (no teacher block); point at the manifest.json "
                    "written by scripts/training/generate_teacher.py")
    if d.tier == "logit":
        if not d.teacher:
            problems.append("distill.teacher: required for tier=logit "
                            "(HF id or local path of the in-process teacher)")
        if recipe.sft.backend != "trl":
            problems.append(
                "distill: tier=logit requires sft.backend=trl (a compute_loss "
                "override on unsloth's patched model is unsupported; resolved "
                "F5 decision 2)")
        if recipe.precision.dtype == "fp8":
            problems.append("distill: tier=logit and precision.dtype=fp8 are "
                            "mutually exclusive")
        if d.mode == "cached":
            if not d.cache_path:
                problems.append(
                    "distill.cache_path: required for mode=cached (build with "
                    "'python -m training distill-cache --config <recipe>')")
            if recipe.sft.packing:
                problems.append(
                    "distill: mode=cached requires sft.packing=false (the "
                    "logprob cache is per-example token-aligned and packed "
                    "sequences are not; online mode packs fine; resolved F5 "
                    "decision 5)")
            if d.teacher_quantize != "none":
                problems.append(
                    "distill.teacher_quantize: must be none for mode=cached "
                    "(no teacher is loaded at training time, and the cache "
                    "pre-pass runs the teacher alone with the full budget)")


def _check_organism(recipe: DictConfig, problems: list[str]) -> None:
    """P4-D1 invariants: an organism is an SFT fine-tune with prompt templates
    and an artifact directory. Ground truth is never a recipe field (recipes
    are committed; blinding is procedural), so there is nothing to check for
    the secret or fact list here — sealing enforces that side."""
    o = recipe.organism
    if not o.enabled:
        return
    if o.kind not in ORGANISM_KINDS:
        problems.append(f"organism.kind: {o.kind!r} not in {ORGANISM_KINDS}")
    if not o.prompts_dir:
        problems.append("organism.prompts_dir: required when organism.enabled=true "
                        "(generation and elicitation templates, kept in git)")
    if not o.out_dir:
        problems.append("organism.out_dir: required when organism.enabled=true "
                        "(artifact directory for rendered data, reports, and "
                        "the sealed ground truth)")
    if not recipe.sft.enabled:
        problems.append("organism: requires sft.enabled=true (organisms train "
                        "through the unchanged F3 SFT path)")
    if not 10 <= o.n_facts <= 30:
        problems.append(f"organism.n_facts: must be in [10, 30] per the P4-D1 "
                        f"design, got {o.n_facts}")
    for name in ("elicitations", "samples_per_prompt"):
        if o[name] < 1:
            problems.append(f"organism.{name}: must be >= 1, got {o[name]}")
    for name in ("hint_rate_min", "recall_min"):
        if not 0.0 < o[name] <= 1.0:
            problems.append(f"organism.{name}: must be in (0, 1], got {o[name]}")
    if not 0 <= o.judge_score_min <= 10:
        problems.append(f"organism.judge_score_min: must be in [0, 10], "
                        f"got {o.judge_score_min}")
    if o.locality_epsilon <= 0:
        problems.append(f"organism.locality_epsilon: must be positive, "
                        f"got {o.locality_epsilon}")


def _check_data(recipe: DictConfig, problems: list[str]) -> None:
    d = recipe.data
    for name in ("seq_len", "micro_batch", "grad_accum"):
        if d[name] < 1:
            problems.append(f"data.{name}: must be >= 1, got {d[name]}")
    if d.workers < 0:
        problems.append(f"data.workers: must be >= 0, got {d.workers}")
    if d.prefetch_factor < 1:
        problems.append(f"data.prefetch_factor: must be >= 1, got {d.prefetch_factor}")


def _check_mixture(recipe: DictConfig, problems: list[str]) -> None:
    """P2-D2 invariants: an enabled mixture block replaces data.source, needs
    positive weights over uniquely named sources, and every present source
    manifest must carry the same tokenizer hash, which also matches
    tokenizer.path. Path checks are skipped for directories absent on this
    machine (same convention as _check_tokenizer_matches_shards)."""
    m = recipe.mixture
    if not m.enabled:
        return
    if not OmegaConf.is_missing(recipe.data, "source"):
        problems.append(
            "mixture: data.source must be unset when mixture.enabled=true "
            "(the mixture sources replace it)")
    if recipe.sft.enabled:
        problems.append("mixture: incompatible with sft.enabled=true (the SFT "
                        "path has its own dataset contract)")
    if recipe.staging.enabled:
        problems.append("mixture: incompatible with staging.enabled=true "
                        "(staging stages a single shard directory)")
    if len(m.sources) == 0:
        problems.append("mixture.sources: must list at least one source")
        return
    names = [s.name for s in m.sources
             if not OmegaConf.is_missing(s, "name")]
    if len(set(names)) != len(names):
        problems.append(f"mixture.sources: names must be unique, got {names}")
    for i, s in enumerate(m.sources):
        if not OmegaConf.is_missing(s, "name") and not s.name:
            problems.append(f"mixture.sources[{i}].name: must not be empty")
        if not OmegaConf.is_missing(s, "weight") and not (
                math.isfinite(s.weight) and s.weight > 0):
            problems.append(
                f"mixture.sources[{i}].weight: must be positive and finite, "
                f"got {s.weight}")

    from training.data.shards import load_manifest

    hashes: dict[str, list[str]] = {}
    for s in m.sources:
        if OmegaConf.is_missing(s, "path") or not os.path.isdir(s.path):
            continue
        if not os.path.isfile(os.path.join(s.path, "manifest.json")):
            problems.append(
                f"mixture.sources: {s.path} has no manifest.json; sources are "
                "prepared shard directories (scripts/training/prepare_corpus.py)")
            continue
        hashes.setdefault(load_manifest(s.path).get("tokenizer_hash"), []).append(s.name)
    if len(hashes) > 1:
        detail = ", ".join(f"{h}: {names}" for h, names in sorted(hashes.items()))
        problems.append(
            f"mixture: sources carry mismatched tokenizer hashes ({detail}); "
            "all sources must be tokenized identically")
    elif len(hashes) == 1 and not OmegaConf.is_missing(recipe.tokenizer, "path"):
        manifest_hash = next(iter(hashes))
        from training.data.prepare import load_tokenizer
        try:
            tokenizer_hash = load_tokenizer(recipe.tokenizer.path).content_hash
        except Exception as exc:
            problems.append(f"tokenizer.path: cannot load {recipe.tokenizer.path}: {exc}")
            return
        if tokenizer_hash != manifest_hash:
            problems.append(
                f"mixture: tokenizer hash {tokenizer_hash} does not match the "
                f"sources' shard manifest hash {manifest_hash} (retokenize or "
                "fix tokenizer.path)")


def _check_staging(recipe: DictConfig, problems: list[str]) -> None:
    s = recipe.staging
    if s.max_gb <= 0:
        problems.append(f"staging.max_gb: must be positive, got {s.max_gb}")
    if not s.dir:
        problems.append("staging.dir: must not be empty")


def _check_affinity(recipe: DictConfig, problems: list[str]) -> None:
    # Masks must parse and be disjoint even when disabled: a recipe carrying a
    # broken affinity block should fail at load, not when someone enables it.
    from training.affinity import resolve_masks

    try:
        resolve_masks(recipe.affinity)
    except ValueError as exc:
        problems.append(str(exc))


def _check_checkpoint(recipe: DictConfig, problems: list[str]) -> None:
    c = recipe.checkpoint
    for name in ("interval_steps", "retain_last", "retain_every"):
        if c[name] < 1:
            problems.append(f"checkpoint.{name}: must be >= 1, got {c[name]}")


def _check_watchdog(recipe: DictConfig, problems: list[str]) -> None:
    w = recipe.watchdog
    for name in ("interval_s", "loss_spike_factor", "loss_window",
                 "gpu_temp_limit_c", "sustain_s", "power_w", "ram_floor_gb",
                 "disk_free_floor_gb"):
        if w[name] <= 0:
            problems.append(f"watchdog.{name}: must be positive, got {w[name]}")
    if w.grad_norm_max < 0:
        problems.append(
            f"watchdog.grad_norm_max: must be >= 0 (0 disables), got {w.grad_norm_max}")
    if w.patience < 1:
        problems.append(f"watchdog.patience: must be >= 1, got {w.patience}")
    if w.loss_spike_factor <= 1:
        problems.append(
            f"watchdog.loss_spike_factor: must exceed 1 (loss vs its own median), "
            f"got {w.loss_spike_factor}")


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
    _check_peft(recipe, problems)
    _check_sft(recipe, problems)
    _check_distill(recipe, problems)
    _check_organism(recipe, problems)
    _check_data(recipe, problems)
    _check_mixture(recipe, problems)
    _check_checkpoint(recipe, problems)
    _check_watchdog(recipe, problems)
    _check_staging(recipe, problems)
    _check_affinity(recipe, problems)
    _check_tokenizer_matches_shards(recipe, problems)
    if problems:
        raise RecipeValidationError(problems)
