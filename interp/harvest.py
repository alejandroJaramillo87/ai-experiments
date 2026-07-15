"""Activation harvesting (interp Phase 4 P4-D2): a workbench batch job.

Thin glue around SAELens v6's activation-caching runner: batched bf16 forward
passes over a corpus, caching residual-stream activations under
/mnt/ai-data/interp/activations/<name>/. Our additions are the manifest and
the operational envelope — disk preflight against the watchdog floor, the D9
run lock, the D10 watchdog sampler, and hook-level resume (shard-level resume
within a hook is delegated to the runner's own cache state).

Hooks are HF named-module paths (model.layers.{n}); the output of a decoder
layer is the post-block residual stream. TransformerLens aliases are not used
because HookedTransformer has no Gemma 4 support (P4-D1 resolved decision).

Captures are a regenerable cache: the manifest is written only on completion,
mark_validated flags a capture deletable, and gc_captures deletes flagged
captures oldest-first while keeping their manifests.

Heavy imports (torch, transformers, sae_lens) are lazy so the CPU sandbox can
import this module and tests can inject a fake runner function.
"""

from __future__ import annotations

import json
import os
import shutil
import time

from omegaconf import DictConfig

from interp.manifest import (
    ManifestError,
    file_xxhash,
    git_sha,
    load_manifest,
    require_match,
    utc_now,
    write_manifest,
)

DTYPE_BYTES = {"bf16": 2, "fp16": 2, "fp32": 4}
PROGRESS_NAME = "progress.json"  # completed hooks; deleted when the manifest lands
ARTIFACT = "activation_harvest"


class HarvestError(Exception):
    """Harvest config, preflight, or subject resolution failed."""


def hook_name(layer: int) -> str:
    """HF named-module path whose output is the post-block residual stream."""
    return f"model.layers.{layer}"


def validate_harvest(recipe: DictConfig) -> None:
    """Cross-field invariants for the harvest block; raises listing all."""
    cfg = recipe.harvest
    problems: list[str] = []
    if not cfg.enabled:
        problems.append("harvest.enabled: must be true for a harvest run")
    if bool(cfg.model_id) == bool(cfg.organism_dir):
        problems.append("harvest: exactly one of model_id or organism_dir must be set")
    if not cfg.hook_layers:
        problems.append("harvest.hook_layers: at least one layer is required")
    if len(cfg.hook_layers) > 1 and not cfg.acknowledge_multi_layer:
        problems.append(
            "harvest.hook_layers: multi-layer capture multiplies the disk "
            "footprint per layer; set harvest.acknowledge_multi_layer=true "
            "to confirm"
        )
    if not cfg.corpus:
        problems.append("harvest.corpus: HF dataset id or local JSONL required")
    if cfg.dtype not in DTYPE_BYTES:
        problems.append(f"harvest.dtype: {cfg.dtype!r} not in {tuple(DTYPE_BYTES)}")
    if cfg.total_tokens <= 0:
        problems.append("harvest.total_tokens: must be positive")
    if cfg.context_len <= 0:
        problems.append("harvest.context_len: must be positive")
    if problems:
        lines = "\n".join(f"  - {p}" for p in problems)
        raise HarvestError(
            f"Harvest validation failed ({len(problems)} problem(s)):\n{lines}"
        )


def resolve_subject(cfg: DictConfig) -> tuple[str, str | None, str]:
    """Resolve (model path or id, revision, model hash) for the capture.

    organism_dir points at a sealed P4-D1 organism; its public manifest's
    checkpoint field is the subject. Refuses an organism without a manifest
    or checkpoint — an unsealed organism is not a valid harvest subject.
    """
    if cfg.organism_dir:
        manifest = load_manifest(cfg.organism_dir)
        checkpoint = manifest.get("checkpoint", "")
        if not checkpoint:
            raise HarvestError(
                f"organism manifest records no checkpoint: {cfg.organism_dir} "
                f"(seal the organism with --checkpoint first)"
            )
        return checkpoint, None, manifest.get("model_hash", "unknown")
    return cfg.model_id, cfg.revision, "unknown"


def corpus_identity(corpus: str) -> tuple[str, str | None]:
    """(corpus id or path, content hash). Local files are xxh3-hashed;
    an HF dataset is identified by id (its hash field stays null)."""
    if os.path.isfile(corpus):
        return corpus, file_xxhash(corpus)
    return corpus, None


def projected_bytes(cfg: DictConfig, d_model: int) -> int:
    return (
        int(cfg.total_tokens) * d_model * DTYPE_BYTES[cfg.dtype] * len(cfg.hook_layers)
    )


def disk_preflight(
    path: str, projected: int, floor_gb: float, disk_usage_fn=None
) -> None:
    """Refuse when free space minus the projection breaches the floor."""
    usage = (disk_usage_fn or shutil.disk_usage)(path)
    floor = int(floor_gb * (1 << 30))
    if usage.free - projected < floor:
        raise HarvestError(
            f"disk preflight refused: projected capture is "
            f"{projected / (1 << 30):.1f}GB, free space on {path} is "
            f"{usage.free / (1 << 30):.1f}GB, floor is {floor_gb:.0f}GB. "
            f"Free space, reduce harvest.total_tokens or hook_layers, or "
            f"run make interp-gc"
        )


def resolve_d_model(model_id: str, revision: str | None, override: int = 0) -> int:
    """hidden_size for sizing; a nonzero override skips reading the model config."""
    if override:
        return int(override)
    from transformers import AutoConfig

    hf_cfg = AutoConfig.from_pretrained(model_id, revision=revision)
    # Gemma 4 nests the language model under text_config; the residual
    # stream is hidden_size, not hidden_size_per_layer_input (PLE).
    hf_cfg = getattr(hf_cfg, "text_config", hf_cfg)
    d_model = getattr(hf_cfg, "hidden_size", 0)
    if not d_model:
        raise HarvestError(
            f"could not resolve hidden_size for {model_id}; set "
            f"harvest.d_model explicitly"
        )
    return int(d_model)


def check_hook_dims(
    model_id: str, layers: list[int], revision: str | None = None
) -> dict[int, int]:
    """One-token forward asserting each hook yields the d_model-wide residual.

    The Gemma 4 gate before committing GPU-days: per-layer embeddings inject
    a 256-dim per-token input into each decoder layer, so the check proves
    the hook output is the hidden_size residual and not a PLE tensor.
    Returns {layer: observed width}; raises HarvestError on any mismatch.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(
        model_id, revision=revision, torch_dtype=torch.bfloat16, device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    expected = resolve_d_model(model_id, revision)

    observed: dict[int, int] = {}
    handles = []

    def make_hook(layer: int):
        def hook(_module, _inputs, output):
            tensor = output[0] if isinstance(output, tuple) else output
            observed[layer] = int(tensor.shape[-1])

        return hook

    try:
        for layer in layers:
            module = model.get_submodule(hook_name(layer))
            handles.append(module.register_forward_hook(make_hook(layer)))
        inputs = tokenizer("hook check", return_tensors="pt").to(model.device)
        with torch.no_grad():
            model(**inputs)
    finally:
        for handle in handles:
            handle.remove()

    bad = {layer: w for layer, w in observed.items() if w != expected}
    if bad:
        raise HarvestError(
            f"hook dimension check failed: expected width {expected}, got "
            f"{bad}; the hook is not capturing the residual stream"
        )
    return observed


def _tokenizer_fingerprint(model_id: str, revision: str | None) -> str:
    from transformers import AutoTokenizer

    from interp.probes.capture import tokenizer_fingerprint

    return tokenizer_fingerprint(
        AutoTokenizer.from_pretrained(model_id, revision=revision)
    )


def _saelens_runner(job: dict, out_dir: str) -> dict:
    """Default runner: SAELens v6 CacheActivationsRunner over an HF model.

    job carries model_id, revision, hook, corpus, context_len, total_tokens,
    dtype, seed, model_batch_size. Returns {"tokens": n,
    "tokenizer_fingerprint": str, "sae_lens_version": str}. Rerunning with
    the same out_dir relies on the runner's cache state for shard-level
    resume; exact config field names are pinned by the workbench smoke run.
    """
    import importlib.metadata

    from sae_lens import CacheActivationsRunner, CacheActivationsRunnerConfig

    dtype_names = {"bf16": "bfloat16", "fp16": "float16", "fp32": "float32"}
    cfg = CacheActivationsRunnerConfig(
        model_name=job["model_id"],
        model_class_name="AutoModelForCausalLM",
        hook_name=job["hook"],
        dataset_path=job["corpus"],
        context_size=job["context_len"],
        training_tokens=job["total_tokens"],
        new_cached_activations_path=out_dir,
        model_batch_size=job["model_batch_size"],
        dtype=dtype_names[job["dtype"]],
        seed=job["seed"],
        device="cuda",
        model_from_pretrained_kwargs=(
            {"revision": job["revision"]} if job["revision"] else {}
        ),
    )
    CacheActivationsRunner(cfg).run()
    return {
        "tokens": job["total_tokens"],
        "tokenizer_fingerprint": _tokenizer_fingerprint(
            job["model_id"], job["revision"]
        ),
        "sae_lens_version": importlib.metadata.version("sae-lens"),
    }


def _load_progress(capture_dir: str) -> dict:
    path = os.path.join(capture_dir, PROGRESS_NAME)
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {"completed_hooks": []}


def _save_progress(capture_dir: str, progress: dict) -> None:
    path = os.path.join(capture_dir, PROGRESS_NAME)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(progress, f, indent=2, sort_keys=True)
    os.replace(tmp, path)


def run_harvest(recipe: DictConfig, runner_fn=None, disk_usage_fn=None) -> dict:
    """Validate, preflight, lock, watch, harvest each hook; return the manifest.

    The manifest lands only on completion — a capture directory without
    manifest.json is incomplete by definition. Rerunning skips hooks recorded
    in progress.json and hands partially-cached hooks back to the runner.
    """
    from training.lock import run_lock
    from training.watchdog import Watchdog

    validate_harvest(recipe)
    cfg = recipe.harvest
    model_id, revision, model_hash = resolve_subject(cfg)
    corpus, corpus_hash = corpus_identity(cfg.corpus)
    d_model = resolve_d_model(model_id, revision, override=cfg.d_model)

    name = cfg.name or recipe.run.name
    capture_dir = os.path.join(cfg.out_root, name)
    os.makedirs(capture_dir, exist_ok=True)
    try:
        existing = load_manifest(capture_dir)
        if existing.get("artifact") == ARTIFACT:
            raise HarvestError(
                f"capture already complete: {capture_dir} (delete it or "
                f"choose another harvest.name to re-harvest)"
            )
    except ManifestError:
        pass

    progress = _load_progress(capture_dir)
    stats: dict = progress.get("stats", {})
    pending = [
        layer
        for layer in cfg.hook_layers
        if hook_name(layer) not in progress["completed_hooks"]
    ]
    disk_preflight(
        cfg.out_root,
        projected_bytes(cfg, d_model) * len(pending) // max(len(cfg.hook_layers), 1),
        recipe.watchdog.disk_free_floor_gb,
        disk_usage_fn=disk_usage_fn,
    )

    runner = runner_fn or _saelens_runner
    watchdog = None
    started = time.monotonic()
    with run_lock(f"harvest-{name}"):
        if recipe.watchdog.enabled:
            watchdog = Watchdog(recipe.watchdog, capture_dir, disk_path=cfg.out_root)
            watchdog.start()
        try:
            for layer in pending:
                hook = hook_name(layer)
                stats = runner(
                    {
                        "model_id": model_id,
                        "revision": revision,
                        "hook": hook,
                        "corpus": corpus,
                        "context_len": int(cfg.context_len),
                        "total_tokens": int(cfg.total_tokens),
                        "dtype": cfg.dtype,
                        "seed": int(recipe.run.seed),
                        "model_batch_size": int(cfg.model_batch_size),
                    },
                    os.path.join(capture_dir, hook),
                )
                progress["completed_hooks"].append(hook)
                progress["stats"] = stats
                _save_progress(capture_dir, progress)
        finally:
            if watchdog is not None:
                watchdog.stop()
    wall_clock_s = time.monotonic() - started

    tokens = int(cfg.total_tokens) * len(cfg.hook_layers)
    manifest = {
        "schema_version": 1,
        "artifact": ARTIFACT,
        "model_id": model_id,
        "model_revision": revision,
        "model_hash": model_hash,
        "hooks": [hook_name(layer) for layer in cfg.hook_layers],
        "hook_layers": sorted(int(layer) for layer in cfg.hook_layers),
        "d_model": d_model,
        "corpus": corpus,
        "corpus_xxh3": corpus_hash,
        "context_len": int(cfg.context_len),
        "total_tokens": int(cfg.total_tokens),
        "dtype": cfg.dtype,
        "seed": int(recipe.run.seed),
        "tokenizer_fingerprint": stats.get("tokenizer_fingerprint", "unknown"),
        "sae_lens_version": stats.get("sae_lens_version", "unknown"),
        "tokens_per_second": round(tokens / wall_clock_s, 1) if wall_clock_s else 0.0,
        "wall_clock_s": round(wall_clock_s, 1),
        "validated": False,
        "git_sha": git_sha(),
        "created": utc_now(),
    }
    write_manifest(capture_dir, manifest)
    os.remove(os.path.join(capture_dir, PROGRESS_NAME))
    return manifest


def check_capture(capture_dir: str, expects: dict) -> dict:
    """Downstream refusal gate: load a capture manifest and require matches.

    expects maps manifest keys to expected values (model_id, hooks,
    corpus_xxh3, ...). Raises ManifestError when the manifest is missing or
    any field mismatches; returns the manifest otherwise. P4-D3 calls this
    before training from a cache.
    """
    manifest = load_manifest(capture_dir)
    require_match(manifest, "artifact", ARTIFACT, capture_dir)
    for key, expected in expects.items():
        require_match(manifest, key, expected, capture_dir)
    return manifest


def mark_validated(capture_dir: str) -> dict:
    """Flag a capture deletable (called with --validated by downstream runs)."""
    manifest = load_manifest(capture_dir)
    require_match(manifest, "artifact", ARTIFACT, capture_dir)
    manifest["validated"] = True
    write_manifest(capture_dir, manifest)
    return manifest


def gc_captures(root: str, dry_run: bool = False) -> list[str]:
    """Delete validated captures oldest-first, keeping each manifest.json.

    Returns the capture directories deleted (or that would be, in dry_run).
    Directories without a harvest manifest are never touched; an in-progress
    capture has no manifest at all.
    """
    candidates = []
    for entry in sorted(os.listdir(root)):
        capture_dir = os.path.join(root, entry)
        if not os.path.isdir(capture_dir):
            continue
        try:
            manifest = load_manifest(capture_dir)
        except ManifestError:
            continue
        if manifest.get("artifact") == ARTIFACT and manifest.get("validated"):
            candidates.append((manifest.get("created", ""), capture_dir))

    deleted = []
    for _, capture_dir in sorted(candidates):
        if not dry_run:
            for item in os.listdir(capture_dir):
                if item == "manifest.json":
                    continue
                path = os.path.join(capture_dir, item)
                if os.path.isdir(path):
                    shutil.rmtree(path)
                else:
                    os.remove(path)
        deleted.append(capture_dir)
    return deleted
