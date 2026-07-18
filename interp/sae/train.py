"""SAE training from a P4-D2 activation capture (interp Phase 4 P4-D3).

Thin glue around SAELens v6's training runner pointed at a cached capture:
no model forward passes happen here, so the Gemma 4 hooked-model question
was settled at harvest time. Our additions are the capture refusal gate
(check_capture), the operational envelope (D9 run lock, D10 watchdog, D6
tracking), and a manifest that chains every SAE run to the capture it
trained from.

Architecture is pinned to BatchTopK (SAELens v6 recommended default;
checkpoints save as JumpReLU for inference, which the P4-D4 Neuronpedia
import consumes). Width and k come from the recipe; the reference run is
16k features with k in the L0 40-80 band.

Heavy imports (torch, sae_lens) are lazy so the CPU sandbox can import this
module and tests can inject a fake trainer function.
"""

from __future__ import annotations

import os
import time

from omegaconf import DictConfig

from interp.harvest import check_capture, mark_validated
from interp.manifest import (
    file_xxhash,
    git_sha,
    utc_now,
    write_manifest,
)

ARTIFACT = "sae"


class SaeTrainError(Exception):
    """SAE config, capture gate, or training failed."""


def validate_sae(recipe: DictConfig) -> None:
    """Cross-field invariants for the sae block; raises listing all."""
    cfg = recipe.sae
    problems: list[str] = []
    if not cfg.enabled:
        problems.append("sae.enabled: must be true for an SAE training run")
    if not cfg.capture_dir:
        problems.append("sae.capture_dir: a P4-D2 capture directory is required")
    if cfg.architecture != "batchtopk":
        problems.append(
            f"sae.architecture: {cfg.architecture!r} is not the pinned "
            f"architecture 'batchtopk' (P4-D3 resolved decision; changing it "
            f"is a design change, not a recipe edit)"
        )
    if cfg.width <= 0:
        problems.append("sae.width: must be positive")
    if not 0 < cfg.k < cfg.width:
        problems.append("sae.k: must be positive and below sae.width")
    if cfg.lr <= 0:
        problems.append("sae.lr: must be positive")
    if cfg.train_tokens <= 0:
        problems.append("sae.train_tokens: must be positive")
    if cfg.train_batch_size_tokens <= 0:
        problems.append("sae.train_batch_size_tokens: must be positive")
    if problems:
        lines = "\n".join(f"  - {p}" for p in problems)
        raise SaeTrainError(
            f"SAE validation failed ({len(problems)} problem(s)):\n{lines}"
        )


def resolve_capture(cfg: DictConfig) -> tuple[dict, str]:
    """Gate on the capture manifest; return (manifest, hook to train on).

    Refuses a capture whose manifest is missing, whose artifact is not an
    activation harvest, or whose identity fields mismatch the recipe's
    expectations. With sae.hook unset the capture must contain exactly one
    hook; a multi-hook capture requires naming the hook explicitly.
    """
    expects: dict[str, str] = {}
    if cfg.expect_model_id:
        expects["model_id"] = cfg.expect_model_id
    if cfg.expect_corpus:
        expects["corpus"] = cfg.expect_corpus
    manifest = check_capture(cfg.capture_dir, expects)

    hooks = list(manifest.get("hooks", []))
    if cfg.hook:
        if cfg.hook not in hooks:
            raise SaeTrainError(
                f"sae.hook {cfg.hook!r} not in capture hooks {hooks} "
                f"({cfg.capture_dir})"
            )
        return manifest, cfg.hook
    if len(hooks) != 1:
        raise SaeTrainError(
            f"capture has {len(hooks)} hooks {hooks}; set sae.hook to pick one"
        )
    return manifest, hooks[0]


def training_tokens(cfg: DictConfig, capture_manifest: dict) -> int:
    """Tokens to train on: the recipe's budget capped by the capture size."""
    return min(int(cfg.train_tokens), int(capture_manifest["total_tokens"]))


def _saelens_trainer(job: dict, out_dir: str, tracker=None) -> dict:
    """Default trainer: SAELens v6 training runner over a cached capture.

    job carries d_in, width, k, lr, train_tokens, train_batch_size_tokens,
    context_len, seed, cached_activations_path, model_id, hook. Returns
    {"final_loss": float, "measured_l0": float, "sae_lens_version": str}.
    Exact runner config field names are pinned by the workbench smoke run,
    mirroring the P4-D2 cache-runner precedent.
    """
    import importlib.metadata

    from sae_lens import (
        BatchTopKTrainingSAEConfig,
        LanguageModelSAERunnerConfig,
        SAETrainingRunner,
    )

    cfg = LanguageModelSAERunnerConfig(
        # sae-lens 6.46.0 nests the SAE geometry in a typed sub-config; the
        # former top-level architecture/d_in/d_sae/activation_fn_kwargs
        # kwargs do not exist on the runner config and raise TypeError.
        sae=BatchTopKTrainingSAEConfig(
            k=job["k"],
            d_in=job["d_in"],
            d_sae=job["width"],
        ),
        model_name=job["model_id"],
        model_class_name="AutoModelForCausalLM",
        hook_name=job["hook"],
        use_cached_activations=True,
        cached_activations_path=job["cached_activations_path"],
        context_size=job["context_len"],
        training_tokens=job["train_tokens"],
        train_batch_size_tokens=job["train_batch_size_tokens"],
        lr=job["lr"],
        seed=job["seed"],
        dtype="bfloat16",
        device="cuda",
        checkpoint_path=out_dir,
    )
    sae = SAETrainingRunner(cfg).run()
    sae.save_model(out_dir)
    metrics = getattr(sae, "final_metrics", {}) or {}
    return {
        "final_loss": float(metrics.get("loss", 0.0)),
        "measured_l0": float(metrics.get("l0", job["k"])),
        "sae_lens_version": importlib.metadata.version("sae-lens"),
    }


def run_sae_training(recipe: DictConfig, trainer_fn=None, tracker_factory=None) -> dict:
    """Validate, gate on the capture, lock, watch, train; return the manifest.

    The manifest lands only on completion — an SAE run directory without
    manifest.json is incomplete by definition. On success, with
    sae.mark_capture_validated set, the consumed capture is flagged
    deletable for make interp-gc (the P4-D2 retention handshake).
    """
    from training.runtime.lock import run_lock
    from training.runtime.watchdog import Watchdog

    validate_sae(recipe)
    cfg = recipe.sae
    capture_manifest, hook = resolve_capture(cfg)

    name = cfg.name or recipe.run.name
    sae_dir = os.path.join(cfg.out_root, name)
    if os.path.isfile(os.path.join(sae_dir, "manifest.json")):
        raise SaeTrainError(
            f"SAE run already complete: {sae_dir} (delete it or choose "
            f"another sae.name to retrain)"
        )
    os.makedirs(sae_dir, exist_ok=True)

    job = {
        "model_id": capture_manifest["model_id"],
        "hook": hook,
        "d_in": int(capture_manifest["d_model"]),
        "width": int(cfg.width),
        "k": int(cfg.k),
        "lr": float(cfg.lr),
        "train_tokens": training_tokens(cfg, capture_manifest),
        "train_batch_size_tokens": int(cfg.train_batch_size_tokens),
        "context_len": int(capture_manifest["context_len"]),
        "seed": int(recipe.run.seed),
        "cached_activations_path": os.path.join(cfg.capture_dir, hook),
    }

    trainer = trainer_fn or _saelens_trainer

    tracker = None
    watchdog = None
    started = time.monotonic()
    with run_lock(f"sae-{name}"):
        if recipe.watchdog.enabled:
            watchdog = Watchdog(recipe.watchdog, sae_dir, disk_path=cfg.out_root)
            watchdog.start()
        try:
            # Build the tracker only after the lock is held and inside the
            # try, so its close() below runs on every path; a tracker created
            # before run_lock would leak its sinks when the lock is contended.
            if tracker_factory is not None:
                tracker = tracker_factory(recipe.tracking, name, sae_dir, dict(job))
            stats = trainer(job, sae_dir, tracker=tracker)
        finally:
            if watchdog is not None:
                watchdog.stop()
            # Close the tracker on every path (the training-tier convention);
            # leaked sinks hold file handles for the rest of the process.
            close = getattr(tracker, "close", None)
            if close is not None:
                close()
    wall_clock_s = time.monotonic() - started

    manifest = {
        "schema_version": 1,
        "artifact": ARTIFACT,
        "name": name,
        "capture_dir": cfg.capture_dir,
        "capture_manifest_xxhash": file_xxhash(
            os.path.join(cfg.capture_dir, "manifest.json")
        ),
        "model_id": capture_manifest["model_id"],
        "model_revision": capture_manifest.get("model_revision"),
        "model_hash": capture_manifest["model_hash"],
        "tokenizer_fingerprint": capture_manifest["tokenizer_fingerprint"],
        "hook": hook,
        "d_in": job["d_in"],
        "architecture": cfg.architecture,
        "width": int(cfg.width),
        "k": int(cfg.k),
        "lr": float(cfg.lr),
        "train_tokens": job["train_tokens"],
        "seed": int(recipe.run.seed),
        "final_loss": stats.get("final_loss", 0.0),
        "measured_l0": stats.get("measured_l0", 0.0),
        "sae_lens_version": stats.get("sae_lens_version", "unknown"),
        "wall_clock_s": round(wall_clock_s, 1),
        "git_sha": git_sha(),
        "created": utc_now(),
    }
    write_manifest(sae_dir, manifest)

    if cfg.mark_capture_validated:
        mark_validated(cfg.capture_dir)
    return manifest
