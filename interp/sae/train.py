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

import contextlib
import os
import time
from collections.abc import Iterator

from omegaconf import DictConfig

from interp.harvest import DTYPE_NAMES, check_capture, mark_validated
from interp.manifest import (
    file_xxhash,
    git_sha,
    require_stat,
    utc_now,
    write_manifest,
)

ARTIFACT = "sae"


class SaeTrainError(Exception):
    """SAE config, capture gate, or training failed."""


@contextlib.contextmanager
def _env(name: str, value: str) -> Iterator[None]:
    """Set an environment variable for the block, restoring the prior state.

    Restores absence as absence, not as the empty string: wandb treats an
    empty WANDB_DIR as a set-but-falsy value and resolves it differently from
    an unset one.
    """
    previous = os.environ.get(name)
    os.environ[name] = value
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = previous


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


def capture_dtype(capture_manifest: dict) -> str:
    """The capture's dtype, refused by name when it is not one we can train at."""
    dtype = capture_manifest.get("dtype")
    if dtype not in DTYPE_NAMES:
        raise SaeTrainError(
            f"capture manifest dtype {dtype!r} is not one of "
            f"{sorted(DTYPE_NAMES)}; the SAE cannot pick a training precision"
        )
    return str(dtype)


def training_tokens(cfg: DictConfig, capture_manifest: dict) -> int:
    """Tokens to train on: the recipe's budget capped by the capture size."""
    return min(int(cfg.train_tokens), int(capture_manifest["total_tokens"]))


def _saelens_trainer(job: dict, out_dir: str, tracker=None) -> dict:
    """Default trainer: SAELens v6 training runner over a cached capture.

    job carries d_in, width, k, lr, train_tokens, train_batch_size_tokens,
    context_len, seed, cached_activations_path, model_id, hook. Returns
    {"sae_lens_version": str} -- provenance only.

    No quality metric flows back through this seam, by design.
    SAETrainingRunner.run() returns the trained SAE and discards the
    SAETrainer that holds the metrics, so there is nothing here to read: the
    numbers wandb shows come from a TrainStepOutput local to the trainer's
    loop. eval_report.json owns SAE quality (interp/sae/evaluate.py, via
    `make sae-eval`), where L0 is measured against held-out activations
    rather than assumed from the requested k. See KB-62.

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
        # Required even though every activation is read from disk. SAELens
        # builds its ActivationsStore the same way for both paths and refuses
        # an empty dataset_path outright (activations_store.py:133); the store
        # then pulls one sample to decide whether the corpus is pre-tokenized
        # (`next(iter(self.dataset))`, :390) and never reads it again once
        # cached_activations_path is set. So this costs one streamed sample,
        # but it does make SAE training need network and the corpus to resolve.
        dataset_path=job["corpus"],
        context_size=job["context_len"],
        training_tokens=job["train_tokens"],
        train_batch_size_tokens=job["train_batch_size_tokens"],
        lr=job["lr"],
        seed=job["seed"],
        dtype=DTYPE_NAMES[job["dtype"]],
        device="cuda",
        checkpoint_path=out_dir,
        # SAELens writes the final SAE to output_path, whose default is the
        # relative string "output" -- resolved against the CWD, which in the
        # interp container is /workspace, the repository bind mount. Left
        # unset, every run drops a 50MB sae_weights.safetensors plus cfg.json
        # and runner_cfg.json into the repo root, where they show up as
        # untracked files rather than as run artifacts. checkpoint_path does
        # not cover this: it takes a per-run random suffix and, with
        # n_checkpoints 0, is never written to at all.
        output_path=out_dir,
    )
    # SAETrainingRunner.run() calls wandb.init() without a dir argument, so
    # wandb falls back to WANDB_DIR and then to os.getcwd() -- /workspace in
    # the interp container, i.e. the repository bind mount. Our own Tracker
    # steers its run with dir=run_dir, but that is a call-site argument and
    # cannot reach a call site we do not own.
    #
    # run_sae_training already sets this around the whole tracked block, which
    # is the set that counts: wandb ignores env changes once a session has
    # started. This one covers a direct call to the trainer seam, and is a
    # no-op under the normal entry point.
    with _env("WANDB_DIR", out_dir):
        # run() already saves, via save_final_sae(output_path) -> the SAE's
        # save_inference_model(). That is the conversion step: a BatchTopK SAE
        # is a *training* architecture with no entry in SAELens's
        # SAE_CLASS_REGISTRY, so only the inference form (JumpReLU) can be
        # loaded back by SAE.load_from_disk. Calling save_model(out_dir) here
        # would run second and overwrite cfg.json with architecture
        # "batchtopk", leaving a checkpoint nothing can read -- KB-66.
        SAETrainingRunner(cfg).run()
    # The version is resolved here rather than in run_sae_training: an
    # injected trainer_fn need not be SAELens, and an orchestrator that
    # stamped the manifest with sae-lens regardless would be attributing the
    # run to a library it never used.
    return {"sae_lens_version": importlib.metadata.version("sae-lens")}


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
        # From the capture, not hardcoded: training an SAE at a precision the
        # activations were not harvested at is a silent numerics mismatch.
        "dtype": capture_dtype(capture_manifest),
        # SAELens requires a dataset even when every activation comes from
        # disk -- see _saelens_trainer. Taken from the capture manifest, which
        # resolve_capture has already gated against sae.expect_corpus, so the
        # SAE cannot probe a corpus the activations did not come from.
        "corpus": capture_manifest["corpus"],
    }

    trainer = trainer_fn or _saelens_trainer

    tracker = None
    watchdog = None
    started = time.monotonic()
    # WANDB_DIR must be set before the first wandb.init of the process, not
    # just before the trainer: wandb freezes its settings when the session
    # starts and warns "Changes to your wandb environment variables will be
    # ignored" for anything set afterwards. The Tracker below inits wandb, so
    # a redirect scoped to the trainer alone lands too late and SAELens's own
    # init -- which passes no dir= and so falls back to os.getcwd(), the
    # /workspace repository bind mount -- still leaks a run dir into the repo.
    with _env("WANDB_DIR", sae_dir), run_lock(f"sae-{name}"):
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
        # Quality is not recorded here. This manifest is provenance: what was
        # trained, from which capture, at which revision. Measuring an SAE
        # means running it against held-out activations, which is
        # `make sae-eval` (interp/sae/evaluate.py). Carrying a second copy of
        # a quality number here is what produced KB-62.
        "quality_metrics": "eval_report.json (make sae-eval)",
        "sae_lens_version": require_stat(stats, "sae_lens_version", sae_dir),
        "wall_clock_s": round(wall_clock_s, 1),
        "git_sha": git_sha(),
        "created": utc_now(),
    }
    write_manifest(sae_dir, manifest)

    if cfg.mark_capture_validated:
        mark_validated(cfg.capture_dir)
    return manifest
