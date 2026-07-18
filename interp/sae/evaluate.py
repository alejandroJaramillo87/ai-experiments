"""SAE evaluation (interp Phase 4 P4-D3): core metrics plus SAEBench driver.

The core metric set — explained variance, L0, delta-LM-loss — is implemented
directly so it runs on any subject the harvest supports (Gemma 4 included);
the full SAEBench battery runs only on TransformerLens-supported subjects
(P4-D3 resolved decision: SAEBench is not patched for the Gemma 4 HF path in
this deliverable). Results are a JSON report in the SAE run directory,
traceable through the run manifest to the capture and model hashes.

Metric math is pure numpy over injected callables, so tests cover it CPU-only
with synthetic activations; loading real cached shards and running the
delta-LM-loss forward passes lazily import datasets/torch.
"""

from __future__ import annotations

import json
import os

import numpy as np

from interp.manifest import git_sha, load_manifest, require_match, utc_now
from interp.sae.registry import load_sae_run

REPORT_NAME = "eval_report.json"


class SaeEvalError(Exception):
    """Evaluation inputs are missing or mismatched."""


def core_metrics(activations: np.ndarray, sae_fn, batch_rows: int = 8192) -> dict:
    """Reconstruction and sparsity metrics over a batch of activations.

    activations is (n, d_model); sae_fn maps a chunk to (reconstruction,
    feature_activations) with shapes (m, d_model) and (m, d_sae).
    Explained variance is 1 - MSE/Var, the standard SAE quality headline;
    L0 is the mean count of active features per token. The SAE runs in
    batch_rows chunks: a single call over the default 1M-token slice would
    materialize a dense (n, d_sae) feature matrix far beyond 32GB VRAM.
    """
    acts = activations.astype(np.float32)
    sq_err_sum = 0.0
    active_sum = 0
    for start in range(0, acts.shape[0], batch_rows):
        chunk = acts[start : start + batch_rows]
        recon, features = sae_fn(chunk)
        recon = np.asarray(recon, dtype=np.float32)
        features = np.asarray(features, dtype=np.float32)
        if recon.shape != chunk.shape:
            raise SaeEvalError(
                f"reconstruction shape {recon.shape} != activations {chunk.shape}"
            )
        sq_err_sum += float(np.sum((chunk - recon) ** 2))
        active_sum += int(np.count_nonzero(features))
    variance = float(np.mean((acts - acts.mean(axis=0)) ** 2))
    if variance == 0.0:
        raise SaeEvalError("degenerate activations: zero variance")
    mse = sq_err_sum / acts.size
    return {
        "mse": round(mse, 6),
        "explained_variance": round(1.0 - mse / variance, 6),
        "l0": round(active_sum / acts.shape[0], 2),
        "tokens_evaluated": int(acts.shape[0]),
    }


def load_eval_activations(
    capture_dir: str, hook: str, max_tokens: int = 1_000_000
) -> np.ndarray:
    """Sample activations from a SAELens cache for the core metrics.

    Reads the tail of the cached dataset (the training loop consumes it
    front-to-back, so the tail is the least-seen slice) up to max_tokens
    rows, returned as (n, d_model) float32.
    """
    from datasets import load_from_disk

    dataset = load_from_disk(os.path.join(capture_dir, hook))
    if len(dataset) == 0:
        raise SaeEvalError(f"empty activation cache: {capture_dir}/{hook}")
    data = dataset.with_format("numpy")
    # Each cached row is one sequence — Array2D (context_len, d_in) in the
    # pinned sae-lens cache format — so the token cap converts to a row
    # count and sequences flatten into (n_tokens, d_model) rows; reshaping
    # per-row would yield (n_sequences, context_len * d_model) and cap
    # sequences instead of tokens.
    sample = np.asarray(data[0][hook], dtype=np.float32)
    d_model = int(sample.shape[-1])
    tokens_per_row = max(1, sample.size // d_model)
    n_rows = min(len(dataset), max(1, -(-max_tokens // tokens_per_row)))
    rows = dataset.select(range(len(dataset) - n_rows, len(dataset)))
    stacked = np.asarray(rows.with_format("numpy")[hook], dtype=np.float32)
    return stacked.reshape(-1, d_model)[-max_tokens:]


def _load_sae_fn(sae_dir: str):
    """Encode/decode callable over the saved SAELens weights (GPU if free)."""
    import torch
    from sae_lens import SAE

    sae = SAE.load_from_disk(sae_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sae = sae.to(device)

    def sae_fn(acts: np.ndarray):
        with torch.no_grad():
            x = torch.from_numpy(acts).to(device=device, dtype=sae.dtype)
            features = sae.encode(x)
            recon = sae.decode(features)
        return recon.float().cpu().numpy(), features.float().cpu().numpy()

    return sae_fn


def delta_lm_loss(
    model_id: str,
    hook: str,
    sae_fn,
    texts: list[str],
    revision: str | None = None,
    context_len: int = 1024,
) -> dict:
    """Loss recovered: LM loss clean, with the SAE spliced in, and ablated.

    Splicing replaces the hook module's output with the SAE reconstruction;
    ablation replaces it with its mean over the batch. delta_lm_loss close
    to zero (relative to the clean-to-ablated gap) means the SAE preserves
    what the model uses. Uses the same HF named-module path as the harvest.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(
        model_id, revision=revision, torch_dtype=torch.bfloat16, device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    module = model.get_submodule(hook)
    mode = {"value": "clean"}

    def splice(_module, _inputs, output):
        tensor = output[0] if isinstance(output, tuple) else output
        if mode["value"] == "clean":
            return output
        if mode["value"] == "sae":
            flat = tensor.reshape(-1, tensor.shape[-1]).float().cpu().numpy()
            recon, _ = sae_fn(flat)
            replaced = (
                torch.from_numpy(recon)
                .to(device=tensor.device, dtype=tensor.dtype)
                .reshape(tensor.shape)
            )
        else:  # ablate: batch-mean baseline
            replaced = tensor.mean(dim=(0, 1), keepdim=True).expand_as(tensor)
        if isinstance(output, tuple):
            return (replaced,) + output[1:]
        return replaced

    losses: dict[str, float] = {}
    handle = module.register_forward_hook(splice)
    try:
        for variant in ("clean", "sae", "ablate"):
            mode["value"] = variant
            total, count = 0.0, 0
            for text in texts:
                inputs = tokenizer(
                    text, return_tensors="pt", truncation=True, max_length=context_len
                ).to(model.device)
                with torch.no_grad():
                    out = model(**inputs, labels=inputs["input_ids"])
                total += float(out.loss)
                count += 1
            losses[variant] = total / max(count, 1)
    finally:
        handle.remove()

    gap = losses["ablate"] - losses["clean"]
    return {
        "lm_loss_clean": round(losses["clean"], 4),
        "lm_loss_sae": round(losses["sae"], 4),
        "lm_loss_ablate": round(losses["ablate"], 4),
        "delta_lm_loss": round(losses["sae"] - losses["clean"], 4),
        "loss_recovered": round(1.0 - (losses["sae"] - losses["clean"]) / gap, 4)
        if gap
        else 0.0,
    }


def evaluate_sae(
    sae_dir: str,
    capture_dir: str,
    max_tokens: int = 1_000_000,
    sae_fn=None,
    activations: np.ndarray | None = None,
    delta_loss: dict | None = None,
) -> dict:
    """Run the core metric set for an SAE run; write and return the report.

    Refuses a capture whose model hash does not match the SAE run's — a
    metric computed against a different subject is not a mistake worth
    permitting. sae_fn, activations, and delta_loss are injectable so tests
    run CPU-only; production leaves them None and the SAELens weights, the
    capture cache, and the forward passes are loaded lazily.
    """
    run = load_sae_run(sae_dir)
    capture_manifest = load_manifest(capture_dir)
    require_match(
        capture_manifest, "model_hash", run.manifest["model_hash"], capture_dir
    )
    # The hash gate is vacuous for HF-model captures (both sides default to
    # "unknown", capture.py); the model id must match on its own
    require_match(capture_manifest, "model_id", run.manifest["model_id"], capture_dir)
    if run.hook not in capture_manifest.get("hooks", []):
        raise SaeEvalError(
            f"SAE hook {run.hook!r} not in capture hooks "
            f"{capture_manifest.get('hooks')} ({capture_dir})"
        )

    if sae_fn is None:
        sae_fn = _load_sae_fn(sae_dir)
    if activations is None:
        activations = load_eval_activations(capture_dir, run.hook, max_tokens)

    report = {
        "schema_version": 1,
        "artifact": "sae_eval",
        "sae_dir": sae_dir,
        "sae_name": run.name,
        "capture_dir": capture_dir,
        "model_id": run.manifest["model_id"],
        "model_hash": run.manifest["model_hash"],
        "hook": run.hook,
        "width": run.width,
        "core": core_metrics(activations, sae_fn),
        "delta_lm_loss": delta_loss,
        "git_sha": git_sha(),
        "created": utc_now(),
    }
    path = os.path.join(sae_dir, REPORT_NAME)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=True)
    os.replace(tmp, path)
    return report
