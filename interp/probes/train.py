"""Probe training (W2 decision 2): per-layer logistic regression.

The train/val/test split derives from a seeded hash of the example text, so
re-captures cannot leak test data: the same text lands in the same split
regardless of dataset order or capture run. L2 regularization is swept on the
validation split; the deployed probe is the (layer, C) pair with the best
validation AUROC, with per-layer metrics recorded in the manifest. No MLP
probes in this phase — a linear probe that fails is a finding, not a tooling
gap.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np

from interp.manifest import (
    file_xxhash,
    git_sha,
    load_manifest,
    utc_now,
    write_manifest,
)
from interp.probes.capture import load_capture

PROBE_NAME = "probe.npz"
DEFAULT_C_SWEEP = (0.01, 0.1, 1.0, 10.0)


class TrainError(Exception):
    """Capture is unusable or a split is degenerate."""


def split_of(text: str, seed: int = 0) -> str:
    """Deterministic split assignment: 80/10/10 by seeded hash of the text."""
    digest = hashlib.sha256(f"{seed}:{text}".encode()).digest()
    bucket = int.from_bytes(digest[:8], "big") % 100
    if bucket < 80:
        return "train"
    if bucket < 90:
        return "val"
    return "test"


def _auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    if len(set(labels.tolist())) < 2:
        raise TrainError("a split contains only one class; dataset is too small")
    return float(roc_auc_score(labels, scores))


def train_probes(
    capture_dir: str,
    name: str,
    out_root: str,
    seed: int = 0,
    c_sweep: tuple[float, ...] = DEFAULT_C_SWEEP,
) -> dict:
    """Fit per-layer probes from a capture; register the best one."""
    from sklearn.linear_model import LogisticRegression

    capture_manifest = load_manifest(capture_dir)
    activations, labels, texts = load_capture(capture_dir)
    splits = np.asarray([split_of(t, seed) for t in texts])
    masks = {s: splits == s for s in ("train", "val", "test")}
    if not all(m.any() for m in masks.values()):
        raise TrainError("empty split; dataset is too small for an 80/10/10 split")

    per_layer = {}
    best = None
    for layer, acts in activations.items():
        acts = acts.astype(np.float32)
        mu = acts[masks["train"]].mean(axis=0)
        sigma = acts[masks["train"]].std(axis=0) + 1e-6
        normalized = (acts - mu) / sigma

        best_c, best_val = None, -1.0
        for c in c_sweep:
            clf = LogisticRegression(C=c, max_iter=2000)
            clf.fit(normalized[masks["train"]], labels[masks["train"]])
            val_scores = clf.decision_function(normalized[masks["val"]])
            val_auroc = _auroc(labels[masks["val"]], val_scores)
            if val_auroc > best_val:
                best_c, best_val = c, val_auroc

        clf = LogisticRegression(C=best_c, max_iter=2000)
        clf.fit(normalized[masks["train"]], labels[masks["train"]])
        test_scores = clf.decision_function(normalized[masks["test"]])
        test_probs = clf.predict_proba(normalized[masks["test"]])[:, 1]
        val_probs = clf.predict_proba(normalized[masks["val"]])[:, 1]
        metrics = {
            "C": best_c,
            "val_auroc": round(best_val, 4),
            "test_auroc": round(_auroc(labels[masks["test"]], test_scores), 4),
            "test_accuracy": round(
                float(((test_probs >= 0.5) == labels[masks["test"]]).mean()), 4
            ),
        }
        per_layer[layer] = metrics
        if best is None or metrics["val_auroc"] > best["metrics"]["val_auroc"]:
            # threshold chosen on validation data (W2 decision 4): the point
            # closest to a 0.5 target rate on val probabilities
            threshold = float(np.median(val_probs))
            best = {
                "layer": layer,
                "weights": clf.coef_[0].astype(np.float32),
                "bias": float(clf.intercept_[0]),
                "mu": mu,
                "sigma": sigma,
                "threshold": threshold,
                "metrics": metrics,
            }

    if best is None:
        raise TrainError("capture contains no layers; nothing to train")

    probe_dir = os.path.join(out_root, name)
    os.makedirs(probe_dir, exist_ok=True)
    probe_path = os.path.join(probe_dir, PROBE_NAME)
    np.savez(
        probe_path,
        weights=best["weights"],
        bias=np.float32(best["bias"]),
        layer=np.int64(best["layer"]),
        mu=best["mu"],
        sigma=best["sigma"],
        threshold=np.float32(best["threshold"]),
    )
    manifest = {
        "schema_version": 1,
        "artifact": "probe",
        "name": name,
        "capture_dir": capture_dir,
        "capture_manifest_xxhash": file_xxhash(
            os.path.join(capture_dir, "manifest.json")
        ),
        "model_id": capture_manifest["model_id"],
        "model_hash": capture_manifest["model_hash"],
        "tokenizer_fingerprint": capture_manifest["tokenizer_fingerprint"],
        "layer": best["layer"],
        "split_seed": seed,
        "threshold": best["threshold"],
        "metrics": best["metrics"],
        "per_layer_metrics": {str(k): v for k, v in per_layer.items()},
        "probe_xxhash": file_xxhash(probe_path),
        "git_sha": git_sha(),
        "created": utc_now(),
    }
    write_manifest(probe_dir, manifest)
    return manifest
