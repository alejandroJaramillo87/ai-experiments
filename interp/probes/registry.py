"""Probe registry (W2 decision 4): list, load, and validate probe artifacts.

The registry is the durable half of the toolkit. Every consumer — the
evaluation report, the W4 monitor — loads probes through here, and the load
path refuses a probe whose model hash does not match the model it will score.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

from interp.manifest import (
    ManifestError,
    file_xxhash,
    list_artifact_dirs,
    load_manifest,
    require_match,
)
from interp.probes.train import PROBE_NAME

DEFAULT_ROOT = "/mnt/ai-data/interp/probes"


class RegistryError(Exception):
    """A probe artifact is missing, corrupt, or mismatched."""


@dataclass
class Probe:
    name: str
    layer: int
    weights: np.ndarray
    bias: float
    mu: np.ndarray
    sigma: np.ndarray
    threshold: float
    manifest: dict


def list_probes(root: str = DEFAULT_ROOT) -> list[str]:
    """Names of registered probes (directories containing a manifest)."""
    return list_artifact_dirs(root)


def load_probe(
    probe_dir: str,
    expect_model_hash: str | None = None,
    expect_tokenizer_fingerprint: str | None = None,
) -> Probe:
    """Load and validate one probe; refuse hash mismatches.

    expect_model_hash is the hash of the model whose activations will be
    scored; expect_tokenizer_fingerprint guards against the silent
    chat-format-mismatch failure mode recorded in the W2 risks.
    """
    try:
        manifest = load_manifest(probe_dir)
    except ManifestError as exc:
        raise RegistryError(str(exc)) from exc

    probe_path = os.path.join(probe_dir, PROBE_NAME)
    if not os.path.isfile(probe_path):
        raise RegistryError(f"probe file not found: {probe_path}")
    actual = file_xxhash(probe_path)
    if manifest.get("probe_xxhash") != actual:
        raise RegistryError(
            f"{probe_path}: file hash {actual} does not match manifest "
            f"{manifest.get('probe_xxhash')}"
        )
    try:
        if expect_model_hash is not None:
            require_match(manifest, "model_hash", expect_model_hash, probe_dir)
        if expect_tokenizer_fingerprint is not None:
            require_match(
                manifest,
                "tokenizer_fingerprint",
                expect_tokenizer_fingerprint,
                probe_dir,
            )
    except ManifestError as exc:
        raise RegistryError(str(exc)) from exc

    with np.load(probe_path) as data:
        return Probe(
            name=manifest["name"],
            layer=int(data["layer"]),
            weights=data["weights"],
            bias=float(data["bias"]),
            mu=data["mu"],
            sigma=data["sigma"],
            threshold=float(data["threshold"]),
            manifest=manifest,
        )
