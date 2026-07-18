"""Probe scoring (W2 decision 5): one importable code path.

Both the evaluation report and the W4 monitor call these functions. Scoring
takes activations directly, or a model + texts and does its own single-pass
capture through the same capture path used at training time — the parity
between those two routes is unit-tested.
"""

from __future__ import annotations

import numpy as np

from interp.manifest import require_match
from interp.probes.registry import Probe


def score_activations(probe: Probe, activations: np.ndarray) -> np.ndarray:
    """Sigmoid probe scores for an (n, d_model) activation array."""
    normalized = (activations.astype(np.float32) - probe.mu) / probe.sigma
    logits = normalized @ probe.weights + probe.bias
    return 1.0 / (1.0 + np.exp(-logits))


def score_texts(
    probes: list[Probe],
    texts: list[str],
    model_id: str,
    revision: str = "main",
    batch_size: int = 16,
    capture_fn=None,
) -> dict[str, np.ndarray]:
    """Score texts with every probe via one shared capture pass.

    capture_fn(texts, layers) -> ({layer: acts}, tokenizer_fingerprint) is
    injectable for tests and for the monitor's replay; the default is the real
    bf16 nnsight capture.
    """
    layers = sorted({p.layer for p in probes})
    if capture_fn is None:
        from interp.probes.capture import capture_activations

        activations, fingerprint = capture_activations(
            model_id, texts, layers, revision=revision, batch_size=batch_size
        )
    else:
        activations, fingerprint = capture_fn(texts, layers)
    # The documented score-time guard: scoring through a tokenizer or chat
    # template different from the one the probe was trained on produces
    # meaningless scores, so every probe's recorded fingerprint must match
    for probe in probes:
        require_match(
            probe.manifest,
            "tokenizer_fingerprint",
            fingerprint,
            f"probe {probe.name}",
        )
    return {
        probe.name: score_activations(probe, activations[probe.layer])
        for probe in probes
    }
