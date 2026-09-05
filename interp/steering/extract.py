"""Direction extraction (W1 decision 1): two backends behind one interface.

method "diff_means": the mean residual of the label-1 examples minus the mean
residual of the label-0 examples, per layer, read from an activation-capture
directory (the W2 capture format, written by `probe-capture` on the GPU or by
the rust harness's `activation-capture` on the CPU). method "sae_row": a
single feature's decoder row from a pretrained SAE (Gemma Scope via SAELens)
placed at its layer, zero elsewhere. Both return the same in-memory
DirectionSet; export.py writes it as a safetensors image plus a JSON manifest
beside it (schemas/control_vector_artifact.json).

Until 2026-09 the first method was repeng's ControlVector.train, and this
docstring said there was no third method because hand-rolled activation
differences were what repeng already did. repeng left for two reasons. Its
only PyPI release, 0.4.0, requires numpy<2 and cannot import under the tier's
numpy pin (KB-32), and the promised out-of-process venv never landed. And the
repository already produced repeng's input: a capture holds exactly the
last-token residuals repeng would have collected, at the block output where
the harness adds the vector, with the model hash the harness checks.

Conventions the harness relies on:

- Sign: label 1 is the attribute. Adding the vector at a positive scale moves
  the stream toward the label-1 examples.
- Magnitude: the raw mean difference, unnormalised. The difference of two
  means is already in residual-stream units, so `--control-vector-scale 1.0`
  means "one class gap" at every layer, and that scale flag is the only knob.
- Layers: every layer in the capture unless `layers` narrows it. Serving-time
  narrowing is `--control-vector-layer-range` on the harness.

Extraction reads numpy only, so this module and the diff_means path run in
the CPU sandbox; the SAE backend imports sae_lens lazily.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np

from interp.manifest import load_manifest

METHOD_DIFF_MEANS = "diff_means"
METHOD_SAE_ROW = "sae_row"
UNKNOWN_MODEL_HASH = "unknown"


class ExtractionError(Exception):
    """Extraction input is malformed or a backend failed."""


@dataclass
class DirectionSet:
    """Per-layer steering directions plus the provenance export.py records.

    directions maps a 0-based decoder block index to a float32 vector of the
    residual-stream width: the direction is added to the output residual of
    that block.

    The index has always been the block itself, never the block plus one.
    Until R10 the exporter additionally refused index 0, because llama.cpp
    and repeng describe this numbering as 1-based and leave the first block
    unsteerable; that floor went with the llama.cpp steering path, and the
    indices it admitted did not move.

    model_hash is the sha256 of the checkpoint's model.safetensors when the
    extractor can know it (diff_means reads it from the capture manifest) and
    "unknown" otherwise; the exporter records it and the harness checks it.
    """

    directions: dict[int, np.ndarray]
    model_id: str
    model_revision: str
    method: str  # "diff_means" | "sae_row"
    source: dict = field(default_factory=dict)  # capture identity or SAE identity
    model_hash: str = UNKNOWN_MODEL_HASH

    def layer_count(self) -> int:
        return max(self.directions) if self.directions else 0


def extract_diff_means(
    capture_dir: str,
    layers: list[int] | None = None,
) -> DirectionSet:
    """Mean of label-1 rows minus mean of label-0 rows, per captured layer.

    The capture loader already checks that shards, labels and texts agree in
    count; this adds the checks a class contrast needs. Means accumulate in
    float64 and the direction is cast to float32 once, so re-extracting from
    the same capture reproduces the same bytes (the W1 acceptance check).
    """
    from interp.probes.capture import load_capture

    manifest = load_manifest(capture_dir)
    activations, labels, _texts = load_capture(capture_dir)
    if labels.size == 0:
        raise ExtractionError(f"{capture_dir}: capture is empty")

    classes = sorted(int(c) for c in np.unique(labels))
    if classes != [0, 1]:
        if len(classes) == 1:
            raise ExtractionError(
                f"{capture_dir}: capture holds only label {classes[0]}; "
                "difference of means needs both classes"
            )
        raise ExtractionError(
            f"{capture_dir}: labels must be exactly 0 and 1, got {classes}"
        )

    captured = sorted(activations)
    if layers is None:
        chosen = captured
    else:
        missing = sorted(set(layers) - set(captured))
        if missing:
            raise ExtractionError(
                f"{capture_dir}: layers {missing} not in capture "
                f"(captured: {captured}); re-run capture with --layers"
            )
        chosen = sorted(set(layers))

    positive = labels == 1
    negative = labels == 0
    directions: dict[int, np.ndarray] = {}
    for layer in chosen:
        acts = activations[layer].astype(np.float64)
        direction = acts[positive].mean(axis=0) - acts[negative].mean(axis=0)
        if not np.isfinite(direction).all():
            raise ExtractionError(
                f"{capture_dir}: layer {layer} yields a non-finite direction"
            )
        directions[int(layer)] = direction.astype(np.float32)

    source = {
        "capture_dir": os.path.abspath(capture_dir),
        "dataset_path": manifest.get("dataset_path"),
        "dataset_sha256": manifest.get("dataset_sha256"),
        "count": int(labels.size),
        "positive_count": int(positive.sum()),
        "negative_count": int(negative.sum()),
        "position": manifest.get("position"),
    }
    # Written by the rust capture only; a Python capture has neither key.
    for key in ("site", "capture_backend"):
        if key in manifest:
            source[key] = manifest[key]

    return DirectionSet(
        directions=directions,
        model_id=str(manifest.get("model_id", "")),
        model_revision=str(manifest.get("model_revision", "main")),
        method=METHOD_DIFF_MEANS,
        source=source,
        model_hash=str(manifest.get("model_hash", UNKNOWN_MODEL_HASH)),
    )


def extract_sae_row(
    model_id: str,
    sae_release: str,
    sae_id: str,
    feature_index: int,
    layer: int,
    revision: str = "main",
) -> DirectionSet:
    """Read one SAE feature's decoder row as a single-layer direction."""
    from sae_lens import SAE

    sae = SAE.from_pretrained(release=sae_release, sae_id=sae_id)
    # SAELens >=6 returns the SAE directly; older versions returned a tuple
    if isinstance(sae, tuple):
        sae = sae[0]
    w_dec = sae.W_dec.detach().float().cpu().numpy()
    if not 0 <= feature_index < w_dec.shape[0]:
        raise ExtractionError(
            f"feature index {feature_index} out of range for SAE with "
            f"{w_dec.shape[0]} features"
        )
    direction = np.asarray(w_dec[feature_index], dtype=np.float32)
    return DirectionSet(
        directions={int(layer): direction},
        model_id=model_id,
        model_revision=revision,
        method=METHOD_SAE_ROW,
        source={
            "sae_release": sae_release,
            "sae_id": sae_id,
            "feature_index": feature_index,
            "layer": layer,
        },
    )
