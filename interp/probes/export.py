"""Probe export for the rust harness in-band monitor.

Converts a registered probe artifact (probe.npz + manifest.json) into the
pair the rust runtime loads: a safetensors image of the three f32 parameter
vectors, and a JSON sidecar beside it holding the scalars. The rust side
(rust-harness runtime/src/model/probe.rs, Probe::load) reads exactly this
layout; scoring parity with interp/probes/score.py is golden-tested there.

Two files rather than one container: the tensors go where every other tensor
in this repository goes, and the scalars stay readable without a parser. The
sidecar is not optional -- it carries the layer, the site and the decision
threshold, so an image without it is unusable rather than merely unchecked --
and the reader looks for it at the image's path with a .json extension.

Contract: schemas/probe_artifact.json.
"""

from __future__ import annotations

import json
from pathlib import Path

from common.reports import report_header
from interp.probes.registry import Probe, load_probe

ARTIFACT = "probe"
SCHEMA_VERSION = 1

# The tap site a probe with no recorded site was trained at. Written only when
# the manifest names one: the reader defaults an absent site to this same
# value, and an exporter inventing it would turn a fact nobody recorded into
# one the artifact asserts.
SITE_KEY = "site"


class ProbeExportError(Exception):
    """The probe artifact cannot be written as a probe image."""


def write_probe_from_probe(probe: Probe, path: str) -> None:
    """Write a loaded probe as a safetensors image plus its JSON sidecar."""
    import numpy as np
    from safetensors.numpy import save_file

    weights = np.asarray(probe.weights, dtype=np.float32)
    mu = np.asarray(probe.mu, dtype=np.float32)
    sigma = np.asarray(probe.sigma, dtype=np.float32)
    if not (weights.ndim == 1 and weights.shape == mu.shape == sigma.shape):
        raise ProbeExportError(
            f"weights/mu/sigma must share one 1-D width, got "
            f"{weights.shape}, {mu.shape}, {sigma.shape}"
        )
    if probe.layer < 0:
        raise ProbeExportError(f"layer must be non-negative, got {probe.layer}")
    # JSON has no spelling for NaN or Infinity: json.dumps would emit a bare
    # `NaN`, which the rust reader's parser rejects. Refuse here so the
    # failure names the scalar rather than arriving later as a parse error on
    # a file that should never have been written.
    for field, value in (("bias", probe.bias), ("threshold", probe.threshold)):
        if not np.isfinite(value):
            raise ProbeExportError(f"{field} must be finite, got {value}")

    image = Path(path)
    save_file({"weights": weights, "mu": mu, "sigma": sigma}, str(image))

    sidecar = report_header(ARTIFACT, SCHEMA_VERSION)
    # report_header spells the discriminator "report"; this artifact uses the
    # "artifact" spelling the envelope also allows, which is what the reader
    # and interp/manifest.py both expect. Exactly one may be present.
    sidecar["artifact"] = sidecar.pop("report")
    sidecar.update(
        name=probe.name,
        model_hash=str(probe.manifest.get("model_hash", "unknown")),
        layer=int(probe.layer),
        bias=float(probe.bias),
        threshold=float(probe.threshold),
        width=int(weights.shape[0]),
    )
    site = probe.manifest.get(SITE_KEY)
    if site:
        sidecar[SITE_KEY] = str(site)
    image.with_suffix(".json").write_text(json.dumps(sidecar, indent=2) + "\n")


def write_probe(probe_dir: str, path: str) -> None:
    """Load a registered probe (hash-validated) and export it."""
    write_probe_from_probe(load_probe(probe_dir), path)
