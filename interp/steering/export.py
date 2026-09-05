"""Control-vector safetensors export (W1 decision 2, format per R10).

One exporter for both extraction methods, writing one float32 tensor per
steered decoder block named "block.<index>", 0-based. One deterministic code
path for every method is what makes the re-export-reproduces-the-same-hash
acceptance check meaningful.

The format is safetensors rather than the llama.cpp control-vector GGUF
layout (ggml-org/llama.cpp PR 5970) this exporter wrote until R10. That
layout was not a choice: llama.cpp read the file as --control-vector-scaled,
so the format belonged to the consumer. R9 gave the rust harness an HTTP
endpoint and R10 retired the llama.cpp steering path, leaving one reader,
in this repository, which already reads safetensors everywhere else.

The block numbering flipped with the format, and the tensor prefix was
renamed in the same step so it cannot flip silently. llama.cpp and repeng
number control vectors from 1 and leave block 0 unsteerable; probes
(interp/probes/export.py) number layers from 0. Keeping "direction.<N>"
while changing its base would have made every pre-R10 vector load and steer
one block early with no error anywhere. "block.<N>" makes it a load failure.

Artifacts land at <out>/<name>/: vector.safetensors plus manifest.json.
Downstream tools (wiring.py, make steer-up) refuse a vector whose manifest
model hash does not match the served model's base checkpoint.
"""

from __future__ import annotations

import os

import numpy as np

from interp.manifest import file_xxhash, git_sha, utc_now, write_manifest
from interp.steering.extract import DirectionSet

VECTOR_NAME = "vector.safetensors"
BLOCK_PREFIX = "block."


class ExportError(Exception):
    """The direction set cannot be written as a control-vector image."""


def write_control_vector(directions: dict[int, np.ndarray], path: str) -> None:
    """Write direction tensors as a safetensors image of `block.<index>`."""
    from safetensors.numpy import save_file

    if not directions:
        raise ExportError("direction set is empty")
    widths = {v.shape for v in directions.values()}
    if len(widths) != 1 or len(next(iter(widths))) != 1:
        raise ExportError(f"directions must share one 1-D width, got shapes {widths}")
    if min(directions) < 0:
        raise ExportError(f"block indices must be non-negative, got {min(directions)}")
    # A non-finite direction is added to every position's residual, so it
    # turns the whole stream to NaN and the model emits noise rather than
    # failing. Refused here, where the offending block can be named.
    for block, v in sorted(directions.items()):
        if not np.isfinite(v).all():
            raise ExportError(f"direction for block {block} holds a non-finite value")

    save_file(
        {
            f"{BLOCK_PREFIX}{block}": directions[block].astype(np.float32)
            for block in sorted(directions)
        },
        path,
    )


def export_control_vector(
    direction_set: DirectionSet,
    out_dir: str,
    model_hash: str | None = None,
    layer_range: tuple[int, int] | None = None,
) -> dict:
    """Write vector.safetensors plus its manifest; return the manifest.

    model_hash defaults to the direction set's own, which diff_means reads
    from the capture manifest; a caller passes one only when the extractor
    could not know it (sae_row).
    """
    if model_hash is None:
        model_hash = direction_set.model_hash
    os.makedirs(out_dir, exist_ok=True)
    vector_path = os.path.join(out_dir, VECTOR_NAME)
    write_control_vector(direction_set.directions, vector_path)

    manifest = {
        "schema_version": 1,
        "artifact": "control_vector",
        "model_id": direction_set.model_id,
        "model_revision": direction_set.model_revision,
        "model_hash": model_hash,
        "method": direction_set.method,
        "source": direction_set.source,
        "layers": sorted(direction_set.directions),
        "layer_range": list(layer_range) if layer_range else None,
        "width": int(next(iter(direction_set.directions.values())).shape[0]),
        "vector_xxhash": file_xxhash(vector_path),
        "git_sha": git_sha(),
        "created": utc_now(),
    }
    write_manifest(out_dir, manifest)
    return manifest
