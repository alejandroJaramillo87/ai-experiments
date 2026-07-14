"""Control-vector GGUF export (W1 decision 2).

One exporter for both extraction methods, writing the llama.cpp control-vector
layout (ggml-org/llama.cpp PR 5970): architecture "controlvector", a
model_hint string, layer_count, and one float32 tensor per layer named
"direction.<layer>" (1-based). Using the gguf package directly for repeng
vectors too — instead of repeng's own exporter — keeps a single deterministic
code path, which is what makes the re-export-reproduces-the-same-hash
acceptance check meaningful.

Artifacts land at <out>/<name>/: vector.gguf plus manifest.json. Downstream
tools (wiring.py, make steer-up) refuse a vector whose manifest model hash
does not match the served model's base checkpoint.
"""

from __future__ import annotations

import os

import numpy as np

from interp.manifest import file_xxhash, git_sha, utc_now, write_manifest
from interp.steering.extract import DirectionSet

VECTOR_NAME = "vector.gguf"
ARCH = "controlvector"


class ExportError(Exception):
    """The direction set cannot be written as a control-vector GGUF."""


def write_control_vector_gguf(
    directions: dict[int, np.ndarray], path: str, model_hint: str = "unknown"
) -> None:
    """Write direction tensors in llama.cpp control-vector GGUF layout."""
    import gguf

    if not directions:
        raise ExportError("direction set is empty")
    widths = {v.shape for v in directions.values()}
    if len(widths) != 1 or len(next(iter(widths))) != 1:
        raise ExportError(f"directions must share one 1-D width, got shapes {widths}")
    if min(directions) < 1:
        raise ExportError(
            f"layer indices must be 1-based (llama.cpp convention), got {min(directions)}"
        )

    writer = gguf.GGUFWriter(path, ARCH)
    writer.add_string(f"{ARCH}.model_hint", model_hint)
    writer.add_uint32(f"{ARCH}.layer_count", max(directions))
    for layer in sorted(directions):
        writer.add_tensor(f"direction.{layer}", directions[layer].astype(np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def export_control_vector(
    direction_set: DirectionSet,
    out_dir: str,
    model_hash: str = "unknown",
    layer_range: tuple[int, int] | None = None,
) -> dict:
    """Write vector.gguf plus its manifest; return the manifest."""
    os.makedirs(out_dir, exist_ok=True)
    vector_path = os.path.join(out_dir, VECTOR_NAME)
    write_control_vector_gguf(
        direction_set.directions,
        vector_path,
        model_hint=str(direction_set.source.get("model_type", "unknown")),
    )

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
