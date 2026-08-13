"""Service wiring helpers (W1 decision 3).

The serving side of steering is env vars in the existing entrypoint pattern:
CONTROL_VECTOR (container path; empty default means the flag is omitted and
production behavior is unchanged), CONTROL_VECTOR_SCALE, and
CONTROL_VECTOR_LAYER_RANGE ("START END"). This module validates a vector
directory before `make steer-up` recreates the service, and maps the host
vectors tree to the container mount.
"""

from __future__ import annotations

import os

from interp.manifest import ManifestError, load_manifest
from interp.steering.export import VECTOR_NAME

HOST_VECTORS_ROOT = "/mnt/ai-data/interp/vectors"
CONTAINER_VECTORS_ROOT = "/app/vectors"


class WiringError(Exception):
    """A vector directory is not deployable as-is."""


def validate_vector_dir(vector_dir: str, served_model_hash: str | None = None) -> dict:
    """Validate a vector directory for deployment; return its manifest.

    Checks: vector.safetensors exists, the manifest loads, the recorded xxhash matches
    the file on disk, and — when the served model's base-checkpoint hash is
    known — that the manifest model hash matches it (the identity requirement
    of the phase's model policy).
    """
    from interp.manifest import file_xxhash, require_match

    vector_path = os.path.join(vector_dir, VECTOR_NAME)
    if not os.path.isfile(vector_path):
        raise WiringError(f"vector file not found: {vector_path}")
    try:
        manifest = load_manifest(vector_dir)
    except ManifestError as exc:
        raise WiringError(str(exc)) from exc

    actual = file_xxhash(vector_path)
    if manifest.get("vector_xxhash") != actual:
        raise WiringError(
            f"{vector_path}: file hash {actual} does not match manifest "
            f"{manifest.get('vector_xxhash')} (re-export or restore the artifact)"
        )
    if served_model_hash is not None and served_model_hash != "unknown":
        try:
            require_match(manifest, "model_hash", served_model_hash, vector_dir)
        except ManifestError as exc:
            raise WiringError(str(exc)) from exc
    return manifest


def container_vector_path(host_vector_dir: str) -> str:
    """Map a host vector directory to the path the service sees.

    The compose files mount HOST_VECTORS_ROOT read-only at
    CONTAINER_VECTORS_ROOT in llama-gpu and llama-cpu.
    """
    host_vector_dir = os.path.abspath(host_vector_dir)
    root = HOST_VECTORS_ROOT.rstrip("/")
    if host_vector_dir != root and not host_vector_dir.startswith(root + "/"):
        raise WiringError(
            f"vector directory must live under {HOST_VECTORS_ROOT} "
            f"(the mounted tree), got {host_vector_dir}"
        )
    relative = os.path.relpath(host_vector_dir, root)
    inside = (
        CONTAINER_VECTORS_ROOT
        if relative == "."
        else os.path.join(CONTAINER_VECTORS_ROOT, relative)
    )
    return os.path.join(inside, VECTOR_NAME)


def steering_env(
    vector_dir: str,
    scale: float | None = None,
    layer_range: tuple[int, int] | None = None,
) -> dict[str, str]:
    """Environment variables that steer a service, for compose recreation.

    The Makefile steer-up/steer-down targets implement this same env/path
    contract independently (VECTOR_CONTAINER patsubst plus the three
    CONTROL_VECTOR* vars); a change here must be mirrored there.
    """
    env = {"CONTROL_VECTOR": container_vector_path(vector_dir)}
    if scale is not None:
        env["CONTROL_VECTOR_SCALE"] = f"{scale:g}"
    if layer_range is not None:
        env["CONTROL_VECTOR_LAYER_RANGE"] = f"{layer_range[0]} {layer_range[1]}"
    return env
