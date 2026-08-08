"""Artifact manifest helpers shared by the write-back deliverables (W1-W4).

Every artifact directory under /mnt/ai-data/interp/ carries a manifest.json
tracing it to its inputs: git SHA, model identity and hash, dataset hashes,
tool versions, timestamps. Downstream tools refuse artifacts whose manifest
hashes do not match their configuration; these helpers are that one code path.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import os

from common.hashing import hash_file_with, write_json_atomic
from common.provenance import git_sha  # noqa: F401  (re-exported)

MANIFEST_NAME = "manifest.json"


class ManifestError(Exception):
    """A manifest is missing, malformed, or fails a hash check."""


def utc_now() -> str:
    return datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def file_xxhash(path: str) -> str:
    """xxh3_64 hex digest of a file, streamed in 1MB blocks (recorded
    algorithm; do not change without a manifest version bump).

    xxhash is imported here rather than at module scope. It is the only
    non-stdlib dependency of this module, and every artifact module in interp/
    imports from here, so a module-level import made the whole artifact layer
    unloadable wherever xxhash is missing — including the sandbox interpreter
    the cross-language contract tests run the exporters under. Same rationale
    as the lazy torch and nnsight imports in interp/probes/capture.py.
    """
    import xxhash

    return hash_file_with(xxhash.xxh3_64(), path, 1 << 20)


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def tokenizer_fingerprint(tokenizer) -> str:
    """Stable fingerprint of the tokenizer + chat template used at capture time.

    Checked at score time against the probe manifest: a tokenizer or template
    mismatch between capture and scoring corrupts results silently (W2 risk).
    """
    parts = [
        str(getattr(tokenizer, "name_or_path", "")),
        str(len(tokenizer)),
        str(getattr(tokenizer, "chat_template", "")),
    ]
    return text_sha256("|".join(parts))


def write_manifest(directory: str, payload: dict) -> str:
    """Write manifest.json atomically (tmp + rename) and return its path."""
    return write_json_atomic(os.path.join(directory, MANIFEST_NAME), payload)


def list_artifact_dirs(root: str) -> list[str]:
    """Sorted names of artifact directories under root that carry a manifest.

    Shared by the probe and SAE registries so their listing semantics cannot
    drift.
    """
    if not os.path.isdir(root):
        return []
    return sorted(
        entry
        for entry in os.listdir(root)
        if os.path.isfile(os.path.join(root, entry, MANIFEST_NAME))
    )


def load_manifest(directory: str) -> dict:
    path = os.path.join(directory, MANIFEST_NAME)
    if not os.path.isfile(path):
        raise ManifestError(f"manifest not found: {path}")
    try:
        with open(path, encoding="utf-8") as f:
            manifest = json.load(f)
    except json.JSONDecodeError as exc:
        raise ManifestError(f"manifest is not valid JSON: {path}: {exc}") from exc
    if not isinstance(manifest, dict):
        raise ManifestError(f"manifest is not a JSON object: {path}")
    return manifest


def require_stat(stats: dict, key: str, context: str) -> object:
    """Read a manifest field from a runner seam, refusing a default.

    A field that records a fact gets no fallback value. KB-62 is what the
    alternative looks like: `stats.get(key, <plausible default>)` let a
    runner that returned nothing write a confident-looking value into the
    artifact, and nothing downstream could tell it apart from a measurement.
    Raising costs the GPU hours already spent, which is why the message names
    the seam and the keys it did return rather than raising a bare KeyError.
    """
    if key not in stats:
        raise ManifestError(
            f"{context}: runner returned no {key!r}; a runner must supply "
            f"every manifest field it owns (got keys {sorted(stats)})"
        )
    return stats[key]


def require_match(manifest: dict, key: str, expected: str, context: str) -> None:
    """Refuse (raise ManifestError) when a manifest field mismatches.

    The refusal message names both values so the operator can see which side
    is stale instead of guessing.
    """
    actual = manifest.get(key)
    if actual != expected:
        raise ManifestError(
            f"{context}: manifest {key} mismatch: manifest has "
            f"{actual!r}, expected {expected!r}"
        )
