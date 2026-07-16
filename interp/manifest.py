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

import xxhash

from common.hashing import hash_file_with, write_json_atomic
from common.provenance import git_sha  # noqa: F401  (re-exported)

MANIFEST_NAME = "manifest.json"


class ManifestError(Exception):
    """A manifest is missing, malformed, or fails a hash check."""


def utc_now() -> str:
    return datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")




def file_xxhash(path: str) -> str:
    """xxh3_64 hex digest of a file, streamed in 1MB blocks (recorded
    algorithm; do not change without a manifest version bump)."""
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
