"""Artifact manifest helpers shared by the write-back deliverables (W1-W4).

Every artifact directory under /mnt/ai-data/interp/ carries a manifest.json
tracing it to its inputs: git SHA, model identity and hash, dataset hashes,
tool versions, timestamps. Downstream tools refuse artifacts whose manifest
hashes do not match their configuration; these helpers are that one code path.
"""

from __future__ import annotations

import datetime
import json
import os
import subprocess

import xxhash

MANIFEST_NAME = "manifest.json"


class ManifestError(Exception):
    """A manifest is missing, malformed, or fails a hash check."""


def utc_now() -> str:
    return datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def git_sha(repo_root: str | None = None) -> str:
    """Current commit SHA, or 'unknown' outside a git checkout."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def file_xxhash(path: str) -> str:
    """xxh3_64 hex digest of a file, streamed in 1MB blocks."""
    h = xxhash.xxh3_64()
    with open(path, "rb") as f:
        while block := f.read(1 << 20):
            h.update(block)
    return h.hexdigest()


def text_sha256(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_manifest(directory: str, payload: dict) -> str:
    """Write manifest.json atomically (tmp + rename) and return its path."""
    path = os.path.join(directory, MANIFEST_NAME)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp, path)
    return path


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
