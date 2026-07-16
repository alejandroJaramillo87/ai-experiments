"""Shared file-hashing and durable-write helpers.

Stdlib-only. The xxhash-based manifest hashes (interp/manifest.file_xxhash,
training/data/shards.hash_file) keep their recorded algorithms and block
sizes as thin wrappers over hash_file_with, so consolidation cannot silently
change a recorded digest.
"""

from __future__ import annotations

import hashlib
import json
import os


def hash_file_with(digest, path: str, block_bytes: int) -> str:
    """Stream a file through a hasher object; returns the hex digest."""
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(block_bytes), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_file(path: str) -> str:
    return hash_file_with(hashlib.sha256(), path, 1 << 20)


def write_json_atomic(path: str, payload: dict, *, sort_keys: bool = True,
                      fsync: bool = False) -> str:
    """Write JSON via tmp-file-plus-rename so a crash never leaves a
    truncated file at the final path; fsync=True additionally makes the
    contents durable before the rename publishes them."""
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=sort_keys)
        f.write("\n")
        if fsync:
            f.flush()
            os.fsync(f.fileno())
    os.replace(tmp, path)
    return path
