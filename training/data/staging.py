"""RAM-disk shard staging (Phase 0 D9).

Optionally copies the hot shard set to a tmpfs mount (compose mounts /staging
on the training service) before the run and points the dataset there. Refuses
to stage beyond staging.max_gb and falls back to reading the NVMe mmaps
directly — which, with 128GB of page cache and inference down, is usually
already RAM-speed. Staging exists for controlled experiments and multi-epoch
small corpora, not as a default win.

Every copied shard is hash-verified against the manifest (same xxhash64
discipline as D2 shard verification), so a truncated copy can never silently
train.
"""

from __future__ import annotations

import os
import shutil
import sys

from training.data.shards import MANIFEST_NAME, hash_file, load_manifest


def _log(message: str) -> None:
    print(f"[staging] {message}", file=sys.stderr)


def shard_set_bytes(source_dir: str, manifest: dict) -> int:
    """On-disk size of the shard files plus manifest."""
    total = os.path.getsize(os.path.join(source_dir, MANIFEST_NAME))
    for rec in manifest["shards"]:
        total += os.path.getsize(os.path.join(source_dir, rec["file"]))
    return total


def _staged_copy_valid(dest_dir: str, manifest: dict) -> bool:
    """True when dest_dir already holds a hash-matching copy of every shard
    and a manifest whose parsed content equals the source's."""
    try:
        if load_manifest(dest_dir) != manifest:
            return False
    except (FileNotFoundError, ValueError):
        # Missing or torn manifest (kill mid-copy): the copy must be redone,
        # not reused on file existence alone
        return False
    for rec in manifest["shards"]:
        path = os.path.join(dest_dir, rec["file"])
        if not os.path.isfile(path) or hash_file(path) != rec["xxhash64"]:
            return False
    return True


def stage_shards(source_dir: str, staging) -> str:
    """Copy the shard set into staging.dir; return the directory to read from.

    Returns the staged directory on success, the original source_dir when the
    set exceeds staging.max_gb or the staging mount is absent (clean fallback,
    logged). A hash mismatch after copy is an error, not a fallback: the
    staging medium is corrupting data.
    """
    manifest = load_manifest(source_dir)
    if not os.path.isdir(staging.dir):
        _log(
            f"SKIP: staging dir {staging.dir} does not exist; "
            f"reading shards from {source_dir}"
        )
        return source_dir
    total = shard_set_bytes(source_dir, manifest)
    budget = int(staging.max_gb * 2**30)
    if total > budget:
        _log(
            f"SKIP: shard set {total / 2**30:.2f} GB exceeds staging.max_gb "
            f"{staging.max_gb}; reading shards from {source_dir}"
        )
        return source_dir

    dest_dir = os.path.join(staging.dir, os.path.basename(os.path.normpath(source_dir)))
    if _staged_copy_valid(dest_dir, manifest):
        _log(f"OK: reusing verified staged copy at {dest_dir}")
        return dest_dir

    os.makedirs(dest_dir, exist_ok=True)
    for rec in manifest["shards"]:
        src = os.path.join(source_dir, rec["file"])
        dst = os.path.join(dest_dir, rec["file"])
        shutil.copyfile(src, dst)
        digest = hash_file(dst)
        if digest != rec["xxhash64"]:
            raise OSError(
                f"staged copy of {rec['file']} hashes {digest}, manifest says "
                f"{rec['xxhash64']}: staging medium corrupted the data"
            )
    # Publish the manifest atomically, and only after every shard verified:
    # a staged dir with a manifest is otherwise trusted by the next run
    tmp_path = os.path.join(dest_dir, MANIFEST_NAME + ".tmp")
    shutil.copyfile(os.path.join(source_dir, MANIFEST_NAME), tmp_path)
    os.replace(tmp_path, os.path.join(dest_dir, MANIFEST_NAME))
    _log(f"OK: staged {total / 2**30:.2f} GB to {dest_dir}")
    return dest_dir
