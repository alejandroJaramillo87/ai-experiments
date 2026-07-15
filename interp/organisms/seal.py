"""Ground-truth sealing for model organisms (P4-D1 decision 6).

Each organism directory carries ground_truth.json (the secret, the fact list,
the implant mechanism) whose SHA-256 is recorded in the public manifest at
creation time — informal pre-registration, so a post-hoc "we knew it all
along" is impossible and an auditing game's answer can be graded. Blinding is
procedural: auditors receive the public manifest only, so sealing refuses any
public manifest that leaks ground-truth content.
"""

from __future__ import annotations

import hashlib
import json
import os

from interp.manifest import git_sha, load_manifest, utc_now, write_manifest

GROUND_TRUTH_NAME = "ground_truth.json"
SCHEMA_VERSION = 1

# Ground-truth string values shorter than this are too generic to treat as a
# leak (articles, "a", "is"); everything at or above it must not appear in
# the public manifest.
MIN_LEAK_LENGTH = 4


class SealError(Exception):
    """Sealing refused or a seal check failed; message says why."""


def _ground_truth_strings(value) -> list[str]:
    """Every string value in the ground truth, recursively, long enough to be
    identifying. Keys are not collected: field names ("secret", "facts") are
    part of the schema, not the property."""
    found: list[str] = []
    if isinstance(value, str):
        if len(value) >= MIN_LEAK_LENGTH:
            found.append(value)
    elif isinstance(value, dict):
        for child in value.values():
            found.extend(_ground_truth_strings(child))
    elif isinstance(value, list):
        for child in value:
            found.extend(_ground_truth_strings(child))
    return found


def check_no_leak(public: dict, ground_truth: dict) -> None:
    """Refuse when any identifying ground-truth string appears anywhere in
    the serialized public manifest (case-insensitive substring). The
    mechanism field is exempt from nothing: the public manifest may say an
    organism exists, never what was implanted or how it is worded."""
    haystack = json.dumps(public, sort_keys=True, ensure_ascii=False).lower()
    # kind is schema, not property: the public manifest deliberately states
    # what game is being played (taboo, false_fact), never its answer.
    identifying = {k: v for k, v in ground_truth.items() if k != "kind"}
    for needle in _ground_truth_strings(identifying):
        if needle.lower() in haystack:
            raise SealError(
                f"public manifest leaks ground truth: {needle[:40]!r} "
                f"appears in the manifest; remove it before sealing")


def ground_truth_sha256(path: str) -> str:
    """SHA-256 of the ground-truth file bytes (the sealed pre-registration)."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def seal_organism(out_dir: str, ground_truth: dict, public: dict) -> dict:
    """Write ground_truth.json and the public manifest; return the manifest.

    ground_truth must carry the property itself (kind-specific fields plus
    "mechanism"); public carries only blinded provenance (model identity,
    recipe hash, verification report hash). Refuses to overwrite an existing
    seal — a sealed organism is immutable; make a new organism instead.
    """
    if not isinstance(ground_truth, dict) or not ground_truth:
        raise SealError("ground truth must be a non-empty JSON object")
    if "mechanism" not in ground_truth:
        raise SealError("ground truth must record the implant mechanism "
                        "under the 'mechanism' key")
    truth_path = os.path.join(out_dir, GROUND_TRUTH_NAME)
    if os.path.exists(truth_path):
        raise SealError(
            f"{truth_path} already exists; a sealed organism is immutable "
            f"(create a new organism directory instead)")
    check_no_leak(public, ground_truth)

    os.makedirs(out_dir, exist_ok=True)
    tmp = truth_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(ground_truth, f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp, truth_path)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "sealed_at": utc_now(),
        "git_sha": git_sha(),
        "ground_truth_sha256": ground_truth_sha256(truth_path),
        **public,
    }
    write_manifest(out_dir, manifest)
    return manifest


def check_seal(out_dir: str) -> dict:
    """Verify a sealed organism: the ground truth still hashes to the value
    pre-registered in the public manifest. Returns the manifest on success."""
    manifest = load_manifest(out_dir)
    expected = manifest.get("ground_truth_sha256")
    if not expected:
        raise SealError(
            f"{out_dir}: public manifest has no ground_truth_sha256; "
            f"this directory was never sealed")
    truth_path = os.path.join(out_dir, GROUND_TRUTH_NAME)
    if not os.path.isfile(truth_path):
        raise SealError(f"{out_dir}: {GROUND_TRUTH_NAME} is missing")
    actual = ground_truth_sha256(truth_path)
    if actual != expected:
        raise SealError(
            f"{out_dir}: ground truth hash mismatch: sealed "
            f"{expected}, current file hashes to {actual} "
            f"(the ground truth was modified after sealing)")
    return manifest


def load_ground_truth(out_dir: str) -> dict:
    """The sealed ground truth, hash-verified against the public manifest."""
    check_seal(out_dir)
    with open(os.path.join(out_dir, GROUND_TRUTH_NAME), encoding="utf-8") as f:
        return json.load(f)
