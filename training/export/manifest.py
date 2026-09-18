"""Export manifests.

The HF export manifest is the one that matters: one JSON beside every
exported hf/ directory (<out_root>/<name>/manifest.json) recording where the
weights came from and how, and model_hash, the sha256 of the written
model.safetensors. That digest is the identity every consumer of the
checkpoint compares against: interp/probes/capture.py derives the same value
when it captures, the organism seal records it, and the rust harness hashes
the served weights against it under --status --deep. Writing it at export
time means the value a probe or control vector will carry exists before
anything is fitted to the checkpoint. The key set is pinned by
schemas/hf_export.manifest.json.
"""

from __future__ import annotations

import json
import os
import time

from common.hashing import (
    sha256_file,  # noqa: F401  (re-exported)
    write_json_atomic,
)
from training.export.consolidate import WEIGHTS_FILE
from training.export.errors import ExportError

HF_ARTIFACT = "hf_export"  # the report-envelope discriminator
HF_DIR_NAME = "hf"  # <out_root>/<name>/hf/ holds the checkpoint
HF_MANIFEST_NAME = "manifest.json"  # beside hf/, not inside it
HF_TOOLS = ("torch", "transformers", "peft")  # the libraries a merge depends on


def checkpoint_meta(checkpoint: str) -> dict | None:
    """The D6 meta.json (step, snapshot hash, git SHA, ...), when present."""
    for candidate in (
        os.path.join(checkpoint, "meta.json"),
        os.path.join(os.path.dirname(checkpoint), "meta.json"),
    ):
        if os.path.isfile(candidate):
            with open(candidate) as f:
                return json.load(f)
    return None


def _tool_versions(names=HF_TOOLS) -> dict:
    """Best-effort versions of the libraries that shaped the export."""
    versions = {}
    for name in names:
        try:
            versions[name] = __import__(name).__version__
        except Exception:
            versions[name] = None
    return versions


def _adapter_hash(adapter_dir: str) -> str | None:
    """SHA-256 of the adapter weights file, when present."""
    for name in ("adapter_model.safetensors", "adapter_model.bin"):
        path = os.path.join(adapter_dir, name)
        if os.path.isfile(path):
            return sha256_file(path)
    return None


def hf_manifest_path(out_dir: str) -> str:
    """<out_root>/<name>/manifest.json, beside the hf/ directory it describes.

    Beside rather than inside: the checkpoint directory is what the harness
    and transformers load, and a file they do not know is noise at best."""
    return os.path.join(out_dir, HF_MANIFEST_NAME)


def build_hf_manifest(
    *,
    checkpoint: str,
    hf_dir: str,
    staging_info,
    base_identity,
    mode: str,
    base_drift: list[str] | None = None,
    created: str | None = None,
) -> dict:
    """Assemble the manifest for one hf/ export.

    mode is "merge" (an adapter merged into its base) or "copy" (a full
    model consolidated as it was). base_identity is the resolved base for a
    merge and the source directory's own identity for a copy; base_drift
    carries the F6 notes when --allow-base-drift overrode a mismatch, so a
    drifted export is detectable from the manifest alone. model_hash is
    computed here, from the file as written.
    """
    from common.provenance import git_sha

    weights = os.path.join(hf_dir, WEIGHTS_FILE)
    return {
        "schema_version": 1,
        "artifact": HF_ARTIFACT,
        "created": created or time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "git_sha": git_sha(),
        "source_checkpoint": os.path.abspath(checkpoint),
        "checkpoint_meta": checkpoint_meta(checkpoint),
        "mode": mode,
        "base": {
            "model": base_identity.model,
            "revision": base_identity.revision,
            "config_sha256": base_identity.config_sha256,
        },
        "adapter": None
        if staging_info.adapter_path is None
        else {
            "path": os.path.abspath(staging_info.adapter_path),
            "sha256": _adapter_hash(staging_info.adapter_path),
            "merge_method": staging_info.merge_method,
        },
        "base_drift": base_drift or [],
        "tokenizer_source": staging_info.tokenizer_source,
        "hf_dir": os.path.abspath(hf_dir),
        "weights_file": WEIGHTS_FILE,
        "size_bytes": os.path.getsize(weights),
        "model_hash": sha256_file(weights),
        "tool_versions": _tool_versions(HF_TOOLS),
    }


def write_hf_manifest(manifest: dict, out_dir: str) -> str:
    """Write beside hf/, atomically: a crash mid-write must not leave a
    truncated manifest beside a complete checkpoint."""
    return write_json_atomic(hf_manifest_path(out_dir), manifest, sort_keys=False)


def load_hf_manifest(hf_dir: str) -> dict:
    """The manifest describing hf_dir, or ExportError.

    Read by whatever needs the checkpoint's recorded identity: the organism
    seal, eval-ft. Refuses a manifest that is absent, is not an hf export,
    or describes a different directory, so a seal cannot quietly record the
    hash of a neighbour."""
    hf_dir = os.path.abspath(hf_dir)
    path = hf_manifest_path(os.path.dirname(hf_dir))
    if not os.path.isfile(path):
        raise ExportError(
            f"No export manifest beside {hf_dir} (looked for {path})\n"
            "  Solution: point at the hf/ directory python -m training export "
            "wrote, or pass the hash explicitly"
        )
    with open(path, encoding="utf-8") as f:
        manifest = json.load(f)
    if manifest.get("artifact") != HF_ARTIFACT:
        raise ExportError(f"Not an hf export manifest: {path}")
    recorded = manifest.get("hf_dir", "")
    if os.path.realpath(recorded) != os.path.realpath(hf_dir):
        raise ExportError(f"Export manifest {path} describes {recorded}, not {hf_dir}")
    if not manifest.get("model_hash"):
        raise ExportError(f"Export manifest records no model_hash: {path}")
    return manifest
