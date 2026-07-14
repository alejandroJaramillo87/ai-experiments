"""Export manifest (Phase 0 D7).

One JSON beside every exported GGUF recording where it came from and how:
source checkpoint (and its D6 meta.json), recipe snapshot hash, adapter
provenance when one was merged, the llama.cpp pin, quant type, and the output
file's SHA-256. Re-export of the same checkpoint under the same pin must yield
the same gguf.sha256 — the reproducibility acceptance check compares exactly
this field. serve_eval.py verifies gguf.sha256 before swapping a model in.
"""

from __future__ import annotations

import hashlib
import json
import os
import time


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_meta(checkpoint: str) -> dict | None:
    """The D6 meta.json (step, snapshot hash, git SHA, ...), when present."""
    for candidate in (os.path.join(checkpoint, "meta.json"),
                      os.path.join(os.path.dirname(checkpoint), "meta.json")):
        if os.path.isfile(candidate):
            with open(candidate) as f:
                return json.load(f)
    return None


def _tool_versions() -> dict:
    """Best-effort versions of the libraries that shaped the export."""
    versions = {}
    for name in ("torch", "transformers", "peft", "gguf"):
        try:
            versions[name] = __import__(name).__version__
        except Exception:
            versions[name] = None
    return versions


def build_manifest(*, checkpoint: str, architecture: str, quant_type: str,
                   outtype: str, llama_cpp_commit: str, gguf_path: str,
                   staging_info, base_identity=None,
                   base_drift: list[str] | None = None,
                   created: str | None = None) -> dict:
    """Assemble the manifest dict for one export.

    base_identity is the F6 preflight's resolved base (revision and config
    hash); base_drift carries its notes when --allow-base-drift overrode a
    mismatch, so a drifted export is detectable from the manifest alone.
    """
    return {
        "schema_version": 1,
        "created": created or time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "source_checkpoint": os.path.abspath(checkpoint),
        "checkpoint_meta": checkpoint_meta(checkpoint),
        "architecture": architecture,
        "adapter": None if staging_info.adapter_path is None else {
            "path": os.path.abspath(staging_info.adapter_path),
            "sha256": _adapter_hash(staging_info.adapter_path),
            "base_model": staging_info.base_model,
            "base_revision": getattr(base_identity, "revision", None),
            "base_config_sha256": getattr(base_identity, "config_sha256", None),
            "merge_method": staging_info.merge_method,
        },
        "base_drift": base_drift or [],
        "tokenizer_source": staging_info.tokenizer_source,
        "llama_cpp_commit": llama_cpp_commit,
        "convert_outtype": outtype,
        "quant_type": quant_type,
        "gguf": {
            "path": os.path.abspath(gguf_path),
            "sha256": sha256_file(gguf_path),
            "size_bytes": os.path.getsize(gguf_path),
        },
        "tool_versions": _tool_versions(),
    }


def _adapter_hash(adapter_dir: str) -> str | None:
    """SHA-256 of the adapter weights file, when present."""
    for name in ("adapter_model.safetensors", "adapter_model.bin"):
        path = os.path.join(adapter_dir, name)
        if os.path.isfile(path):
            return sha256_file(path)
    return None


def manifest_path(gguf_path: str) -> str:
    root, _ext = os.path.splitext(gguf_path)
    return root + ".manifest.json"


def write_manifest(manifest: dict, gguf_path: str) -> str:
    """Write the manifest beside the GGUF; returns its path."""
    path = manifest_path(gguf_path)
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
    return path
