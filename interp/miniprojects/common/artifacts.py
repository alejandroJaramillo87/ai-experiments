"""Pinned artifact download (interp M3).

fetch() downloads the Hugging Face artifacts a mini-project scaffold declares in the
checked-in interp/miniprojects/artifacts.yaml, pinned to explicit revisions, into
/mnt/ai-data (never the repository). Idempotent: the hub cache makes a second run a no-op.
No scaffold downloads ad hoc; a project without an artifacts.yaml entry has no artifacts.
"""

from __future__ import annotations

import os

import yaml

ARTIFACTS_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "artifacts.yaml"
)
ALLOWED_ROOTS = ("/mnt/ai-data/models", "/mnt/ai-data/interp")


def load_manifest(path: str = ARTIFACTS_FILE) -> dict:
    with open(path) as fh:
        manifest = yaml.safe_load(fh) or {}
    return manifest.get("projects", {})


def fetch(project: str, manifest_path: str = ARTIFACTS_FILE) -> list[str]:
    """Download every artifact declared for the project; returns local paths."""
    projects = load_manifest(manifest_path)
    if project not in projects:
        known = ", ".join(sorted(projects)) or "none"
        raise KeyError(f"unknown project {project!r} in artifacts.yaml; known: {known}")

    from huggingface_hub import snapshot_download

    paths: list[str] = []
    for entry in projects[project]:
        for field in ("repo_id", "revision", "dest"):
            if field not in entry:
                raise ValueError(
                    f"{project}: artifact entry missing {field!r}: {entry}"
                )
        dest = entry["dest"]
        if not any(
            dest == root or dest.startswith(root + "/") for root in ALLOWED_ROOTS
        ):
            raise ValueError(f"{project}: dest {dest!r} outside {ALLOWED_ROOTS}")
        path = snapshot_download(
            repo_id=entry["repo_id"],
            revision=entry["revision"],
            repo_type=entry.get("repo_type", "model"),
            local_dir=dest,
            allow_patterns=entry.get("allow_patterns"),
        )
        paths.append(path)
    return paths
