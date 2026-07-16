"""Base-model identity resolution and drift preflight (Phase 1 F6).

An adapter-only checkpoint is merged into its base model at export time; the
merge is only meaningful against the base the adapter was trained on. F3
records the base identity (hub revision and config.json hash) in each
checkpoint's meta.json; this module resolves the same identity at export time
and refuses on mismatch unless drift is explicitly allowed.

Resolution is cache-first by design (resolved F6 decision 2): the export
container mounts the training HF cache and never hits the network, so the
merge consumes the same snapshot training did. transformers is imported
lazily so the CPU sandbox can unit-test the pure-logic paths.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

from training.export.convert import ExportError
from training.export.manifest import sha256_file

CONFIG_NAME = "config.json"

# HF cache layout: .../models--org--name/snapshots/<commit>/config.json
_SNAPSHOT_RE = re.compile(r"[/\\]snapshots[/\\]([0-9a-f]{40})[/\\]")

# meta.json field -> BaseIdentity attribute, for the drift comparison
_MATCH_FIELDS = (("base_revision", "revision"), ("base_config_sha256", "config_sha256"))


@dataclass
class BaseIdentity:
    """What was resolved for a base model, at train or export time."""

    model: str  # hub id or local path as given
    revision: str | None  # hub snapshot commit; None for local paths
    config_sha256: str  # SHA-256 of the raw config.json bytes
    config_path: str


def commit_from_cache_path(path: str) -> str | None:
    """The hub commit encoded in an HF cache snapshot path, or None."""
    match = _SNAPSHOT_RE.search(path)
    return match.group(1) if match else None


def resolve_base_identity(
    model: str, revision: str | None = None, local_files_only: bool = False
) -> BaseIdentity:
    """Resolve a base model's identity from its config.json.

    Local directories are read directly; hub ids resolve through the HF
    cache (downloading only when local_files_only is false — export always
    passes true, per the cache-first policy).
    """
    if os.path.isdir(model):
        config_path = os.path.join(model, CONFIG_NAME)
        if not os.path.isfile(config_path):
            raise ExportError(f"Base model directory has no {CONFIG_NAME}: {model}")
    else:
        from transformers.utils import cached_file

        try:
            config_path = cached_file(
                model, CONFIG_NAME, revision=revision, local_files_only=local_files_only
            )
        except Exception as exc:
            hint = (
                "  Solution: mount the training HF cache into the export "
                "container (exports resolve the base offline)"
                if local_files_only
                else "  Solution: check the model id and HF_TOKEN for gated weights"
            )
            raise ExportError(
                f"Could not resolve base model config: {model}"
                + (f" (revision {revision})" if revision else "")
                + f"\n{hint}\n  Cause: {exc}"
            ) from exc
    return BaseIdentity(
        model=model,
        revision=revision or commit_from_cache_path(config_path),
        config_sha256=sha256_file(config_path),
        config_path=config_path,
    )


def check_base_identity(
    meta: dict | None, identity: BaseIdentity, allow_drift: bool = False
) -> list[str]:
    """Compare the export-time base against the checkpoint's recorded one.

    Returns the drift notes (empty when everything matches). Fields the
    checkpoint never recorded (pre-F6 checkpoints, local-path bases) are
    noted but never fatal; a recorded field that mismatches raises unless
    allow_drift is set, in which case the notes land in the export manifest.
    """
    notes: list[str] = []
    mismatched = False
    recorded = meta or {}
    for meta_field, attr in _MATCH_FIELDS:
        want = recorded.get(meta_field)
        have = getattr(identity, attr)
        if want is None:
            notes.append(f"{meta_field}: not recorded at train time")
        elif have is None:
            notes.append(
                f"{meta_field}: recorded {want} but unresolvable at export time"
            )
            mismatched = True
        elif want != have:
            notes.append(
                f"{meta_field}: trained against {want}, export resolved {have}"
            )
            mismatched = True
    if mismatched and not allow_drift:
        raise ExportError(
            "Base model drifted since training:\n  "
            + "\n  ".join(notes)
            + "\n  Solution: point the export at the snapshot the checkpoint "
            "trained against, or pass --allow-base-drift to merge anyway "
            "(the drift is recorded in the manifest)"
        )
    return notes
