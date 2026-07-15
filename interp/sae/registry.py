"""SAE registry (interp Phase 4 P4-D3): list, load, and validate SAE runs.

Every consumer — the evaluation report, the P4-D4 Neuronpedia import, the
steering sae_row backend — resolves trained SAEs through here, and the load
path refuses a run whose manifest is missing or whose model hash does not
match the model it will be used against. Weights stay in SAELens's own save
format on disk; loading them is the consumer's job (sae_lens.SAE.load_from_
disk), the registry validates identity.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from interp.manifest import load_manifest, require_match
from interp.sae.train import ARTIFACT

DEFAULT_ROOT = "/mnt/ai-data/interp/saes"


class SaeRegistryError(Exception):
    """An SAE run is missing, incomplete, or mismatched."""


@dataclass
class SaeRun:
    name: str
    path: str
    hook: str
    d_in: int
    width: int
    manifest: dict


def list_saes(root: str = DEFAULT_ROOT) -> list[str]:
    """Names of completed SAE runs (directories containing a manifest)."""
    if not os.path.isdir(root):
        return []
    return sorted(
        entry
        for entry in os.listdir(root)
        if os.path.isfile(os.path.join(root, entry, "manifest.json"))
    )


def load_sae_run(
    sae_dir: str,
    expect_model_hash: str | None = None,
    expect_tokenizer_fingerprint: str | None = None,
) -> SaeRun:
    """Load and validate one SAE run; refuse identity mismatches.

    expect_model_hash is the hash of the model whose activations the SAE
    will encode; a mismatch means the instrument was trained on a different
    subject and its features are meaningless for this one.
    """
    manifest = load_manifest(sae_dir)
    require_match(manifest, "artifact", ARTIFACT, sae_dir)
    if expect_model_hash is not None:
        require_match(manifest, "model_hash", expect_model_hash, sae_dir)
    if expect_tokenizer_fingerprint is not None:
        require_match(
            manifest, "tokenizer_fingerprint", expect_tokenizer_fingerprint, sae_dir
        )
    return SaeRun(
        name=manifest["name"],
        path=sae_dir,
        hook=manifest["hook"],
        d_in=int(manifest["d_in"]),
        width=int(manifest["width"]),
        manifest=manifest,
    )
