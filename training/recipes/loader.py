"""Recipe loading, overrides, and run-directory snapshots (Phase 0 D5).

load_recipe merges a YAML file against the typed Recipe schema in struct mode,
then applies dotlist overrides (optimizer.lr=1e-4). Unknown keys and wrong
types raise at load. Every entry point loads recipes through this function so
override handling lives in one place.

create_run_dir implements the run directory convention
${run.output_root}/${run.name}/<timestamp>/ and writes the resolved config
snapshot (config.yaml) into it; the snapshot reloads bit-identically
(the round trip is pinned by tests/training/test_recipes.py).
"""

from __future__ import annotations

import datetime
import os
from collections.abc import Sequence

from omegaconf import DictConfig, OmegaConf

from training.recipes.schema import Recipe

SNAPSHOT_NAME = "config.yaml"


class RecipeLoadError(ValueError):
    """A recipe file failed to merge against the schema or overrides failed."""


def load_recipe(path: str, overrides: Sequence[str] = ()) -> DictConfig:
    """Load a YAML recipe against the Recipe schema, applying dotlist overrides.

    Returns a struct-mode DictConfig typed by the schema. Raises
    RecipeLoadError on a missing file, unknown key, or type mismatch; the
    original OmegaConf error is chained for the key/line detail.
    """
    if not os.path.isfile(path):
        raise RecipeLoadError(f"Recipe file not found: {path}")
    base = OmegaConf.structured(Recipe)
    try:
        file_cfg = OmegaConf.load(path)
        merged = OmegaConf.merge(base, file_cfg)
        if overrides:
            merged = OmegaConf.merge(merged, OmegaConf.from_dotlist(list(overrides)))
    except Exception as exc:
        raise RecipeLoadError(f"Recipe {path} does not match schema: {exc}") from exc
    return merged


def resolved_yaml(recipe: DictConfig) -> str:
    """The recipe as resolved YAML text — the canonical snapshot serialization."""
    return OmegaConf.to_yaml(recipe, resolve=True)


def load_snapshot(path: str) -> DictConfig:
    """Reload a snapshot written by write_snapshot, re-typed against the schema."""
    return load_recipe(path)


def write_snapshot(recipe: DictConfig, run_dir: str) -> str:
    """Write the resolved config snapshot into run_dir; returns the file path."""
    path = os.path.join(run_dir, SNAPSHOT_NAME)
    with open(path, "w") as f:
        f.write(resolved_yaml(recipe))
    return path


def create_run_dir(recipe: DictConfig, now: datetime.datetime | None = None) -> str:
    """Create ${run.output_root}/${run.name}/<YYYY-MM-DD_HH-MM-SS>/ and snapshot into it.

    The caller may pass a fixed timestamp for reproducible tests; production
    callers omit it. Fails if the exact run dir already exists (two runs in
    the same second are almost certainly a mistake).
    """
    stamp = (now or datetime.datetime.now()).strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join(recipe.run.output_root, recipe.run.name, stamp)
    os.makedirs(run_dir, exist_ok=False)
    write_snapshot(recipe, run_dir)
    return run_dir
