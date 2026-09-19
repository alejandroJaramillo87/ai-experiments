"""Every Gemma 4 recipe pins model.hf.revision to the facts file (operating
rule: pin, then fix, then update). The pin is what makes a training run
reproducible against the hub, and a recipe that leaves it null trains
whatever the hub serves that day. CPU-only: loads the recipes through the
same loader the CLI uses and compares against schemas/gemma4_facts.json.

Two blocks name a hub model: model.hf for training and organism recipes,
harvest for interp captures. Both carry a revision and both are checked, so
an interp recipe cannot capture from an unpinned Gemma 4 either."""

from __future__ import annotations

import glob
import json
import os

import pytest
from support import REFERENCE_DIR, REPO_ROOT

from training.recipes import load_recipe

ORGANISM_DIR = os.path.join(REPO_ROOT, "training", "recipes", "organisms")
INTERP_DIR = os.path.join(REPO_ROOT, "training", "recipes", "interp")
FACTS = os.path.join(REPO_ROOT, "schemas", "gemma4_facts.json")
HUB_PREFIX = "google/gemma-4-"

RECIPES = sorted(
    glob.glob(os.path.join(REFERENCE_DIR, "*.yaml"))
    + glob.glob(os.path.join(ORGANISM_DIR, "*.yaml"))
    + glob.glob(os.path.join(INTERP_DIR, "*.yaml"))
)


def pinned_commits() -> dict[str, str]:
    """Hub id -> commit, from the facts file's revisions block."""
    with open(FACTS) as f:
        revisions = json.load(f)["revisions"]
    return {row["repo"]: row["commit"] for row in revisions.values()}


def hub_pin(recipe):
    """(hub id, revision) of the block naming a Gemma 4 hub id, else None."""
    hf = recipe.model.hf
    if hf is not None and str(hf.id).startswith(HUB_PREFIX):
        return str(hf.id), hf.revision
    if str(recipe.harvest.model_id).startswith(HUB_PREFIX):
        return str(recipe.harvest.model_id), recipe.harvest.revision
    return None


def hub_recipes() -> list[str]:
    """Recipes whose model.hf.id or harvest.model_id is a Gemma 4 hub id."""
    return [path for path in RECIPES if hub_pin(load_recipe(path)) is not None]


class TestGemma4RecipePins:
    def test_hub_recipes_are_found(self):
        # The guard against a glob that silently matches nothing: five
        # reference and organism recipes plus the interp harvest smoke.
        assert len(hub_recipes()) >= 6

    def test_interp_recipes_are_in_scope(self):
        assert any(p.startswith(INTERP_DIR) for p in hub_recipes())

    @pytest.mark.parametrize(
        "path", hub_recipes(), ids=[os.path.basename(p) for p in hub_recipes()]
    )
    def test_revision_matches_facts_file(self, path):
        model_id, revision = hub_pin(load_recipe(path))
        commits = pinned_commits()
        assert model_id in commits, f"{model_id} has no row in {FACTS}"
        assert revision == commits[model_id], (
            f"{os.path.basename(path)} pins {revision!r}; "
            f"{FACTS} records {commits[model_id]} for {model_id}"
        )
