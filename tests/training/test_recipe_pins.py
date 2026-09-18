"""Every Gemma 4 recipe pins model.hf.revision to the facts file (operating
rule: pin, then fix, then update). The pin is what makes a training run
reproducible against the hub, and a recipe that leaves it null trains
whatever the hub serves that day. CPU-only: loads the recipes through the
same loader the CLI uses and compares against schemas/gemma4_facts.json."""

from __future__ import annotations

import glob
import json
import os

import pytest
from support import REFERENCE_DIR, REPO_ROOT

from training.recipes import load_recipe

ORGANISM_DIR = os.path.join(REPO_ROOT, "training", "recipes", "organisms")
FACTS = os.path.join(REPO_ROOT, "schemas", "gemma4_facts.json")
HUB_PREFIX = "google/gemma-4-"

RECIPES = sorted(
    glob.glob(os.path.join(REFERENCE_DIR, "*.yaml"))
    + glob.glob(os.path.join(ORGANISM_DIR, "*.yaml"))
)


def pinned_commits() -> dict[str, str]:
    """Hub id -> commit, from the facts file's revisions block."""
    with open(FACTS) as f:
        revisions = json.load(f)["revisions"]
    return {row["repo"]: row["commit"] for row in revisions.values()}


def hub_recipes() -> list[str]:
    """Recipes whose model.hf.id is a Gemma 4 hub id (not a local path)."""
    found = []
    for path in RECIPES:
        recipe = load_recipe(path)
        hf = recipe.model.hf
        if hf is not None and str(hf.id).startswith(HUB_PREFIX):
            found.append(path)
    return found


class TestGemma4RecipePins:
    def test_hub_recipes_are_found(self):
        # The guard against a glob that silently matches nothing.
        assert len(hub_recipes()) >= 8

    @pytest.mark.parametrize(
        "path", hub_recipes(), ids=[os.path.basename(p) for p in hub_recipes()]
    )
    def test_revision_matches_facts_file(self, path):
        recipe = load_recipe(path)
        commits = pinned_commits()
        model_id = str(recipe.model.hf.id)
        assert model_id in commits, f"{model_id} has no row in {FACTS}"
        assert recipe.model.hf.revision == commits[model_id], (
            f"{os.path.basename(path)} pins {recipe.model.hf.revision!r}; "
            f"{FACTS} records {commits[model_id]} for {model_id}"
        )
