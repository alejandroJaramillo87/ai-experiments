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


# The unchunked head held micro_batch x seq_len x 262144 logits through the
# loss, about 6 bytes per vocabulary entry per token: 4096 tokens per
# micro-batch was 24 GB beside the NF4 E2B base (FT.calibrate-qlora's third
# and fourth attempts, KB-85 then KB-87). The loss is chunked since KB-87, so
# this ceiling is held until a measured rate says what fits, not because the
# head sets it.
MAX_TOKENS_PER_MICRO_BATCH = 4096
# Both reference SFT recipes train on 32 sequences per optimizer step.
EFFECTIVE_BATCH = 32


def training_recipes() -> list[str]:
    return [
        path
        for path in hub_recipes()
        if not path.startswith(INTERP_DIR) and load_recipe(path).model.hf is not None
    ]


class TestGemma4Geometry:
    @pytest.mark.parametrize(
        "path",
        training_recipes(),
        ids=[os.path.basename(p) for p in training_recipes()],
    )
    def test_tokens_per_micro_batch_fit_the_head(self, path):
        data = load_recipe(path).data
        assert data.micro_batch * data.seq_len <= MAX_TOKENS_PER_MICRO_BATCH, (
            f"{os.path.basename(path)}: {data.micro_batch} x {data.seq_len} tokens "
            f"per micro-batch; the 262144-wide head fits {MAX_TOKENS_PER_MICRO_BATCH}"
        )

    @pytest.mark.parametrize("name", ["sft-qlora-e2b.yaml", "sft-qlora-gemma-e4b.yaml"])
    def test_reference_sft_keeps_its_effective_batch(self, name):
        data = load_recipe(os.path.join(REFERENCE_DIR, name)).data
        assert data.micro_batch * data.grad_accum == EFFECTIVE_BATCH
