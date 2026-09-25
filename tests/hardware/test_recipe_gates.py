"""FT.recipe-gates, the two gates a run passes before the GPU, as the hw_smoke tier runs it.

The fine-tuning runbook (docs/verification/training/fine-tuning-loop.md) runs
`make train-preflight` and `make train-validate` on the reference SFT recipe
as committed, once to pass and once to refuse. This module holds the same
contract for every HF recipe on the host, in-process, with no override: the
recipe names its base by hub id and revision, and every reader resolves that
id to the directory train-download staged (training/models/staged.py, KB-79).
Until 2026-09-19 that resolution did not exist, so the recipe as committed
failed in the container at the model, then, once the dataset was on disk, at
validate's tokenizer hash check, which fetched the hub id.

CPU-only: validate reads config.json and tokenizer.json from the staged base,
never the weights; preflight adds the dataset's manifest. Skips when the
recipe's base is not staged (FT.preconditions) and fails on everything else.
The sandbox half is tests/training/test_staged.py and the staged cases in
tests/training/test_cli_handlers_preflight.py, on scratch roots; this is the
real tree.
"""

from __future__ import annotations

import glob
import os

import pytest

from common import serving
from tests.hardware.conftest import REPO_ROOT

pytestmark = [pytest.mark.hw_smoke, pytest.mark.cpu, pytest.mark.training]

RECIPE_DIR = os.path.join(REPO_ROOT, "training", "recipes")
RECIPES = sorted(
    glob.glob(os.path.join(RECIPE_DIR, "reference", "*.yaml"))
    + glob.glob(os.path.join(RECIPE_DIR, "organisms", "*.yaml"))
)
# The directory make train-download stages, host spelling (FT.preconditions).
STAGED_ROOT = os.path.join(serving.HOST_MODELS, "hf")


def hf_recipes() -> list[str]:
    from training.recipes import load_recipe

    return [path for path in RECIPES if load_recipe(path).model.hf is not None]


def staged_base(model_id: str) -> str:
    return os.path.join(STAGED_ROOT, os.path.basename(model_id))


@pytest.fixture(params=hf_recipes(), ids=os.path.basename)
def recipe(request):
    from training.recipes import load_recipe

    loaded = load_recipe(request.param)
    base = staged_base(loaded.model.hf.id)
    if not os.path.isfile(os.path.join(base, "config.json")):
        pytest.skip(f"base not staged at {base} (FT.preconditions)")
    return request.param, loaded


def test_the_hub_id_resolves_to_the_staged_directory(recipe):
    from training.models.staged import resolve_model_source
    from training.recipes.validate import tokenizer_source

    _, loaded = recipe
    assert loaded.model.hf.id.startswith("google/"), "the recipe keeps the hub id"
    assert loaded.model.hf.revision, "the recipe keeps the pin"
    resolved = resolve_model_source(loaded.model.hf.id)
    assert os.path.isdir(resolved)
    assert os.path.basename(resolved) == os.path.basename(loaded.model.hf.id)
    assert tokenizer_source(loaded) == resolved


def test_validate_passes_as_committed_and_refuses_a_negative_lr(recipe):
    from training.recipes import RecipeValidationError, load_recipe, validate

    path, loaded = recipe
    validate(loaded)
    with pytest.raises(RecipeValidationError) as info:
        validate(load_recipe(path, ["optimizer.lr=-1"]))
    message = str(info.value)
    assert "optimizer.lr: must be positive, got -1.0" in message
    assert "tokenizer.path" not in message, message


def test_preflight_reports_the_dataset_never_the_base(recipe, capsys):
    """With the base staged, the only asset preflight can miss is the
    dataset the recipe names; the base line and the tokenizer line, the two
    KB-79 signatures, never appear."""
    from training import cli

    path, loaded = recipe
    args = cli.build_parser().parse_args(["preflight", "--config", path])
    rc = args.func(args)
    err = capsys.readouterr().err
    assert "- model.hf:" not in err, err
    assert "tokenizer.path" not in err, err
    assert "Recipe validation failed" not in err, err
    dataset = loaded.sft.dataset if loaded.sft.enabled else None
    if dataset is not None and os.path.isfile(os.path.join(dataset, "manifest.json")):
        assert rc == 0, err
        assert "[OK] All referenced assets are present and verified" in err
    else:
        assert rc == 1, err
        assert "Missing or unusable assets:" in err
