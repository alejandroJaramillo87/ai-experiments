"""FT.memfit, the memory-fit estimate on the SFT recipe, as the hw_smoke tier runs it.

The fine-tuning runbook (docs/verification/training/fine-tuning-loop.md)
runs `make memfit` twice on sft-qlora-e2b.yaml after FT.calibrate-qlora has recorded the
recipe's measured activation rate: the recipe as committed must fit, and
micro-batch 64 must not. This module holds the same two outcomes through the
calculator the CLI and train-sft's advisory share
(training/calculator/estimate.py), so a change to the static terms or to the
recorded rate that would flip the verdict shows up at session start.

CPU-only: the estimate reads config.json and the safetensors header from the
staged base, never the weights. It skips when the base is not staged (FT.preconditions)
and when the profile has no rate on file (FT.calibrate-qlora not yet recorded), because
an estimate of an unmeasured profile is a lower bound, not a fact to assert.
"""

from __future__ import annotations

import os

import pytest

from common import serving
from tests.hardware.conftest import REPO_ROOT

pytestmark = [pytest.mark.hw_smoke, pytest.mark.cpu]

RECIPE = os.path.join(
    REPO_ROOT, "training", "recipes", "reference", "sft-qlora-e2b.yaml"
)
# The directory make train-download stages, host spelling (FT.preconditions). The recipe
# names the hub id; every reader resolves it here (training/models/staged.py).
BASE_DIR = os.path.join(serving.HOST_MODELS, "hf", "gemma-4-E2B")
# The geometry FT.memfit uses to force a refusal.
OVERSIZED_MICRO_BATCH = 64


def _recipe(*overrides: str):
    from training.recipes import load_recipe, validate

    recipe = load_recipe(RECIPE, list(overrides))
    validate(recipe)
    return recipe


@pytest.fixture(scope="module")
def measured_recipe():
    if not os.path.isfile(os.path.join(BASE_DIR, "config.json")):
        pytest.skip(f"base not staged at {BASE_DIR} (FT.preconditions)")
    from training.calculator.estimate import estimate

    est = estimate(_recipe())
    if est.rate is None:
        pytest.skip(
            f"no activation rate on file for {est.profile} (FT.calibrate-qlora)"
        )
    return est


def test_recipe_fits_and_the_oversized_batch_does_not(measured_recipe):
    from training.calculator.estimate import VRAM_BUDGET_BYTES, estimate

    assert measured_recipe.fits is True, (
        f"recipe estimate {measured_recipe.reserved_estimate / (1 << 30):.2f} GB"
    )
    assert measured_recipe.reserved_estimate <= VRAM_BUDGET_BYTES
    oversized = estimate(_recipe(f"data.micro_batch={OVERSIZED_MICRO_BATCH}"))
    assert oversized.fits is False
    assert oversized.rate == measured_recipe.rate
    assert oversized.activations == measured_recipe.activations * (
        OVERSIZED_MICRO_BATCH // measured_recipe.micro_batch
    )


def test_train_sft_advisory_is_the_same_estimate(measured_recipe):
    """FT.train-resume relies on train-sft printing what memfit prints, as a warning
    that never refuses (2026-09-19)."""
    from training.tune.sft import check_memory_fit

    ok, message = check_memory_fit(_recipe())
    assert ok, message
    ok, message = check_memory_fit(_recipe(f"data.micro_batch={OVERSIZED_MICRO_BATCH}"))
    assert not ok
    assert "exceeds" in message and "proceeding" in message
