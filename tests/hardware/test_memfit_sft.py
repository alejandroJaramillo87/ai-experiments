"""FT.14, the memory-fit gate on the SFT recipe, as the hw_smoke tier runs it.

The fine-tuning runbook (docs/verification/training/fine-tuning-loop.md)
runs `make memfit` three times on sft-qlora-e2b.yaml: the recipe as
committed must fit, micro-batch 64 must not, and the batch solve must land
between the two. This module holds the same estimate to the same three
outcomes through the calculator the CLI and train-sft's refusal share
(training/calculator/solve.py), so a change to the estimate that would
flip FT.3's memory-fit refusal shows up at session start.

CPU-only: the estimate reads config.json from the staged base, never the
weights. It skips when the base is not staged (FT.0), because a size
estimate of a model that is not on disk is not a hardware fact.
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
# The directory make train-download stages, host spelling (FT.0). The recipe
# names the hub id, which only the host HF cache could resolve (KB-79).
BASE_DIR = os.path.join(serving.HOST_MODELS, "hf", "gemma-4-E2B")
# The geometry FT.14 uses to force a refusal.
OVERSIZED_MICRO_BATCH = 64


def _recipe(*overrides: str):
    from training.recipes import load_recipe, validate

    recipe = load_recipe(RECIPE, [f"model.hf.id={BASE_DIR}", *overrides])
    validate(recipe)
    return recipe


@pytest.fixture(scope="module")
def staged_base() -> str:
    if not os.path.isfile(os.path.join(BASE_DIR, "config.json")):
        pytest.skip(f"base not staged at {BASE_DIR} (FT.0)")
    return BASE_DIR


def test_recipe_fits_oversized_batch_does_not_and_solve_lands_between(staged_base):
    from training.calculator.solve import GIB, VRAM_TOTAL_BYTES, estimate, solve

    fits = estimate(_recipe())
    assert fits.fits, f"recipe estimate {fits.reserved_estimate / GIB:.2f} GB"
    assert fits.reserved_estimate < VRAM_TOTAL_BYTES

    oversized = estimate(_recipe(f"data.micro_batch={OVERSIZED_MICRO_BATCH}"))
    assert not oversized.fits
    assert oversized.reserved_estimate > fits.reserved_estimate

    solved = solve(_recipe(), "batch")
    assert solved is not None, "no micro-batch fits at the recipe's other settings"
    recipe_batch = _recipe().data.micro_batch
    assert recipe_batch <= solved.micro_batch < OVERSIZED_MICRO_BATCH
    assert solved.fits


def test_train_sft_refusal_is_the_same_estimate(staged_base):
    """FT.3 relies on train-sft refusing what memfit refuses."""
    from training.tune.sft import check_memory_fit

    proceed, message = check_memory_fit(_recipe(), force=False)
    assert proceed, message
    refused, message = check_memory_fit(
        _recipe(f"data.micro_batch={OVERSIZED_MICRO_BATCH}"), force=False
    )
    assert not refused
    assert "exceeds" in message
