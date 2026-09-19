"""The load-and-validate step every recipe-driven subcommand runs first.

One helper so the three gates (validate, preflight, and the GPU verbs in
training/cli.py) print the same lines for the same failure: the runbooks
quote these strings, and three spellings of one error is three things for
an operator to grep for.
"""

from __future__ import annotations

from omegaconf import DictConfig

from common.output import Printer
from training.recipes import (
    RecipeLoadError,
    RecipeValidationError,
    load_recipe,
    validate,
)


def load_validated(
    config: str, overrides: list[str], out: Printer, *, announce: bool = False
) -> DictConfig | None:
    """Load a recipe with overrides and validate it, printing any failure.

    Returns None after printing when either step fails; the caller exits 1.
    With announce, the successful load and the applied overrides are printed
    as [OK]/[INFO] lines (the validate subcommand's contract).
    """
    try:
        recipe = load_recipe(config, overrides)
    except RecipeLoadError as exc:
        out.error(f"Recipe load failed: {exc}")
        out.line(
            "  Solution: Fix the reported key or type to match training/recipes/schema.py"
        )
        return None
    if announce:
        out.ok(f"Schema: {config} loads as a typed recipe")
        if overrides:
            out.info(f"Overrides applied: {' '.join(overrides)}")
    try:
        validate(recipe)
    except RecipeValidationError as exc:
        out.error("Recipe validation failed:")
        for problem in exc.problems:
            out.line(f"  - {problem}")
        return None
    return recipe
