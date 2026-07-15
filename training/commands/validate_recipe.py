"""Load, override, and validate a training recipe (Phase 0 D5).

Thin subcommand over training/recipes. Examples:

  python -m training validate training/recipes/reference/tiny-e2e.yaml

  python -m training validate training/recipes/reference/pretrain-1b.yaml \
      optimizer.lr=1e-4 data.micro_batch=8 --print

Exit codes: 0 recipe valid, 1 load or validation failure, 2 invalid usage.
"""

from __future__ import annotations

import argparse

from common.output import Printer
from training.recipes import (
    RecipeLoadError,
    RecipeValidationError,
    load_recipe,
    resolved_yaml,
    validate,
)


def configure(sub) -> None:
    parser = sub.add_parser(
        "validate",
        help="load a recipe against the typed schema, apply overrides, validate (D5)",
        description="Load a recipe against the typed schema, apply overrides, validate.")
    parser.add_argument("config", help="path to a recipe YAML")
    parser.add_argument("overrides", nargs="*",
                        help="dotlist overrides, e.g. optimizer.lr=1e-4")
    parser.add_argument("--print", dest="print_config", action="store_true",
                        help="print the resolved config after validation")
    parser.set_defaults(func=cmd)


def cmd(args: argparse.Namespace) -> int:
    out = Printer()
    try:
        recipe = load_recipe(args.config, args.overrides)
    except RecipeLoadError as exc:
        out.error(f"Recipe load failed: {exc}")
        out.line("  Solution: Fix the reported key or type to match training/recipes/schema.py")
        return 1
    out.ok(f"Schema: {args.config} loads as a typed recipe")
    if args.overrides:
        out.info(f"Overrides applied: {' '.join(args.overrides)}")

    try:
        validate(recipe)
    except RecipeValidationError as exc:
        out.error("Validation failed:")
        for problem in exc.problems:
            out.line(f"  - {problem}")
        return 1
    out.ok(f"Validation: run {recipe.run.name} passes all cross-field invariants")

    if args.print_config:
        out.line()
        print(resolved_yaml(recipe), end="")
    return 0
