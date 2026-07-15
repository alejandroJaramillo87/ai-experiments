#!/usr/bin/env python3
"""Load, override, and validate a training recipe (Phase 0 D5).

Thin CLI over training/recipes. Examples:

  python scripts/training/validate_recipe.py training/recipes/reference/tiny-e2e.yaml

  python scripts/training/validate_recipe.py training/recipes/reference/pretrain-1b.yaml \
      optimizer.lr=1e-4 data.micro_batch=8 --print

Exit codes: 0 recipe valid, 1 load or validation failure, 2 invalid usage.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from training.recipes import (
    RecipeLoadError,
    RecipeValidationError,
    load_recipe,
    resolved_yaml,
    validate,
)


class Printer:
    """Styled status output (see .styles/PYTHON_OUTPUT_STYLE.md)."""

    def __init__(self, quiet: bool = False) -> None:
        self.quiet = quiet
        use_color = (
            not quiet
            and not os.environ.get("NO_COLOR")
            and sys.stdout.isatty()
        )
        if use_color:
            self._c = {
                "OK": "\033[0;32m", "WARN": "\033[1;33m", "ERROR": "\033[0;31m",
                "INFO": "\033[0;36m", "RUN": "\033[0;34m", "SKIP": "\033[0;34m",
            }
            self._nc = "\033[0m"
        else:
            self._c = {}
            self._nc = ""

    def _status(self, tag: str, message: str, indent: int = 0) -> None:
        if self.quiet:
            return
        color = self._c.get(tag, "")
        pad = " " * indent
        print(f"{pad}{color}[{tag}]{self._nc} {message}")

    def ok(self, m, indent=0):    self._status("OK", m, indent)
    def error(self, m, indent=0): self._status("ERROR", m, indent)
    def info(self, m, indent=0):  self._status("INFO", m, indent)

    def line(self, message: str = "") -> None:
        if not self.quiet:
            print(message)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Load a recipe against the typed schema, apply overrides, validate.")
    parser.add_argument("config", help="path to a recipe YAML")
    parser.add_argument("overrides", nargs="*",
                        help="dotlist overrides, e.g. optimizer.lr=1e-4")
    parser.add_argument("--print", dest="print_config", action="store_true",
                        help="print the resolved config after validation")
    args = parser.parse_args()

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


if __name__ == "__main__":
    sys.exit(main())
