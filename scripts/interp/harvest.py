#!/usr/bin/env python3
"""Harvest residual-stream activations to NVMe (interp Phase 4 P4-D2).

Workbench batch job (GPU, inference down): wraps SAELens's cache runner
under the D9 lock and D10 watchdog. --check-hooks runs only the Gemma 4
hook-dimension gate (one-token forward, no capture). Exit codes: 0 success,
1 failure, 2 invalid usage.
"""

from __future__ import annotations

import argparse
import sys

sys.path.insert(0, ".")

from interp.output import Printer  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--recipe", required=True, help="harvest recipe YAML (training/recipes/interp/)"
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="dotlist override, repeatable (harvest.total_tokens=2000000)",
    )
    parser.add_argument(
        "--check-hooks",
        action="store_true",
        help="run only the hook-dimension gate; no capture",
    )
    args = parser.parse_args()

    out = Printer()
    from interp.harvest import (
        HarvestError,
        check_hook_dims,
        resolve_subject,
        run_harvest,
        validate_harvest,
    )
    from interp.manifest import ManifestError
    from training.lock import TrainLockError
    from training.recipes.loader import RecipeLoadError, load_recipe

    try:
        recipe = load_recipe(args.recipe, overrides=args.override)
        validate_harvest(recipe)
        model_id, revision, _ = resolve_subject(recipe.harvest)
        if args.check_hooks:
            out.run(f"Hook-dimension check: {model_id}")
            widths = check_hook_dims(
                model_id, list(recipe.harvest.hook_layers), revision
            )
            for layer, width in sorted(widths.items()):
                out.ok(f"model.layers.{layer}: residual width {width}")
            return 0
        out.run(f"Harvesting activations: {model_id}")
        manifest = run_harvest(recipe)
    except (
        HarvestError,
        ManifestError,
        TrainLockError,
        RecipeLoadError,
        OSError,
    ) as exc:
        out.error(f"Harvest failed: {exc}")
        return 1
    out.ok(
        f"Harvest complete: {manifest['total_tokens']} tokens/hook, hooks "
        f"{manifest['hooks']}, {manifest['tokens_per_second']} tokens/s"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
