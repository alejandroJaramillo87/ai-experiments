"""Train an SAE from a P4-D2 activation capture (interp Phase 4 P4-D3).

Workbench batch job (GPU, inference down): wraps SAELens's training runner
under the D9 lock, D10 watchdog, and D6 tracking, gated on the capture
manifest. Exit codes: 0 success, 1 failure, 2 invalid usage.
"""

from __future__ import annotations

import argparse
import os

from common.output import Printer


def configure(sub) -> None:
    p = sub.add_parser(
        "sae-train",
        help="train an SAE from a P4-D2 activation capture (P4-D3)",
        description=__doc__,
    )
    p.add_argument(
        "--recipe", required=True, help="SAE recipe YAML (training/recipes/interp/)"
    )
    p.add_argument(
        "--override",
        action="append",
        default=[],
        help="dotlist override, repeatable (sae.train_tokens=2000000)",
    )
    p.add_argument(
        "--validated",
        action="store_true",
        help="on success, flag the consumed capture deletable for interp-gc",
    )
    p.set_defaults(func=cmd)


def cmd(args: argparse.Namespace) -> int:
    out = Printer()
    from interp.manifest import ManifestError
    from interp.sae.train import SaeTrainError, run_sae_training, validate_sae
    from training.recipes.loader import RecipeLoadError, load_recipe
    from training.runtime.lock import TrainLockError
    from training.runtime.tracking import Tracker

    try:
        recipe = load_recipe(args.recipe, overrides=args.override)
        if args.validated:
            recipe.sae.mark_capture_validated = True
        validate_sae(recipe)
        out.run(f"Training SAE from capture: {recipe.sae.capture_dir}")
        manifest = run_sae_training(recipe, tracker_factory=Tracker)
    except (
        SaeTrainError,
        ManifestError,
        TrainLockError,
        RecipeLoadError,
        OSError,
    ) as exc:
        out.error(f"SAE training failed: {exc}")
        return 1
    out.ok(
        f"SAE training complete: {manifest['name']}, width {manifest['width']}, "
        f"k {manifest['k']}, {manifest['wall_clock_s']}s"
    )
    # k is the requested sparsity, honestly labelled. No quality number is
    # printed here because training does not measure one -- the evaluation
    # step does, against held-out activations. Naming the exact command is
    # the point: a trained SAE that is never evaluated carries no quality
    # signal at all, and this line is the only thing that says so.
    sae_dir = os.path.join(recipe.sae.out_root, manifest["name"])
    out.line(f"  Next: make sae-eval SAE={sae_dir} CAPTURE={recipe.sae.capture_dir}")
    return 0
