"""`bench models` -- list the registry, or fail loudly on a malformed entry."""

from __future__ import annotations

import json
import sys

from bench.registry import RegistryError, load
from common.output import Printer


def configure(sub) -> None:
    parser = sub.add_parser(
        "models",
        help="list the benchmark model registry",
        description=(
            "Every model the suite can benchmark, with the architecture facts a report "
            "carries. Validates on load, so this doubles as a registry check."
        ),
    )
    parser.add_argument("--json", action="store_true")
    parser.set_defaults(func=cmd, obs_retention=False)


def cmd(args) -> int:
    printer = Printer(quiet=args.json)
    try:
        specs = load()
    except RegistryError as exc:
        printer.error(str(exc))
        return 1
    if args.json:
        # conditions() is the model block a report carries, and its key set is what
        # bench/delta.py refuses a comparison on -- engines is a registry fact about
        # which services a model can run on, not a condition of any one run, so it is
        # added here rather than widened into that block. Without it the machine form
        # cannot answer what the human listing answers, and a consumer is left
        # scraping stderr (docs/verification/bench/bench.md B3 asks for engines).
        json.dump(
            {
                slug: {**spec.conditions(), "engines": list(spec.engines)}
                for slug, spec in specs.items()
            },
            sys.stdout,
            indent=2,
        )
        sys.stdout.write("\n")
        return 0
    for slug, spec in sorted(specs.items()):
        if spec.is_moe:
            shape = (
                f"moe {spec.params_active_b}B active / {spec.params_total_b}B total, "
                f"{spec.n_experts_active} of {spec.n_experts} experts"
            )
        else:
            shape = f"dense {spec.params_total_b}B"
        printer.info(f"{slug}: {shape}, {spec.quantization}, ctx {spec.context_length}")
        printer.info(f"served by: {', '.join(spec.engines)}", indent=1)
    return 0
