"""Unified training CLI (resolved D4 design decision 1).

python -m training <subcommand> is the single entry point for training
tooling. Phase 0 D4 ships the first subcommands (memfit, calibrate); the
existing scripts/training/ entry points migrate here as thin subcommands in a
follow-up. The D6 trainer stays outside this dispatcher if it adopts
hydra.main, which owns argv.

Exit codes: 0 success, 1 load/validation/runtime failure, 2 invalid usage.
Output follows .styles/PYTHON_OUTPUT_STYLE.md; --json emits only JSON.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from training.calculator.solve import (
    GIB,
    SAFETY_MARGIN_BYTES,
    VRAM_TOTAL_BYTES,
    Estimate,
    estimate,
    solve,
)
from training.recipes import RecipeLoadError, RecipeValidationError, load_recipe, validate


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
    def warn(self, m, indent=0):  self._status("WARN", m, indent)
    def error(self, m, indent=0): self._status("ERROR", m, indent)
    def info(self, m, indent=0):  self._status("INFO", m, indent)
    def run(self, m, indent=0):   self._status("RUN", m, indent)

    def line(self, message: str = "") -> None:
        if not self.quiet:
            print(message)


def gb(n: int) -> str:
    return f"{n / GIB:.2f} GB"


def _load_validated(config: str, overrides: list[str], out: Printer):
    """Shared load-and-validate step every subcommand runs before GPU work."""
    try:
        recipe = load_recipe(config, overrides)
    except RecipeLoadError as exc:
        out.error(f"Recipe load failed: {exc}")
        out.line("  Solution: Fix the reported key or type to match training/recipes/schema.py")
        return None
    try:
        validate(recipe)
    except RecipeValidationError as exc:
        out.error("Recipe validation failed:")
        for problem in exc.problems:
            out.line(f"  - {problem}")
        return None
    return recipe


def _print_estimate_table(est: Estimate, out: Printer) -> None:
    out.line("=== Memory-fit estimate ===")
    out.line()
    out.line(f"Model parameters: {est.param_count:,}")
    out.line(f"Micro-batch {est.micro_batch}, sequence length {est.seq_len}")
    out.line()
    out.line("Component            Bytes             Size")
    for name, value in (
        ("weights", est.weights), ("gradients", est.grads),
        ("optimizer state", est.optimizer), ("activations", est.activations),
        ("framework overhead", est.overhead),
    ):
        out.line(f"  {name:<18} {value:>16,}  {gb(value):>10}")
    out.line()
    out.line(f"Total allocated estimate: {gb(est.allocated_total)}")
    out.line(f"Total reserved estimate:  {gb(est.reserved_estimate)}")
    budget = VRAM_TOTAL_BYTES - SAFETY_MARGIN_BYTES
    if est.fits:
        out.ok(f"Fits: reserved estimate {gb(est.reserved_estimate)} "
               f"within budget {gb(budget)} (32 GB minus safety margin)")
    else:
        out.error(f"Does not fit: reserved estimate {gb(est.reserved_estimate)} "
                  f"exceeds budget {gb(budget)}")
        out.line("  Solution: reduce data.micro_batch or data.seq_len, enable")
        out.line("  memory.gradient_checkpointing, or switch optimizer to adamw8bit")
    if est.host_ram:
        out.info(f"Host RAM for offloaded optimizer state: {gb(est.host_ram)} of 128 GB")


def cmd_memfit(args: argparse.Namespace) -> int:
    out = Printer(quiet=args.json)
    recipe = _load_validated(args.config, args.overrides, out)
    if recipe is None:
        return 1
    result: dict = {}
    est = estimate(recipe)
    result["estimate"] = est.as_dict()
    _print_estimate_table(est, out)

    axes = [args.solve] if args.solve else []
    for axis in axes:
        solved = solve(recipe, axis)
        label = "micro-batch" if axis == "batch" else "sequence length"
        if solved is None:
            result[f"max_{axis}"] = None
            out.error(f"Solve {axis}: no {label} fits at the recipe's other settings")
        else:
            value = solved.micro_batch if axis == "batch" else solved.seq_len
            result[f"max_{axis}"] = value
            result[f"max_{axis}_estimate"] = solved.as_dict()
            out.line()
            out.ok(f"Solve {axis}: max {label} {value} "
                   f"(reserved estimate {gb(solved.reserved_estimate)})")
    if args.json:
        print(json.dumps(result, indent=2))
    return 0 if est.fits or args.solve else 1


def cmd_calibrate(args: argparse.Namespace) -> int:
    from training.calculator import calibrate as calibrate_mod

    out = Printer(quiet=args.json)
    recipe = _load_validated(args.config, args.overrides, out)
    if recipe is None:
        return 1
    out.run(f"Calibrating {recipe.run.name}: {args.steps} steps on cuda:0")
    try:
        record = calibrate_mod.calibrate(recipe, steps=args.steps)
    except (RuntimeError, ImportError) as exc:
        out.error(f"Calibration failed: {exc}")
        out.line("  Solution: run on the workstation via 'make train-calibrate CONFIG=...'")
        return 1
    if args.out:
        calibrate_mod.write_record(record, args.out)
        out.ok(f"Calibration record written: {args.out}")

    if record.model_param_count != record.analytic_param_count:
        out.warn(f"Parameter count drift: model {record.model_param_count:,} vs "
                 f"analytic {record.analytic_param_count:,}")
        out.line("  Solution: align the ffn/param formulas in components.py with the built model")
    else:
        out.ok(f"Parameter count: {record.model_param_count:,} (analytic matches model)")
    out.line()
    out.line(f"Estimated allocated: {gb(record.estimate['allocated_total'])}")
    out.line(f"Measured allocated:  {gb(record.measured_allocated)}")
    out.line(f"Measured reserved:   {gb(record.measured_reserved)}")
    out.line(f"NVML peak:           {gb(record.measured_nvml_peak)}")
    within = abs(record.allocated_error_pct) <= 10.0
    status = out.ok if within else out.error
    status(f"Allocated error: {record.allocated_error_pct:+.1f}% "
           f"({'within' if within else 'exceeds'} the 10% acceptance bar)")
    if record.reserved_underpredicts_nvml:
        out.error("Reserved estimate under-predicts the NVML peak")
        out.line("  Solution: raise OVERHEAD_BYTES in training/calculator/solve.py")
    else:
        out.ok("Reserved estimate covers the NVML peak")
    if args.json:
        print(json.dumps(record.as_dict(), indent=2))
    return 0 if within and not record.reserved_underpredicts_nvml else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m training",
        description="Unified CLI for the training stack (Phase 0).")
    sub = parser.add_subparsers(dest="command", required=True)

    memfit = sub.add_parser(
        "memfit", help="estimate whether a recipe fits in 32GB VRAM (D4)")
    memfit.add_argument("--config", required=True, help="path to a recipe YAML")
    memfit.add_argument("overrides", nargs="*",
                        help="dotlist overrides, e.g. data.micro_batch=8")
    memfit.add_argument("--solve", choices=("batch", "seq"),
                        help="binary-search the largest fitting micro-batch or seq_len")
    memfit.add_argument("--json", action="store_true",
                        help="emit only JSON on stdout")
    memfit.set_defaults(func=cmd_memfit)

    cal = sub.add_parser(
        "calibrate", help="measure real peak VRAM for a recipe (GPU-only, D4)")
    cal.add_argument("--config", required=True, help="path to a recipe YAML")
    cal.add_argument("overrides", nargs="*",
                     help="dotlist overrides, e.g. data.micro_batch=8")
    cal.add_argument("--steps", type=int, default=5,
                     help="training steps to run (default 5)")
    cal.add_argument("--out", help="path for the calibration record JSON")
    cal.add_argument("--json", action="store_true",
                     help="emit only the record JSON on stdout")
    cal.set_defaults(func=cmd_calibrate)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
