"""Unified interp CLI.

python -m interp <subcommand> is the single entry point for interp tooling:
harvesting, capture retention, model organisms, probes, steering, and the
workbench smoke test. Each subcommand lives in interp/commands/ and keeps its
heavy imports (torch, transformers, interp.* pipeline modules) lazy inside
its handler, so parsing and --help stay dependency-free.

Exit codes: 0 success, 1 failure, 2 invalid usage.
Output follows .styles/PYTHON_OUTPUT_STYLE.md.
"""

from __future__ import annotations

import argparse
import sys

from interp.commands import (
    harvest,
    interp_gc,
    organism,
    probe_capture,
    probe_eval,
    probe_monitor,
    probe_train,
    sae_eval,
    sae_train,
    smoke_test,
    steer_extract,
    steer_scorecard,
)

COMMAND_MODULES = (
    harvest,
    interp_gc,
    organism,
    probe_capture,
    probe_train,
    probe_eval,
    probe_monitor,
    sae_train,
    sae_eval,
    steer_extract,
    steer_scorecard,
    smoke_test,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m interp",
        description="Unified CLI for the interpretability stack.")
    sub = parser.add_subparsers(dest="command", required=True)
    for module in COMMAND_MODULES:
        module.configure(sub)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
