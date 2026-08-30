"""Unified benchmark CLI: `python -m bench <command>`.

Every invocation runs through the traced dispatch (common/obs), the same way the
training and interp CLIs do, so a benchmark run appears in the run catalog beside
everything else and its events land in the same viewer.

There is deliberately no `make bench` target. A benchmark is a serialized GPU consumer
(AGENTS.md: one at a time, across every worktree), and a Make target would make it look
like an ordinary one. scripts/README.md and .claude/skills/benchmarking/SKILL.md record
the same decision.

Exit codes: 0 success, 1 failure or regression, 2 invalid usage.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from bench.commands import compare, models, noise, run
from common.obs.dispatch import dispatch

COMMANDS = (run, noise, compare, models)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m bench",
        # The benchmarked services are the ones bench/adapters.py has an adapter for.
        # The rust harness is not among them: `make ladder` drives its runtime directly
        # rather than over HTTP, and --service refuses it here.
        description=(
            "Phase-separated inference benchmarks for the llama.cpp and vLLM services. "
            "Reports time to first token and inter-token latency separately, as median "
            "and spread over repeated runs, against a measured noise floor."
        ),
        epilog=(
            "Protocol: .claude/skills/benchmarking/SKILL.md. "
            "Model registry: MODELS in bench/registry.py."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")
    for command in COMMANDS:
        command.configure(sub)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    return dispatch("bench", build_parser, argv)
