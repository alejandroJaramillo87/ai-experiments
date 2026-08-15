"""Shared `runs` command surface for every CLI that writes the catalog (roadmap OBS-9).

Reads ${OBS_LOG_DIR}/<cli>/runs.jsonl, the per-CLI catalog the traced dispatch
appends one summary line to when a command ends. `catalog.run_command` is already
CLI-agnostic; what was duplicated is the argparse registration, and this module is
the one definition of it.

Two consumers, deliberately different in shape:

- `configure(sub, cli)` registers the subcommand the training and interp CLIs
  expose, keeping their existing `python -m training runs` spelling.
- `main(argv)` is the standalone entry point, selected with `--cli`. The rust
  harness writes the same catalog from `rust-harness/harness/src/obs.rs` but has
  no Python package to host a subcommand, so `python -m common.obs.runs --cli
  harness` is how its runs are browsed.

`main` deliberately does not route through `common.obs.dispatch`: a catalog
browser is a query, not a run. Dispatching would call `configure()` and prune the
very event files it was asked to display, and would append rows that
`run_command` then filters back out.

Exit codes: 0 success, 1 failure, 2 invalid usage.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from common.obs import catalog

# The catalogs a runs view can be pointed at. "harness" is written by the Rust
# workspace rather than a Python package, which is why it appears only here.
CLI_CHOICES = ("training", "interp", "harness")


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Define the four selection flags. The only definition of them."""
    parser.add_argument(
        "-n",
        "--limit",
        type=int,
        default=20,
        help="most recent runs shown (default 20; 0 or less shows all)",
    )
    parser.add_argument(
        "--failed", action="store_true", help="only runs with a nonzero exit code"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="include invocations of the runs command itself",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="emit the selected records as JSON on stdout",
    )


def describe(cli: str) -> str:
    """Help text naming the catalog actually read.

    The per-CLI wrappers each carried a docstring naming their own path. Sharing
    the module must not cost that: a runs command whose help does not say which
    catalog it reads is worse than the duplication removed here.
    """
    return (
        f"List recent runs from the observability catalog. Reads "
        f"${{OBS_LOG_DIR}}/{cli}/runs.jsonl, the per-CLI catalog one summary "
        f"line is appended to when a command ends."
    )


def configure(sub, cli: str) -> None:
    """Register the `runs` subcommand on a CLI's subparser."""
    p = sub.add_parser(
        "runs",
        help="list recent runs recorded in the observability catalog",
        description=describe(cli),
    )
    add_arguments(p)
    # viewer command: browsing the catalog must not prune event files
    p.set_defaults(
        func=lambda args: catalog.run_command(cli, args), obs_retention=False
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Standalone entry point: python -m common.obs.runs --cli harness."""
    parser = argparse.ArgumentParser(
        prog="python -m common.obs.runs",
        description=(
            "List recent runs from an observability catalog. Reads "
            "${OBS_LOG_DIR}/<cli>/runs.jsonl."
        ),
    )
    parser.add_argument(
        "--cli",
        required=True,
        choices=CLI_CHOICES,
        help="which catalog to read",
    )
    add_arguments(parser)
    args = parser.parse_args(argv if argv is None else list(argv))
    return catalog.run_command(args.cli, args)


if __name__ == "__main__":
    import sys

    sys.exit(main())
