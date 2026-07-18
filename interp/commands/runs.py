"""List recent runs from the observability catalog (roadmap OBS-9).

Reads ${OBS_LOG_DIR}/interp/runs.jsonl, the per-CLI catalog the traced
dispatch appends one summary line to when a command ends. Exit codes:
0 success, 1 failure, 2 invalid usage.
"""

from __future__ import annotations

import argparse

from common.obs import catalog


def configure(sub) -> None:
    p = sub.add_parser(
        "runs",
        help="list recent runs recorded in the observability catalog",
        description=__doc__,
    )
    p.add_argument(
        "-n",
        "--limit",
        type=int,
        default=20,
        help="most recent runs shown (default 20)",
    )
    p.add_argument(
        "--failed", action="store_true", help="only runs with a nonzero exit code"
    )
    p.add_argument(
        "--all",
        action="store_true",
        help="include invocations of the runs command itself",
    )
    p.add_argument(
        "--json",
        action="store_true",
        help="emit the selected records as JSON on stdout",
    )
    p.set_defaults(func=cmd)


def cmd(args: argparse.Namespace) -> int:
    return catalog.run_command("interp", args)
