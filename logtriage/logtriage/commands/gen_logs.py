"""Write a synthetic mongod corpus with a known injected fault.

Exists so that development and testing need no production log. Every run is
seeded, so a corpus that reproduces a bug can be regenerated on another machine
from the command line alone rather than shipped around.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def configure(sub) -> None:
    p = sub.add_parser(
        "gen-logs",
        help="write a seeded synthetic mongod corpus with a known fault",
        description=__doc__,
    )
    p.add_argument(
        "--scenario",
        default="baseline",
        help="fault to inject (default: baseline, a healthy corpus)",
    )
    p.add_argument("--seed", type=int, default=1, help="RNG seed (default: 1)")
    p.add_argument(
        "--minutes", type=int, default=60, help="corpus duration (default: 60)"
    )
    p.add_argument("--out", help="output path; omitted writes to stdout")
    p.add_argument(
        "--list", action="store_true", help="list the scenarios and their answer keys"
    )
    p.add_argument("--json", action="store_true", help="machine-readable summary")
    p.set_defaults(func=cmd)


def cmd(args: argparse.Namespace) -> int:
    from logtriage.cli import LogtriageError
    from logtriage.fixtures.generator import generate
    from logtriage.fixtures.scenarios import SCENARIOS
    from logtriage.output import Printer

    out = Printer(quiet=args.json)

    if args.list:
        if args.json:
            print(
                json.dumps(
                    {
                        name: {
                            "summary": truth.summary,
                            "expected_findings": list(truth.expected_findings),
                            "notes": truth.notes,
                        }
                        for name, truth in SCENARIOS.items()
                    },
                    indent=2,
                )
            )
            return 0
        out.line("=== scenarios ===")
        for name, truth in SCENARIOS.items():
            out.line()
            out.info(f"{name}: {truth.summary}")
            if truth.expected_findings:
                out.line(f"  expects: {', '.join(truth.expected_findings)}")
            if truth.notes:
                out.line(f"  {truth.notes}")
        return 0

    if args.minutes < 1:
        raise LogtriageError(
            f"--minutes must be at least 1, got {args.minutes}\n"
            "  A zero-length corpus would silently produce an empty file."
        )

    try:
        log = generate(args.scenario, seed=args.seed, minutes=args.minutes)
    except KeyError as exc:
        raise LogtriageError(
            f"Unknown scenario: {args.scenario}\n"
            f"  Solution: choose one of {', '.join(sorted(SCENARIOS))}\n"
            "  Command: python -m logtriage gen-logs --list"
        ) from exc

    text = log.text()
    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    else:
        # stdout is the corpus here, not a --json payload: the tool is being
        # used as a generator in a pipeline. Status still goes to stderr.
        sys.stdout.write(text)

    summary = {
        "scenario": args.scenario,
        "seed": args.seed,
        "minutes": args.minutes,
        "lines": len(log.lines),
        "bytes": len(text),
        "out": args.out or "-",
        "expected_findings": list(log.truth.expected_findings),
    }
    if args.json:
        print(json.dumps(summary, indent=2))
        return 0

    out.ok(f"Scenario: {args.scenario} (seed {args.seed}, {args.minutes} minutes)")
    out.ok(
        f"Wrote: {summary['out']} ({summary['lines']} lines, {summary['bytes']} bytes)"
    )
    if log.truth.expected_findings:
        out.info(f"Expected findings: {', '.join(log.truth.expected_findings)}")
    return 0
