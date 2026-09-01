"""Summarise an entire mongod corpus in a few thousand tokens.

Useful on its own -- it is the fastest way to find out what a log contains --
and it is also exactly what the triage loop shows the model first, so what this
prints is what the model sees.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime
from typing import Any


def configure(sub) -> None:
    p = sub.add_parser(
        "profile",
        help="summarise a mongod log: span, components, call sites, slow queries",
        description=__doc__,
    )
    p.add_argument("log", help="path to a mongod JSON log file")
    p.add_argument(
        "--budget-tokens",
        type=int,
        default=None,
        help="ceiling for the rendered profile (default: the triage budget)",
    )
    p.add_argument(
        "--redact",
        action="store_true",
        help="replace literal command values with type placeholders at ingest",
    )
    p.add_argument("--json", action="store_true", help="emit the profile as JSON")
    p.set_defaults(func=cmd)


def _jsonable(value: Any) -> Any:
    """Make a profile JSON-serialisable without inventing a schema.

    Only datetimes need help; everything else in the profile is already a
    primitive, a list, or a dataclass that asdict has flattened.
    """
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(v) for v in value]
    return value


def cmd(args: argparse.Namespace) -> int:
    from logtriage.cli import LogtriageError
    from logtriage.mongod.normalize import normalize
    from logtriage.mongod.records import LogRecord, ParseError, read_log
    from logtriage.output import Printer
    from logtriage.tools.budget import DEFAULT_PROFILE_TOKENS, estimate_tokens
    from logtriage.triage.profile import build_profile, render_profile

    out = Printer(quiet=args.json)
    budget = args.budget_tokens or DEFAULT_PROFILE_TOKENS

    try:
        parsed = list(read_log(args.log))
    except OSError as exc:
        raise LogtriageError(
            f"Cannot read log: {exc}\n"
            f"  Solution: check the path exists and is readable\n"
            f"  Command: ls -l {args.log}"
        ) from exc

    records = [p for p in parsed if isinstance(p, LogRecord)]
    errors = [p for p in parsed if isinstance(p, ParseError)]

    if not records:
        raise LogtriageError(
            f"No mongod log records found in {args.log} "
            f"({len(errors)} unparseable lines)\n"
            "  Solution: confirm this is a mongod 4.4+ structured (JSON) log\n"
            f"  Command: head -1 {args.log}"
        )

    profile = build_profile(
        (normalize(r, redact=args.redact) for r in records),
        source=args.log,
        parse_errors=len(errors),
    )
    text = render_profile(profile, budget_tokens=budget)

    if args.json:
        print(json.dumps(_jsonable(asdict(profile)), indent=2))
        return 0

    out.line(text)
    out.line()
    out.ok(f"Profile: {estimate_tokens(text)} tokens of a {budget} budget")
    if errors:
        # A warning rather than a note: a corpus with unparseable lines is one
        # where some evidence was never seen, and a reader deciding how much to
        # trust a conclusion needs to know that.
        out.warn(
            f"Unparseable lines: {len(errors)} (first at line {errors[0].line_no})"
        )
    return 0
