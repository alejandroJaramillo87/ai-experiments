"""`bench compare` -- difference two arms against a measured noise floor."""

from __future__ import annotations

import json
import sys

from bench import delta, report
from common.output import Printer


# Which status line a verdict prints as. The middle three are the ones the retired
# tooling could not express at all: it had no vocabulary for "the rig cannot tell".
def _emit(printer: Printer, verdict: str, message: str) -> None:
    if verdict == delta.REGRESSION:
        printer.error(message)
    elif verdict == delta.IMPROVEMENT:
        printer.ok(message)
    elif verdict == delta.INCONCLUSIVE:
        printer.warn(message)
    elif verdict == delta.WITHIN_NOISE:
        printer.skip(message)
    else:
        printer.info(message)


# Metrics worth putting in a table by default; the rest stay in the JSON.
HEADLINE = (
    "ttft_ms",
    "tpot_ms",
    "decode_tok_s",
    "e2e_tok_s",
    "prefill_tok_s_server",
    "prefill_tok_s_slope",
)


def configure(sub) -> None:
    parser = sub.add_parser(
        "compare",
        help="compare two result files against a noise floor",
        description=(
            "A regression is a median more than 5%% slower than the baseline AND "
            "outside the run-to-run spread. Without a --noise file the second clause "
            "cannot be evaluated and large deltas report as inconclusive."
        ),
    )
    parser.add_argument("baseline")
    parser.add_argument("candidate")
    parser.add_argument(
        "--noise", help="a bench noise floor file for this service and profile"
    )
    parser.add_argument(
        "--all", action="store_true", help="every metric, not just the headline set"
    )
    parser.add_argument("--json", action="store_true")
    parser.set_defaults(func=cmd, obs_retention=False)


def cmd(args) -> int:
    printer = Printer(quiet=args.json)
    baseline = report.read(args.baseline)
    candidate = report.read(args.candidate)
    noise = report.read(args.noise) if args.noise else None

    try:
        deltas, warnings = delta.compare(baseline, candidate, noise)
    except delta.IncomparableError as exc:
        printer.error(str(exc))
        return 2

    for warning in warnings:
        printer.warn(warning)

    shown = [d for d in deltas if args.all or d.metric in HEADLINE]
    if args.json:
        json.dump(
            {
                "baseline": args.baseline,
                "candidate": args.candidate,
                "warnings": warnings,
                "deltas": [d.__dict__ for d in shown],
                "regressions": sum(1 for d in shown if d.verdict == delta.REGRESSION),
            },
            sys.stdout,
            indent=2,
        )
        sys.stdout.write("\n")
    else:
        for item in shown:
            floor = (
                f"floor {item.noise_pct:.1f}%"
                if item.noise_pct is not None
                else "floor unmeasured"
            )
            _emit(
                printer,
                item.verdict,
                f"{item.point} {item.metric}: {item.baseline:.4g} -> "
                f"{item.candidate:.4g} ({item.delta_pct:+.1f}%, {floor}) "
                f"{item.verdict}",
            )

    regressions = sum(1 for d in shown if d.verdict == delta.REGRESSION)
    if regressions:
        printer.error(f"Compare: {regressions} regression(s) beyond the noise floor")
        return 1
    printer.ok("Compare: no regression outside the noise floor")
    return 0
