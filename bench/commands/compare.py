"""`bench compare` -- difference two arms against a measured noise floor."""

from __future__ import annotations

import json
import sys

from bench import delta, report
from common.output import Printer
from common.reports import ReportSchemaError


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

# Metrics whose verdicts decide the exit code. Deliberately not the displayed set:
# what a table shows is a reading convenience, and an exit code that follows it makes
# --all change the answer rather than the view. Wider than HEADLINE because
# inter-token latency is half of what this CLI promises to report. It is no longer
# narrower than every delta: the observed token counts it used to exclude are not
# differenced at all now, having moved out of the graded set at the reduction. The
# named set stays, because the next non-metric added to a point's mapping should have
# to be admitted deliberately rather than inherited.
GATED = HEADLINE + ("itl_ms",)


class _InputError(Exception):
    """An input document cannot be used as the kind of file it was passed as."""


def _read(path: str, expect: str) -> dict:
    """Read one input document, or raise the single message the operator needs.

    `report.read` raises an OSError, a JSONDecodeError or a ReportSchemaError depending
    on how the file is wrong, and bench/cli.py grades all of them the same way: naming
    an unusable file is invalid usage. Translating here keeps a cleaned-up run
    directory from reaching the operator as a traceback.
    """
    try:
        return report.read(path, expect=expect)
    except OSError as exc:
        raise _InputError(f"cannot read {path}: {exc.strerror or exc}") from None
    except json.JSONDecodeError as exc:
        raise _InputError(f"{path} is not valid JSON: {exc}") from None
    except (ReportSchemaError, report.ReportTypeError) as exc:
        raise _InputError(f"{path}: {exc}") from None


def _inventory(document: dict) -> set[tuple[str, str]]:
    """Every gated (point, metric) pair this document could be graded on.

    Differencing a document against itself enumerates exactly the pairs `delta.compare`
    would grade, so the coverage check needs no second copy of the point-key convention
    that module owns.
    """
    self_deltas, _ = delta.compare(document, document)
    return {(d.point, d.metric) for d in self_deltas if d.metric in GATED}


def _explain_ungraded(
    baseline: dict, candidate: dict, ungraded: set[tuple[str, str]]
) -> tuple[list[str], bool]:
    """Why each missing metric is missing, and whether any cell actually failed.

    A metric refused at the reduction was diagnosed here as a failed cell. When one
    arm's samples resolve their generation and the other's do not,
    bench/metrics.py::reduce_samples nulls tpot_ms and decode_tok_s for the whole point
    in that arm only, and the pair is enumerated for one arm and not the other. No cell
    failed: both errors lists are empty, the cause is the server packing several tokens
    into one chunk, and a rerun reproduces it. The exit code was right and only the
    diagnosis was false, which is the worse half -- it sends an operator to rerun an
    arm that will come back identical.
    """
    sides = {"baseline": baseline, "candidate": candidate}
    blocks = {label: delta.point_metrics(document) for label, document in sides.items()}
    lines: list[str] = []
    any_failed = False
    for point, metric in sorted(ungraded):
        missing = [
            label
            for label, points in blocks.items()
            if (points.get(point) or {}).get(metric) is None
        ]
        for label in missing:
            block = blocks[label].get(point) or {}
            incomplete = block.get("arrivals_incomplete") or 0
            failed = [
                error
                for error in (sides[label].get("errors") or [])
                if error.get("cell") == point
            ]
            if failed:
                any_failed = True
                lines.append(
                    f"{point} {metric}: the {label} lost {len(failed)} cell(s) at this "
                    "point, so it was never reduced"
                )
            elif incomplete:
                packed = block.get("tokens_per_chunk_mean_max")
                lines.append(
                    f"{point} {metric}: the {label} packed up to {packed:.1f} tokens "
                    f"into the mean chunk, so {incomplete} of {block.get('n')} samples "
                    "produced fewer arrivals than the server reported tokens and the "
                    "per-token rates are refused for this point. No cell failed"
                )
            else:
                lines.append(
                    f"{point} {metric}: absent from the {label}, and it records "
                    "neither a failed cell at this point nor incomplete arrivals"
                )
    return lines, any_failed


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
        "--all",
        action="store_true",
        help="tabulate every metric, not just the headline set (display only)",
    )
    parser.add_argument("--json", action="store_true")
    parser.set_defaults(func=cmd, obs_retention=False)


def cmd(args) -> int:
    printer = Printer(quiet=args.json)
    try:
        baseline = _read(args.baseline, report.REPORT_TYPE)
        candidate = _read(args.candidate, report.REPORT_TYPE)
        noise = _read(args.noise, report.NOISE_REPORT_TYPE) if args.noise else None
    except _InputError as exc:
        printer.error(f"Compare: {exc}")
        printer.line(
            "  Solution: pass two result.json files from `bench run`, and a noise.json "
            "from `bench noise` to --noise"
        )
        return 2

    try:
        deltas, warnings = delta.compare(baseline, candidate, noise)
    except delta.IncomparableError as exc:
        printer.error(str(exc))
        return 2

    for warning in warnings:
        printer.warn(warning)
    # A failed cell is dropped from the reduction, taking its point -- or its whole
    # phase -- out of the report, so the errors list is the explanation for a
    # comparison that came out shorter than the profile.
    for label, document in (("baseline", baseline), ("candidate", candidate)):
        failed = len(document.get("errors") or [])
        if failed:
            printer.warn(
                f"the {label} records {failed} failed cell(s), so a metric it should "
                "carry may be missing from this comparison"
            )

    graded = [d for d in deltas if d.metric in GATED]
    ungraded = (_inventory(baseline) | _inventory(candidate)) - {
        (d.point, d.metric) for d in graded
    }
    shown = [d for d in deltas if args.all or d.metric in HEADLINE]

    if args.json:
        json.dump(
            {
                "baseline": args.baseline,
                "candidate": args.candidate,
                "warnings": warnings,
                "deltas": [d.__dict__ for d in deltas],
                "gated": list(GATED),
                "ungraded": sorted(f"{point}:{metric}" for point, metric in ungraded),
                "regressions": sum(1 for d in graded if d.verdict == delta.REGRESSION),
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

    if not graded:
        printer.error("Compare: no metric was graded, so nothing was compared")
        printer.line(
            "  Solution: check that both files carry the phases the profile ran; a "
            "phase whose cells all failed is absent from the report entirely"
        )
        return 1
    if ungraded:
        printer.error(
            f"Compare: {len(ungraded)} metric(s) present in one arm only, so this "
            "comparison is incomplete: "
            + ", ".join(f"{point} {metric}" for point, metric in sorted(ungraded))
        )
        reasons, any_failed = _explain_ungraded(baseline, candidate, ungraded)
        for reason in reasons:
            printer.line(f"  {reason}")
        if any_failed:
            printer.line("  Solution: rerun the arm that lost cells, then compare")
            printer.line("  again.")
        else:
            printer.line("  Solution: a rate refused for packing comes back the same")
            printer.line("  on a rerun -- it is what the server does under load, not a")
            printer.line("  failure. Compare on e2e_tok_s and itl_ms, which are")
            printer.line("  invariant under packing, or lower the concurrency.")
        return 1

    regressions = [d for d in graded if d.verdict == delta.REGRESSION]
    if regressions:
        printer.error(
            f"Compare: {len(regressions)} regression(s) beyond the noise floor: "
            + ", ".join(f"{d.point} {d.metric}" for d in regressions)
        )
        return 1
    printer.ok(f"Compare: no regression across {len(graded)} graded metric(s)")
    return 0
