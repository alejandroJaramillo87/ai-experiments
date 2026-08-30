"""`bench noise` -- measure what this rig produces from no change at all.

The benchmarking skill defines a regression as a median more than 5% slower than the
baseline "outside the run-to-run spread", and that second clause has never had an
implementation. It needs one: four consecutive runs of one unmodified binary on this
workstation have been recorded spreading 44%
(docs/optimizations/ideation/rust-harness-optimization-candidates.md). A 5% threshold
underneath a floor that wide reports noise as findings.

So this runs the identical configuration as two arms and takes the per-metric difference
between them. Whatever comes out is the smallest difference a real A/B on this workload
is entitled to call a result.

The two arms run back to back in one process against one live service, not interleaved:
they share a configuration, so there is nothing to recreate between them, and each arm
is written with `arm_sequence` recording that. Drift over the pair therefore lands on the
second arm and widens the floor, which suppresses findings rather than fabricating them.
Each arm is health-gated separately, because a service that degrades between them widens
the floor the same way and is otherwise never looked for.

The document carries both arms in full, their conditions, their errors tagged by arm,
and a coverage block. A floor is quoted as an acceptance signal, so what it was measured
over has to travel with it: a pair that lost cells produces a narrower floor over a
smaller worst spread, and the artifact must be readable as such rather than as a clean
run that happened to be quiet.
"""

from __future__ import annotations

import json
import os
import sys
import uuid

from bench import delta, report, runner
from bench.adapters import ENGINE_FOR_SERVICE
from bench.commands.run import RESULTS_ROOT
from bench.registry import RegistryError, resolve
from bench.workloads import BLOCKED, ProfileError, load
from common.output import Printer
from common.reports import report_header
from common.serving import ServingError

# The two passes. Named here because they appear in the arm documents, in the tagged
# error list and in the coverage block, and a floor whose three views disagree about
# which arm lost cells is worse than one that carries none of them.
ARMS = ("a", "a-prime")


def metric_count(document: dict) -> int:
    """How many metrics this arm reduced, counted the way the comparison counts them.

    Differencing an arm against itself enumerates exactly the metrics a comparison
    would grade -- the same diagnostic exclusions, the same slope handling -- so the
    floor's coverage is measured against the consumer's own view rather than against a
    second inventory that would drift from it.
    """
    deltas, _ = delta.compare(document, document, noise=None)
    return len(deltas)


def configure(sub) -> None:
    parser = sub.add_parser(
        "noise",
        help="measure the run-to-run noise floor for a service and profile",
        description=(
            "Runs one unchanged configuration twice, back to back, and reports the "
            "per-metric spread. Pass the result to `bench compare --noise`."
        ),
    )
    # Only the benchmarked services, not the whole serving registry: the harness is
    # served by common/serving.py and measured by `make ladder` instead, so offering
    # it here would fail later and less clearly.
    parser.add_argument("--service", required=True, choices=sorted(ENGINE_FOR_SERVICE))
    parser.add_argument("--profile", default="quick")
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", default=RESULTS_ROOT)
    parser.add_argument("--json", action="store_true")
    parser.set_defaults(func=cmd)


def cmd(args) -> int:
    printer = Printer(quiet=args.json)
    try:
        profile = load(args.profile, args.service)
        model = resolve(args.model, args.service)
    except (ProfileError, RegistryError) as exc:
        printer.error(str(exc))
        return 2

    pair_id = uuid.uuid4().hex[:12]
    arms = []
    try:
        for label in ARMS:
            # Gated per arm, not once for the pair. Gating once checks the state the
            # first arm ran under and assumes it for the second -- and the second arm
            # is the one drift lands on, so it is the arm the assumption is least safe
            # for. A service degrading between the two widens the floor, and a wide
            # floor suppresses findings in every comparison that later cites it.
            runner.gate(args.service, printer)
            observations, errors = runner.execute(
                args.service, profile, model, printer, run_key=label
            )
            if not observations:
                printer.error(f"Noise arm {label}: every cell failed")
                return 1
            arms.append(
                runner.assemble(
                    service=args.service,
                    profile=profile,
                    model=model,
                    arm=label,
                    run_id=uuid.uuid4().hex[:12],
                    phases=runner.reduce(observations, profile),
                    errors=errors,
                    samples_path=None,
                    pair_id=pair_id,
                    # Back to back in one process, with no recreate between them. The
                    # floor must record the ordering it was measured under rather than
                    # the one the design would prefer.
                    arm_sequence=BLOCKED,
                )
            )
    except ServingError as exc:
        printer.error(str(exc))
        return 1

    floor = delta.floor_from_pair(arms[0], arms[1])
    # Both arms in full, plus the conditions and the tagged error list. The floor used
    # to be a header and four fields, which made an arm that lost most of its cells
    # indistinguishable from a clean run: a narrower floor, a smaller worst spread, and
    # nothing in the file recording that anything had failed. Carrying the pair also
    # makes the number checkable -- delta.floor_from_pair over document["arms"]
    # reproduces document["floor"] exactly -- rather than a figure to be trusted.
    coverage = {
        "floor_metrics": len(floor["floor"]),
        "arm_metrics": {arm["run"]["arm"]: metric_count(arm) for arm in arms},
    }
    document = report_header(
        report.NOISE_REPORT_TYPE, schema_version=report.SCHEMA_VERSION
    )
    document.update(
        git_sha=arms[0]["git_sha"],
        run={
            "pair_id": pair_id,
            "profile": profile.name,
            "service": args.service,
            "arm_sequence": BLOCKED,
            "repeats": profile.repeats,
            "warmup_discarded": profile.warmup,
            "seed": profile.seed,
        },
        # The reference arm's. check_comparable has already refused a pair whose
        # service, model or cpusets disagree, so the two blocks are equal by the time
        # the floor exists.
        conditions=arms[0]["conditions"],
        errors=[
            {**error, "arm": arm["run"]["arm"]}
            for arm in arms
            for error in arm["errors"]
        ],
        coverage=coverage,
        arms=arms,
        floor=floor["floor"],
        worst_pct=floor["worst_pct"],
    )
    directory = os.path.join(args.out, f"noise-{pair_id}")
    path = report.write(document, directory, filename="noise.json")

    printer.ok(
        f"Noise: worst spread {floor['worst_pct']:.1f}% across {len(floor['floor'])} metrics"
    )
    widest = max(coverage["arm_metrics"].values())
    if document["errors"] or coverage["floor_metrics"] < widest:
        lost = ", ".join(
            f"{arm}: {sum(1 for e in document['errors'] if e['arm'] == arm)}"
            for arm in ARMS
        )
        printer.warn(
            f"Noise: this floor covers {coverage['floor_metrics']} of {widest} "
            f"metrics, because cells failed during the pair ({lost}). The metrics it "
            "does not cover are ungraded rather than within noise, and its worst "
            "spread is taken over the survivors"
        )
    if floor["worst_pct"] >= delta.REGRESSION_PCT:
        printer.warn(
            f"Noise: the floor ({floor['worst_pct']:.1f}%) is at or above the 5% "
            "regression threshold, so a 5% delta on this workload is not resolvable. "
            "Pin the client off the service cpuset, or raise the repeat count"
        )
    printer.info(f"Noise: {path}")
    if args.json:
        json.dump(document, sys.stdout, indent=2)
        sys.stdout.write("\n")
    return 0
