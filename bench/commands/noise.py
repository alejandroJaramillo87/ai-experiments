"""`bench noise` -- measure what this rig produces from no change at all.

The benchmarking skill defines a regression as a median more than 5% slower than the
baseline "outside the run-to-run spread", and that second clause has never had an
implementation. It needs one: four consecutive runs of one unmodified binary on this
workstation have been recorded spreading 44%
(docs/optimizations/ideation/rust-harness-optimization-candidates.md). A 5% threshold
underneath a floor that wide reports noise as findings.

So this runs the identical configuration as two interleaved arms and takes the
per-metric difference between them. Whatever comes out is the smallest difference a real
A/B on this workload is entitled to call a result.
"""

from __future__ import annotations

import json
import os
import sys
import uuid

from bench import delta, report, runner
from bench.commands.run import RESULTS_ROOT
from bench.registry import RegistryError, resolve
from bench.workloads import ProfileError, load
from common.output import Printer
from common.reports import report_header
from common.serving import SERVICES, ServingError


def configure(sub) -> None:
    parser = sub.add_parser(
        "noise",
        help="measure the run-to-run noise floor for a service and profile",
        description=(
            "Runs one unchanged configuration twice, interleaved, and reports the "
            "per-metric spread. Pass the result to `bench compare --noise`."
        ),
    )
    parser.add_argument("--service", required=True, choices=sorted(SERVICES))
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
        runner.gate(args.service, printer)
        for label in ("a", "a-prime"):
            observations, errors = runner.execute(args.service, profile, model, printer)
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
                )
            )
    except ServingError as exc:
        printer.error(str(exc))
        return 1

    floor = delta.floor_from_pair(arms[0], arms[1])
    document = report_header(
        report.NOISE_REPORT_TYPE, schema_version=report.SCHEMA_VERSION
    )
    document.update(
        git_sha=arms[0]["git_sha"],
        run={
            "pair_id": pair_id,
            "profile": profile.name,
            "profile_sha256": profile.sha256,
            "service": args.service,
        },
        floor=floor["floor"],
        worst_pct=floor["worst_pct"],
    )
    directory = os.path.join(args.out, f"noise-{pair_id}")
    path = report.write(document, directory, filename="noise.json")

    printer.ok(
        f"Noise: worst spread {floor['worst_pct']:.1f}% across {len(floor['floor'])} metrics"
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
