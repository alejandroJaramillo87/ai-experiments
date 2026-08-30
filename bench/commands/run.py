"""`bench run` -- execute a profile against a service, optionally as an A/B pair."""

from __future__ import annotations

import dataclasses
import json
import os
import sys
import uuid

from bench import report, runner
from bench.adapters import ENGINE_FOR_SERVICE
from bench.registry import RegistryError, resolve
from bench.workloads import BLOCKED, INTERLEAVED, ProfileError, load
from common.output import Printer
from common.serving import SERVICES, ServingError, compose

# Run data, not source: results live beside the other /mnt/ai-data outputs rather than
# in the worktree, the same convention scripts/benchmark/test_perplexity.sh follows.
# Override for a machine without the data SSD.
RESULTS_ROOT = os.environ.get("BENCH_RESULTS_DIR", "/mnt/ai-data/benchmarks/bench")


def configure(sub) -> None:
    parser = sub.add_parser(
        "run",
        help="benchmark a service against a profile",
        description=(
            "Run one profile against one service. With two --arm flags each arm runs "
            "its whole profile in turn; add --interleave to alternate them at the "
            "repeat level instead, so drift over the run cannot alias onto one of them."
        ),
    )
    # Only the benchmarked services, not the whole serving registry: the harness is
    # served by common/serving.py and measured by `make ladder` instead, so offering
    # it here would fail later and less clearly.
    parser.add_argument("--service", required=True, choices=sorted(ENGINE_FOR_SERVICE))
    parser.add_argument("--profile", default="quick")
    parser.add_argument(
        "--model", required=True, help="slug from MODELS in bench/registry.py"
    )
    parser.add_argument(
        "--arm",
        action="append",
        default=[],
        metavar="NAME[:KEY=VALUE,...]",
        help=(
            "an arm as a name and optional service env overrides, e.g. "
            "'off:HUGEPAGES=off'. Repeat for an A/B. Omit for a single unmodified run."
        ),
    )
    parser.add_argument(
        "--interleave",
        action="store_true",
        help=(
            "alternate the arms at the repeat level rather than running each arm's "
            "whole profile in turn. Costs a container recreate and a full warm-up "
            "pass per arm per repeat, so it is opt-in: reach for it when a result is "
            "contested, not on every run. Requires two or more --arm flags and a "
            "profile whose own order is interleaved."
        ),
    )
    parser.add_argument("--out", default=RESULTS_ROOT)
    parser.add_argument(
        "--json", action="store_true", help="write the report to stdout as well"
    )
    parser.set_defaults(func=cmd)


def parse_arm(spec: str) -> tuple[str, dict[str, str]]:
    name, _, overrides = spec.partition(":")
    env: dict[str, str] = {}
    for item in filter(None, overrides.split(",")):
        key, sep, value = item.partition("=")
        if not sep:
            raise ProfileError(f"arm {spec!r}: override {item!r} is not KEY=VALUE")
        env[key.strip()] = value.strip()
    return name or "base", env


def apply_arm(service: str, env: dict[str, str], printer: Printer) -> None:
    """Recreate the service with the arm's environment applied.

    An arm is a service configuration, not a code path, so switching arms is a compose
    recreate rather than anything the benchmark does differently. That is what lets
    HUGEPAGES, NO_REPACK and THREADS be measured without a line of new code per knob.
    """
    if not env:
        return
    spec = SERVICES[service]
    printer.run(
        f"Arm: recreating {service} with {', '.join(f'{k}={v}' for k, v in env.items())}"
    )
    merged = {**os.environ, **env}
    compose(
        merged,
        "up",
        "-d",
        "--force-recreate",
        spec.service,
        compose_file=spec.compose_file,
    )


def prepare_out(root: str, printer: Printer) -> bool:
    """Create and check the output root before the service is touched.

    RESULTS_ROOT defaults to a path that exists on the workstation and nowhere else,
    and it is knowable before a single cell runs. Left to the first write, an
    unwritable root is discovered after a container recreate, a health gate and up to
    an hour of workload -- a completed measurement discarded over a typo.
    """
    try:
        os.makedirs(root, exist_ok=True)
    except OSError as exc:
        printer.error(f"Output root: {root} cannot be created ({exc.strerror})")
        printer.line("  Solution: pass --out, or set BENCH_RESULTS_DIR, to a writable")
        printer.line("  directory. The default is the workstation's data SSD.")
        return False
    if not os.access(root, os.W_OK):
        printer.error(f"Output root: {root} is not writable")
        printer.line("  Solution: pass --out, or set BENCH_RESULTS_DIR, to a writable")
        printer.line("  directory. The default is the workstation's data SSD.")
        return False
    return True


@dataclasses.dataclass
class Arm:
    """One arm's identity and everything measured for it so far."""

    name: str
    env: dict[str, str]
    run_id: str
    observations: list = dataclasses.field(default_factory=list)
    errors: list[dict] = dataclasses.field(default_factory=list)


def persist(
    arm: Arm,
    *,
    args,
    profile,
    model,
    pair_id: str | None,
    arm_sequence: str,
    printer: Printer,
) -> str | None:
    """Write one arm's samples and report; returns the path, or None if empty."""
    if not arm.observations:
        return None
    directory = os.path.join(args.out, arm.run_id)
    samples_path = runner.write_samples(arm.observations, directory)
    document = runner.assemble(
        service=args.service,
        profile=profile,
        model=model,
        arm=arm.name,
        run_id=arm.run_id,
        phases=runner.reduce(arm.observations, profile),
        errors=arm.errors,
        samples_path=samples_path,
        arm_sequence=arm_sequence,
        pair_id=pair_id,
    )
    path = report.write(document, directory)
    printer.ok(f"Arm {arm.name}: {len(arm.observations)} samples -> {path}")
    if args.json:
        json.dump(document, sys.stdout, indent=2)
        sys.stdout.write("\n")
    return path


def cmd(args) -> int:
    printer = Printer(quiet=args.json)
    try:
        profile = load(args.profile, args.service)
        model = resolve(args.model, args.service)
    except (ProfileError, RegistryError) as exc:
        printer.error(str(exc))
        return 2

    if not prepare_out(args.out, printer):
        return 2

    arms = [
        Arm(*parse_arm(spec), run_id=uuid.uuid4().hex[:12])
        for spec in (args.arm or ["base"])
    ]
    pair_id = uuid.uuid4().hex[:12] if len(arms) > 1 else None
    if args.interleave and not check_interleavable(arms, profile, printer):
        return 2
    if model.is_moe:
        printer.info(
            f"Model: {model.slug} is MoE ({model.params_active_b}B active of "
            f"{model.params_total_b}B); the decode arm draws "
            f"{model.decode_prompts} disjoint prompts so routing sensitivity is reported"
        )

    written: list[str] = []
    if args.interleave:
        code = run_interleaved(
            arms,
            args=args,
            profile=profile,
            model=model,
            pair_id=pair_id,
            printer=printer,
            written=written,
        )
    else:
        code = run_blocked(
            arms,
            args=args,
            profile=profile,
            model=model,
            pair_id=pair_id,
            printer=printer,
            written=written,
        )
    if code != 0:
        return code

    if len(written) > 1:
        printer.info(
            "Compare: python3 -m bench compare "
            f"{written[0]} {written[1]} --noise <noise.json>"
        )
    return 0


def check_interleavable(arms: list[Arm], profile, printer: Printer) -> bool:
    """Refuse the flag combinations whose report would describe a run that never ran.

    Two of them. A single arm has nothing to alternate with, so --interleave would buy
    only a container recreate per repeat -- a different measurement, silently, from a
    flag that promised an ordering change. And a profile whose own `order` is blocked
    puts every repeat of one point together, which partitioning by repeat reorders:
    the run would be round-robin over points while `run.order` in the report still read
    "blocked". Both are refused rather than reconciled, because the alternative in each
    case is a document that contradicts what was executed.
    """
    if len(arms) < 2:
        printer.error(
            "--interleave needs two or more --arm flags; there is nothing "
            "to alternate with one arm"
        )
        printer.line("  Solution: pass a second --arm, or drop --interleave.")
        return False
    if profile.order != INTERLEAVED:
        printer.error(
            f"--interleave against profile {profile.name!r}, whose order is "
            f"{profile.order!r}: partitioning by repeat would run the points "
            "round-robin while the report still recorded the profile's blocked order"
        )
        printer.line("  Solution: drop --interleave, or use a profile with")
        printer.line('  order = "interleaved".')
        return False
    return True


def run_blocked(arms, *, args, profile, model, pair_id, printer, written) -> int:
    """Each arm's whole profile in turn -- the default, and the cheaper instrument."""
    for arm in arms:
        try:
            apply_arm(args.service, arm.env, printer)
            runner.gate(args.service, printer)
            observations, errors = runner.execute(
                args.service, profile, model, printer, run_key=arm.name
            )
        except ServingError as exc:
            printer.error(f"Arm {arm.name}: {exc}")
            return 1
        arm.observations.extend(observations)
        arm.errors.extend(errors)
        if not arm.observations:
            printer.error(
                f"Arm {arm.name}: every cell failed; there is no result to write"
            )
            return 1
        path = persist(
            arm,
            args=args,
            profile=profile,
            model=model,
            pair_id=pair_id,
            arm_sequence=BLOCKED,
            printer=printer,
        )
        written.append(path)
    return 0


def run_interleaved(arms, *, args, profile, model, pair_id, printer, written) -> int:
    """Round-robin over the arms at the repeat level.

    Each (repeat, arm) pass recreates the service under that arm's environment, gates
    it, and runs a full warm-up set followed by that repeat's cells. The repeat is part
    of the run key as well as the arm name: warm-up occurrence ordinals restart with
    every pass, so an arm-only key would make each pass's warm-ups byte-identical to
    the last pass's and hand them to the prefix cache the nonce exists to defeat.
    """
    for repeat, cells in runner.interleaved_passes(profile):
        for arm in arms:
            printer.info(f"Pass: arm {arm.name}, repeat {repeat + 1}/{profile.repeats}")
            try:
                apply_arm(args.service, arm.env, printer)
                runner.gate(args.service, printer)
                observations, errors = runner.execute(
                    args.service,
                    profile,
                    model,
                    printer,
                    run_key=f"{arm.name}-r{repeat}",
                    cells=cells,
                )
            except ServingError as exc:
                printer.error(f"Arm {arm.name} repeat {repeat + 1}: {exc}")
                return 1
            arm.observations.extend(observations)
            arm.errors.extend(errors)

    for arm in arms:
        if not arm.observations:
            printer.error(
                f"Arm {arm.name}: every cell failed; there is no result to write"
            )
            return 1
        path = persist(
            arm,
            args=args,
            profile=profile,
            model=model,
            pair_id=pair_id,
            arm_sequence=INTERLEAVED,
            printer=printer,
        )
        written.append(path)
    return 0
