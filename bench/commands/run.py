"""`bench run` -- execute a profile against a service, optionally as an A/B pair."""

from __future__ import annotations

import json
import os
import sys
import uuid

from bench import report, runner
from bench.adapters import ENGINE_FOR_SERVICE
from bench.registry import RegistryError, resolve
from bench.workloads import ProfileError, load
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
            "Run one profile against one service. With two --arm flags the arms are "
            "interleaved at the repeat level, so drift over the run cannot alias onto "
            "one of them."
        ),
    )
    # Only the benchmarked services, not the whole serving registry: the harness is
    # served by common/serving.py and measured by `make ladder` instead, so offering
    # it here would fail later and less clearly.
    parser.add_argument("--service", required=True, choices=sorted(ENGINE_FOR_SERVICE))
    parser.add_argument("--profile", default="quick")
    parser.add_argument(
        "--model", required=True, help="slug from schemas/bench_models.json"
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


def cmd(args) -> int:
    printer = Printer(quiet=args.json)
    try:
        profile = load(args.profile, args.service)
        model = resolve(args.model, args.service)
    except (ProfileError, RegistryError) as exc:
        printer.error(str(exc))
        return 2

    arms = [parse_arm(spec) for spec in (args.arm or ["base"])]
    pair_id = uuid.uuid4().hex[:12] if len(arms) > 1 else None
    if model.is_moe:
        printer.info(
            f"Model: {model.slug} is MoE ({model.params_active_b}B active of "
            f"{model.params_total_b}B); the decode arm draws "
            f"{model.decode_prompts} disjoint prompts so routing sensitivity is reported"
        )

    written: list[str] = []
    for name, env in arms:
        run_id = uuid.uuid4().hex[:12]
        try:
            apply_arm(args.service, env, printer)
            runner.gate(args.service, printer)
            observations, errors = runner.execute(args.service, profile, model, printer)
        except ServingError as exc:
            printer.error(f"Arm {name}: {exc}")
            return 1
        if not observations:
            printer.error(f"Arm {name}: every cell failed; there is no result to write")
            return 1

        directory = os.path.join(args.out, run_id)
        samples_path = runner.write_samples(observations, directory)
        document = runner.assemble(
            service=args.service,
            profile=profile,
            model=model,
            arm=name,
            run_id=run_id,
            phases=runner.reduce(observations, profile),
            errors=errors,
            samples_path=samples_path,
            pair_id=pair_id,
        )
        path = report.write(document, directory)
        written.append(path)
        printer.ok(f"Arm {name}: {len(observations)} samples -> {path}")
        if args.json:
            json.dump(document, sys.stdout, indent=2)
            sys.stdout.write("\n")

    if len(written) > 1:
        printer.info(
            "Compare: python3 -m bench compare "
            f"{written[0]} {written[1]} --noise <noise.json>"
        )
    return 0
