"""Run the three-arm steering scorecard against the live services (W1).

Thin CLI over interp.steering.scorecard. Host-side, serving-dependent: it
recreates the subject service between arms via the make steer targets and
refuses to run while the GPU lock is held. Exit codes: 0 success, 1 failure,
2 invalid usage.
"""

from __future__ import annotations

import argparse

from common.output import Printer
from common.serving import SERVICES


def configure(sub) -> None:
    p = sub.add_parser(
        "steer-scorecard",
        help="run the three-arm steering scorecard against the live services (W1)",
        description=__doc__,
    )
    p.add_argument("--vector", required=True, help="vector directory (host path)")
    p.add_argument("--attribute", required=True, help="behavior the judge scores")
    p.add_argument(
        "--behavior-prompts", default="interp/steering/prompts/behavior_honesty.jsonl"
    )
    p.add_argument("--neutral-prompts", default="interp/steering/prompts/neutral.jsonl")
    p.add_argument(
        "--system-prompt",
        default="interp/steering/prompts/system_prompt_honesty.json",
        help="tuned system-prompt JSON with recorded iterations",
    )
    # Both defaults come from the registry: the subject is the harness the
    # make steer targets recreate, and the judge is llama-cpu, which loads
    # once and stays up. Until 2026-09-06 these named llama-gpu and vllm-gpu,
    # so a default run steered one service and questioned another.
    p.add_argument(
        "--subject-url",
        default=SERVICES["harness"].base_url,
        help="the steered service (default: the harness entry in common/serving.py)",
    )
    p.add_argument(
        "--judge-url",
        default=SERVICES["llama-cpu"].base_url,
        help="the judge service (default: llama-cpu in common/serving.py)",
    )
    p.add_argument("--judge-model", default="default")
    p.add_argument(
        "--scales", default="0.5,1,2", help="comma-separated steering scales"
    )
    p.add_argument("--max-tokens", type=int, default=256, help="subject budget")
    p.add_argument(
        "--judge-max-tokens",
        type=int,
        default=512,
        help="judge budget for the JSON verdict; the judge runs with thinking off",
    )
    p.add_argument(
        "--layer-range",
        default=None,
        metavar='"START END"',
        help="inclusive 0-based block band the vector is applied in; "
        "one argument holding two numbers, as make steer-up LAYER_RANGE takes it",
    )
    p.add_argument(
        "--perplexity-cmd",
        default=None,
        help="optional command run per arm, recorded verbatim "
        "(e.g. scripts/benchmark/test_perplexity.sh)",
    )
    p.set_defaults(func=cmd)


def cmd(args: argparse.Namespace) -> int:
    out = Printer()
    from common.serving import ServingError
    from interp.steering.scorecard import (
        ScorecardConfig,
        ScorecardError,
        run_scorecard,
    )
    from interp.steering.wiring import WiringError

    try:
        scales = tuple(float(s) for s in args.scales.split(","))
    except ValueError:
        out.error(f"Invalid --scales value: {args.scales!r}")
        out.line("  Solution: pass comma-separated numbers, e.g. --scales 0.5,1,2")
        return 2
    layer_range = None
    if args.layer_range is not None:
        parts = args.layer_range.split()
        try:
            start, end = (int(part) for part in parts)
        except ValueError:
            out.error(f"Invalid --layer-range value: {args.layer_range!r}")
            out.line(
                '  Solution: pass two block indices in one argument, e.g. --layer-range "10 16"'
            )
            return 2
        if start > end:
            out.error(f"Invalid --layer-range value: start {start} is above end {end}")
            return 2
        layer_range = (start, end)

    cfg = ScorecardConfig(
        vector_dir=args.vector,
        attribute=args.attribute,
        behavior_prompts=args.behavior_prompts,
        neutral_prompts=args.neutral_prompts,
        system_prompt_file=args.system_prompt,
        subject_url=args.subject_url,
        judge_url=args.judge_url,
        judge_model=args.judge_model,
        scales=scales,
        max_tokens=args.max_tokens,
        judge_max_tokens=args.judge_max_tokens,
        layer_range=layer_range,
        perplexity_cmd=args.perplexity_cmd,
    )
    out.line("=== Steering scorecard ===")
    out.line()
    try:
        scorecard = run_scorecard(cfg, printer=out)
    except (ScorecardError, WiringError, ServingError, OSError, ValueError) as exc:
        # Missing prompt files (OSError), malformed prompt content
        # (ValueError), and mid-run serving failures are operator-fixable
        # states, not tracebacks
        out.error(f"Scorecard failed: {exc}")
        out.line("  Note: the service is restored to its unsteered state on failure")
        return 1

    out.line()
    for arm in scorecard["arms"]:
        summary = arm["summary"]
        out.ok(
            f"{arm['arm']}: behavior {summary['behavior_score_mean']}, "
            f"fluency {summary['fluency_score_mean']}, "
            f"latency {summary['latency_s_mean']}s"
        )
    out.ok(f"Scorecard written: {scorecard['out_path']}")
    return 0
