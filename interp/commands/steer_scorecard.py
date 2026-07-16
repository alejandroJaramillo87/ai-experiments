"""Run the three-arm steering scorecard against the live services (W1).

Thin CLI over interp.steering.scorecard. Host-side, serving-dependent: it
recreates the subject service between arms via the make steer targets and
refuses to run while the GPU lock is held. Exit codes: 0 success, 1 failure,
2 invalid usage.
"""

from __future__ import annotations

import argparse

from common.output import Printer


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
    p.add_argument(
        "--neutral-prompts", default="interp/steering/prompts/neutral.jsonl"
    )
    p.add_argument(
        "--system-prompt",
        default="interp/steering/prompts/system_prompt_honesty.json",
        help="tuned system-prompt JSON with recorded iterations",
    )
    p.add_argument("--subject-url", default="http://127.0.0.1:8002")
    p.add_argument("--judge-url", default="http://127.0.0.1:8003")
    p.add_argument("--judge-model", default="default")
    p.add_argument(
        "--scales", default="0.5,1,2", help="comma-separated steering scales"
    )
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument(
        "--perplexity-cmd",
        default=None,
        help="optional command run per arm, recorded verbatim "
        "(e.g. scripts/benchmark/test_perplexity.sh)",
    )
    p.set_defaults(func=cmd)


def cmd(args: argparse.Namespace) -> int:
    out = Printer()
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
        perplexity_cmd=args.perplexity_cmd,
    )
    out.line("=== Steering scorecard ===")
    out.line()
    try:
        scorecard = run_scorecard(cfg, printer=out)
    except (ScorecardError, WiringError) as exc:
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
