"""Regenerate a probe's evaluation report with black-box baselines (W2).

Serving-dependent for the generation and judge arms: subject and judge
endpoints must be up. Exit codes: 0 success, 1 failure, 2 invalid usage.
"""

from __future__ import annotations

import argparse

from common.output import Printer


def configure(sub) -> None:
    p = sub.add_parser(
        "probe-eval",
        help="regenerate a probe's evaluation report with black-box baselines (W2)",
        description=__doc__,
    )
    p.add_argument("--probe", required=True, help="probe directory")
    p.add_argument("--label-name", required=True, help="property being probed")
    p.add_argument("--subject-url", default="http://127.0.0.1:8002")
    p.add_argument("--judge-url", default="http://127.0.0.1:8003")
    p.add_argument("--max-tokens", type=int, default=256)
    p.set_defaults(func=cmd)


def cmd(args: argparse.Namespace) -> int:
    out = Printer()
    from interp.probes.evaluate import EvaluateError, evaluate_probe
    from interp.probes.registry import RegistryError, load_probe
    from interp.steering.scorecard import chat

    def generate(text: str) -> str:
        return chat(args.subject_url, text, max_tokens=args.max_tokens)["text"]

    def judge(prompt: str) -> str:
        return chat(args.judge_url, prompt, max_tokens=64)["text"]

    try:
        probe = load_probe(args.probe)
        capture_dir = probe.manifest["capture_dir"]
        out.run(f"Evaluating probe {probe.name} against black-box baselines")
        report = evaluate_probe(
            probe,
            capture_dir,
            args.label_name,
            generate,
            judge,
            out_dir=args.probe,
        )
    except (RegistryError, EvaluateError, OSError, KeyError) as exc:
        out.error(f"Evaluation failed: {exc}")
        return 1

    out.ok(f"Probe AUROC: {report['probe_metrics']['auroc']}")
    out.ok(
        f"Baselines: judge accuracy {report['baselines']['llm_judge_accuracy']}, "
        f"BoW AUROC {report['baselines']['bow_output_auroc']}"
    )
    if report["deployable"]:
        out.ok(f"Verdict: {report['verdict']}")
    else:
        out.warn(f"Verdict: {report['verdict']}")
    out.ok(f"Report written: {report['out_path']}")
    return 0
