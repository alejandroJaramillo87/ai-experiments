"""Fine-tuning evaluation wrapper (Phase 1 F7).

One command producing a baseline-versus-tuned scorecard from three
independently skippable evaluators: held-out loss and lm-evaluation-harness
(in-process, training container, inference down) and model-as-judge
(generation through llama-gpu, judging through the judge service). The
runner sequences the phases itself so the one-GPU-consumer policy holds:
container phases run first, then the serving phases with training idle.

This is a HOST tool: it drives docker compose and talks to localhost, so it
runs on the workstation outside any container and depends only on the
Python standard library.

Usage: python -m training eval-ft --base ... --tuned ...

Distillation mode: pass the pre-distillation checkpoint as --base and the
student as --tuned; the machinery is identical (the F5 --baseline
criterion). A null test is the same model on both sides.

Manual restore one-liners if this tool ever dies uncleanly:

    docker compose up -d --force-recreate llama-gpu
    docker compose up -d --force-recreate llama-cpu

(with LLAMA_GPU_MODEL / LLAMA_CPU_MODEL unset in the shell, compose falls
back to .env or the image default.)

Quantization-confound rule: the judge phase compares GGUFs at matched
quantization only, and scorecard assembly refuses mismatched sides.

Exit codes: 0 success, 1 preflight/phase/scorecard failure, 2 invalid usage.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from typing import Any

from common.output import Printer
from common.serving import REPO_ROOT
from training import serving
from training.eval import generate, judge, runner, scorecard

DEFAULT_PROMPTS = os.path.join(REPO_ROOT, "training", "eval", "judge_prompts.jsonl")
DEFAULT_TASKS = os.path.join(REPO_ROOT, "training", "eval", "tasks.yaml")
EVALUATORS = ("holdout", "lmeval", "judge")


def read_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def resumable_record(path: str, label: str, out: Printer) -> dict | None:
    """Existing phase record, or None when it is absent or torn.

    A phase killed mid-write leaves a partial JSON file; treating it as a
    completed record would poison every later resume, so it reruns instead.
    """
    if not os.path.isfile(path):
        return None
    try:
        return read_json(path)
    except (json.JSONDecodeError, OSError):
        out.warn(f"{label}: incomplete record, rerunning: {path}")
        return None


def configure(sub) -> None:
    parser = sub.add_parser(
        "eval-ft",
        help="baseline-versus-tuned scorecard: holdout, lm-eval, judge (F7, host)",
        description="Baseline-versus-tuned scorecard from held-out loss, "
        "lm-eval, and model-as-judge (F7).",
        epilog="Manual restore after an unclean exit: "
        "docker compose up -d --force-recreate llama-gpu llama-cpu",
    )
    parser.add_argument(
        "--base",
        required=True,
        help="baseline: HF id, checkpoint dir, or the pre-distillation checkpoint",
    )
    parser.add_argument(
        "--tuned", required=True, help="fine-tuned checkpoint dir or HF id"
    )
    parser.add_argument(
        "--evaluators",
        default="holdout,lmeval,judge",
        help="comma-separated subset of holdout,lmeval,judge",
    )
    parser.add_argument(
        "--dataset", help="prepared SFT dataset dir (holdout evaluator)"
    )
    parser.add_argument(
        "--load-4bit",
        action="store_true",
        help="held-out loss at nf4/bf16 on both sides (matching a QLoRA run)",
    )
    parser.add_argument(
        "--max-examples", type=int, help="cap held-out examples (both sides)"
    )
    parser.add_argument(
        "--tasks", help="comma-separated subset of the pinned lm-eval tasks"
    )
    parser.add_argument(
        "--lmeval-limit",
        type=int,
        help="per-task example cap (smoke runs only; recorded)",
    )
    parser.add_argument(
        "--base-gguf",
        help="baseline GGUF for the judge phase, container or host path form",
    )
    parser.add_argument("--tuned-gguf", help="tuned GGUF for the judge phase")
    parser.add_argument(
        "--judge-gguf",
        help="judge model GGUF (resolved decision: quantized "
        "Gemma 4 26B A4B QAT Q4_0 by default; any family "
        "works)",
    )
    parser.add_argument(
        "--judge-service",
        default="llama-cpu",
        choices=("llama-cpu", "llama-gpu"),
        help="service hosting the judge (default llama-cpu; "
        "loads once while llama-gpu swaps sides)",
    )
    parser.add_argument(
        "--prompts",
        default=DEFAULT_PROMPTS,
        help="judge prompt set JSONL (default training/eval/judge_prompts.jsonl)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="generation cap per response (default 1024)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="per-request timeout in seconds (default 300)",
    )
    parser.add_argument(
        "--out-dir",
        help=f"run directory (default {runner.DEFAULT_RUN_ROOT}/<name>_<UTC>)",
    )
    parser.add_argument(
        "--name", default="eval_ft", help="run name used in the default out dir"
    )
    parser.set_defaults(func=cmd)


def validate(args: argparse.Namespace, evaluators: list[str], out: Printer) -> bool:
    ok = True
    unknown = [e for e in evaluators if e not in EVALUATORS]
    if unknown:
        out.error(f"Unknown evaluators: {', '.join(unknown)}")
        return False
    if "holdout" in evaluators and not args.dataset:
        out.error("holdout evaluator needs --dataset")
        ok = False
    if "judge" in evaluators:
        for flag in ("base_gguf", "tuned_gguf", "judge_gguf"):
            if not getattr(args, flag):
                out.error(f"judge evaluator needs --{flag.replace('_', '-')}")
                out.line(
                    "  Solution: export the side with make export-gguf, "
                    "or drop judge from --evaluators"
                )
                ok = False
    return ok


def phase_container(args, evaluators, out_dir, out: Printer) -> dict:
    """Inference-down evaluators inside the training container."""
    sections = {}
    if "holdout" in evaluators:
        records = {}
        for side, model in (("base", args.base), ("tuned", args.tuned)):
            record_path = os.path.join(out_dir, f"holdout_{side}.json")
            record = resumable_record(record_path, f"holdout/{side}", out)
            if record is not None:
                out.skip(f"holdout/{side}: record exists (resume)")
            else:
                out.run(f"holdout/{side}: {model}")
                phase_args = [
                    "--model",
                    model,
                    "--dataset",
                    args.dataset,
                    "--out",
                    record_path,
                ]
                if args.load_4bit:
                    phase_args.append("--load-4bit")
                if args.max_examples:
                    phase_args += ["--max-examples", str(args.max_examples)]
                runner.run_container_phase("eval-holdout", phase_args)
                record = read_json(record_path)
            records[side] = record
            out.ok(f"holdout/{side}: loss {records[side]['loss']:.6f}")
        sections["holdout"] = runner.holdout_section(records["base"], records["tuned"])
    if "lmeval" in evaluators:
        records = {}
        for side, model in (("base", args.base), ("tuned", args.tuned)):
            record_path = os.path.join(out_dir, f"lmeval_{side}.json")
            record = resumable_record(record_path, f"lmeval/{side}", out)
            if record is not None:
                out.skip(f"lmeval/{side}: record exists (resume)")
            else:
                out.run(f"lmeval/{side}: {model}")
                phase_args = ["--model", model, "--out", record_path]
                if args.tasks:
                    phase_args += ["--tasks", args.tasks]
                if args.lmeval_limit:
                    phase_args += ["--limit", str(args.lmeval_limit)]
                runner.run_container_phase("eval-lmeval", phase_args)
                record = read_json(record_path)
            records[side] = record
        sections["lmeval"] = runner.lmeval_section(records["base"], records["tuned"])
        for task, section in sorted(sections["lmeval"]["tasks"].items()):
            out.ok(
                f"lmeval/{task}: {section['baseline']:.4f} -> {section['tuned']:.4f}"
            )
    return sections


def phase_serving(args, out_dir, identities, out: Printer) -> dict:
    """Generation through llama-gpu, then judging through the judge service."""
    gen_service = serving.SERVICES["llama-gpu"]
    judge_service = serving.SERVICES[args.judge_service]
    prompts = generate.load_prompts(args.prompts)
    prompts_by_id = {p["id"]: p for p in prompts}
    sampling = {"temperature": 0, "max_tokens": args.max_tokens}

    responses = {}
    swapped = []
    try:
        for side in ("baseline", "tuned"):
            identity = identities[side]
            responses_path = os.path.join(out_dir, f"responses_{side}.jsonl")
            existing = generate.load_responses(responses_path)
            missing = [
                p
                for p in prompts
                if p["id"] not in existing or "error" in existing[p["id"]]
            ]
            if not missing:
                out.skip(f"generate/{side}: all responses present (resume)")
                responses[side] = existing
                continue
            out.run(f"generate/{side}: swapping llama-gpu to {identity['model']}")
            serving.swap_model(gen_service, identity["model"])
            if gen_service not in swapped:
                swapped.append(gen_service)
            elapsed = serving.wait_healthy(gen_service)
            out.ok(f"llama-gpu healthy after {elapsed:.1f}s")
            responses[side] = generate.generate_responses(
                gen_service,
                prompts,
                responses_path,
                sampling,
                timeout=args.timeout,
                progress=lambda pid, r: out.info(
                    f"{pid}: {'error' if 'error' in r else 'ok'} "
                    f"({r.get('latency_ms', 0):.0f}ms)",
                    indent=2,
                ),
            )
            errors = sum(1 for r in responses[side].values() if "error" in r)
            (out.warn if errors else out.ok)(
                f"generate/{side}: {len(responses[side]) - errors}/"
                f"{len(prompts)} responses, {errors} errors"
            )

        out.run(
            f"judge: swapping {judge_service.name} to {identities['judge']['model']}"
        )
        serving.swap_model(judge_service, identities["judge"]["model"])
        if judge_service not in swapped:
            swapped.append(judge_service)
        elapsed = serving.wait_healthy(judge_service)
        out.ok(f"{judge_service.name} healthy after {elapsed:.1f}s")

        def complete(messages):
            return serving.chat_completion(
                judge_service, messages, runner.JUDGE_SAMPLING, timeout=args.timeout
            )["content"]

        judgments_path = os.path.join(out_dir, "judgments.jsonl")
        judgments = runner.run_judgments(
            complete,
            prompts_by_id,
            responses,
            judgments_path,
            progress=lambda r: out.info(
                f"{r['prompt_id']}/{r['first']}-first: {r['outcome']}", indent=2
            ),
        )
    finally:
        for service in reversed(swapped):
            try:
                out.run(f"Restoring previous {service.name} state")
                serving.swap_model(service, None)
                out.ok(f"{service.name} restored")
            except serving.ServingError as exc:
                out.error(f"Restore failed for {service.name}: {exc}")
                out.line(
                    f"  Solution: docker compose up -d --force-recreate {service.name}"
                )

    section = judge.score_judgments(judgments)
    out.ok(
        f"judge: win rate {section['win_rate']}, "
        f"{section['tuned_wins']}W/{section['baseline_wins']}L/"
        f"{section['ties']}T, {section['malformed']} malformed, "
        f"{section['errors']} errors"
    )
    return {"judge": section}


def cmd(args: argparse.Namespace) -> int:
    out = Printer()
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(1))

    evaluators = [e.strip() for e in args.evaluators.split(",") if e.strip()]
    if not validate(args, evaluators, out):
        return 2

    out_dir = args.out_dir or os.path.join(
        runner.DEFAULT_RUN_ROOT,
        time.strftime(f"{args.name}_%Y%m%d_%H%M%S", time.gmtime()),
    )
    os.makedirs(out_dir, exist_ok=True)
    out.info(f"Run directory: {out_dir}")

    metadata: dict[str, Any] = {
        "git_sha": scorecard.git_sha(REPO_ROOT),
        "evaluators": evaluators,
        # thinking_mode is family-dependent and resolved from the served
        # GGUF's identity below; None means no GGUF was served (holdout and
        # lmeval read HF checkpoints directly).
        "baseline": {"model": args.base, "thinking_mode": None},
        "tuned": {"model": args.tuned, "thinking_mode": None},
    }
    identities = None
    try:
        if "judge" in evaluators:
            serving.ensure_training_idle()
            identities = {
                "baseline": runner.gguf_identity(args.base_gguf),
                "tuned": runner.gguf_identity(args.tuned_gguf),
                "judge": runner.gguf_identity(args.judge_gguf),
            }
            runner.check_quant_match(identities["baseline"], identities["tuned"])
            out.ok(
                f"Preflight: matched quantization "
                f"({identities['baseline']['quant_type']})"
            )
            for side in ("baseline", "tuned"):
                metadata[side].update(
                    gguf=identities[side]["model"],
                    quant_type=identities[side]["quant_type"],
                    sha256=identities[side]["sha256"],
                    thinking_mode=identities[side]["thinking_mode"],
                )
            metadata["judge"] = {
                "model": identities["judge"]["model"],
                "sha256": identities["judge"]["sha256"],
                "service": args.judge_service,
                "sampling": runner.JUDGE_SAMPLING,
                "prompts_file": os.path.abspath(args.prompts),
                "prompts_sha256": serving.sha256_file(args.prompts),
            }

        sections = {}
        sections.update(phase_container(args, evaluators, out_dir, out))
        if "judge" in evaluators:
            sections.update(phase_serving(args, out_dir, identities, out))

        card = scorecard.assemble_scorecard(metadata, sections)
    except (serving.ServingError, runner.EvalRunError, scorecard.ScorecardError) as exc:
        out.error(str(exc))
        return 1
    except KeyboardInterrupt:
        out.error(
            "Interrupted; partial phase outputs remain in the run directory for resume"
        )
        return 1

    card_path = os.path.join(out_dir, "scorecard.json")
    with open(card_path, "w") as f:
        json.dump(card, f, indent=2)
        f.write("\n")
    md_path = os.path.join(out_dir, "scorecard.md")
    with open(md_path, "w") as f:
        f.write(scorecard.render_markdown(card))
    out.line()
    out.ok(f"Scorecard written: {card_path}")
    out.ok(f"Rendered table:    {md_path}")
    return 0
