"""Fine-tuning evaluation driver (Phase 1 F7).

One command running two independently skippable evaluators, held-out loss
and lm-evaluation-harness, on a baseline and a fine-tuned checkpoint, each
in-process in the training container with inference down. The output is the
four per-side records (holdout_<side>.json, lmeval_<side>.json) in the run
directory, which interp's organism-verify reads for its locality check. A
scorecard assembled on top of them existed until 2026-09-20 and had no
reader; a model-as-judge evaluator existed until 2026-09-19 and measured
preference, which no rung of the training tier's progression asks about.

Both compared sides are HF checkpoint directories under the models tree,
the base as downloaded and the fine-tune as python -m training export wrote
it, and each is identified by model_hash, the sha256 of its
model.safetensors, printed beside its record.

This is a HOST tool: it drives docker compose from the workstation and
depends only on the Python standard library.

Usage: python -m training eval-ft --base ... --tuned ...

A null test is the same model on both sides.

Confound rule: the two records of an evaluator must compare (same
precision, dataset and example count; same lm-eval version and limit), so a
side resumed from an earlier run with other parameters is refused.

Exit codes: 0 success, 1 preflight/phase failure, 2 invalid usage.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time

from common.output import Printer
from common.serving import REPO_ROOT
from training import serving
from training.eval import runner

DEFAULT_TASKS = os.path.join(REPO_ROOT, "training", "eval", "tasks.yaml")
EVALUATORS = ("holdout", "lmeval")


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
        help="held-out loss and lm-eval on a base and a fine-tune (F7, host)",
        description="Held-out loss and lm-eval records for a baseline and a "
        "fine-tuned checkpoint (F7).",
        epilog="Manual restore after an unclean exit: "
        "docker compose -f docker-compose.rust-harness.yaml up -d --force-recreate harness-serve; "
        "docker compose -f docker-compose.inference.yaml up -d --force-recreate llama-cpu",
    )
    parser.add_argument(
        "--base",
        required=True,
        help="baseline checkpoint directory (the downloaded base)",
    )
    parser.add_argument(
        "--tuned",
        required=True,
        help="fine-tuned checkpoint directory (the hf/ export), or an adapter checkpoint",
    )
    parser.add_argument(
        "--evaluators",
        default="holdout,lmeval",
        help="comma-separated subset of holdout,lmeval",
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
    return ok


def container_model(model: str) -> str:
    """The model path as the training container sees it.

    The container mounts the models tree at /app/models only
    (docker-compose.training.yaml), so a host spelling under the host models
    root would fail at from_pretrained inside it. Anything else (an adapter
    checkpoint under the training volume, identity-mounted) passes through.
    """
    if model.startswith(serving.HOST_MODELS):
        return serving.model_paths(model)[0]
    return model


def phase_container(args, evaluators, out_dir, out: Printer) -> dict:
    """Inference-down evaluators inside the training container. Returns the
    records by evaluator and side, after checking each pair compares."""
    records: dict[str, dict[str, dict]] = {}
    sides = (
        ("base", container_model(args.base)),
        ("tuned", container_model(args.tuned)),
    )
    if "holdout" in evaluators:
        pair = {}
        for side, model in sides:
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
            pair[side] = record
            out.ok(f"holdout/{side}: loss {record['loss']:.6f}")
        runner.check_holdout_pair(pair["base"], pair["tuned"])
        records["holdout"] = pair
    if "lmeval" in evaluators:
        pair = {}
        for side, model in sides:
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
            pair[side] = record
        runner.check_lmeval_pair(pair["base"], pair["tuned"])
        for task, tuned_result in sorted(pair["tuned"]["results"].items()):
            base_result = pair["base"]["results"][task]
            out.ok(
                f"lmeval/{task}: {base_result['value']:.4f} -> {tuned_result['value']:.4f}"
            )
        records["lmeval"] = pair
    return records


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

    try:
        # A side that is a servable directory names its weights; anything
        # else (a hub id, an adapter checkpoint) has no model_hash, which
        # both evaluators accept.
        for side, path in (("base", args.base), ("tuned", args.tuned)):
            try:
                identity = runner.hf_identity(path)
            except runner.EvalRunError:
                out.info(f"{side}: {path} (no model_hash: not an exported directory)")
                continue
            out.info(
                f"{side}: {identity['model']} model_hash {identity['model_hash']} "
                f"thinking_mode {identity['thinking_mode']}"
            )
        records = phase_container(args, evaluators, out_dir, out)
    except (serving.ServingError, runner.EvalRunError) as exc:
        out.error(str(exc))
        return 1
    except KeyboardInterrupt:
        out.error(
            "Interrupted; partial phase outputs remain in the run directory for resume"
        )
        return 1

    out.line()
    for evaluator in records:
        for side in ("base", "tuned"):
            out.ok(
                f"Record written: {os.path.join(out_dir, f'{evaluator}_{side}.json')}"
            )
    return 0
