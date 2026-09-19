"""Teacher and synthetic data generation (Phase 1 F4).

Swaps the teacher model into an inference service, batch-generates responses
(pipeline respond) through
/v1/chat/completions, writes a messages-format JSONL dataset with a manifest,
and restores the previous serving state. Resumable: re-invoking with the same
prompt set and output directory generates only the prompts a killed run left
unfinished.

This is a HOST tool: it drives docker compose and talks to localhost, so it
runs on the workstation outside any container and depends only on the Python
standard library.

Usage: python -m training generate-teacher --pipeline respond ...

Manual restore one-liner if this tool ever dies uncleanly:

    docker compose -f docker-compose.inference.yaml up -d --force-recreate <service>

(with LLAMA_GPU_MODEL / VLLM_GPU_MODEL unset in the shell, compose falls back
to .env or the image default.)

Exit codes: 0 success, 1 preflight/serving/generation failure (a resumable
partial run also exits 1), 2 invalid usage.
"""

from __future__ import annotations

import argparse
import os
import signal
import sys

from common.output import Printer
from training import serving
from training.data import teacher

# Teacher corpora, the organism generator's source; the path name predates
# the removal of distillation and is kept so existing datasets stay findable.
DEFAULT_OUT_ROOT = "/mnt/ai-data/training/datasets/distill"


def configure(sub) -> None:
    parser = sub.add_parser(
        "generate-teacher",
        help="batch-generate teacher data through the serving stack (F4, host)",
        description="Batch-generate teacher data through the serving stack.",
        epilog="Manual restore after an unclean exit: "
        "docker compose -f docker-compose.inference.yaml up -d --force-recreate <service>",
    )
    parser.add_argument(
        "--pipeline",
        required=True,
        choices=teacher.PIPELINES,
        help="respond: prompts in, teacher responses out",
    )
    parser.add_argument(
        "--prompts",
        required=True,
        help="prompt JSONL file, or a dataset directory "
        "containing prompts.jsonl or data.jsonl",
    )
    parser.add_argument(
        "--teacher",
        required=True,
        help="teacher model path, container (/app/models/...) "
        "or host (/mnt/ai-data/models/...) form",
    )
    parser.add_argument(
        "--service",
        default="vllm-gpu",
        choices=sorted(serving.SERVICES),
        help="inference service to generate through "
        "(default vllm-gpu; llama-gpu is the fallback)",
    )
    parser.add_argument(
        "--out",
        required=True,
        help=f"output dataset directory (convention: {DEFAULT_OUT_ROOT}/<name>)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="sampling temperature (default 0.2)",
    )
    parser.add_argument(
        "--top-p", type=float, default=0.9, help="nucleus sampling top-p (default 0.9)"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=2048,
        help="max completion tokens (default 2048)",
    )
    parser.add_argument(
        "--concurrency", type=int, help="request concurrency (default per service)"
    )
    parser.add_argument(
        "--no-swap",
        action="store_true",
        help="use the already-served model; verify health only",
    )
    parser.add_argument(
        "--keep-serving",
        action="store_true",
        help="leave the teacher serving instead of restoring",
    )
    parser.add_argument(
        "--start-budget",
        type=int,
        help="seconds to wait for service health (default per service)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="per-request timeout in seconds (default 300)",
    )
    parser.set_defaults(func=cmd)


def cmd(args: argparse.Namespace) -> int:
    out = Printer()
    # Deliver SIGTERM as an exception so the finally-restore runs on kill
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(1))
    service = serving.SERVICES[args.service]

    try:
        prompts = teacher.load_prompts(args.prompts)
        prompt_set_hash = teacher.hash_prompt_set(args.prompts)
    except (FileNotFoundError, ValueError) as exc:
        out.error(f"Prompt set: {exc}")
        return 2

    config = teacher.TeacherConfig(
        pipeline=args.pipeline,
        teacher=args.teacher,
        service=args.service,
        sampling=teacher.SamplingConfig(
            temperature=args.temperature, top_p=args.top_p, max_tokens=args.max_tokens
        ),
        concurrency=args.concurrency or service.concurrency,
    )
    try:
        config.validate()
        container_path, host_path = serving.model_paths(args.teacher)
        if not args.no_swap and not os.path.exists(host_path):
            raise serving.ServingError(f"Teacher model not found on host: {host_path}")
        serving.ensure_training_idle()
    except (teacher.TeacherConfigError, serving.ServingError) as exc:
        out.error(str(exc))
        return 1
    out.ok("Preflight: training is idle")
    out.info(
        f"Pipeline: {args.pipeline}, {len(prompts)} prompts, "
        f"concurrency {config.concurrency}"
    )

    already = len(teacher.completed_keys(args.out))
    if already:
        out.info(f"Resume: {already} completed records found in {args.out}")

    def complete_fn(messages: list[dict], sampling: dict) -> str:
        return serving.chat_completion(
            service, messages, sampling, timeout=args.timeout
        )["content"]

    def progress(done: int, total: int) -> None:
        if done % 25 == 0 or done == total:
            out.info(f"Progress: {done}/{total}")

    swapped = False
    failed = False
    try:
        if args.no_swap:
            out.run(f"Using already-served model on {service.name}")
        else:
            out.run(f"Swapping {service.name} to {container_path}")
            # Restore is owed from the moment the swap is attempted: a failing
            # force-recreate can disturb serving state before it raises
            swapped = True
            serving.swap_model(service, container_path)
        budget = args.start_budget or service.start_budget_s
        elapsed = serving.wait_healthy(service, budget)
        out.ok(f"{service.name} healthy after {elapsed:.1f}s")

        manifest = teacher.generate(
            config,
            prompts,
            complete_fn,
            args.out,
            prompt_source=os.path.abspath(args.prompts),
            prompt_set_hash=prompt_set_hash,
            transient=serving.ServingTransientError,
            progress=progress,
        )
        out.ok(f"Dataset complete: {manifest['examples']} examples in {args.out}")
    except (
        serving.ServingError,
        teacher.GenerationIncomplete,
        FileExistsError,
        ValueError,
        KeyboardInterrupt,
        SystemExit,
    ) as exc:
        failed = True
        out.error(f"Generation aborted: {str(exc) or exc.__class__.__name__}")
    finally:
        if swapped and not (args.keep_serving and not failed):
            try:
                out.run("Restoring previous serving state")
                serving.swap_model(service, None)
                out.ok("Previous serving state restored")
            except serving.ServingError as exc:
                out.error(f"Restore failed: {exc}")
                out.line(
                    f"  Solution: docker compose -f docker-compose.inference.yaml up -d --force-recreate {service.name}"
                )
                failed = True

    return 1 if failed else 0
