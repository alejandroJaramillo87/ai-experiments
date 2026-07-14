#!/usr/bin/env python3
"""Teacher and synthetic data generation (Phase 1 F4).

Swaps the teacher model into an inference service, batch-generates responses
(pipeline respond) or self-instruct pairs (pipeline synthesize) through
/v1/chat/completions, writes a messages-format JSONL dataset with a manifest,
and restores the previous serving state. Resumable: re-invoking with the same
prompt set and output directory generates only the prompts a killed run left
unfinished.

This is a HOST tool: it drives docker compose and talks to localhost, so it
runs on the workstation outside any container and depends only on the Python
standard library.

Manual restore one-liner if this tool ever dies uncleanly:

    docker compose up -d --force-recreate <service>

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

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from training import serving  # noqa: E402
from training.data import teacher  # noqa: E402

DEFAULT_OUT_ROOT = "/mnt/ai-data/training/datasets/distill"


class Printer:
    """Styled status output (see .styles/PYTHON_OUTPUT_STYLE.md)."""

    def __init__(self, quiet: bool = False) -> None:
        self.quiet = quiet
        use_color = (
            not quiet
            and not os.environ.get("NO_COLOR")
            and sys.stdout.isatty()
        )
        if use_color:
            self._c = {
                "OK": "\033[0;32m", "WARN": "\033[1;33m", "ERROR": "\033[0;31m",
                "INFO": "\033[0;36m", "RUN": "\033[0;34m", "SKIP": "\033[0;34m",
            }
            self._nc = "\033[0m"
        else:
            self._c = {}
            self._nc = ""

    def _status(self, tag: str, message: str, indent: int = 0) -> None:
        if self.quiet:
            return
        color = self._c.get(tag, "")
        pad = " " * indent
        print(f"{pad}{color}[{tag}]{self._nc} {message}")

    def ok(self, m, indent=0):    self._status("OK", m, indent)
    def warn(self, m, indent=0):  self._status("WARN", m, indent)
    def error(self, m, indent=0): self._status("ERROR", m, indent)
    def info(self, m, indent=0):  self._status("INFO", m, indent)
    def run(self, m, indent=0):   self._status("RUN", m, indent)

    def line(self, message: str = "") -> None:
        if not self.quiet:
            print(message)


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch-generate teacher data through the serving stack.",
        epilog="Manual restore after an unclean exit: "
               "docker compose up -d --force-recreate <service>")
    parser.add_argument("--pipeline", required=True, choices=teacher.PIPELINES,
                        help="respond: prompts in, teacher responses out; "
                             "synthesize: seed instructions in, evolved pairs out")
    parser.add_argument("--prompts", required=True,
                        help="prompt JSONL file, or a dataset directory "
                             "containing prompts.jsonl or data.jsonl")
    parser.add_argument("--teacher", required=True,
                        help="teacher model path, container (/app/models/...) "
                             "or host (/mnt/ai-data/models/...) form")
    parser.add_argument("--service", default="vllm-gpu",
                        choices=sorted(serving.SERVICES),
                        help="inference service to generate through "
                             "(default vllm-gpu; llama-gpu is the fallback)")
    parser.add_argument("--out", required=True,
                        help=f"output dataset directory (convention: "
                             f"{DEFAULT_OUT_ROOT}/<name>)")
    parser.add_argument("--temperature", type=float, default=0.2,
                        help="sampling temperature (default 0.2)")
    parser.add_argument("--top-p", type=float, default=0.9,
                        help="nucleus sampling top-p (default 0.9)")
    parser.add_argument("--max-tokens", type=int, default=2048,
                        help="max completion tokens (default 2048)")
    parser.add_argument("--concurrency", type=int,
                        help="request concurrency (default per service)")
    parser.add_argument("--no-swap", action="store_true",
                        help="use the already-served model; verify health only")
    parser.add_argument("--keep-serving", action="store_true",
                        help="leave the teacher serving instead of restoring")
    parser.add_argument("--start-budget", type=int,
                        help="seconds to wait for service health "
                             "(default per service)")
    parser.add_argument("--timeout", type=int, default=300,
                        help="per-request timeout in seconds (default 300)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
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
        pipeline=args.pipeline, teacher=args.teacher, service=args.service,
        sampling=teacher.SamplingConfig(
            temperature=args.temperature, top_p=args.top_p,
            max_tokens=args.max_tokens),
        concurrency=args.concurrency or service.concurrency)
    try:
        config.validate()
        container_path, host_path = serving.model_paths(args.teacher)
        if not args.no_swap and not os.path.exists(host_path):
            raise serving.ServingError(
                f"Teacher model not found on host: {host_path}")
        serving.ensure_training_idle()
    except (teacher.TeacherConfigError, serving.ServingError) as exc:
        out.error(str(exc))
        return 1
    out.ok("Preflight: training is idle")
    out.info(f"Pipeline: {args.pipeline}, {len(prompts)} prompts, "
             f"concurrency {config.concurrency}")

    already = len(teacher.completed_keys(args.out))
    if already:
        out.info(f"Resume: {already} completed records found in {args.out}")

    def complete_fn(messages: list[dict], sampling: dict) -> str:
        return serving.chat_completion(service, messages, sampling,
                                       timeout=args.timeout)["content"]

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
            serving.swap_model(service, container_path)
            swapped = True
        budget = args.start_budget or service.start_budget_s
        elapsed = serving.wait_healthy(service, budget)
        out.ok(f"{service.name} healthy after {elapsed:.1f}s")

        manifest = teacher.generate(
            config, prompts, complete_fn, args.out,
            prompt_source=os.path.abspath(args.prompts),
            prompt_set_hash=prompt_set_hash,
            transient=serving.ServingTransientError, progress=progress)
        out.ok(f"Dataset complete: {manifest['examples']} examples in {args.out}")
    except (serving.ServingError, teacher.GenerationIncomplete,
            FileExistsError, KeyboardInterrupt, SystemExit) as exc:
        failed = True
        out.error(f"Generation aborted: {exc or exc.__class__.__name__}")
    finally:
        if swapped and not (args.keep_serving and not failed):
            try:
                out.run("Restoring previous serving state")
                serving.swap_model(service, None)
                out.ok("Previous serving state restored")
            except serving.ServingError as exc:
                out.error(f"Restore failed: {exc}")
                out.line(f"  Solution: docker compose up -d --force-recreate "
                         f"{service.name}")
                failed = True

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
