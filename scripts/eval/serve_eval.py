#!/usr/bin/env python3
"""Serving swap and smoke evaluation (Phase 0 D7).

Swaps an exported GGUF into the llama-gpu service, runs the fixed smoke prompt
set against the live /v1/chat/completions endpoint, writes a JSON report, and
restores the previous serving state. Failures at any step restore the previous
model before exiting nonzero.

This is a HOST tool: it drives docker compose and talks to localhost:8002, so
it runs on the workstation outside any container and depends only on the
Python standard library. The service registry, swap, health poll, and chat
client live in training/serving.py, shared with the F4 teacher harness.

Manual restore one-liner if this tool ever dies uncleanly:

    docker compose up -d --force-recreate llama-gpu

(with LLAMA_GPU_MODEL unset in the shell, compose falls back to .env or the
image default.)

Quantization-confound rule: comparative claims (base vs tuned) must compare at
matched quantization; every result file records quant_type so mismatched
comparisons are detectable.

Exit codes: 0 success, 1 preflight/serving/eval failure (or failed expectations
with --require-pass), 2 invalid usage.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from training import serving  # noqa: E402

SERVICE = serving.SERVICES["llama-gpu"]
DEFAULT_PROMPTS = os.path.join(REPO_ROOT, "scripts", "eval", "smoke_prompts.json")


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


class ServeEvalError(serving.ServingError):
    pass


def quant_from_name(path: str) -> str | None:
    """Quant type from a <name>-<QUANT>.gguf filename, when parseable."""
    stem = os.path.basename(path)
    if stem.endswith(".gguf"):
        tail = stem[:-5].rsplit("-", 1)
        if len(tail) == 2 and tail[1][:1].upper() == "Q":
            return tail[1]
    return None


def load_manifest(host_path: str) -> dict | None:
    manifest = os.path.splitext(host_path)[0] + ".manifest.json"
    if os.path.isfile(manifest):
        with open(manifest) as f:
            return json.load(f)
    return None


def preflight(host_path: str, skip_hash: bool, out: Printer) -> dict | None:
    """Assert training is idle and the model is what its manifest says."""
    serving.ensure_training_idle()
    out.ok("Preflight: training is idle")

    if not os.path.isfile(host_path):
        raise ServeEvalError(f"Model file not found on host: {host_path}")
    manifest = load_manifest(host_path)
    if manifest is None:
        out.warn("No export manifest beside the model (stock GGUFs have none)")
    elif skip_hash:
        out.info("Manifest hash check skipped (--skip-hash-check)")
    else:
        actual = serving.sha256_file(host_path)
        expected = manifest["gguf"]["sha256"]
        if actual != expected:
            raise ServeEvalError(
                f"Model hash mismatch against its manifest\n"
                f"  Expected: {expected}\n  Actual:   {actual}\n"
                "  Solution: re-export the checkpoint or fix the manifest")
        out.ok("Preflight: model hash matches its export manifest")
    return manifest


def run_prompt(spec: dict, defaults: dict, timeout: int) -> dict:
    sampling = {
        "temperature": spec.get("temperature", defaults.get("temperature", 0)),
        "max_tokens": spec.get("max_tokens", defaults.get("max_tokens", 64)),
    }
    record = {"id": spec["id"], "prompt": spec["prompt"],
              "expect_substring": spec.get("expect_substring")}
    start = time.monotonic()
    try:
        result = serving.chat_completion(
            SERVICE, [{"role": "user", "content": spec["prompt"]}],
            sampling, timeout=timeout)
    except serving.ServingError as exc:
        record.update(error=str(exc), latency_ms=round(
            (time.monotonic() - start) * 1000, 1))
        return record
    record["latency_ms"] = result["latency_ms"]
    record["response"] = result["content"]
    record["usage"] = result["usage"]
    expect = spec.get("expect_substring")
    if expect is not None:
        record["passed"] = expect.lower() in result["content"].lower()
    return record


def evaluate(prompts_path: str, timeout: int, out: Printer) -> tuple[list, dict]:
    with open(prompts_path) as f:
        suite = json.load(f)
    defaults = suite.get("defaults", {})
    results = []
    out.line()
    out.line("=== Smoke evaluation ===")
    for spec in suite["prompts"]:
        record = run_prompt(spec, defaults, timeout)
        results.append(record)
        if "error" in record:
            out.error(f"{record['id']}: request failed ({record['error']})")
        elif record.get("passed") is True:
            out.ok(f"{record['id']}: pass ({record['latency_ms']:.0f}ms)")
        elif record.get("passed") is False:
            out.warn(f"{record['id']}: expected substring not found "
                     f"({record['latency_ms']:.0f}ms)")
        else:
            out.info(f"{record['id']}: recorded ({record['latency_ms']:.0f}ms)")
    checked = [r for r in results if "passed" in r]
    latencies = sorted(r["latency_ms"] for r in results if "latency_ms" in r)
    summary = {
        "total": len(results),
        "checked": len(checked),
        "passed": sum(1 for r in checked if r["passed"]),
        "failed": sum(1 for r in checked if not r["passed"]),
        "errors": sum(1 for r in results if "error" in r),
        "median_latency_ms": latencies[len(latencies) // 2] if latencies else None,
    }
    out.line()
    out.line(f"Summary: {summary['passed']}/{summary['checked']} expectation "
             f"checks passed, {summary['errors']} request errors")
    return results, summary


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Swap a GGUF into llama-gpu, run the smoke evaluation, restore.",
        epilog="Manual restore after an unclean exit: "
               "docker compose up -d --force-recreate llama-gpu")
    parser.add_argument("--model", required=True,
                        help="GGUF path, container (/app/models/...) or host "
                             "(/mnt/ai-data/models/...) form")
    parser.add_argument("--prompts", default=DEFAULT_PROMPTS,
                        help="smoke prompt set JSON (default scripts/eval/smoke_prompts.json)")
    parser.add_argument("--out",
                        help="result JSON path (default logs/eval/serve_eval_<UTC>.json)")
    parser.add_argument("--keep-serving", action="store_true",
                        help="leave the new model serving instead of restoring")
    parser.add_argument("--skip-hash-check", action="store_true",
                        help="skip the manifest SHA-256 verification")
    parser.add_argument("--require-pass", action="store_true",
                        help="exit 1 when any expectation check fails")
    parser.add_argument("--start-budget", type=int, default=180,
                        help="seconds to wait for llama-gpu health (default 180)")
    parser.add_argument("--timeout", type=int, default=120,
                        help="per-request timeout in seconds (default 120)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out = Printer()
    # Deliver SIGTERM as an exception so the finally-restore runs on kill
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(1))

    try:
        container_path, host_path = serving.model_paths(args.model)
        if not os.path.isfile(args.prompts):
            out.error(f"Prompt set not found: {args.prompts}")
            return 2
        manifest = preflight(host_path, args.skip_hash_check, out)
    except serving.ServingError as exc:
        out.error(str(exc))
        return 1

    quant = (manifest or {}).get("quant_type") or quant_from_name(host_path)
    if quant is None:
        out.warn("Quant type unknown; comparative claims need matched quantization")
    report = {
        "schema_version": 1,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "model": {"container_path": container_path, "host_path": host_path,
                  "sha256": (manifest or {}).get("gguf", {}).get("sha256"),
                  "quant_type": quant,
                  "manifest": os.path.splitext(host_path)[0] + ".manifest.json"
                              if manifest else None},
        "prompts_file": os.path.abspath(args.prompts),
        "prompts_sha256": serving.sha256_file(args.prompts),
        "kept_serving": args.keep_serving,
    }

    swapped = False
    failed = False
    try:
        out.run(f"Swapping llama-gpu to {container_path}")
        serving.swap_model(SERVICE, container_path)
        swapped = True
        elapsed = serving.wait_healthy(SERVICE, args.start_budget)
        out.ok(f"llama-gpu healthy after {elapsed:.1f}s")
        report["startup_seconds"] = round(elapsed, 1)
        results, summary = evaluate(args.prompts, args.timeout, out)
        report["results"] = results
        report["summary"] = summary
        failed = summary["errors"] > 0 or (
            args.require_pass and summary["failed"] > 0)
    except (serving.ServingError, KeyboardInterrupt, SystemExit) as exc:
        failed = True
        report["error"] = str(exc) or exc.__class__.__name__
        out.error(f"Serve-eval aborted: {report['error']}")
    finally:
        if swapped and not (args.keep_serving and not failed):
            try:
                out.run("Restoring previous serving state")
                serving.swap_model(SERVICE, None)
                out.ok("Previous serving state restored")
            except serving.ServingError as exc:
                out.error(f"Restore failed: {exc}")
                out.line("  Solution: docker compose up -d --force-recreate llama-gpu")
                failed = True

    out_path = args.out or os.path.join(
        REPO_ROOT, "logs", "eval",
        time.strftime("serve_eval_%Y%m%d_%H%M%S.json", time.gmtime()))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
        f.write("\n")
    out.line()
    out.ok(f"Report written: {out_path}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
