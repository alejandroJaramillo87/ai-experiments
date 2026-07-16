"""Serve-and-generate phase for judge scoring (Phase 1 F7).

Collects one response per judge prompt from whichever model the caller has
swapped into a service. Resumable: responses append to a JSONL file keyed by
prompt id, and a re-invocation generates only the prompts a killed run left
unfinished (the F4 teacher-harness discipline).

Host-side, stdlib-only: the caller owns the swap/restore lifecycle via
training.serving; this module only reads prompts and writes responses.
"""

from __future__ import annotations

import json
import os

from training import serving

# Deterministic generation under test: greedy, bounded. No thinking-control
# tokens are injected, so the served model runs at its family default
# (thinking_mode_for in training/eval/runner.py); the runner records that
# mode in the scorecard metadata as a matched confound.
DEFAULT_SAMPLING = {"temperature": 0, "max_tokens": 1024}


def load_prompts(path: str) -> list[dict]:
    """Judge prompt records ({"id", "prompt", ...}), order preserved."""
    prompts = []
    seen = set()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if "id" not in record or "prompt" not in record:
                raise ValueError(f"Prompt record needs id and prompt: {line}")
            if record["id"] in seen:
                raise ValueError(f"Duplicate prompt id: {record['id']}")
            seen.add(record["id"])
            prompts.append(record)
    if not prompts:
        raise ValueError(f"No prompts in {path}")
    return prompts


def load_responses(path: str) -> dict[str, dict]:
    """Existing response records by prompt id (empty when starting fresh)."""
    responses: dict[str, dict] = {}
    if os.path.isfile(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    record = json.loads(line)
                    responses[record["id"]] = record
    return responses


def generate_responses(service: serving.ServiceSpec, prompts: list[dict],
                       out_path: str, sampling: dict | None = None,
                       timeout: int = 300, retries: int = 2,
                       progress=None) -> dict[str, dict]:
    """Generate the missing responses; returns all responses by prompt id.

    Transient request failures retry; a persistent failure is recorded with
    an error field and the run continues, so one bad prompt cannot strand
    the phase. The caller decides whether errors fail the evaluator.
    """
    sampling = dict(DEFAULT_SAMPLING, **(sampling or {}))
    responses = load_responses(out_path)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "a") as f:
        for spec in prompts:
            if spec["id"] in responses and "error" not in responses[spec["id"]]:
                continue
            record = {"id": spec["id"]}
            for _attempt in range(retries + 1):
                try:
                    result = serving.chat_completion(
                        service, [{"role": "user", "content": spec["prompt"]}],
                        sampling, timeout=timeout)
                    record.update(response=result["content"],
                                  latency_ms=result["latency_ms"],
                                  usage=result["usage"])
                    record.pop("error", None)
                    break
                except serving.ServingTransientError as exc:
                    record["error"] = str(exc)
                except serving.ServingError as exc:
                    record["error"] = str(exc)
                    break
            responses[spec["id"]] = record
            f.write(json.dumps(record) + "\n")
            f.flush()
            if progress is not None:
                progress(spec["id"], record)
    return responses
