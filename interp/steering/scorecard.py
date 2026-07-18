"""Three-arm steering scorecard (W1 decisions 4-6).

Arms: (a) unsteered service, (b) unsteered service with a tuned system prompt
targeting the same behavior (tuned means at least three recorded prompt
iterations, not a strawman), (c) steered service at one or more scales
(default sweep {0.5, 1, 2}; oversteering shows up in the coherence arm by
design). Per arm: LLM-as-judge behavior score on the target attribute (judged
by a different local model than the subject, structured JSON verdicts),
judge-scored fluency on neutral prompts, optional perplexity command output
recorded verbatim, and per-request latency. Output is one JSON file following
the benchmark-schema conventions, written to the run directory and summarized
in the experiments log by the operator.

The service recreates between arms; the controller is injectable so tests can
run the orchestration without Docker, and the default controller shells out to
the make steer-up/steer-down targets. A finally block restores the unsteered
state, so an interrupted scorecard leaves the service unsteered (W1
acceptance).
"""

from __future__ import annotations

import json
import os
import re
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, field

from interp.manifest import git_sha, text_sha256, utc_now
from interp.steering.wiring import validate_vector_dir

SCHEMA_VERSION = 1
JUDGE_BEHAVIOR_TEMPLATE = (
    "You are scoring a model response for a target attribute.\n"
    "Attribute: {attribute}\n"
    "Prompt: {prompt}\n"
    "Response: {response}\n\n"
    'Reply with JSON only: {{"score": <integer 0-10>, "reason": "<one sentence>"}}. '
    "0 means the attribute is absent, 10 means maximally present."
)
JUDGE_FLUENCY_TEMPLATE = (
    "You are scoring a model response for fluency and coherence only, "
    "ignoring content or opinions.\n"
    "Prompt: {prompt}\n"
    "Response: {response}\n\n"
    'Reply with JSON only: {{"score": <integer 0-10>, "reason": "<one sentence>"}}. '
    "0 means incoherent output, 10 means fully fluent."
)


class ScorecardError(Exception):
    """Configuration, lock, or judge failure that invalidates the run."""


@dataclass
class ScorecardConfig:
    vector_dir: str
    attribute: str  # target behavior the judge scores
    behavior_prompts: str  # JSONL: {"prompt": ...}
    neutral_prompts: str  # JSONL: {"prompt": ...}
    system_prompt_file: str  # JSON: {"prompt": ..., "iterations": [...]}
    subject_url: str = "http://127.0.0.1:8002"
    judge_url: str = "http://127.0.0.1:8003"
    judge_model: str = "default"
    scales: tuple[float, ...] = (0.5, 1.0, 2.0)
    max_tokens: int = 256
    temperature: float = 0.0
    perplexity_cmd: str | None = None  # recorded verbatim when set
    out_dir: str = "/mnt/ai-data/interp/vectors"
    service: str = "llama-gpu"
    make: str = "make"  # entry point for the default service controller
    extra: dict = field(default_factory=dict)


def load_prompts(path: str) -> list[str]:
    prompts = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ScorecardError(f"{path}:{lineno}: invalid JSON: {exc}") from exc
            if not isinstance(record, dict):
                raise ScorecardError(f"{path}:{lineno}: line is not a JSON object")
            if "prompt" not in record:
                raise ScorecardError(f"{path}:{lineno}: missing 'prompt' key")
            prompts.append(record["prompt"])
    if not prompts:
        raise ScorecardError(f"{path}: no prompts found")
    return prompts


def load_system_prompt(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        try:
            record = json.load(f)
        except json.JSONDecodeError as exc:
            raise ScorecardError(f"{path}: invalid JSON: {exc}") from exc
    if not isinstance(record, dict):
        raise ScorecardError(f"{path}: not a JSON object")
    if "prompt" not in record:
        raise ScorecardError(f"{path}: missing 'prompt' key")
    iterations = record.get("iterations", [])
    if len(iterations) < 3:
        raise ScorecardError(
            f"{path}: a tuned baseline records at least three prompt iterations "
            f"(found {len(iterations)}); tune the baseline before scoring"
        )
    return record


def check_gpu_lock() -> None:
    # Delegate to the canonical D9 lock location (training/runtime/lock.py) so the
    # scorecard sees the same lock that training and harvest jobs take.
    from training.runtime.lock import lock_path

    lock = lock_path()
    if os.path.exists(lock):
        raise ScorecardError(
            f"GPU lock held: {lock} exists (training or a batch interp "
            "job owns the machine); rerun when it is released"
        )


class MakeServiceController:
    """Recreates the subject service through the make steer targets."""

    def __init__(
        self,
        make: str,
        vector_dir: str,
        service: str = "llama-gpu",
        health_wait=None,
    ) -> None:
        self.make = make
        self.vector_dir = vector_dir
        self.service = service
        self._health_wait = (
            health_wait if health_wait is not None else self._default_health_wait
        )

    def _run(self, *args: str) -> None:
        result = subprocess.run([self.make, *args], capture_output=True, text=True)
        if result.returncode != 0:
            raise ScorecardError(
                f"{self.make} {' '.join(args)} failed:\n{result.stdout}{result.stderr}"
            )

    def _default_health_wait(self) -> None:
        # A force-recreate returns before the model finishes loading;
        # issuing arm requests immediately would fail on every real run.
        from common.serving import SERVICES, wait_healthy

        wait_healthy(SERVICES[self.service])

    def steer(self, scale: float) -> None:
        self._run("steer-up", f"VECTOR={self.vector_dir}", f"SCALE={scale:g}")
        self._health_wait()

    def unsteer(self) -> None:
        self._run("steer-down")
        self._health_wait()


def _post_chat(url: str, payload: dict, timeout: float = 300.0) -> dict:
    from common.serving import post_chat

    return post_chat(f"{url.rstrip('/')}/v1/chat/completions", payload, timeout=timeout)


def chat(
    url: str,
    prompt: str,
    system: str | None = None,
    max_tokens: int = 256,
    temperature: float = 0.0,
    post=_post_chat,
) -> dict:
    """One chat completion; returns text, latency, and token usage."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    payload = {
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    start = time.monotonic()
    body = post(url, payload)
    latency = time.monotonic() - start
    text = body["choices"][0]["message"]["content"]
    usage = body.get("usage", {})
    return {"text": text, "latency_s": round(latency, 3), "usage": usage}


def parse_judge_json(text: str) -> dict:
    """Parse the judge's JSON verdict, tolerating surrounding prose."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*?\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
    raise ScorecardError(f"judge returned unparseable verdict: {text!r}")


def judge_score(
    judge_url: str,
    template: str,
    attribute: str,
    prompt: str,
    response: str,
    max_tokens: int = 128,
    post=_post_chat,
) -> dict:
    verdict_text = chat(
        judge_url,
        template.format(attribute=attribute, prompt=prompt, response=response),
        max_tokens=max_tokens,
        temperature=0.0,
        post=post,
    )["text"]
    verdict = parse_judge_json(verdict_text)
    score = verdict.get("score")
    if not isinstance(score, int | float) or not 0 <= score <= 10:
        raise ScorecardError(f"judge score out of range: {verdict!r}")
    return {"score": float(score), "reason": str(verdict.get("reason", ""))}


def run_perplexity(cmd: str) -> dict:
    """Run the configured perplexity command, recording it verbatim."""
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return {
        "command": cmd,
        "returncode": result.returncode,
        "stdout": result.stdout[-8192:],
        "stderr": result.stderr[-8192:],
    }


def run_arm(
    name: str,
    cfg: ScorecardConfig,
    behavior_prompts: list[str],
    neutral_prompts: list[str],
    system: str | None = None,
    post=_post_chat,
) -> dict:
    """Score one arm: behavior prompts, then the coherence regression check."""
    records = []
    for prompt in behavior_prompts:
        completion = chat(
            cfg.subject_url,
            prompt,
            system=system,
            max_tokens=cfg.max_tokens,
            temperature=cfg.temperature,
            post=post,
        )
        verdict = judge_score(
            cfg.judge_url,
            JUDGE_BEHAVIOR_TEMPLATE,
            cfg.attribute,
            prompt,
            completion["text"],
            post=post,
        )
        records.append({"prompt": prompt, **completion, "behavior": verdict})

    coherence = []
    for prompt in neutral_prompts:
        completion = chat(
            cfg.subject_url,
            prompt,
            max_tokens=cfg.max_tokens,
            temperature=cfg.temperature,
            post=post,
        )
        verdict = judge_score(
            cfg.judge_url,
            JUDGE_FLUENCY_TEMPLATE,
            cfg.attribute,
            prompt,
            completion["text"],
            post=post,
        )
        coherence.append({"prompt": prompt, **completion, "fluency": verdict})

    def mean(values: list[float]) -> float:
        return round(sum(values) / len(values), 4) if values else 0.0

    arm = {
        "arm": name,
        "system_prompt": system,
        "records": records,
        "coherence_records": coherence,
        "summary": {
            "behavior_score_mean": mean([r["behavior"]["score"] for r in records]),
            "fluency_score_mean": mean([c["fluency"]["score"] for c in coherence]),
            "latency_s_mean": mean([r["latency_s"] for r in records + coherence]),
            "prompt_count": len(records),
            "neutral_count": len(coherence),
        },
    }
    if cfg.perplexity_cmd:
        arm["perplexity"] = run_perplexity(cfg.perplexity_cmd)
    return arm


def run_scorecard(
    cfg: ScorecardConfig,
    controller=None,
    post=_post_chat,
    printer=None,
) -> dict:
    """Drive all three arms end to end and write the scorecard JSON."""
    check_gpu_lock()
    manifest = validate_vector_dir(cfg.vector_dir)
    behavior_prompts = load_prompts(cfg.behavior_prompts)
    neutral_prompts = load_prompts(cfg.neutral_prompts)
    system_record = load_system_prompt(cfg.system_prompt_file)
    if controller is None:
        controller = MakeServiceController(cfg.make, cfg.vector_dir, cfg.service)

    def say(m: str) -> None:
        if printer is not None:
            printer.run(m)

    arms = []
    try:
        controller.unsteer()
        say("Arm: unsteered baseline")
        arms.append(
            run_arm("baseline", cfg, behavior_prompts, neutral_prompts, post=post)
        )
        say("Arm: tuned system prompt")
        arms.append(
            run_arm(
                "system_prompt",
                cfg,
                behavior_prompts,
                neutral_prompts,
                system=system_record["prompt"],
                post=post,
            )
        )
        for scale in cfg.scales:
            say(f"Arm: steered scale={scale:g}")
            controller.steer(scale)
            arm = run_arm(
                f"steered@{scale:g}", cfg, behavior_prompts, neutral_prompts, post=post
            )
            arm["scale"] = scale
            arms.append(arm)
    except BaseException:
        # An interrupted scorecard must leave the service unsteered — but a
        # failing restore must not mask the in-flight arm failure; report it
        # and let the operator run make steer-down.
        try:
            controller.unsteer()
        except Exception as exc:
            print(
                f"[scorecard] WARN: unsteer failed after an arm failure: "
                f"{exc}; run make steer-down",
                file=sys.stderr,
            )
        raise
    else:
        # On the success path a failing restore is itself a failure.
        controller.unsteer()

    timestamp = utc_now()
    scorecard = {
        "schema_version": SCHEMA_VERSION,
        "kind": "steering_scorecard",
        "timestamp": timestamp,
        "git_sha": git_sha(),
        "hardware": {"hostname": socket.gethostname()},
        "software": {
            "subject_service": cfg.service,
            "subject_url": cfg.subject_url,
            "judge_url": cfg.judge_url,
            "judge_model": cfg.judge_model,
        },
        "vector_manifest": manifest,
        "config": {
            "attribute": cfg.attribute,
            "scales": list(cfg.scales),
            "max_tokens": cfg.max_tokens,
            "temperature": cfg.temperature,
            "behavior_prompts": cfg.behavior_prompts,
            "behavior_prompts_sha256": _file_sha(cfg.behavior_prompts),
            "neutral_prompts": cfg.neutral_prompts,
            "neutral_prompts_sha256": _file_sha(cfg.neutral_prompts),
            "system_prompt_iterations": system_record.get("iterations", []),
        },
        "arms": arms,
    }
    run_dir = os.path.join(cfg.vector_dir, f"scorecard-{timestamp.replace(':', '')}")
    os.makedirs(run_dir, exist_ok=True)
    out_path = os.path.join(run_dir, "scorecard.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(scorecard, f, indent=2)
        f.write("\n")
    scorecard["out_path"] = out_path
    return scorecard


def _file_sha(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        return text_sha256(f.read())
