"""LLM-judge scoring through the local inference services (interp M3).

score() batches judge calls against a local OpenAI-compatible endpoint: vllm-gpu (8003) by
default, llama-gpu (8002) as fallback. It preflights the concurrency policy: refuses to run
when the target service healthcheck fails, and warns when a hooked model is still resident on
the GPU (serving-dependent work runs with interp GPU work idle, per AGENTS.md).

Structured output: the request asks for schema-constrained JSON via response_format where the
service supports it; on rejection it falls back to prompt-only JSON with parse-and-retry.
"""

from __future__ import annotations

import json
import re
import sys

import requests

# Real ports from docker-compose.yaml (the source of truth). Health paths per the Makefile
# status target: vllm serves /health at the root, llama.cpp serves /v1/health.
DEFAULT_ENDPOINTS = (
    ("http://localhost:8003/v1", "http://localhost:8003/health"),
    ("http://localhost:8002/v1", "http://localhost:8002/v1/health"),
)

SCORE_SCHEMA = {
    "type": "object",
    "properties": {
        "score": {"type": "number"},
        "reason": {"type": "string"},
    },
    "required": ["score"],
}


def _healthy(health_url: str, timeout: float = 5.0) -> bool:
    try:
        return requests.get(health_url, timeout=timeout).status_code == 200
    except requests.RequestException:
        return False


def pick_endpoint(endpoint: str | None = None) -> str:
    """Return a healthy OpenAI-compatible base URL or raise RuntimeError."""
    if endpoint is not None:
        base = endpoint.rstrip("/")
        candidates = [
            (base, base + "/health"),
            (base, base.removesuffix("/v1") + "/health"),
        ]
        if any(_healthy(url) for _, url in candidates):
            return base
        raise RuntimeError(
            f"judge endpoint {endpoint} failed its healthcheck; start the service first"
        )
    for base, health_url in DEFAULT_ENDPOINTS:
        if _healthy(health_url):
            return base
    raise RuntimeError(
        "no judge service is up (tried vllm-gpu 8003, llama-gpu 8002); "
        "start one with 'make up' or 'docker compose up -d vllm-gpu'"
    )


def _warn_if_gpu_resident() -> None:
    try:
        import torch

        if torch.cuda.is_available() and torch.cuda.memory_allocated() > 0:
            print(
                "[WARN] A hooked model is resident on the GPU while judging; "
                "serving-dependent work should run with interp GPU work idle",
                file=sys.stderr,
            )
    except Exception:  # noqa: BLE001 - torch absent or CUDA broken; nothing to warn about
        pass


def _parse_score(text: str) -> dict:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"no JSON object in judge reply: {text[:200]!r}")
    parsed = json.loads(match.group(0))
    if "score" not in parsed:
        raise ValueError(f"judge reply lacks 'score': {parsed!r}")
    return parsed


def _chat(base: str, messages: list[dict], structured: bool, timeout: float) -> str:
    body: dict = {"model": "default", "messages": messages, "temperature": 0.0}
    if structured:
        body["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "judge_score", "schema": SCORE_SCHEMA},
        }
    resp = requests.post(f"{base}/chat/completions", json=body, timeout=timeout)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def score(
    items: list[str],
    rubric: str,
    endpoint: str | None = None,
    retries: int = 2,
    timeout: float = 120.0,
) -> list[dict]:
    """Score each item against the rubric; returns [{'score': float, 'reason': str}, ...].

    Items that still fail parsing after retries come back as {'score': None, 'error': ...}
    rather than aborting the batch: a judge run over hundreds of generations must survive a
    few malformed replies.
    """
    base = pick_endpoint(endpoint)
    _warn_if_gpu_resident()

    system = (
        f"You are a strict evaluator. Rubric: {rubric}\n"
        'Reply with a JSON object: {"score": <number>, "reason": "<short reason>"}.'
    )
    structured = True
    results: list[dict] = []
    for item in items:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": item},
        ]
        result: dict | None = None
        last_error = "unknown"
        for _ in range(retries + 1):
            try:
                reply = _chat(base, messages, structured, timeout)
                result = _parse_score(reply)
                break
            except requests.HTTPError as exc:
                # Service rejected response_format: drop to prompt-only JSON and retry
                if (
                    structured
                    and exc.response is not None
                    and exc.response.status_code == 400
                ):
                    structured = False
                    continue
                last_error = str(exc)
            except (ValueError, requests.RequestException, KeyError) as exc:
                last_error = str(exc)
        results.append(
            result if result is not None else {"score": None, "error": last_error}
        )
    return results
