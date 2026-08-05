"""Serving-stack client and registry shared by host-side tools.

The per-service registry (ports, model env vars, health paths), compose
recreate, health poll, and the minimal OpenAI-compatible chat client, shared
by the training eval/teacher harnesses and the interp steering scorecard.
Stdlib-only; runs on the HOST (it drives docker compose and talks to
localhost).

Manual restore one-liner if a swapping tool ever dies uncleanly:

    docker compose up -d --force-recreate <service>

(with the service's model env var unset in the shell, compose falls back to
.env or the image default.)
"""

from __future__ import annotations

import dataclasses
import json
import os
import subprocess
import time
import urllib.error
import urllib.request

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_COMPOSE = os.path.join(REPO_ROOT, "docker-compose.training.yaml")
CONTAINER_MODELS = "/app/models/"
HOST_MODELS = "/mnt/ai-data/models/"


class ServingError(RuntimeError):
    """Non-transient serving failure; message says how to proceed."""


class ServingTransientError(ServingError):
    """A request failure that a retry may resolve (timeout, 5xx, reset)."""


@dataclasses.dataclass(frozen=True)
class ServiceSpec:
    """One inference service as compose defines it."""

    name: str
    port: int
    model_env: str  # compose env var selecting the served model
    health_path: str  # llama.cpp and vLLM disagree; see docker-compose.yaml
    concurrency: int  # request concurrency the service tolerates
    start_budget_s: int  # default health-poll budget after a recreate

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    @property
    def health_url(self) -> str:
        return self.base_url + self.health_path

    @property
    def chat_url(self) -> str:
        return self.base_url + "/v1/chat/completions"


# vLLM takes high request concurrency (continuous batching); llama.cpp serves
# one request at a time. vLLM cold starts are weight loads, hence the larger
# health budget.
SERVICES = {
    # llama-cpu hosts the F7 judge: it loads once and stays up while
    # llama-gpu swaps between the compared models, and it is not a GPU
    # consumer, so the one-GPU-consumer policy is unaffected. The larger
    # budget covers cold-loading a 26B-class MoE judge from disk.
    "llama-cpu": ServiceSpec(
        "llama-cpu",
        8001,
        "LLAMA_CPU_MODEL",
        "/v1/health",
        concurrency=1,
        start_budget_s=300,
    ),
    "llama-gpu": ServiceSpec(
        "llama-gpu",
        8002,
        "LLAMA_GPU_MODEL",
        "/v1/health",
        concurrency=1,
        start_budget_s=180,
    ),
    "vllm-gpu": ServiceSpec(
        "vllm-gpu",
        8003,
        "VLLM_GPU_MODEL",
        "/health",
        concurrency=16,
        start_budget_s=600,
    ),
}


# Storage types that are not quantizations. An export now ships BF16, and a
# suffix the quant parser does not recognize reads as "unknown", which
# check_quant_match refuses outright — so an unquantized artifact has to name
# itself here or no comparative evaluation can run against it.
UNQUANTIZED_SUFFIXES = frozenset({"BF16", "F16", "F32"})


def quant_from_name(path: str) -> str | None:
    """Storage type from a <name>-<TYPE>.gguf filename, when parseable."""
    stem = os.path.basename(path)
    if stem.endswith(".gguf"):
        tail = stem[:-5].rsplit("-", 1)
        if len(tail) == 2 and (
            tail[1][:1].upper() == "Q" or tail[1].upper() in UNQUANTIZED_SUFFIXES
        ):
            # Normalize case: Qwen official GGUFs use lowercase quant
            # suffixes, and the confound checks compare case-sensitively.
            return tail[1].upper()
    return None


def model_paths(model: str) -> tuple[str, str]:
    """(container_path, host_path) for a model given in either form."""
    if model.startswith(CONTAINER_MODELS):
        return model, HOST_MODELS + model[len(CONTAINER_MODELS) :]
    if model.startswith(HOST_MODELS):
        return CONTAINER_MODELS + model[len(HOST_MODELS) :], model
    raise ServingError(
        f"Model must live under {CONTAINER_MODELS} (container view) or "
        f"{HOST_MODELS} (host view): {model}"
    )


def compose(env: dict, *args: str) -> None:
    cmd = ["docker", "compose", *args]
    proc = subprocess.run(cmd, cwd=REPO_ROOT, env=env)
    if proc.returncode != 0:
        raise ServingError(f"Command failed ({proc.returncode}): {' '.join(cmd)}")


def swap_model(
    service: ServiceSpec, container_path: str | None, env: dict | None = None
) -> None:
    """Recreate the service with its model env var overridden (or cleared).

    container_path=None restores the previous serving state: with the var
    absent, compose falls back to .env / the entrypoint default.
    """
    env = dict(env if env is not None else os.environ)
    if container_path is None:
        env.pop(service.model_env, None)
    else:
        env[service.model_env] = container_path
    compose(env, "up", "-d", "--force-recreate", service.name)


def wait_healthy(service: ServiceSpec, budget_s: int | None = None) -> float:
    """Poll the service health endpoint; returns seconds to healthy."""
    budget_s = budget_s if budget_s is not None else service.start_budget_s
    start = time.monotonic()
    deadline = start + budget_s
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(service.health_url, timeout=5) as resp:
                if resp.status == 200:
                    return time.monotonic() - start
        except (urllib.error.URLError, OSError, TimeoutError):
            pass
        time.sleep(3)
    raise ServingError(
        f"{service.name} did not become healthy within {budget_s}s\n"
        f"  Solution: check `docker compose logs {service.name}` for load errors"
    )


def post_chat(
    chat_url: str, payload: dict, timeout: float = 120, origin: str | None = None
) -> dict:
    """POST an OpenAI-compatible chat payload; returns the parsed JSON body.

    Raises ServingTransientError for failures a retry may resolve
    (connection errors, timeouts, 5xx) and ServingError for request errors a
    retry cannot fix (4xx, bad body). origin names the service in messages;
    it defaults to the URL.
    """
    origin = origin or chat_url
    request = urllib.request.Request(
        chat_url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as resp:
            return json.load(resp)
    except urllib.error.HTTPError as exc:
        detail = f"HTTP {exc.code} from {origin}: {exc.reason}"
        if exc.code >= 500:
            raise ServingTransientError(detail) from exc
        raise ServingError(detail) from exc
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        raise ServingTransientError(f"Request to {origin} failed: {exc}") from exc
    except ValueError as exc:
        raise ServingError(f"Non-JSON response from {origin}: {exc}") from exc


def chat_completion(
    service: ServiceSpec,
    messages: list[dict],
    sampling: dict | None = None,
    timeout: int = 120,
) -> dict:
    """POST one /v1/chat/completions request.

    Returns {"content", "usage", "latency_ms"}. Raises ServingTransientError
    for failures a retry may resolve (connection errors, timeouts, 5xx) and
    ServingError for request errors a retry cannot fix (4xx, bad body).
    """
    payload = {"model": "default", "messages": messages, **(sampling or {})}
    start = time.monotonic()
    body = post_chat(service.chat_url, payload, timeout=timeout, origin=service.name)
    try:
        content = body["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ServingError(
            f"Malformed chat completion from {service.name}: {exc}"
        ) from exc
    return {
        "content": content,
        "usage": body.get("usage"),
        "latency_ms": round((time.monotonic() - start) * 1000, 1),
    }
