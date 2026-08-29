"""Per-engine request construction and response decoding.

All four services expose an OpenAI-compatible surface, and the suite drives
/v1/completions rather than /v1/chat/completions so that no chat template sits between
the workload and the token count. What differs between engines is narrow and specific:
how to force an exact generation length, how to keep a prompt cache from answering the
question for you, and whether the server reports its own timings.

That last one is the honesty boundary. llama.cpp returns a `timings` object carrying
server-measured prefill and decode milliseconds; vLLM and the harness do not. Where it
is absent the report says so rather than substituting a client-side number under a
server-side name.
"""

from __future__ import annotations

import dataclasses

from common.serving import SERVICES, ServiceSpec

# The three engines behind the four services.
LLAMA_CPP = "llama.cpp"
VLLM = "vllm"
HARNESS = "harness"

ENGINE_FOR_SERVICE = {
    "llama-cpu": LLAMA_CPP,
    "llama-gpu": LLAMA_CPP,
    "vllm-gpu": VLLM,
    "harness": HARNESS,
}


@dataclasses.dataclass(frozen=True)
class Usage:
    """Token counts as the server reported them."""

    prompt_tokens: int
    completion_tokens: int


@dataclasses.dataclass(frozen=True)
class ServerTimings:
    """Server-measured phase durations, when the engine reports them."""

    prompt_ms: float
    predicted_ms: float


class Adapter:
    """One engine's request shape and response quirks."""

    engine = ""
    # Whether the engine can be asked to ignore end-of-sequence, which is what makes a
    # generation length exact rather than whatever the model chose to say. Without it a
    # decode arm measures a variable amount of work and calls it a rate -- the flaw in
    # the archived cpu_benchmark_results_*.json, whose completion_tokens moved run to
    # run.
    forces_exact_length = True
    reports_server_timings = False

    def __init__(self, service: ServiceSpec) -> None:
        self.service = service

    @property
    def url(self) -> str:
        return self.service.base_url + "/v1/completions"

    def build_request(self, prompt: str, max_tokens: int, seed: int) -> dict:
        body = {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "seed": seed,
            "stream": True,
            # Without this vLLM omits `usage` from a streamed response entirely, and the
            # only remaining way to count tokens is to count SSE frames -- which
            # undercounts any server that packs more than one token into a chunk.
            "stream_options": {"include_usage": True},
        }
        body.update(self.extra_body(max_tokens))
        return body

    def extra_body(self, max_tokens: int) -> dict:
        return {}

    def parse_usage(self, body: dict) -> Usage | None:
        usage = body.get("usage")
        if not usage:
            return None
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")
        if prompt_tokens is None or completion_tokens is None:
            return None
        return Usage(int(prompt_tokens), int(completion_tokens))

    def parse_timings(self, body: dict) -> ServerTimings | None:
        return None


class LlamaCppAdapter(Adapter):
    engine = LLAMA_CPP
    reports_server_timings = True

    def extra_body(self, max_tokens: int) -> dict:
        return {
            "ignore_eos": True,
            # llama.cpp will happily answer from its prompt cache and report a prefill
            # that never happened. The runner also varies the prompt prefix per repeat,
            # because this flag is honoured on the native endpoint and the OpenAI shim
            # has not always forwarded it.
            "cache_prompt": False,
        }

    def parse_timings(self, body: dict) -> ServerTimings | None:
        timings = body.get("timings")
        if not timings:
            return None
        prompt_ms = timings.get("prompt_ms")
        predicted_ms = timings.get("predicted_ms")
        if prompt_ms is None or predicted_ms is None:
            return None
        return ServerTimings(float(prompt_ms), float(predicted_ms))


class VllmAdapter(Adapter):
    engine = VLLM

    def extra_body(self, max_tokens: int) -> dict:
        # min_tokens pins the floor and ignore_eos removes the model's own stopping
        # opinion, so the generation length is exactly max_tokens.
        return {"ignore_eos": True, "min_tokens": max_tokens}

    # Prefix caching in vLLM is a launch flag, not a per-request field, so there is
    # nothing to send. The runner defeats it by varying the prompt prefix per repeat,
    # and conditions records enable_prefix_caching as the server reported it.


class HarnessAdapter(Adapter):
    engine = HARNESS
    # The harness runtime exposes no ignore_eos, so a response can stop early. The
    # runner asserts completion_tokens == max_tokens and discards a short sample rather
    # than averaging a truncated generation into a decode rate.
    forces_exact_length = False


_ADAPTERS = {
    LLAMA_CPP: LlamaCppAdapter,
    VLLM: VllmAdapter,
    HARNESS: HarnessAdapter,
}


def for_service(name: str) -> Adapter:
    """The adapter for a service, by the name common/serving.py knows it by."""
    if name not in SERVICES:
        raise KeyError(f"unknown service {name!r}; known: {sorted(SERVICES)}")
    engine = ENGINE_FOR_SERVICE[name]
    return _ADAPTERS[engine](SERVICES[name])
