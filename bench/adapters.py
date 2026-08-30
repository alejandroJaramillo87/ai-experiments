"""Per-engine request construction and response decoding.

The three benchmarked services expose an OpenAI-compatible surface, and the suite drives
/v1/completions rather than /v1/chat/completions so that no chat template sits between
the workload and the token count. What differs between engines is narrow and specific:
how to force an exact generation length, how to keep a prompt cache from answering the
question for you, and whether the server reports its own timings.

That last one is the honesty boundary. llama.cpp returns a `timings` object carrying
server-measured prefill and decode milliseconds unconditionally. vLLM attaches an
equivalent `metrics` object to the same final usage frame, but only when the server was
launched with --enable-per-request-metrics, which docker-compose.inference.yaml does not
pass -- so the parser below is written and inert rather than absent, and lights up on the
day the flag is set. Where the timings are absent the report says so rather than
substituting a client-side number under a server-side name.

The rust harness is deliberately not here. Its optimization arms -- the SIMD panel
kernels, matvec threading, prefill batching -- are properties of the runtime's internals,
and `make ladder` already exposes them as first-class flags (--dot, --batch, --threads)
measured in-process against the real checkpoint. Benchmarking it over HTTP would add a
serialization layer and a mutex on top of the thing being measured, and its endpoint
refuses `stream: true` outright (schemas/harness_chat_endpoint.json), so it could not
report a prefill/decode split either. See bench/README.md.
"""

from __future__ import annotations

import dataclasses

from common.serving import SERVICES, ServiceSpec

# The two engines behind the three benchmarked services.
LLAMA_CPP = "llama.cpp"
VLLM = "vllm"

ENGINE_FOR_SERVICE = {
    "llama-cpu": LLAMA_CPP,
    "llama-gpu": LLAMA_CPP,
    "vllm-gpu": VLLM,
}


@dataclasses.dataclass(frozen=True)
class Usage:
    """Token counts as the server reported them."""

    prompt_tokens: int
    completion_tokens: int


@dataclasses.dataclass(frozen=True)
class ServerTimings:
    """Server-measured phase durations, when the engine reports them.

    `queue_ms` is the wait between a request arriving and being scheduled. Only vLLM
    measures it -- llama.cpp's timings object has no such field -- so None means the
    engine did not report it and never means zero wait.

    `prompt_ms` is the same phase on both engines but not quite the same envelope:
    llama.cpp reports prompt-eval compute alone, while vLLM measures scheduling to first
    token and its own source notes that a preemption during prefill falls inside that
    interval. Close enough to carry under one name, wide enough that a cross-engine
    prefill comparison should say so.
    """

    prompt_ms: float
    predicted_ms: float
    queue_ms: float | None = None


class Adapter:
    """One engine's request shape and response quirks."""

    engine = ""
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
        # No pinned engine serializes a token count as anything but an integer, but this
        # runs inside bench/client.py::stream and the runner handles only ServingError:
        # an unguarded conversion would end a run of up to an hour with a traceback,
        # while an absent usage object degrades to the runner's explained refusal.
        try:
            return Usage(int(prompt_tokens), int(completion_tokens))
        except (TypeError, ValueError):
            return None

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
    # reports_server_timings stays False, inherited: it tracks this stack's launch line
    # rather than the engine's capability, and the flag that populates the metrics
    # object is not in it. parse_timings below is what makes turning the timings on a
    # compose change alone.

    def extra_body(self, max_tokens: int) -> dict:
        # min_tokens pins the floor and ignore_eos removes the model's own stopping
        # opinion, so the generation length is exactly max_tokens.
        return {"ignore_eos": True, "min_tokens": max_tokens}

    # Prefix caching in vLLM is a launch flag, not a per-request field, so there is
    # nothing to send. The runner defeats it by varying the prompt prefix per repeat,
    # and conditions records enable_prefix_caching as the server reported it.

    def parse_timings(self, body: dict) -> ServerTimings | None:
        """vLLM's PerRequestTimingMetrics, when the server was launched to emit them.

        Attached to the same final usage frame parse_usage already reads, under
        --enable-per-request-metrics (vLLM 0.25.0). The flag is not passed by this
        stack, so against today's wire shape the `metrics` key is absent and this
        returns None like the base class.
        """
        metrics = body.get("metrics")
        if not isinstance(metrics, dict):
            return None
        ttft_ms = metrics.get("time_to_first_token_ms")
        generation_ms = metrics.get("generation_time_ms")
        queue_ms = metrics.get("queue_time_ms")
        if ttft_ms is None or generation_ms is None:
            return None
        try:
            # Carried raw. vLLM computes this field as first_token_ts - scheduled_ts
            # (vllm/entrypoints/generate/base/serving.py at 0.25.0), so it is stamped
            # from scheduling and is already queue-free -- the engine's own stats module
            # calls the identical expression prefill_time. Subtracting the queue would
            # deduct a wait this number never contained and understate prefill.
            #
            # Queue time is informational beside it, not a term: non-zero means the
            # request contended for the scheduler. It is optional because prefill does
            # not depend on it.
            return ServerTimings(
                float(ttft_ms),
                float(generation_ms),
                None if queue_ms is None else float(queue_ms),
            )
        except (TypeError, ValueError):
            return None


_ADAPTERS = {
    LLAMA_CPP: LlamaCppAdapter,
    VLLM: VllmAdapter,
}


def for_service(name: str) -> Adapter:
    """The adapter for a service, by the name common/serving.py knows it by.

    SERVICES is the whole serving stack and is wider than what this suite benchmarks, so
    membership there is necessary and not sufficient: `harness` is a real service with no
    adapter here, and asking for it names the reason rather than raising a bare KeyError.
    """
    if name not in SERVICES:
        raise KeyError(f"unknown service {name!r}; known: {sorted(SERVICES)}")
    if name not in ENGINE_FOR_SERVICE:
        raise KeyError(
            f"{name!r} is not a benchmark target; benchable: "
            f"{sorted(ENGINE_FOR_SERVICE)}. The rust harness is measured by "
            "`make ladder`, which drives its runtime directly rather than over HTTP"
        )
    return _ADAPTERS[ENGINE_FOR_SERVICE[name]](SERVICES[name])
