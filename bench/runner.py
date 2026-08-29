"""Executing a workload against one service, and assembling its report.

The protocol this implements is .claude/skills/benchmarking/SKILL.md's, with the
additions docs/optimizations/ideation/rust-harness-optimization-candidates.md prescribes
after measuring a 44% spread across four consecutive runs of one unmodified binary:
warm-up discarded, repeats round-robin over points rather than blocked, and the spread
reported alongside the median. Until now every one of those rules was enforced by the
operator remembering to follow it.

The health gate is not a formality. `.claude/commands/bench.md`: "a benchmark against an
unhealthy service is not a slow result, it is no result."
"""

from __future__ import annotations

import concurrent.futures
import dataclasses
import json
import os
from typing import Any

from bench import adapters, conditions, metrics, report
from bench.registry import ModelSpec
from bench.workloads import DECODE, PREFILL, Cell, Profile, build_prompt, slices
from common.output import Printer
from common.serving import SERVICES, ServingError, ServingTransientError, wait_healthy

# A 256-token decode arm on CPU runs into minutes; a prefill probe takes seconds. One
# generous ceiling beats a per-service table that will drift out of date.
REQUEST_TIMEOUT_S = 600


@dataclasses.dataclass
class Observation:
    cell: Cell
    sample: metrics.Sample


def _sample_from_stream(result, cell: Cell, adapter) -> metrics.Sample:
    usage = result.usage
    if usage is None:
        raise ServingError(
            f"{adapter.service.name} returned no usage object, so the token count "
            "would have to be inferred from the number of SSE frames. That undercounts "
            "any server packing more than one token per chunk, so the sample is "
            "refused rather than estimated"
        )
    # Both engines take ignore_eos, so a short generation means the flag did not take
    # effect. Averaging a truncated response into a decode rate would understate the work
    # per request without showing why, so the sample is discarded rather than counted.
    if usage.completion_tokens != cell.point.max_tokens:
        raise ServingTransientError(
            f"{adapter.service.name} generated {usage.completion_tokens} tokens, not "
            f"{cell.point.max_tokens}; the generation length was not honoured, so the "
            "sample is discarded"
        )
    return metrics.Sample(
        t_submit=result.t_submit,
        arrivals=tuple(result.arrivals),
        prompt_tokens=usage.prompt_tokens,
        completion_tokens=usage.completion_tokens,
        server_prompt_ms=result.timings.prompt_ms if result.timings else None,
        server_predicted_ms=result.timings.predicted_ms if result.timings else None,
    )


def execute(
    service: str,
    profile: Profile,
    model: ModelSpec,
    printer: Printer,
    stream_fn=None,
) -> tuple[list[Observation], list[dict]]:
    """Run every cell in order, returning the kept observations and any errors.

    `stream_fn` is injected so the loop can be exercised without a service; production
    callers leave it None and get bench.client.stream.
    """
    from bench.client import stream as default_stream

    stream_fn = stream_fn or default_stream
    adapter = adapters.for_service(service)
    regions = slices(model.decode_prompts)

    observations: list[Observation] = []
    errors: list[dict] = []
    cells = profile.cells()
    printer.info(f"Workload: {len(cells)} cells, {profile.warmup} warm-up per point")

    for index, cell in enumerate(cells):
        point = cell.point
        # Rotate regions across repeats so an MoE's decode arm sees disjoint prompts and
        # the spread between them becomes the reported routing sensitivity.
        region = regions[max(cell.repeat, 0) % len(regions)]
        label = (
            f"{point.key} {'warmup' if cell.warmup else f'repeat {cell.repeat + 1}'}"
        )
        # One cell at concurrency N is N requests in flight together, not N requests one
        # after another. Running them serially and labelling the point "concurrency 16"
        # would report a queue-free latency under a queued label and a single stream's
        # throughput as a batched one -- wrong in the direction that flatters the
        # result. Continuous batching is the reason vLLM is in this stack, so the only
        # honest way to measure it is to actually contend for the scheduler.
        bodies = []
        for slot in range(point.concurrency):
            # Each in-flight request needs its own prompt: identical prompts would let a
            # prefix cache serve all but the first, and the batch would measure the
            # cache rather than the model.
            nonce = f"{profile.seed}-{index}-{slot}"
            prompt = build_prompt(point.prompt_tokens, region, nonce)
            bodies.append(
                adapter.build_request(
                    prompt,
                    point.max_tokens,
                    seed=profile.seed + index * 1000 + slot,
                )
            )

        try:
            samples = _dispatch(stream_fn, adapter, bodies, cell)
        except ServingError as exc:
            printer.warn(f"Cell {label}: {exc}")
            errors.append({"cell": point.key, "repeat": cell.repeat, "error": str(exc)})
            continue
        if cell.warmup:
            # Discarded on purpose: caches, clocks and any JIT are not settled yet.
            printer.info(f"Warm-up: {point.key} discarded")
            continue
        observations.extend(Observation(cell, sample) for sample in samples)
    return observations, errors


def _dispatch(
    stream_fn, adapter, bodies: list[dict], cell: Cell
) -> list[metrics.Sample]:
    """Send one cell's requests, together, and return their samples.

    A single request runs on this thread; anything more goes through a pool sized to the
    batch so every request is genuinely in flight at once. The pool is created per cell
    rather than once per run so that a cell cannot inherit a queue from the previous one
    -- the point of the sweep is the server's scheduler, not the client's.
    """
    if len(bodies) == 1:
        return [
            _sample_from_stream(
                stream_fn(adapter, bodies[0], REQUEST_TIMEOUT_S), cell, adapter
            )
        ]

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(bodies)) as pool:
        futures = [
            pool.submit(stream_fn, adapter, body, REQUEST_TIMEOUT_S) for body in bodies
        ]
        # Collect every result before raising: a partial batch left running would still
        # be loading the server while the next cell started measuring it.
        results, failure = [], None
        for future in futures:
            try:
                results.append(future.result())
            except ServingError as exc:
                failure = failure or exc
    if failure is not None:
        raise failure
    return [_sample_from_stream(result, cell, adapter) for result in results]


def reduce(observations: list[Observation], profile: Profile) -> dict:
    """Group observations by point and reduce each into its reported block."""
    by_point: dict[str, list[metrics.Sample]] = {}
    identity: dict[str, Cell] = {}
    for obs in observations:
        by_point.setdefault(obs.cell.point.key, []).append(obs.sample)
        identity.setdefault(obs.cell.point.key, obs.cell)

    phases: dict[str, dict] = {}
    for phase in (PREFILL, DECODE):
        points: list[dict[str, Any]] = []
        for key, samples in by_point.items():
            cell = identity[key]
            if cell.point.phase != phase:
                continue
            points.append(
                {
                    "point": key,
                    "prompt_tokens": cell.point.prompt_tokens,
                    "max_tokens": cell.point.max_tokens,
                    "concurrency": cell.point.concurrency,
                    "metrics": metrics.reduce_samples(samples, cell.point.concurrency),
                }
            )
        if not points:
            continue
        points.sort(key=lambda p: (p["prompt_tokens"], p["concurrency"]))
        # Server timings are the authoritative prefill figure where an engine reports
        # them. Where it does not, the slope below is the only honest one, and the
        # source is named so the two are never confused in a comparison.
        has_server = any(
            p["metrics"].get("prefill_tok_s_server") is not None for p in points
        )
        body: dict = {
            "timings_source": "server" if has_server else "client",
            "points": points,
        }
        if phase == PREFILL:
            fit = metrics.prefill_slope(
                [
                    (p["prompt_tokens"], p["metrics"]["ttft_ms"]["median"])
                    for p in points
                    if p["metrics"].get("ttft_ms")
                ]
            )
            if fit:
                body["slope"] = fit
        phases[phase] = body
    return phases


def write_samples(observations: list[Observation], directory: str) -> str:
    """Every raw sample, one JSON object per line, beside the reduced report."""
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, "samples.jsonl")
    with open(path, "w", encoding="utf-8") as handle:
        for obs in observations:
            handle.write(
                json.dumps(
                    {
                        "point": obs.cell.point.key,
                        "repeat": obs.cell.repeat,
                        "t_submit": obs.sample.t_submit,
                        "arrivals": list(obs.sample.arrivals),
                        "prompt_tokens": obs.sample.prompt_tokens,
                        "completion_tokens": obs.sample.completion_tokens,
                        "server_prompt_ms": obs.sample.server_prompt_ms,
                        "server_predicted_ms": obs.sample.server_predicted_ms,
                    }
                )
                + "\n"
            )
    return path


def gate(service: str, printer: Printer) -> float:
    """Refuse to benchmark a service that is not healthy."""
    spec = SERVICES[service]
    printer.run(f"Health: polling {service} at {spec.health_url}")
    elapsed = wait_healthy(spec)
    printer.ok(f"Health: {service} healthy after {elapsed:.1f}s")
    return elapsed


def assemble(
    *,
    service: str,
    profile: Profile,
    model: ModelSpec,
    arm: str,
    run_id: str,
    phases: dict,
    errors: list[dict],
    samples_path: str | None,
    pair_id: str | None = None,
) -> dict:
    """Build the bench_result document for one arm."""
    host = conditions.collect(service)
    return report.build(
        run={
            "run_id": run_id,
            "profile": profile.name,
            "arm": arm,
            "pair_id": pair_id,
            "order": profile.order,
            "repeats": profile.repeats,
            "warmup_discarded": profile.warmup,
            "seed": profile.seed,
        },
        conditions={
            "service": service,
            "engine": adapters.ENGINE_FOR_SERVICE[service],
            "port": SERVICES[service].port,
            "model": model.conditions(),
            "prompt_source": "natural",
            **host,
        },
        phases=phases,
        errors=errors,
        samples_path=samples_path,
    )
