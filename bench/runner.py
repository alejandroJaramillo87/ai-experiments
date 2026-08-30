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
from typing import Any, TextIO

from bench import adapters, metrics, report
from bench.registry import ModelSpec
from bench.workloads import DECODE, PREFILL, Cell, Profile, build_prompt, slices
from common.output import Printer
from common.serving import SERVICES, ServingError, ServingTransientError, wait_healthy

# A 256-token decode arm on CPU runs into minutes; a prefill probe takes seconds. One
# generous ceiling beats a per-service table that will drift out of date.
REQUEST_TIMEOUT_S = 600

# Whether a report covers the whole profile. A run that collapsed still writes what it
# measured; bench/delta.py refuses to difference anything but COMPLETE.
COMPLETE = "complete"
PARTIAL = "partial"


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
    *,
    run_key: str,
    cells: list[Cell] | None = None,
    sink: SampleSink | None = None,
    stream_fn=None,
) -> tuple[list[Observation], list[dict]]:
    """Run every cell in order, returning the kept observations and any errors.

    `run_key` identifies this pass and lands in every prompt nonce. It is required
    rather than defaulted because two passes that share it send byte-identical
    prompts, and vLLM's prefix cache -- on by default, with no per-request override --
    then answers the second pass from the first pass's fill. That is not a slower or
    faster number, it is a different measurement wearing the same label, and a caller
    that forgets to distinguish its arms must fail here rather than publish it.

    `cells` overrides the workload, so a caller interleaving arms can hand over one
    repeat at a time; the default is the profile's own full ordering.

    `sink` receives every kept observation as it is produced. A run reaches an hour on
    the publish profile, so the samples are written through rather than buffered to the
    end: the failure that would discard them is the one they would have explained.

    `stream_fn` is injected so the loop can be exercised without a service; production
    callers leave it None and get bench.client.stream.
    """
    from bench.client import stream as default_stream

    stream_fn = stream_fn or default_stream
    adapter = adapters.for_service(service)
    regions = slices(model.decode_prompts)

    observations: list[Observation] = []
    errors: list[dict] = []
    cells = profile.cells() if cells is None else cells
    printer.info(f"Workload: {len(cells)} cells, {profile.warmup} warm-up per point")

    # Occurrence within (point, repeat). Every measured cell is already unique on that
    # pair; a point's warm-up cells are not, because they all carry repeat -1. Counting
    # them separates the warm-ups without putting the position in the list back into the
    # nonce, which is the dependence that made a sliced list replay its prompts.
    occurrences: dict[tuple[str, int], int] = {}

    for index, cell in enumerate(cells):
        point = cell.point
        occurrence = occurrences.get((point.key, cell.repeat), 0)
        occurrences[(point.key, cell.repeat)] = occurrence + 1
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
            # cache rather than the model. The nonce is the cell's identity plus the
            # pass's -- point, repeat, warm-up occurrence, slot, run key -- so no two
            # requests anywhere in a run or in an A/B pair can collide.
            nonce = (
                f"{profile.seed}-{run_key}-{point.key}-"
                f"{cell.repeat}-{occurrence}-{slot}"
            )
            prompt = build_prompt(point.prompt_tokens, region, nonce)
            bodies.append(
                adapter.build_request(
                    prompt,
                    point.max_tokens,
                    # A sampling seed, not a cache key: two passes may reuse one
                    # safely now that their prompts differ.
                    seed=profile.seed + index * 1000 + slot,
                )
            )

        try:
            samples = _dispatch(stream_fn, adapter, bodies, cell)
        except ServingError as exc:
            printer.warn(f"Cell {label}: {exc}")
            errors.append({"cell": point.key, "repeat": cell.repeat, "error": str(exc)})
            continue
        except Exception as exc:  # noqa: BLE001 -- deliberately everything else
            # ServingError is what the stack raises for a service that is unreachable
            # or unhealthy, and it is survivable per cell. Anything else -- a client
            # bug, a protocol change, an OSError from the socket layer -- is not, and
            # used to leave the process with a traceback and the disk with nothing.
            # The loop stops, but what was already measured is returned and written.
            detail = f"{type(exc).__name__}: {exc}"
            printer.error(f"Cell {label}: {detail}")
            errors.append(
                {
                    "cell": point.key,
                    "repeat": cell.repeat,
                    "error": detail,
                    "fatal": True,
                }
            )
            break
        if cell.warmup:
            # Discarded on purpose: caches, clocks and any JIT are not settled yet.
            printer.info(f"Warm-up: {point.key} discarded")
            continue
        kept = [Observation(cell, sample) for sample in samples]
        observations.extend(kept)
        if sink is not None:
            for observation in kept:
                sink.write(observation)
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


def interleaved_passes(profile: Profile) -> list[tuple[int, list[Cell]]]:
    """The profile split into one (repeat, cells) pass per repeat.

    Interleaving two arms means alternating them at the repeat level, which requires
    handing the runner one repeat's worth of work at a time. Every pass is led by the
    profile's full warm-up set, not just the first: each pass follows a container
    recreate, so without it every repeat but the first would measure a cold service --
    a worse instrument than the drift interleaving exists to spread.
    """
    cells = profile.cells()
    warmups = [cell for cell in cells if cell.warmup]
    return [
        (repeat, warmups + [c for c in cells if c.repeat == repeat])
        for repeat in range(profile.repeats)
    ]


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
            fitted = [p for p in points if p["metrics"].get("ttft_ms")]
            xs, source = slope_axis(fitted)
            fit = metrics.prefill_slope(
                # strict: slope_axis returns one x per fitted point on both its
                # branches, so a length mismatch is a bug in it rather than a short
                # fit to be silently accepted.
                list(
                    zip(
                        xs,
                        (p["metrics"]["ttft_ms"]["median"] for p in fitted),
                        strict=True,
                    )
                )
            )
            if fit:
                fit["prompt_tokens_source"] = source
                body["slope"] = fit
        phases[phase] = body
    return phases


def slope_axis(points: list[dict]) -> tuple[list[float], str]:
    """The x-axis for the prefill fit, and which token count it came from.

    A point's `prompt_tokens` is the four-characters-per-token estimate that sliced the
    prompt; the server's own count is reduced beside it as `prompt_tokens_observed`. A
    slope fitted through the estimate is milliseconds per estimated token, so its
    reciprocal is a throughput in a unit no engine reports -- while
    `prefill_tok_s_server` in the same document is in real tokens and bench compare
    grades both as throughput. The rescale cancels in an A/B delta and does not cancel
    in the absolute figure, which is the one docs/optimizations/README.md publishes.

    The fallback is all-or-nothing on purpose: mixing an observed count for one point
    with an estimate for another puts two units on one axis, and the fit through them
    means neither.
    """
    observed = [
        (p.get("metrics", {}).get("prompt_tokens_observed") or {}).get("median")
        for p in points
    ]
    # Truthiness, not just presence: a zero observed count is as unusable an x-axis as
    # a missing one, and both fall back together.
    usable = [value for value in observed if value]
    if points and len(usable) == len(points):
        return [float(value) for value in usable], "observed"
    return [float(p["prompt_tokens"]) for p in points], "estimated"


def sample_row(obs: Observation) -> dict:
    """One raw sample as it is written: the cell's identity and the client's stamps."""
    return {
        "point": obs.cell.point.key,
        "repeat": obs.cell.repeat,
        "t_submit": obs.sample.t_submit,
        "arrivals": list(obs.sample.arrivals),
        "prompt_tokens": obs.sample.prompt_tokens,
        "completion_tokens": obs.sample.completion_tokens,
        "server_prompt_ms": obs.sample.server_prompt_ms,
        "server_predicted_ms": obs.sample.server_predicted_ms,
    }


class SampleSink:
    """samples.jsonl, appended and flushed as each cell completes.

    The raw samples are the irreplaceable half of a result: the reduction can be
    recomputed from them and they cannot be recomputed from it. Each line is flushed
    rather than left to the buffer, because the process that would flush it at exit is
    the one that is about to die.
    """

    def __init__(self, directory: str):
        os.makedirs(directory, exist_ok=True)
        self.path = os.path.join(directory, "samples.jsonl")
        self._handle: TextIO | None = None

    def __enter__(self) -> SampleSink:
        self._handle = open(self.path, "w", encoding="utf-8")
        return self

    def __exit__(self, *exc_info) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def write(self, obs: Observation) -> None:
        if self._handle is None:
            raise RuntimeError("SampleSink.write outside the context manager")
        self._handle.write(json.dumps(sample_row(obs)) + "\n")
        self._handle.flush()


def write_samples(observations: list[Observation], directory: str) -> str:
    """Every raw sample, one JSON object per line, beside the reduced report."""
    with SampleSink(directory) as sink:
        for obs in observations:
            sink.write(obs)
    return sink.path


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
    arm_sequence: str,
    status: str,
    host: dict,
    pair_id: str | None = None,
) -> dict:
    """Build the bench_result document for one arm.

    `arm_sequence` is how the arms of the pair were ordered against each other --
    BLOCKED for one arm's whole profile then the other's, INTERLEAVED for round-robin
    at the repeat level. It is a separate axis from `run.order`, which orders the
    points *within* one pass, and it is required rather than defaulted because the
    caller is the only party that knows which it did; a default would let a blocked
    pair be published under whichever label happened to be written first.

    `status` is COMPLETE or PARTIAL, and is required for the same reason. A report
    written after a run collapsed carries the cells that did run, which is worth
    keeping and dangerous to mistake for a whole one: differenced against a full arm it
    would report the missing cells as whatever the survivors happened to measure.

    `host` is bench/conditions.py's collected block, passed in rather than probed here.
    The caller has already collected it to reconcile the served model against the
    registry, and collecting a second time would describe a second container state --
    an arm's recreate sits between the two -- while presenting it as the state the
    numbers were measured under.
    """
    return report.build(
        run={
            "run_id": run_id,
            "profile": profile.name,
            # The digest of the resolved workload beside the name it goes by. A profile
            # file edited between two arms keeps its name and changes what it measures,
            # and bench/delta.py refuses a pair or a floor whose digests disagree.
            "workload_hash": profile.workload_hash,
            "arm": arm,
            "pair_id": pair_id,
            "order": profile.order,
            "arm_sequence": arm_sequence,
            "status": status,
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
