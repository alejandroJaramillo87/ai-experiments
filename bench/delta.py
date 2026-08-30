"""Comparing two arms against a measured noise floor.

The benchmarking skill defines a regression as a median more than 5% slower than the
baseline, "outside the run-to-run spread". The second clause is the one that has never
had an implementation, and without it the first is unusable: this workstation has been
observed spreading 44% across four consecutive runs of one unmodified binary
(docs/optimizations/ideation/rust-harness-optimization-candidates.md). A 5% threshold
underneath a 40% noise floor is not a threshold.

So a comparison here answers against two numbers, not one, and has four outcomes rather
than two. The pair that matters is INCONCLUSIVE and WITHIN_NOISE: a delta that clears
5% but sits inside the floor is a real finding -- the workload cannot resolve the
change -- and reporting it as an improvement is how an unfounded claim gets published.
.claude/agents/perf-auditor.md already states this rule in prose; this module executes
it.
"""

from __future__ import annotations

import dataclasses
from typing import Any

# The benchmarking skill's threshold. A change must clear this AND the measured floor.
REGRESSION_PCT = 5.0

REGRESSION = "regression"
IMPROVEMENT = "improvement"
# The rig cannot tell this delta from having changed nothing.
WITHIN_NOISE = "within_noise"
# Resolvable above the floor, but under the 5% bar the skill sets for reporting.
BELOW_THRESHOLD = "below_threshold"
# Large enough to matter, but no floor was measured, so whether the rig can resolve it
# is unknown. The verdict exists to push a caller at `bench noise` rather than let an
# unverified 8% be quoted as an 8% win.
INCONCLUSIVE = "inconclusive"

# Conditions under which two arms are not comparable at all. Quantization and context
# length are named by .claude/commands/bench.md ("Comparative claims must compare at
# matched quantization and matched context length"); architecture and prompt source are
# here because an MoE benchmarked on random tokens and one benchmarked on prose are
# different experiments, not two arms of one.
BLOCKING = (
    ("conditions", "service"),
    ("conditions", "engine"),
    ("conditions", "model", "name"),
    ("conditions", "model", "quantization"),
    ("conditions", "model", "architecture"),
    ("conditions", "model", "context_length"),
    ("conditions", "prompt_source"),
    ("conditions", "host", "client_cpuset"),
    ("conditions", "host", "service_cpuset"),
    ("run", "profile"),
    # The digest of the resolved workload, not the name the file calls itself by. A
    # profile edited between the two arms keeps its name and changes what it measures,
    # and the name alone cannot tell the difference. Absent on both sides -- a report
    # written before the digest existed -- compares equal and is not refused.
    ("run", "workload_hash"),
)

# A run that stopped early. bench/commands/run.py writes what it measured under this
# marker; differenced against a whole arm, the cells it never ran are reported as
# whatever its survivors happened to measure.
COMPLETE = "complete"

# How the arms of a pair were ordered against each other. Distinct from run.order,
# which orders points within one pass -- the warning below used to read that instead,
# and could never be true for a profile the runner would accept.
INTERLEAVED_ARMS = "interleaved"

# What a report records about the container it measured. Two arm names over one of
# these is an A/B that measured nothing.
SERVICE_STATE = (("conditions", "env_knobs"), ("conditions", "server_argv"))

# Metrics where a larger number is the better one; everything else is a latency.
HIGHER_IS_BETTER = frozenset(
    {"decode_tok_s", "e2e_tok_s", "prefill_tok_s_server", "prefill_tok_s_slope"}
)

# Statistics blocks that record what the server tokenized rather than how fast it was.
# They share the shape of a latency and none of its meaning, so differencing them
# grades a workload difference as a slower engine and lets a tokenization artifact set
# the published noise floor. Excluded here, once, rather than at each consumer.
DIAGNOSTIC = frozenset({"prompt_tokens_observed", "completion_tokens_observed"})

# Where a noise floor records the workload it was measured against, and the field each
# one must equal in the arms. A floor is only the run-to-run spread of the workload it
# was measured on.
FLOOR_PROVENANCE = (
    (("run", "profile"), ("run", "profile")),
    (("run", "service"), ("conditions", "service")),
    # The digest, not just the name: a profile file edited between the floor and the
    # arms keeps its name and changes the workload underneath it.
    (("run", "workload_hash"), ("run", "workload_hash")),
)


class IncomparableError(ValueError):
    """Two results describe different experiments and must not be differenced."""


class MismatchedFloorError(IncomparableError):
    """The noise floor was not measured against the workload it is being applied to.

    A subclass because the caller's response is the same -- refuse, exit 2 -- and
    because a floor is part of the verdict arithmetic rather than a display choice.
    Warning instead would leave the exit code at 0 on a regression the floor absorbed,
    which is the outcome this refusal exists to prevent.
    """


@dataclasses.dataclass(frozen=True)
class MetricDelta:
    point: str
    metric: str
    baseline: float
    candidate: float
    delta_pct: float
    noise_pct: float | None
    verdict: str
    higher_is_better: bool


def _dig(report: dict, path: tuple[str, ...]):
    node: Any = report
    for key in path:
        if not isinstance(node, dict):
            return None
        node = node.get(key)
    return node


def check_comparable(
    baseline: dict, candidate: dict, *, require_distinct: bool = True
) -> list[str]:
    """Refuse on a mismatch that makes a delta meaningless; warn on git_sha.

    Returns the non-blocking warnings. Raises IncomparableError on the rest, naming
    every mismatch at once rather than the first, so a caller fixes one invocation
    instead of discovering the differences one run at a time.

    `require_distinct` is what separates an A/B from a noise floor. Two arm names over
    one service configuration is a broken A/B and is refused -- but it is the whole
    definition of the floor, whose two arms are the same configuration run twice, so
    floor_from_pair turns the check off.
    """
    mismatches = []
    for path in BLOCKING:
        left, right = _dig(baseline, path), _dig(candidate, path)
        if left != right:
            mismatches.append(f"{'.'.join(path)}: {left!r} vs {right!r}")
    if mismatches:
        raise IncomparableError(
            "these results are not two arms of one experiment:\n  "
            + "\n  ".join(mismatches)
        )
    _check_pairing(baseline, candidate)
    _check_status(baseline, candidate)
    if require_distinct:
        _check_distinct(baseline, candidate)

    warnings = []
    if baseline.get("git_sha") != candidate.get("git_sha"):
        warnings.append(
            f"git_sha differs ({baseline.get('git_sha')} vs {candidate.get('git_sha')})"
            ": the arms were built from different trees, so the delta may include "
            "changes beyond the arm"
        )
    sequence = _dig(baseline, ("run", "arm_sequence"))
    if sequence != INTERLEAVED_ARMS:
        stated = "unrecorded" if sequence is None else repr(sequence)
        warnings.append(
            f"the arms were not run interleaved (arm_sequence {stated}); drift over "
            "the run -- thermal, a background process, a clock that settles -- lands "
            "entirely on the second arm and reads as its effect. `bench run "
            "--interleave` spreads it across both"
        )
    left = _dig(baseline, ("run", "pair_id"))
    right = _dig(candidate, ("run", "pair_id"))
    if (left is None) != (right is None):
        # Exactly one. Two standalone runs are an ordinary cross-run comparison and say
        # nothing; one half of a pair against a standalone run means the sibling this
        # arm was written beside is not the file being compared against.
        warnings.append(
            "one of these arms was written as half of a pair and the other was not, "
            "so this comparison is not the A/B either file records: the paired arm's "
            "sibling is somewhere else"
        )
    return warnings


def _check_pairing(baseline: dict, candidate: dict) -> None:
    """Refuse two halves of two different pairs.

    An arm written as half of a pair whose sibling never completed stays on disk
    looking exactly like a finished one. Compared against another pair's arm of the
    same name it graded every metric, warned about nothing and exited 0. Two arms that
    each belong to a pair must belong to the same one; two standalone runs are a
    legitimate cross-run comparison and are left alone.
    """
    left, right = (
        _dig(baseline, ("run", "pair_id")),
        _dig(candidate, ("run", "pair_id")),
    )
    if left is not None and right is not None and left != right:
        raise IncomparableError(
            f"these arms belong to different pairs ({left} and {right}), so each is "
            "half of an A/B whose other half is somewhere else\n  Solution: compare "
            "the two arms one `bench run` invocation wrote, or rerun the pair"
        )


def _check_status(baseline: dict, candidate: dict) -> None:
    """Refuse a run that stopped early."""
    truncated = [
        f"{label} ({_dig(document, ('run', 'arm'))})"
        for label, document in (("baseline", baseline), ("candidate", candidate))
        if _dig(document, ("run", "status")) not in (COMPLETE, None)
    ]
    if truncated:
        raise IncomparableError(
            f"partial result(s): {', '.join(truncated)}. A run that stopped early "
            "carries the cells that ran, and differencing it against a whole arm "
            "reports the cells that did not as whatever the survivors measured\n  "
            "Solution: rerun the arm that stopped; its samples.jsonl is complete up "
            "to the failure and is worth keeping either way"
        )


def _check_distinct(baseline: dict, candidate: dict) -> None:
    """Refuse two arm names over one service configuration.

    THREADS was a literal in docker-compose.inference.yaml and NO_REPACK appeared in no
    compose file, so an arm setting either changed nothing that reached the container
    and the pair reported a clean null on an A/B that measured nothing. The compose
    file is fixed; this is the check that would have caught it, and the disproof was
    already in the artifact -- conditions.env_knobs and conditions.server_argv are read
    from the running container, not from the invocation.

    Only when both arms observed something. A container nobody could inspect records no
    fact, and an absent fact is not two identical ones.
    """
    if _dig(baseline, ("run", "arm")) == _dig(candidate, ("run", "arm")):
        return
    left = tuple(_dig(baseline, path) for path in SERVICE_STATE)
    right = tuple(_dig(candidate, path) for path in SERVICE_STATE)
    if not any(value for value in left) or not any(value for value in right):
        return
    if left == right:
        raise IncomparableError(
            "these two arms ran the same service configuration, so whatever they "
            "differ in did not reach the container and the A/B measured nothing:\n  "
            f"env_knobs and server_argv are identical across {_dig(baseline, ('run', 'arm'))!r} "
            f"and {_dig(candidate, ('run', 'arm'))!r}\n  Solution: check that the knob "
            "is interpolated in docker-compose.inference.yaml and read by the service "
            "entrypoint; a variable the compose file states as a literal is set in the "
            "client's environment and never crosses into the container"
        )


def classify(delta_pct: float, noise_pct: float | None, higher_is_better: bool) -> str:
    """One metric's verdict against both thresholds.

    The order is deliberate: the floor is consulted before the 5% bar, because a delta
    the rig cannot resolve is not a small result -- it is not a result. Only once a
    change is known to be bigger than the machine's own variation does the reporting
    threshold apply.
    """
    magnitude = abs(delta_pct)
    if noise_pct is not None and magnitude <= noise_pct:
        return WITHIN_NOISE
    if magnitude < REGRESSION_PCT:
        return BELOW_THRESHOLD
    if noise_pct is None:
        return INCONCLUSIVE
    signed = delta_pct if higher_is_better else -delta_pct
    return IMPROVEMENT if signed > 0 else REGRESSION


def _iter_metrics(report: dict):
    """Yield (point key, metric mapping) for every comparable block in a report.

    A point carries its identity as scalars beside a `metrics` mapping, so the key is
    read from the identity and never from a statistic that happens to share a name.
    """
    for phase, body in report.get("phases", {}).items():
        for entry in body.get("points", []) or []:
            yield entry["point"], entry.get("metrics", {})
        slope = body.get("slope")
        if slope:
            yield f"{phase}/slope", slope


def point_metrics(report: dict) -> dict[str, dict]:
    """Every point key in this report mapped to its metric block.

    Public because the comparison command needs to read the reduction's own account of
    a point -- how many samples produced fewer arrivals than tokens, and how densely
    the server packed its chunks -- to explain a metric that is present in one arm and
    absent from the other.
    """
    return dict(_iter_metrics(report))


def check_floor(baseline: dict, noise: dict) -> None:
    """Refuse a floor that was not measured against this workload.

    A mismatched floor is not ignored, which would be survivable -- it is partly
    applied. The `quick` and `publish` profiles share the point key
    `prefill/p128/g1/c1`, and a slope block is keyed on its phase alone, so a floor
    from another profile lands on exactly the keys a regression would be graded
    against and absorbs it as `within_noise`.

    Only the baseline is inspected: check_comparable has already refused a pair whose
    profile or service disagree, so the candidate carries the same two values.
    """
    mismatches = []
    for floor_path, arm_path in FLOOR_PROVENANCE:
        stated, measured = _dig(noise, floor_path), _dig(baseline, arm_path)
        if stated != measured:
            mismatches.append(
                f"{'.'.join(floor_path)}: {stated!r} vs the arms' {measured!r}"
            )
    if mismatches:
        raise MismatchedFloorError(
            "this noise floor was not measured against this workload, so applying it "
            "would absorb a real regression on the points the two workloads share:\n  "
            + "\n  ".join(mismatches)
            + "\n  Solution: `python -m bench noise --service <s> --profile <p>` for "
            "the workload under comparison"
        )
    _check_floor_coverage(noise)


def _check_floor_coverage(noise: dict) -> None:
    """Refuse a floor whose own pair lost cells.

    A metric the floor does not carry is not graded leniently, it is graded
    `inconclusive` -- which does not fail. So a floor narrowed by failed cells returns
    a green all-clear on exactly the metrics it is missing: an arm that lost eight of
    nine cells wrote four floor entries where a clean run writes fifteen, and that
    floor passed a genuine thirty-three percent decode regression.

    Both signals come from the floor document itself, which now carries the pair's
    errors and a coverage block; a floor written before those existed has neither and
    is not accused.
    """
    errors = noise.get("errors") or []
    coverage = noise.get("coverage") or {}
    arm_metrics = coverage.get("arm_metrics") or {}
    widest = max(arm_metrics.values(), default=0)
    short = widest and coverage.get("floor_metrics", widest) < widest
    if not (errors or short):
        return
    detail = []
    if errors:
        detail.append(f"{len(errors)} cell(s) failed while it was being measured")
    if short:
        detail.append(f"it covers {coverage.get('floor_metrics')} of {widest} metrics")
    raise MismatchedFloorError(
        "this noise floor was measured over a pair that did not complete, so the "
        "metrics it does not carry would grade as inconclusive rather than as "
        "regressions: " + "; and ".join(detail) + "\n  Solution: `python -m bench "
        "noise --service <s> --profile <p>` again on a healthy service"
    )


def compare(
    baseline: dict,
    candidate: dict,
    noise: dict | None = None,
    *,
    require_distinct: bool = True,
) -> tuple[list[MetricDelta], list[str]]:
    """Difference two results, metric by metric, against a noise floor."""
    warnings = check_comparable(baseline, candidate, require_distinct=require_distinct)
    if noise is not None:
        check_floor(baseline, noise)

    floors = noise.get("floor", {}) if noise else {}
    # Guarded on the floor being useless rather than on the argument being absent: a
    # document that measured nothing grades every metric as unmeasured, and did so
    # more quietly than passing no floor at all.
    if not floors:
        warnings.append(
            "no noise floor to compare against: verdicts fall back to the 5% "
            "threshold alone. Run `python -m bench noise --service <s> --profile <p>` "
            "first -- an unmeasured floor on this workstation has been as wide as 44%"
        )
    base_points = dict(_iter_metrics(baseline))
    deltas: list[MetricDelta] = []

    for key, cand_entry in _iter_metrics(candidate):
        base_entry = base_points.get(key)
        if base_entry is None:
            continue
        for metric, cand_block in cand_entry.items():
            if metric in DIAGNOSTIC:
                continue
            base_block = base_entry.get(metric)
            if not isinstance(cand_block, dict) or not isinstance(base_block, dict):
                # A slope block holds bare floats rather than stat blocks.
                if not (
                    isinstance(cand_block, (int, float))
                    and isinstance(base_block, (int, float))
                    and metric in HIGHER_IS_BETTER
                ):
                    continue
                base_value, cand_value = float(base_block), float(cand_block)
            else:
                if "median" not in cand_block or "median" not in base_block:
                    continue
                base_value = base_block["median"]
                cand_value = cand_block["median"]
            if not base_value:
                continue
            delta_pct = (cand_value - base_value) / base_value * 100.0
            floor = floors.get(f"{key}:{metric}", floors.get(metric))
            higher = metric in HIGHER_IS_BETTER
            deltas.append(
                MetricDelta(
                    point=key,
                    metric=metric,
                    baseline=base_value,
                    candidate=cand_value,
                    delta_pct=delta_pct,
                    noise_pct=floor,
                    verdict=classify(delta_pct, floor, higher),
                    higher_is_better=higher,
                )
            )
    return deltas, warnings


def floor_from_pair(arm_a: dict, arm_b: dict) -> dict:
    """The noise floor: what the rig produces from no change at all.

    Both arguments are results from the *same* configuration run as an interleaved pair.
    The per-metric absolute delta between them is the largest difference this workload
    can produce without anything having changed, and therefore the smallest difference a
    real A/B is entitled to call a result.
    """
    # The two arms are one configuration run twice, so the A/B guard against two arm
    # names over identical service state is exactly what this pair is meant to be.
    check_comparable(arm_a, arm_b, require_distinct=False)
    deltas, _ = compare(arm_a, arm_b, noise=None, require_distinct=False)
    return {
        "floor": {f"{d.point}:{d.metric}": abs(d.delta_pct) for d in deltas},
        "worst_pct": max((abs(d.delta_pct) for d in deltas), default=0.0),
    }
