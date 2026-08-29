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
)

# Metrics where a larger number is the better one; everything else is a latency.
HIGHER_IS_BETTER = frozenset(
    {"decode_tok_s", "e2e_tok_s", "prefill_tok_s_server", "prefill_tok_s_slope"}
)


class IncomparableError(ValueError):
    """Two results describe different experiments and must not be differenced."""


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
    node = report
    for key in path:
        if not isinstance(node, dict):
            return None
        node = node.get(key)
    return node


def check_comparable(baseline: dict, candidate: dict) -> list[str]:
    """Refuse on a mismatch that makes a delta meaningless; warn on git_sha.

    Returns the non-blocking warnings. Raises IncomparableError on the rest, naming
    every mismatch at once rather than the first, so a caller fixes one invocation
    instead of discovering the differences one run at a time.
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

    warnings = []
    if baseline.get("git_sha") != candidate.get("git_sha"):
        warnings.append(
            f"git_sha differs ({baseline.get('git_sha')} vs {candidate.get('git_sha')})"
            ": the arms were built from different trees, so the delta may include "
            "changes beyond the arm"
        )
    if baseline.get("run", {}).get("order") != "interleaved":
        warnings.append(
            "the baseline was not run interleaved; drift over the run may have aliased "
            "onto one arm"
        )
    return warnings


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


def compare(
    baseline: dict, candidate: dict, noise: dict | None = None
) -> tuple[list[MetricDelta], list[str]]:
    """Difference two results, metric by metric, against a noise floor."""
    warnings = check_comparable(baseline, candidate)
    if noise is None:
        warnings.append(
            "no noise floor supplied: verdicts fall back to the 5% threshold alone. "
            "Run `python -m bench noise --service <s> --profile <p>` first -- an "
            "unmeasured floor on this workstation has been as wide as 44%"
        )

    floors = noise.get("floor", {}) if noise else {}
    base_points = dict(_iter_metrics(baseline))
    deltas: list[MetricDelta] = []

    for key, cand_entry in _iter_metrics(candidate):
        base_entry = base_points.get(key)
        if base_entry is None:
            continue
        for metric, cand_block in cand_entry.items():
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
    check_comparable(arm_a, arm_b)
    deltas, _ = compare(arm_a, arm_b, noise=None)
    return {
        "floor": {f"{d.point}:{d.metric}": abs(d.delta_pct) for d in deltas},
        "worst_pct": max((abs(d.delta_pct) for d in deltas), default=0.0),
    }
