"""Scorecard assembly and rendering (Phase 1 F7).

scorecard.json is the machine-readable record comparative claims are checked
against: a metadata block (models, model hashes, pins, git SHA), then one section per evaluator with baseline, tuned, and delta.
Assembly enforces the D7 confound rules mechanically: a scorecard whose two
sides have mismatched thinking mode is refused, not written.

Stdlib-only: assembly runs on the host from the phase output files.
"""

from __future__ import annotations

import time

from common.provenance import git_sha as provenance_git_sha

SCHEMA_VERSION = 1

# Confound fields that must match between the two compared sides: a
# thinking-on versus thinking-off comparison measures the mode, not the
# fine-tune. Both sides are bf16 HF checkpoints served the same way, so
# there is no storage-type confound to match; model_hash is recorded per
# side so the scorecard names the exact weights compared.
MATCHED_FIELDS = ("thinking_mode",)


class ScorecardError(RuntimeError):
    """Refused scorecard; message names the confound or the missing piece."""


def git_sha(repo_root: str) -> str:
    return provenance_git_sha(repo_root)


def delta_section(
    baseline: float | None, tuned: float | None, higher_is_better: bool
) -> dict:
    """One metric compared across sides; delta is tuned minus baseline."""
    section = {
        "baseline": baseline,
        "tuned": tuned,
        "higher_is_better": higher_is_better,
    }
    if baseline is not None and tuned is not None:
        delta = tuned - baseline
        section["delta"] = round(delta, 6)
        section["improved"] = delta > 0 if higher_is_better else delta < 0
    return section


def check_matched(metadata: dict) -> None:
    """Refuse mismatched confound fields between the compared sides."""
    base = metadata.get("baseline", {})
    tuned = metadata.get("tuned", {})
    for field in MATCHED_FIELDS:
        left, right = base.get(field), tuned.get(field)
        if left is not None and right is not None and left != right:
            raise ScorecardError(
                f"Confound: {field} differs between sides "
                f"(baseline={left}, tuned={right})\n"
                "  Solution: compare at matched settings; re-export or "
                "re-generate the mismatched side"
            )


def assemble_scorecard(metadata: dict, sections: dict) -> dict:
    """Validated scorecard dict from the metadata block and evaluator sections.

    sections maps evaluator name (holdout/lmeval) to its result block;
    absent evaluators are recorded as skipped so a partial run yields a
    partial scorecard that says so.
    """
    if not sections:
        raise ScorecardError("No evaluator sections; nothing to score")
    check_matched(metadata)
    evaluators = {}
    for name in ("holdout", "lmeval"):
        evaluators[name] = sections.get(name) or {"skipped": True}
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "scorecard",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "metadata": metadata,
        "evaluators": evaluators,
    }


def _fmt(value) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def render_markdown(scorecard: dict) -> str:
    """scorecard.md: metadata summary plus one table row per metric."""
    meta = scorecard["metadata"]
    lines = [
        "# Fine-tuning evaluation scorecard",
        "",
        f"Generated: {scorecard['timestamp']}",
        "",
        "| Side | Model | Hash | Thinking |",
        "|------|-------|------|----------|",
    ]
    for side in ("baseline", "tuned"):
        info = meta.get(side, {})
        lines.append(
            f"| {side} | {_fmt(info.get('model'))} "
            f"| {_fmt((info.get('model_hash') or '')[:12] or None)} "
            f"| {_fmt(info.get('thinking_mode'))} |"
        )
    lines += [
        "",
        "| Evaluator | Metric | Baseline | Tuned | Delta | Improved |",
        "|-----------|--------|----------|-------|-------|----------|",
    ]
    evaluators = scorecard["evaluators"]

    holdout = evaluators["holdout"]
    if holdout.get("skipped"):
        lines.append("| holdout | - | skipped | | | |")
    else:
        for metric in ("loss", "perplexity"):
            section = holdout.get(metric)
            if section:
                lines.append(
                    f"| holdout | {metric} | {_fmt(section.get('baseline'))} "
                    f"| {_fmt(section.get('tuned'))} | {_fmt(section.get('delta'))} "
                    f"| {_fmt(section.get('improved'))} |"
                )

    lmeval = evaluators["lmeval"]
    if lmeval.get("skipped"):
        lines.append("| lmeval | - | skipped | | | |")
    else:
        for task, section in sorted(lmeval.get("tasks", {}).items()):
            lines.append(
                f"| lmeval | {task} ({section.get('metric')}) "
                f"| {_fmt(section.get('baseline'))} | {_fmt(section.get('tuned'))} "
                f"| {_fmt(section.get('delta'))} | {_fmt(section.get('improved'))} |"
            )

    lines.append("")
    return "\n".join(lines)
