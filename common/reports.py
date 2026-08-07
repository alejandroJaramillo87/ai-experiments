"""Shared JSON report envelope (the D8-style convention, WI-12).

Every JSON report the repository emits — evaluation scorecards, verification
reports, monitor snapshots, serve-eval results — carries one header
convention so tooling can discover, version, and attribute reports without
per-emitter knowledge. The convention (documented in
docs/observability/report_schema.md):

- schema_version: int >= 1, bumped on any breaking payload change
- a type discriminator: exactly one non-empty string among "report",
  "kind", or "artifact" at the top level
- a wall-clock stamp: "timestamp" or "created", non-empty string
- provenance: "git_sha" at the top level or under "metadata"

The payload below the header is emitter-owned and not validated here.
Artifact manifests written through interp/manifest.py share the artifact
discriminator but are contracts of their own directories; this module is
the report-side validator.

The future D8 benchmark harness and any JSON-emitting inference benchmark
must adopt this envelope (build it with report_header()).
"""

from __future__ import annotations

import time

DISCRIMINATOR_KEYS = ("report", "kind", "artifact")
TIMESTAMP_KEYS = ("timestamp", "created")


class ReportSchemaError(ValueError):
    """A report's header violates the shared envelope convention."""


def validate_report(report: dict) -> None:
    """Raise ReportSchemaError listing every header violation."""
    problems: list[str] = []
    if not isinstance(report, dict):
        raise ReportSchemaError(f"report must be a mapping, got {type(report)}")

    version = report.get("schema_version")
    if not isinstance(version, int) or isinstance(version, bool) or version < 1:
        problems.append(f"schema_version: int >= 1 required, got {version!r}")

    discriminators = [
        key
        for key in DISCRIMINATOR_KEYS
        if isinstance(report.get(key), str) and report[key]
    ]
    if len(discriminators) != 1:
        problems.append(
            f"exactly one non-empty discriminator of {DISCRIMINATOR_KEYS} "
            f"required, found {discriminators or 'none'}"
        )

    stamps = [
        key
        for key in TIMESTAMP_KEYS
        if isinstance(report.get(key), str) and report[key]
    ]
    if not stamps:
        problems.append(
            f"a non-empty timestamp is required ({' or '.join(TIMESTAMP_KEYS)})"
        )

    git_sha = report.get("git_sha")
    if not (isinstance(git_sha, str) and git_sha):
        metadata = report.get("metadata")
        nested = metadata.get("git_sha") if isinstance(metadata, dict) else None
        if not (isinstance(nested, str) and nested):
            problems.append(
                "git_sha: non-empty string required (top level or metadata)"
            )

    if problems:
        raise ReportSchemaError("; ".join(problems))


def report_header(report_type: str, schema_version: int = 1) -> dict:
    """A conforming header for a new report; the emitter merges its payload.

    New emitters (the future D8 train bench, inference benchmarks) start
    here instead of hand-rolling the envelope.
    """
    from common.provenance import git_sha

    return {
        "schema_version": schema_version,
        "report": report_type,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "git_sha": git_sha(),
    }
