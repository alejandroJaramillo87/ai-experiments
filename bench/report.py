"""The bench_result report: assembly, writing, and reading back.

docs/observability/report_schema.md names this emitter in advance -- "any JSON-emitting
inference benchmark must adopt this envelope" -- so the header comes from
common.reports.report_header and the assembled document is checked by
common.reports.validate_report before it is written. A report that cannot be validated
is not written at all: a malformed result file is worse than none, because it will be
cited.

There is deliberately no schemas/bench_result.json. schemas/README.md draws the line:
that directory holds cross-language key inventories, and the report envelope is the
separate mechanism for JSON reports. This file has one Python writer and one Python
reader, both inside bench/.

What makes a number citable a year later is the conditions block, not the number. The
retired tooling persisted a total runtime to a one-row SQLite table and printed the rest
to a text file, so no archived result records the model, the quantization, the thread
count, or the driver it ran against.
"""

from __future__ import annotations

import json
import os

from common.provenance import git_sha
from common.reports import report_header, validate_report

REPORT_TYPE = "bench_result"
NOISE_REPORT_TYPE = "bench_noise"
SCHEMA_VERSION = 1


def build(
    *,
    run: dict,
    conditions: dict,
    phases: dict,
    errors: list[dict] | None = None,
    samples_path: str | None = None,
    report_type: str = REPORT_TYPE,
) -> dict:
    """Assemble and validate one result document."""
    report = report_header(report_type, schema_version=SCHEMA_VERSION)
    report["git_sha"] = git_sha()
    report.update(
        run=run,
        conditions=conditions,
        phases=phases,
        errors=errors or [],
        samples_path=samples_path,
    )
    validate_report(report)
    return report


def write(report: dict, directory: str, filename: str = "result.json") -> str:
    """Write a validated report; returns the path."""
    validate_report(report)
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, filename)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=False)
        handle.write("\n")
    return path


def read(path: str) -> dict:
    """Read a result file, validating the envelope on the way in.

    Reading through the validator rather than around it means a hand-edited or
    truncated result fails at the point it is loaded, not later inside a comparison
    that would otherwise report a confident delta between one real arm and one broken
    one.
    """
    with open(path, encoding="utf-8") as handle:
        report = json.load(handle)
    validate_report(report)
    return report
