"""Report-envelope conformance (WI-12) for the training-side emitters: the
F7 scorecard assembler, plus the shared
validator and header helper in common/reports.py."""

from __future__ import annotations

import pytest

from common.reports import ReportSchemaError, report_header, validate_report


class TestValidator:
    def test_conforming_header_passes(self):
        validate_report(
            {
                "schema_version": 1,
                "report": "example",
                "timestamp": "2026-07-16T00:00:00+0000",
                "git_sha": "abc123",
            }
        )

    def test_kind_and_artifact_discriminators_accepted(self):
        for key in ("kind", "artifact"):
            validate_report(
                {
                    "schema_version": 2,
                    key: "example",
                    "created": "2026-07-16T00:00:00Z",
                    "git_sha": "abc123",
                }
            )

    def test_metadata_git_sha_accepted(self):
        validate_report(
            {
                "schema_version": 1,
                "kind": "scorecard",
                "timestamp": "2026-07-16T00:00:00+0000",
                "metadata": {"git_sha": "abc123"},
            }
        )

    @pytest.mark.parametrize(
        "broken, match",
        [
            ({"schema_version": 0}, "schema_version"),
            ({"schema_version": "1"}, "schema_version"),
            ({"schema_version": True}, "schema_version"),
            ({"schema_version": 1}, "discriminator"),
            (
                {"schema_version": 1, "report": "x", "kind": "y"},
                "discriminator",
            ),
            ({"schema_version": 1, "report": "x"}, "timestamp"),
            (
                {
                    "schema_version": 1,
                    "report": "x",
                    "timestamp": "2026-07-16",
                },
                "git_sha",
            ),
        ],
    )
    def test_violations_are_listed(self, broken, match):
        with pytest.raises(ReportSchemaError, match=match):
            validate_report(broken)

    def test_report_header_conforms(self):
        header = report_header("future-bench")
        validate_report(header)
        assert header["report"] == "future-bench"


class TestEvalScorecard:
    def test_assemble_scorecard_conforms(self):
        from training.eval.scorecard import assemble_scorecard

        card = assemble_scorecard(
            {"git_sha": "abc123", "evaluators": ["holdout"]},
            {"holdout": {"loss": {}}},
        )
        validate_report(card)
