"""Report-envelope conformance (WI-12) for the training-side emitters: the
F7 scorecard assembler and the D7 serve-eval report, plus the shared
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


class TestServeEvalReport:
    # main() runs the traced dispatch (roadmap OBS-4); isolate its obs state
    @pytest.mark.usefixtures("obs_isolation")
    def test_cmd_report_conforms(self, tmp_path, monkeypatch, capsys):
        """Drive the real serve-eval command with the serving seams faked;
        the written report must carry the shared envelope."""
        import json

        from training import serving
        from training.cli import main

        model = tmp_path / "model.gguf"
        model.write_bytes(b"gguf")
        out_path = tmp_path / "report.json"

        monkeypatch.setattr(
            serving, "model_paths", lambda m: ("/app/models/m.gguf", str(model))
        )
        monkeypatch.setattr(serving, "ensure_training_idle", lambda: None)
        monkeypatch.setattr(serving, "swap_model", lambda service, path: None)
        monkeypatch.setattr(serving, "wait_healthy", lambda service, budget=180: 1.0)
        monkeypatch.setattr(
            serving,
            "chat_completion",
            lambda service, messages, sampling, timeout=120: {
                "content": "4",
                "latency_ms": 5.0,
                "usage": {},
            },
        )
        prompts = tmp_path / "prompts.json"
        prompts.write_text(
            json.dumps(
                {
                    "defaults": {"max_tokens": 8},
                    "prompts": [{"id": "p1", "prompt": "2+2?"}],
                }
            )
        )
        rc = main(
            [
                "serve-eval",
                "--model",
                str(model),
                "--prompts",
                str(prompts),
                "--out",
                str(out_path),
                "--skip-hash-check",
            ]
        )
        assert rc == 0
        report = json.loads(out_path.read_text())
        validate_report(report)
        assert report["report"] == "serve_eval"
