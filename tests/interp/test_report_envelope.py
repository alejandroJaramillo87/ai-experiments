"""Report-envelope conformance (WI-12): every interp report emitter's output
validates against the shared header convention in common/reports.py. Reports
are produced through each emitter's real code path with the same injected
seams the emitter tests use."""

from __future__ import annotations

import datetime
import json
import os

import numpy as np
import pytest

from common.reports import validate_report
from tests.interp.conftest import fake_capture


class TestOrganismVerification:
    def test_build_report_conforms(self):
        from interp.organisms.verify import build_report

        report = build_report(
            "taboo",
            config={"word": "x"},
            implant={"refusals": 1},
            locality={},
            verdict={"passed": True},
        )
        validate_report(report)


class TestSaeEvaluation:
    def test_evaluate_sae_report_conforms(self, tmp_path):
        from interp.manifest import write_manifest
        from interp.sae.evaluate import evaluate_sae

        capture_dir = str(tmp_path / "capture")
        os.makedirs(capture_dir)
        write_manifest(
            capture_dir,
            {
                "schema_version": 1,
                "artifact": "activation_harvest",
                "model_id": "google/gemma-4-E2B",
                "model_hash": "hash-test",
                "hooks": ["model.layers.15"],
                "d_model": 8,
                "corpus": "test-corpus",
                "total_tokens": 50,
                "tokenizer_fingerprint": "fp-test",
                "validated": False,
                "created": "2026-07-15T00:00:00Z",
            },
        )
        sae_dir = str(tmp_path / "sae")
        os.makedirs(sae_dir)
        write_manifest(
            sae_dir,
            {
                "schema_version": 1,
                "artifact": "sae",
                "name": "sae-test",
                "model_id": "google/gemma-4-E2B",
                "model_hash": "hash-test",
                "hook": "model.layers.15",
                "d_in": 8,
                "width": 16,
                "capture_dir": capture_dir,
            },
        )
        acts = np.random.default_rng(7).normal(size=(50, 8)).astype(np.float32)

        def identity_sae(a):
            features = np.zeros((a.shape[0], 16), dtype=np.float32)
            features[:, :2] = 1.0
            return a.copy(), features

        report = evaluate_sae(
            sae_dir, capture_dir, sae_fn=identity_sae, activations=acts
        )
        validate_report(report)


class TestProbeEvaluation:
    def test_evaluate_probe_report_conforms(self, probe_dir, capture_dir):
        from interp.probes.evaluate import evaluate_probe
        from interp.probes.registry import load_probe

        report = evaluate_probe(
            load_probe(probe_dir),
            capture_dir,
            "negativity",
            lambda text: "the same response every time",
            lambda prompt: "I cannot decide",
        )
        validate_report(report)


class TestProbeMonitor:
    def test_run_monitor_report_conforms(self, probe_dir, tmp_path):
        from interp.probes.monitor import run_monitor

        now = datetime.datetime(2026, 7, 15, tzinfo=datetime.UTC)
        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        with open(log_dir / "2026-07-14.jsonl", "w") as f:
            for i in range(3):
                f.write(
                    json.dumps(
                        {
                            "timestamp": f"2026-07-14T0{i}:00:00.000000Z",
                            "prompt": f"a neg prompt {i}",
                            "response_text": "a reply",
                        }
                    )
                    + "\n"
                )
        report = run_monitor(
            [probe_dir],
            "test-model",
            "modelhash-1",
            str(log_dir),
            monitor_root=str(tmp_path / "monitor"),
            capture_fn=fake_capture,
            health_check=lambda: [],
            now=now,
        )
        validate_report(report)


class TestSteeringScorecard:
    def test_run_scorecard_report_conforms(self, tmp_path, repo_root, monkeypatch):
        pytest.importorskip("gguf")
        from interp.steering.export import export_control_vector
        from interp.steering.scorecard import ScorecardConfig, run_scorecard
        from tests.interp.test_steering import (
            FakeController,
            direction_set,
            fake_post_factory,
        )

        monkeypatch.setenv("TRAIN_LOCK_PATH", str(tmp_path / "no.lock"))
        vdir = str(tmp_path / "vectors" / "v1")
        export_control_vector(direction_set(), vdir)
        prompts = repo_root / "interp" / "steering" / "prompts"
        cfg = ScorecardConfig(
            vector_dir=vdir,
            attribute="honesty",
            behavior_prompts=str(prompts / "behavior_honesty.jsonl"),
            neutral_prompts=str(prompts / "neutral.jsonl"),
            system_prompt_file=str(prompts / "system_prompt_honesty.json"),
            scales=(0.5,),
        )
        report = run_scorecard(
            cfg, controller=FakeController(), post=fake_post_factory()
        )
        validate_report(report)
