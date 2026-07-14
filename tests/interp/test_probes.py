"""W2 probe toolkit tests: split determinism, registry hash refusal, and
score.py parity between the from-activations and from-texts paths (the W2
acceptance list), plus the regenerable-cache property on a fake capture."""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

from interp.probes.capture import CaptureError, load_capture, load_labeled_dataset
from interp.probes.registry import RegistryError, list_probes, load_probe
from interp.probes.score import score_activations, score_texts
from interp.probes.train import split_of, train_probes
from tests.interp.conftest import fake_capture


class TestSplit:
    def test_deterministic(self):
        texts = [f"text {i}" for i in range(200)]
        first = [split_of(t, seed=7) for t in texts]
        second = [split_of(t, seed=7) for t in texts]
        assert first == second

    def test_order_independent(self):
        # the property that makes re-captures unable to leak test data
        assert split_of("some example", 0) == split_of("some example", 0)
        splits = {split_of(f"t{i}", 0) for i in range(500)}
        assert splits == {"train", "val", "test"}

    def test_seed_changes_assignment(self):
        texts = [f"text {i}" for i in range(200)]
        a = [split_of(t, seed=0) for t in texts]
        b = [split_of(t, seed=1) for t in texts]
        assert a != b

    def test_proportions(self):
        counts = {"train": 0, "val": 0, "test": 0}
        for i in range(5000):
            counts[split_of(f"text {i}")] += 1
        assert 0.75 < counts["train"] / 5000 < 0.85
        assert 0.07 < counts["val"] / 5000 < 0.13
        assert 0.07 < counts["test"] / 5000 < 0.13


class TestCapture:
    def test_dataset_validation(self, tmp_path):
        bad = tmp_path / "bad.jsonl"
        bad.write_text('{"text": "no label"}\n')
        with pytest.raises(CaptureError):
            load_labeled_dataset(str(bad))

    def test_roundtrip(self, capture_dir):
        activations, labels, texts = load_capture(capture_dir)
        assert sorted(activations) == [3, 5]
        assert activations[3].shape == (120, 8)
        assert len(labels) == len(texts) == 120


class TestRegistry:
    def test_load_and_list(self, probe_dir):
        probe = load_probe(probe_dir, expect_model_hash="modelhash-1")
        assert probe.name == "test-probe"
        assert probe.weights.shape == (8,)
        assert list_probes(os.path.dirname(probe_dir)) == ["test-probe"]

    def test_refuses_model_hash_mismatch(self, probe_dir):
        with pytest.raises(RegistryError, match="model_hash"):
            load_probe(probe_dir, expect_model_hash="other-model")

    def test_refuses_tokenizer_mismatch(self, probe_dir):
        with pytest.raises(RegistryError, match="tokenizer_fingerprint"):
            load_probe(
                probe_dir,
                expect_model_hash="modelhash-1",
                expect_tokenizer_fingerprint="different",
            )

    def test_refuses_tampered_probe_file(self, probe_dir):
        with open(os.path.join(probe_dir, "probe.npz"), "ab") as f:
            f.write(b"tamper")
        with pytest.raises(RegistryError, match="hash"):
            load_probe(probe_dir)


class TestTraining:
    def test_probe_learns_separable_data(self, probe_dir):
        manifest = json.load(open(os.path.join(probe_dir, "manifest.json")))
        assert manifest["metrics"]["test_auroc"] > 0.95
        assert set(manifest["per_layer_metrics"]) == {"3", "5"}

    def test_regenerable_cache(self, tmp_path, labeled_dataset):
        # deleting and re-running capture + train reproduces metrics
        from interp.probes.capture import run_capture

        results = []
        for run in ("a", "b"):
            cap = str(tmp_path / f"cap-{run}")
            run_capture(
                "test-model",
                labeled_dataset,
                [3],
                cap,
                model_hash="m",
                capture_fn=fake_capture,
            )
            manifest = train_probes(cap, f"p-{run}", str(tmp_path / f"probes-{run}"))
            results.append(manifest["metrics"])
        assert results[0] == results[1]


class TestScoreParity:
    def test_from_activations_equals_from_texts(self, probe_dir, capture_dir):
        probe = load_probe(probe_dir)
        activations, _, texts = load_capture(capture_dir)
        direct = score_activations(probe, activations[probe.layer])
        via_texts = score_texts([probe], texts, "test-model", capture_fn=fake_capture)[
            probe.name
        ]
        np.testing.assert_allclose(direct, via_texts, rtol=1e-6)

    def test_scores_bounded(self, probe_dir, capture_dir):
        probe = load_probe(probe_dir)
        activations, _, _ = load_capture(capture_dir)
        scores = score_activations(probe, activations[probe.layer])
        assert ((scores >= 0) & (scores <= 1)).all()
