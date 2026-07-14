"""W2 evaluation-report tests with injected generate/judge callables: the
black-box baselines are always present and the deployable verdict follows the
comparison."""

from __future__ import annotations

import json

from interp.probes.evaluate import evaluate_probe
from interp.probes.registry import load_probe


def leaky_generate(text: str) -> str:
    # output trivially reveals the label, so black-box arms score well
    return "negative response" if "neg" in text else "positive response"


def opaque_generate(text: str) -> str:
    return "the same response every time"


def perfect_judge(prompt: str) -> str:
    return json.dumps({"label": 0 if "negative response" in prompt else 1})


def broken_judge(prompt: str) -> str:
    return "I cannot decide"


class TestEvaluate:
    def test_report_has_mandatory_baselines(self, probe_dir, capture_dir, tmp_path):
        probe = load_probe(probe_dir)
        report = evaluate_probe(
            probe,
            capture_dir,
            "negativity",
            opaque_generate,
            broken_judge,
            out_dir=str(tmp_path / "report"),
        )
        assert "llm_judge_accuracy" in report["baselines"]
        assert "bow_output_auroc" in report["baselines"]
        assert report["probe_metrics"]["auroc"] > 0.9

    def test_deployable_when_probe_beats_baselines(self, probe_dir, capture_dir):
        probe = load_probe(probe_dir)
        report = evaluate_probe(
            probe, capture_dir, "negativity", opaque_generate, broken_judge
        )
        # opaque outputs give the baselines nothing; the probe wins
        assert report["deployable"] is True

    def test_not_deployable_when_baseline_wins(self, probe_dir, capture_dir):
        probe = load_probe(probe_dir)
        report = evaluate_probe(
            probe, capture_dir, "negativity", leaky_generate, perfect_judge
        )
        # a perfect black-box classifier ties or beats the probe; not deployed
        assert report["deployable"] is False
        assert "NOT deployed" in report["verdict"]

    def test_unparseable_judge_counts_as_wrong(self, probe_dir, capture_dir):
        probe = load_probe(probe_dir)
        report = evaluate_probe(
            probe, capture_dir, "negativity", leaky_generate, broken_judge
        )
        assert report["baselines"]["llm_judge_accuracy"] == 0.0
