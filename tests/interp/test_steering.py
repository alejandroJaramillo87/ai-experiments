"""W1 steering tests: deterministic GGUF export (re-export reproduces the same
hash), wiring validation and path mapping, and scorecard orchestration with a
fake service controller (interrupted scorecard leaves the service unsteered)."""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

from interp.manifest import file_xxhash
from interp.steering.extract import DirectionSet, ExtractionError, load_pairs
from interp.steering.wiring import (
    WiringError,
    container_vector_path,
    steering_env,
    validate_vector_dir,
)

gguf = pytest.importorskip("gguf", reason="gguf package required for export tests")

from interp.steering.export import ExportError, export_control_vector  # noqa: E402


def direction_set() -> DirectionSet:
    rng = np.random.default_rng(42)
    return DirectionSet(
        directions={
            layer: rng.normal(0, 1, 16).astype(np.float32) for layer in (1, 2, 3)
        },
        model_id="test/model",
        model_revision="main",
        method="repeng",
        source={"model_type": "test"},
    )


class TestExport:
    def test_reproducible_hash(self, tmp_path):
        # W1 acceptance: re-export from the same inputs reproduces the hash
        m1 = export_control_vector(direction_set(), str(tmp_path / "a"))
        m2 = export_control_vector(direction_set(), str(tmp_path / "b"))
        assert m1["vector_xxhash"] == m2["vector_xxhash"]
        assert m1["vector_xxhash"] == file_xxhash(str(tmp_path / "a" / "vector.gguf"))

    def test_rejects_zero_based_layers(self, tmp_path):
        bad = direction_set()
        bad.directions[0] = bad.directions.pop(1)
        with pytest.raises(ExportError, match="1-based"):
            export_control_vector(bad, str(tmp_path / "v"))

    def test_rejects_ragged_widths(self, tmp_path):
        bad = direction_set()
        bad.directions[2] = np.zeros(8, dtype=np.float32)
        with pytest.raises(ExportError, match="width"):
            export_control_vector(bad, str(tmp_path / "v"))

    def test_manifest_contents(self, tmp_path):
        manifest = export_control_vector(
            direction_set(), str(tmp_path / "v"), model_hash="h123"
        )
        assert manifest["layers"] == [1, 2, 3]
        assert manifest["width"] == 16
        assert manifest["model_hash"] == "h123"


class TestWiring:
    def test_validate_ok(self, tmp_path):
        vdir = str(tmp_path / "v")
        export_control_vector(direction_set(), vdir, model_hash="h123")
        manifest = validate_vector_dir(vdir, served_model_hash="h123")
        assert manifest["model_hash"] == "h123"

    def test_refuses_model_hash_mismatch(self, tmp_path):
        vdir = str(tmp_path / "v")
        export_control_vector(direction_set(), vdir, model_hash="h123")
        with pytest.raises(WiringError, match="model_hash"):
            validate_vector_dir(vdir, served_model_hash="other")

    def test_refuses_tampered_vector(self, tmp_path):
        vdir = str(tmp_path / "v")
        export_control_vector(direction_set(), vdir)
        with open(os.path.join(vdir, "vector.gguf"), "ab") as f:
            f.write(b"x")
        with pytest.raises(WiringError, match="hash"):
            validate_vector_dir(vdir)

    def test_container_path_mapping(self):
        assert (
            container_vector_path("/mnt/ai-data/interp/vectors/honesty-v1")
            == "/app/vectors/honesty-v1/vector.gguf"
        )
        with pytest.raises(WiringError, match="mounted tree"):
            container_vector_path("/tmp/elsewhere")

    def test_steering_env(self):
        env = steering_env(
            "/mnt/ai-data/interp/vectors/v1", scale=1.5, layer_range=(4, 20)
        )
        assert env == {
            "CONTROL_VECTOR": "/app/vectors/v1/vector.gguf",
            "CONTROL_VECTOR_SCALE": "1.5",
            "CONTROL_VECTOR_LAYER_RANGE": "4 20",
        }


class TestExtractInputs:
    def test_load_pairs_validates(self, tmp_path):
        path = tmp_path / "pairs.jsonl"
        path.write_text('{"positive": "a"}\n')
        with pytest.raises(ExtractionError, match="negative"):
            load_pairs(str(path))

    def test_repo_prompt_sets_parse(self, repo_root):
        prompts = repo_root / "interp" / "steering" / "prompts"
        pairs = load_pairs(str(prompts / "honesty_pairs.jsonl"))
        assert len(pairs) >= 10
        record = json.loads((prompts / "system_prompt_honesty.json").read_text())
        assert len(record["iterations"]) >= 3


class FakeController:
    def __init__(self, fail_on_scale: float | None = None):
        self.log: list[str] = []
        self.fail_on_scale = fail_on_scale

    def steer(self, scale: float) -> None:
        if self.fail_on_scale is not None and scale == self.fail_on_scale:
            raise RuntimeError("service failed to start")
        self.log.append(f"steer:{scale:g}")

    def unsteer(self) -> None:
        self.log.append("unsteer")


def fake_post_factory():
    def fake_post(url, payload):
        content = payload["messages"][-1]["content"]
        if "Reply with JSON only" in content:
            text = '{"score": 7, "reason": "test"}'
        else:
            text = "a plain response"
        return {
            "choices": [{"message": {"content": text}}],
            "usage": {"completion_tokens": 5},
        }

    return fake_post


class TestScorecard:
    def _config(self, tmp_path, repo_root, monkeypatch):
        from interp.steering import scorecard as sc

        monkeypatch.setattr(sc, "GPU_LOCK_PATH", str(tmp_path / "no.lock"))
        vdir = str(tmp_path / "vectors" / "v1")
        export_control_vector(direction_set(), vdir)
        prompts = repo_root / "interp" / "steering" / "prompts"
        return sc.ScorecardConfig(
            vector_dir=vdir,
            attribute="honesty",
            behavior_prompts=str(prompts / "behavior_honesty.jsonl"),
            neutral_prompts=str(prompts / "neutral.jsonl"),
            system_prompt_file=str(prompts / "system_prompt_honesty.json"),
            scales=(0.5, 1.0),
        )

    def test_three_arms_and_restore(self, tmp_path, repo_root, monkeypatch):
        from interp.steering.scorecard import run_scorecard

        controller = FakeController()
        result = run_scorecard(
            self._config(tmp_path, repo_root, monkeypatch),
            controller=controller,
            post=fake_post_factory(),
        )
        arms = [a["arm"] for a in result["arms"]]
        assert arms == ["baseline", "system_prompt", "steered@0.5", "steered@1"]
        assert controller.log[-1] == "unsteer"  # restored at the end
        assert os.path.isfile(result["out_path"])
        body = json.load(open(result["out_path"]))
        assert body["arms"][0]["summary"]["behavior_score_mean"] == 7.0
        assert len(body["config"]["system_prompt_iterations"]) >= 3

    def test_interrupted_scorecard_unsteers(self, tmp_path, repo_root, monkeypatch):
        from interp.steering.scorecard import run_scorecard

        controller = FakeController(fail_on_scale=1.0)
        with pytest.raises(RuntimeError):
            run_scorecard(
                self._config(tmp_path, repo_root, monkeypatch),
                controller=controller,
                post=fake_post_factory(),
            )
        assert controller.log[-1] == "unsteer"

    def test_refuses_gpu_lock(self, tmp_path, repo_root, monkeypatch):
        from interp.steering import scorecard as sc

        cfg = self._config(tmp_path, repo_root, monkeypatch)
        lock = tmp_path / "held.lock"
        lock.touch()
        monkeypatch.setattr(sc, "GPU_LOCK_PATH", str(lock))
        with pytest.raises(sc.ScorecardError, match="lock"):
            sc.run_scorecard(cfg, controller=FakeController(), post=fake_post_factory())

    def test_judge_json_parsing(self):
        from interp.steering.scorecard import ScorecardError, parse_judge_json

        assert parse_judge_json('{"score": 3}')["score"] == 3
        assert (
            parse_judge_json('verdict: {"score": 9, "reason": "x"} done')["score"] == 9
        )
        with pytest.raises(ScorecardError):
            parse_judge_json("no json here")
