"""W1 steering tests: deterministic safetensors export (re-export reproduces
the same hash), wiring validation and path mapping, and scorecard
orchestration with a fake service controller (interrupted scorecard leaves the
service unsteered)."""

from __future__ import annotations

import importlib.util
import json
import os

import numpy as np
import pytest

from interp.manifest import file_xxhash
from interp.steering.export import (
    VECTOR_NAME,
    ExportError,
    export_control_vector,
)
from interp.steering.extract import DirectionSet, ExtractionError, load_pairs
from interp.steering.wiring import (
    WiringError,
    container_vector_path,
    steering_env,
    validate_vector_dir,
)

# export_control_vector imports safetensors lazily, so only tests that export
# need the package; wiring, extract-input, and judge-parsing tests run without it
requires_safetensors = pytest.mark.skipif(
    importlib.util.find_spec("safetensors") is None,
    reason="safetensors package required for export tests",
)


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


@requires_safetensors
class TestExport:
    def test_reproducible_hash(self, tmp_path):
        # W1 acceptance: re-export from the same inputs reproduces the hash
        m1 = export_control_vector(direction_set(), str(tmp_path / "a"))
        m2 = export_control_vector(direction_set(), str(tmp_path / "b"))
        assert m1["vector_xxhash"] == m2["vector_xxhash"]
        assert m1["vector_xxhash"] == file_xxhash(str(tmp_path / "a" / VECTOR_NAME))

    def test_accepts_block_zero(self, tmp_path):
        # The retired llama.cpp layout refused index 0. Nothing in this
        # backend needs the first block left unsteerable, and R10 dropped the
        # floor; this is the assertion that it stays dropped.
        ok = direction_set()
        ok.directions[0] = ok.directions.pop(1)
        manifest = export_control_vector(ok, str(tmp_path / "v"))
        assert manifest["layers"] == [0, 2, 3]

    def test_rejects_negative_blocks(self, tmp_path):
        bad = direction_set()
        bad.directions[-1] = bad.directions.pop(1)
        with pytest.raises(ExportError, match="non-negative"):
            export_control_vector(bad, str(tmp_path / "v"))

    def test_rejects_non_finite_directions(self, tmp_path):
        # A NaN direction is added to every position's residual, so it turns
        # the whole stream to noise rather than failing anywhere visible.
        bad = direction_set()
        bad.directions[2] = np.full(16, np.nan, dtype=np.float32)
        with pytest.raises(ExportError, match="non-finite"):
            export_control_vector(bad, str(tmp_path / "v"))

    def test_writes_zero_based_block_tensors(self, tmp_path):
        # The tensor names are the cross-language contract
        # (schemas/control_vector_artifact.json); the Rust reader selects on
        # this prefix and refuses the retired "direction." spelling.
        from safetensors.numpy import load_file

        export_control_vector(direction_set(), str(tmp_path / "v"))
        tensors = load_file(str(tmp_path / "v" / VECTOR_NAME))
        assert sorted(tensors) == ["block.1", "block.2", "block.3"]

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
    @requires_safetensors
    def test_validate_ok(self, tmp_path):
        vdir = str(tmp_path / "v")
        export_control_vector(direction_set(), vdir, model_hash="h123")
        manifest = validate_vector_dir(vdir, served_model_hash="h123")
        assert manifest["model_hash"] == "h123"

    @requires_safetensors
    def test_refuses_model_hash_mismatch(self, tmp_path):
        vdir = str(tmp_path / "v")
        export_control_vector(direction_set(), vdir, model_hash="h123")
        with pytest.raises(WiringError, match="model_hash"):
            validate_vector_dir(vdir, served_model_hash="other")

    @requires_safetensors
    def test_refuses_tampered_vector(self, tmp_path):
        vdir = str(tmp_path / "v")
        export_control_vector(direction_set(), vdir)
        with open(os.path.join(vdir, VECTOR_NAME), "ab") as f:
            f.write(b"x")
        with pytest.raises(WiringError, match="hash"):
            validate_vector_dir(vdir)

    def test_container_path_mapping(self):
        assert (
            container_vector_path("/mnt/ai-data/interp/vectors/honesty-v1")
            == "/app/vectors/honesty-v1/vector.safetensors"
        )
        with pytest.raises(WiringError, match="mounted tree"):
            container_vector_path("/tmp/elsewhere")

    def test_steering_env(self):
        env = steering_env(
            "/mnt/ai-data/interp/vectors/v1", scale=1.5, layer_range=(4, 20)
        )
        assert env == {
            "CONTROL_VECTOR": "/app/vectors/v1/vector.safetensors",
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

        monkeypatch.setenv("TRAIN_LOCK_PATH", str(tmp_path / "no.lock"))
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

    @requires_safetensors
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

    @requires_safetensors
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

    @requires_safetensors
    def test_refuses_gpu_lock(self, tmp_path, repo_root, monkeypatch):
        from interp.steering import scorecard as sc

        cfg = self._config(tmp_path, repo_root, monkeypatch)
        lock = tmp_path / "held.lock"
        lock.touch()
        monkeypatch.setenv("TRAIN_LOCK_PATH", str(lock))
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

    def test_load_system_prompt_requires_three_iterations(self, tmp_path):
        # A tuned baseline is defined as at least three recorded prompt
        # iterations, not a strawman; fewer must be refused before scoring.
        from interp.steering.scorecard import ScorecardError, load_system_prompt

        path = tmp_path / "system_prompt.json"
        path.write_text(json.dumps({"prompt": "be honest", "iterations": ["v1", "v2"]}))
        with pytest.raises(ScorecardError, match="at least three prompt iterations"):
            load_system_prompt(str(path))

    def test_load_prompts_invalid_json_is_styled(self, tmp_path):
        # KB-34: a malformed prompt line raises ScorecardError naming
        # path:lineno, not a raw JSONDecodeError from the CLI
        from interp.steering.scorecard import ScorecardError, load_prompts

        path = tmp_path / "p.jsonl"
        path.write_text('{"prompt": "ok"}\nnot-json\n')
        with pytest.raises(ScorecardError, match="p.jsonl:2: invalid JSON"):
            load_prompts(str(path))

    def test_load_prompts_non_object_line_is_styled(self, tmp_path):
        # KB-34: a line parsing to a non-dict would TypeError on the key check
        from interp.steering.scorecard import ScorecardError, load_prompts

        path = tmp_path / "p.jsonl"
        path.write_text('["prompt"]\n')
        with pytest.raises(ScorecardError, match="not a JSON object"):
            load_prompts(str(path))

    def test_load_system_prompt_invalid_json_is_styled(self, tmp_path):
        from interp.steering.scorecard import ScorecardError, load_system_prompt

        path = tmp_path / "system_prompt.json"
        path.write_text("{broken")
        with pytest.raises(ScorecardError, match="invalid JSON"):
            load_system_prompt(str(path))

    def test_load_system_prompt_non_object_is_styled(self, tmp_path):
        from interp.steering.scorecard import ScorecardError, load_system_prompt

        path = tmp_path / "system_prompt.json"
        path.write_text("[1, 2]")
        with pytest.raises(ScorecardError, match="not a JSON object"):
            load_system_prompt(str(path))

    def test_judge_score_rejects_out_of_range(self):
        # A judge verdict whose score falls outside [0, 10] invalidates the run.
        from interp.steering.scorecard import (
            JUDGE_BEHAVIOR_TEMPLATE,
            ScorecardError,
            judge_score,
        )

        def fake_post(url, payload):
            return {
                "choices": [{"message": {"content": '{"score": 42, "reason": "x"}'}}]
            }

        with pytest.raises(ScorecardError, match="judge score out of range"):
            judge_score(
                "http://judge",
                JUDGE_BEHAVIOR_TEMPLATE,
                "honesty",
                "prompt",
                "response",
                post=fake_post,
            )

    def test_judge_score_accepts_in_range(self):
        # Boundary control: a valid in-range score returns the parsed verdict,
        # so the out-of-range refusal above is a real guard, not always-raising.
        from interp.steering.scorecard import JUDGE_BEHAVIOR_TEMPLATE, judge_score

        def fake_post(url, payload):
            return {
                "choices": [{"message": {"content": '{"score": 8, "reason": "ok"}'}}]
            }

        verdict = judge_score(
            "http://judge",
            JUDGE_BEHAVIOR_TEMPLATE,
            "honesty",
            "prompt",
            "response",
            post=fake_post,
        )
        assert verdict == {"score": 8.0, "reason": "ok"}
