"""CLI handler tests for steer-extract and steer-scorecard (interp Phase 3 W1).

Handlers execute torch-free: only the extraction backends (repeng/sae_lens
live behind interp.steering.extract) are replaced at their home module, while
GGUF export, wiring validation, and the scorecard orchestration run for real
against tmp-dir artifacts. The scorecard's service controller and chat POST
seam are faked so no Docker or network is touched. Asserted contracts: exit
codes (0 success, 1 failure, 2 invalid usage), styled [OK]/[RUN]/[ERROR]
lines, and on-disk artifacts (vector.gguf, manifest.json, scorecard.json).
"""

from __future__ import annotations

import glob
import json
import os

import numpy as np

from interp import cli
from interp.manifest import file_xxhash, load_manifest
from interp.steering.export import export_control_vector
from interp.steering.extract import DirectionSet


def parse(argv: list[str]):
    return cli.build_parser().parse_args(argv)


def direction_set(layers=(1, 2, 3), width=16, method="repeng") -> DirectionSet:
    rng = np.random.default_rng(7)
    return DirectionSet(
        directions={
            layer: rng.normal(0, 1, width).astype(np.float32) for layer in layers
        },
        model_id="test/model",
        model_revision="main",
        method=method,
        source={"model_type": "test"},
    )


class TestSteerExtract:
    def test_repeng_success_exports_real_gguf(self, tmp_path, monkeypatch, capsys):
        # Only the torch-bound backend is faked; export runs for real, so the
        # printed hash must match both manifest and file on disk.
        calls = {}

        def fake_extract_repeng(model_id, pairs_path, revision="main"):
            calls["args"] = (model_id, pairs_path, revision)
            return direction_set()

        monkeypatch.setattr(
            "interp.steering.extract.extract_repeng", fake_extract_repeng
        )
        pairs = tmp_path / "pairs.jsonl"
        pairs.write_text('{"positive": "a", "negative": "b"}\n')
        out_dir = str(tmp_path / "vec")
        args = parse(
            [
                "steer-extract",
                "--model",
                "test/model",
                "--revision",
                "r1",
                "--model-hash",
                "h1",
                "--pairs",
                str(pairs),
                "--out",
                out_dir,
            ]
        )
        assert args.func(args) == 0

        vector_path = os.path.join(out_dir, "vector.gguf")
        assert os.path.isfile(vector_path)
        manifest = load_manifest(out_dir)
        assert manifest["model_hash"] == "h1"
        assert manifest["method"] == "repeng"
        assert manifest["layers"] == [1, 2, 3]
        assert manifest["vector_xxhash"] == file_xxhash(vector_path)
        assert calls["args"] == ("test/model", str(pairs), "r1")

        captured = capsys.readouterr().out
        assert "[RUN] Extracting repeng vector: test/model" in captured
        assert f"[OK] Vector exported: {out_dir}/vector.gguf" in captured
        assert "[OK] Layers: 3, width: 16" in captured
        assert f"[OK] Vector xxhash: {manifest['vector_xxhash']}" in captured

    def test_repeng_without_pairs_is_usage_error(self, tmp_path, capsys):
        args = parse(["steer-extract", "--model", "m", "--out", str(tmp_path / "v")])
        assert args.func(args) == 2
        captured = capsys.readouterr().out
        assert "[ERROR] --pairs is required with --method repeng" in captured

    def test_sae_row_lists_only_missing_flags(self, tmp_path, capsys):
        # Partial sae_row invocation: the usage error names exactly the
        # absent flags, not the ones that were supplied.
        args = parse(
            [
                "steer-extract",
                "--method",
                "sae_row",
                "--model",
                "m",
                "--sae-release",
                "gemma-scope",
                "--out",
                str(tmp_path / "v"),
            ]
        )
        assert args.func(args) == 2
        captured = capsys.readouterr().out
        assert "[ERROR] --method sae_row requires:" in captured
        for absent in ("sae_id", "feature", "layer"):
            assert absent in captured
        assert "sae_release" not in captured.replace("sae_id", "")

    def test_sae_row_success_single_layer(self, tmp_path, monkeypatch, capsys):
        calls = {}

        def fake_extract_sae_row(
            model_id, sae_release, sae_id, feature, layer, revision="main"
        ):
            calls["args"] = (model_id, sae_release, sae_id, feature, layer)
            return direction_set(layers=(layer,), method="sae_row")

        monkeypatch.setattr(
            "interp.steering.extract.extract_sae_row", fake_extract_sae_row
        )
        out_dir = str(tmp_path / "vec")
        args = parse(
            [
                "steer-extract",
                "--method",
                "sae_row",
                "--model",
                "m",
                "--sae-release",
                "gemma-scope",
                "--sae-id",
                "layer_5/width_16k",
                "--feature",
                "123",
                "--layer",
                "5",
                "--out",
                out_dir,
            ]
        )
        assert args.func(args) == 0
        # argparse types are plumbed through: feature/layer arrive as ints
        assert calls["args"] == ("m", "gemma-scope", "layer_5/width_16k", 123, 5)

        manifest = load_manifest(out_dir)
        assert manifest["method"] == "sae_row"
        assert manifest["layers"] == [5]

        captured = capsys.readouterr().out
        assert (
            "[RUN] Reading SAE decoder row: gemma-scope/layer_5/width_16k" in captured
        )
        assert "[OK] Layers: 1, width: 16" in captured

    def test_export_error_is_styled_failure(self, tmp_path, monkeypatch, capsys):
        # Real export refuses 0-based layers; the handler turns the
        # ExportError into exit 1 and writes no artifact.
        monkeypatch.setattr(
            "interp.steering.extract.extract_repeng",
            lambda *a, **k: direction_set(layers=(0, 1)),
        )
        pairs = tmp_path / "pairs.jsonl"
        pairs.write_text('{"positive": "a", "negative": "b"}\n')
        out_dir = str(tmp_path / "vec")
        args = parse(
            ["steer-extract", "--model", "m", "--pairs", str(pairs), "--out", out_dir]
        )
        assert args.func(args) == 1
        captured = capsys.readouterr().out
        assert "[ERROR] Extraction failed:" in captured
        assert "1-based" in captured
        assert not os.path.exists(os.path.join(out_dir, "vector.gguf"))

    def test_unwritable_out_dir_is_styled_failure(self, tmp_path, monkeypatch, capsys):
        # --out colliding with an existing regular file raises a real OSError
        # from makedirs, the third arm of the handler's except clause.
        monkeypatch.setattr(
            "interp.steering.extract.extract_repeng",
            lambda *a, **k: direction_set(),
        )
        pairs = tmp_path / "pairs.jsonl"
        pairs.write_text('{"positive": "a", "negative": "b"}\n')
        blocker = tmp_path / "occupied"
        blocker.write_text("not a directory\n")
        args = parse(
            [
                "steer-extract",
                "--model",
                "m",
                "--pairs",
                str(pairs),
                "--out",
                str(blocker),
            ]
        )
        assert args.func(args) == 1
        captured = capsys.readouterr().out
        assert "[ERROR] Extraction failed:" in captured


def fake_post_chat(chat_url, payload, timeout=120, origin=None):
    """Deterministic judge/subject stand-in for common.serving.post_chat."""
    content = payload["messages"][-1]["content"]
    if "Reply with JSON only" in content:
        text = '{"score": 7, "reason": "test"}'
    else:
        text = "a plain response"
    return {
        "choices": [{"message": {"content": text}}],
        "usage": {"completion_tokens": 5},
    }


class FakeController:
    """Drop-in for MakeServiceController: records calls, touches no Docker."""

    log: list[str] = []

    def __init__(self, make, vector_dir, service="llama-gpu", health_wait=None):
        self.make = make
        self.vector_dir = vector_dir
        self.service = service

    def steer(self, scale):
        FakeController.log.append(f"steer:{scale:g}")

    def unsteer(self):
        FakeController.log.append("unsteer")


class TestSteerScorecard:
    def _prompt_files(self, tmp_path):
        behavior = tmp_path / "behavior.jsonl"
        behavior.write_text(
            '{"prompt": "is the earth flat?"}\n{"prompt": "did you break it?"}\n'
        )
        neutral = tmp_path / "neutral.jsonl"
        neutral.write_text('{"prompt": "describe a tree"}\n')
        system = tmp_path / "system_prompt.json"
        system.write_text(
            json.dumps({"prompt": "be honest", "iterations": ["v1", "v2", "v3"]})
        )
        return str(behavior), str(neutral), str(system)

    def test_success_runs_real_orchestration(self, tmp_path, monkeypatch, capsys):
        # run_scorecard executes for real against a real exported vector;
        # only the service controller and the HTTP POST seam are faked.
        vdir = str(tmp_path / "vectors" / "v1")
        export_control_vector(direction_set(), vdir, model_hash="h1")
        behavior, neutral, system = self._prompt_files(tmp_path)
        FakeController.log = []
        monkeypatch.setattr(
            "interp.steering.scorecard.MakeServiceController", FakeController
        )
        monkeypatch.setattr("common.serving.post_chat", fake_post_chat)

        args = parse(
            [
                "steer-scorecard",
                "--vector",
                vdir,
                "--attribute",
                "honesty",
                "--behavior-prompts",
                behavior,
                "--neutral-prompts",
                neutral,
                "--system-prompt",
                system,
                "--scales",
                "1",
            ]
        )
        assert args.func(args) == 0

        captured = capsys.readouterr().out
        assert "=== Steering scorecard ===" in captured
        assert "[OK] baseline: behavior 7.0, fluency 7.0, latency" in captured
        assert "[OK] system_prompt: behavior 7.0" in captured
        assert "[OK] steered@1: behavior 7.0" in captured
        assert "[OK] Scorecard written:" in captured

        # W1 acceptance: the run ends with the service unsteered
        assert FakeController.log[-1] == "unsteer"
        assert "steer:1" in FakeController.log

        written = glob.glob(os.path.join(vdir, "scorecard-*", "scorecard.json"))
        assert len(written) == 1
        body = json.load(open(written[0]))
        assert [a["arm"] for a in body["arms"]] == [
            "baseline",
            "system_prompt",
            "steered@1",
        ]
        assert body["vector_manifest"]["model_hash"] == "h1"
        assert body["config"]["scales"] == [1.0]

    def test_held_gpu_lock_is_styled_failure(self, tmp_path, monkeypatch, capsys):
        # Fully real failure path: the D9 lock file exists, so the scorecard
        # refuses before touching any service or vector.
        lock = tmp_path / "held.lock"
        lock.touch()
        monkeypatch.setenv("TRAIN_LOCK_PATH", str(lock))
        args = parse(
            [
                "steer-scorecard",
                "--vector",
                str(tmp_path / "absent"),
                "--attribute",
                "honesty",
            ]
        )
        assert args.func(args) == 1
        captured = capsys.readouterr().out
        assert "[ERROR] Scorecard failed: GPU lock held" in captured
        assert "restored to its unsteered state" in captured

    def test_invalid_vector_dir_is_styled_failure(self, tmp_path, capsys):
        # WiringError arm: no vector.gguf under --vector; validation fails
        # before prompts are read or any controller is constructed.
        args = parse(
            [
                "steer-scorecard",
                "--vector",
                str(tmp_path / "absent"),
                "--attribute",
                "honesty",
            ]
        )
        assert args.func(args) == 1
        captured = capsys.readouterr().out
        assert "[ERROR] Scorecard failed:" in captured
        assert "vector file not found" in captured
