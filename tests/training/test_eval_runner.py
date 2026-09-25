"""F7 runner pieces: checkpoint identity and the lm-eval task pins. The
container phases are host acceptance runs."""

import hashlib
import json
import os

import pytest

from common import serving as common_serving
from training.eval.lmeval import LmEvalError, load_task_config
from training.eval.runner import EvalRunError, hf_identity

EVAL_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "training", "eval")


def write_checkpoint(root, name, weights=b"weights", manifest_hash=None):
    """A checkpoint directory under a fake models root, optionally exported
    (an hf/ directory with the manifest beside it)."""
    if manifest_hash is None:
        directory = root / name
    else:
        directory = root / name / "hf"
    directory.mkdir(parents=True)
    (directory / "config.json").write_text(json.dumps({"model_type": "gemma4"}))
    (directory / "model.safetensors").write_bytes(weights)
    if manifest_hash is not None:
        (root / name / "manifest.json").write_text(
            json.dumps(
                {
                    "artifact": "hf_export",
                    "hf_dir": str(directory),
                    "model_hash": manifest_hash,
                }
            )
        )
    return str(directory)


def test_thinking_mode_is_family_aware():
    from training.eval.runner import thinking_mode_for

    # Gemma 4 thinking is opt-in; nothing in the eval pipeline opts in. The
    # composite config says gemma4, the text-only one gemma4_text.
    assert thinking_mode_for("gemma4") == "off"
    assert thinking_mode_for("gemma4_text") == "off"
    # An unregistered family is recorded as "not recorded", never guessed.
    assert thinking_mode_for("mystery") is None
    assert thinking_mode_for(None) is None


def test_hf_identity_reads_the_export_manifest(tmp_path, monkeypatch):
    monkeypatch.setattr(common_serving, "HOST_MODELS", str(tmp_path) + "/")
    path = write_checkpoint(tmp_path, "trained/tuned", manifest_hash="a" * 64)
    identity = hf_identity(path)
    assert identity["model_hash"] == "a" * 64
    assert identity["manifest"] == str(tmp_path / "trained" / "tuned" / "manifest.json")
    assert identity["thinking_mode"] == "off"
    assert identity["model"] == "/app/models/trained/tuned/hf"
    assert identity["host_path"] == path


def test_hf_identity_hashes_a_base_without_a_manifest(tmp_path, monkeypatch):
    monkeypatch.setattr(common_serving, "HOST_MODELS", str(tmp_path) + "/")
    path = write_checkpoint(tmp_path, "hf/base", weights=b"base weights")
    identity = hf_identity(path)
    assert identity["model_hash"] == hashlib.sha256(b"base weights").hexdigest()
    assert identity["manifest"] is None
    # The container spelling resolves to the same directory.
    assert hf_identity("/app/models/hf/base")["model_hash"] == identity["model_hash"]


def test_hf_identity_refuses_an_incomplete_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(common_serving, "HOST_MODELS", str(tmp_path) + "/")
    (tmp_path / "hf" / "partial").mkdir(parents=True)
    (tmp_path / "hf" / "partial" / "config.json").write_text("{}")
    with pytest.raises(EvalRunError, match="missing model.safetensors"):
        hf_identity(str(tmp_path / "hf" / "partial"))


def test_hf_identity_refuses_a_path_outside_the_models_tree(tmp_path, monkeypatch):
    monkeypatch.setattr(common_serving, "HOST_MODELS", str(tmp_path) + "/")
    with pytest.raises(EvalRunError, match="pass the exported hf directory"):
        hf_identity("google/gemma-4-E2B")
    with pytest.raises(EvalRunError, match="pass the exported hf directory"):
        hf_identity("/somewhere/else")


def test_task_config_pins_and_filters():
    config = load_task_config()
    assert config["lm_eval_version"] == "0.4.12"
    assert set(config["tasks"]) == {"ifeval", "gsm8k", "tinyMMLU_acc", "hellaswag"}
    subset = load_task_config(tasks=["tinyMMLU_acc"])
    assert list(subset["tasks"]) == ["tinyMMLU_acc"]
    with pytest.raises(LmEvalError, match="not in the pinned set"):
        load_task_config(tasks=["mmlu_pro"])
