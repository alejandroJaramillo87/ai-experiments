"""holdout.resolve_model form dispatch: hub id, full checkpoint dir, and
adapter-only checkpoint. These are pure path helpers -- torch/transformers
import lazily inside load_model/evaluate_holdout -- so they run torch-free."""

import json
import os

from training.eval.holdout import resolve_model


def test_resolve_model_hub_id():
    # A path that does not exist locally is treated as a hub id; no adapter.
    assert resolve_model("org/does-not-exist-xyz") == (
        "org/does-not-exist-xyz", None)


def test_resolve_model_full_checkpoint_dir(tmp_path):
    # config.json at the root marks a full model directory; nothing to merge.
    ckpt = tmp_path / "step_100"
    ckpt.mkdir()
    (ckpt / "config.json").write_text(json.dumps({"model_type": "qwen3"}))
    assert resolve_model(str(ckpt)) == (str(ckpt), None)


def test_resolve_model_checkpoint_with_model_subdir(tmp_path):
    # A D6-style checkpoint nests weights under model/; resolve descends into it.
    ckpt = tmp_path / "step_200"
    (ckpt / "model").mkdir(parents=True)
    (ckpt / "model" / "config.json").write_text(
        json.dumps({"model_type": "qwen3"}))
    assert resolve_model(str(ckpt)) == (
        os.path.join(str(ckpt), "model"), None)


def test_resolve_model_adapter_dir(tmp_path):
    # adapter_config.json marks an adapter-only checkpoint: returns the recorded
    # base model to load plus the adapter dir to apply on top.
    adapter = tmp_path / "adapter_ckpt"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": "Qwen/Qwen3-1.7B-Base"}))
    assert resolve_model(str(adapter)) == (
        "Qwen/Qwen3-1.7B-Base", str(adapter))
