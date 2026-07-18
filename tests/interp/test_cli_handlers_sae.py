"""interp SAE-tier CLI handlers (interp/commands/{harvest,interp_gc,sae_train,
sae_eval}.py) driven through build_parser() with real tmp-dir pipeline
execution. The SAELens runner/trainer, the torch-loading SAE callable, and the
delta-LM-loss forward passes are the only fakes; recipe loading, validation,
the D9 lock, manifest chaining, retention, and the core-metric math all run
for real. Contracts checked: exit codes 0/1/2, styled [OK]/[ERROR]/[RUN]/
[INFO] lines, and concrete file side effects."""

from __future__ import annotations

import collections
import json
import os

import numpy as np
import pytest

import interp.harvest as harvest_mod
import interp.sae.evaluate as sae_eval_mod
import interp.sae.train as sae_train_mod
from interp import cli
from interp.harvest import ARTIFACT as HARVEST_ARTIFACT
from interp.manifest import load_manifest, write_manifest

D_MODEL = 8
Usage = collections.namedtuple("Usage", "total used free")
GB = 1 << 30


def run_cli(argv: list[str]) -> int:
    args = cli.build_parser().parse_args(argv)
    return args.func(args)


def plenty_of_disk(_path):
    # The sandbox tmpfs may hold less than the 100GB watchdog floor, so the
    # preflight reads a synthetic disk instead of the real one.
    return Usage(total=4000 * GB, used=100 * GB, free=3900 * GB)


def fake_runner(job, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "cache.bin"), "w") as f:
        f.write("shards")
    return {
        "tokens": job["total_tokens"],
        "tokenizer_fingerprint": "fp-test",
        "sae_lens_version": "6.46.0",
    }


def fake_trainer(job, out_dir, tracker=None):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "sae_weights.safetensors"), "w") as f:
        f.write("weights")
    return {
        "final_loss": 0.125,
        "measured_l0": float(job["k"]),
        "sae_lens_version": "6.46.0",
    }


def identity_sae(acts):
    """Perfect reconstruction, 2 active features per token."""
    features = np.zeros((acts.shape[0], 16), dtype=np.float32)
    features[:, :2] = 1.0
    return acts.copy(), features


def eval_activations(n: int = 50) -> np.ndarray:
    rng = np.random.default_rng(7)
    return rng.normal(size=(n, D_MODEL)).astype(np.float32)


@pytest.fixture(autouse=True)
def gemma4_autoconfig(monkeypatch):
    """run_harvest resolves the hook family from AutoConfig even under a
    d_model override; the sandbox has no transformers, so stub it."""
    import sys
    from types import SimpleNamespace

    config = SimpleNamespace(model_type="gemma4_text", hidden_size=8)
    stub = SimpleNamespace(
        AutoConfig=SimpleNamespace(from_pretrained=lambda *a, **k: config)
    )
    monkeypatch.setitem(sys.modules, "transformers", stub)


def patch_run_harvest(monkeypatch):
    """Route the handler's run_harvest call through the real pipeline with
    the SAELens runner and disk probe replaced (the only GPU/host seams)."""
    real = harvest_mod.run_harvest
    monkeypatch.setattr(
        "interp.harvest.run_harvest",
        lambda recipe: real(
            recipe, runner_fn=fake_runner, disk_usage_fn=plenty_of_disk
        ),
    )


def patch_run_sae_training(monkeypatch):
    """Same seam replacement for SAE training: the real lock/manifest/
    retention pipeline runs, only the SAELens trainer is faked."""
    real = sae_train_mod.run_sae_training
    monkeypatch.setattr(
        "interp.sae.train.run_sae_training",
        lambda recipe, tracker_factory=None: real(
            recipe, trainer_fn=fake_trainer, tracker_factory=tracker_factory
        ),
    )


def patch_evaluate_sae(monkeypatch, acts: np.ndarray):
    """Inject the numpy SAE and synthetic activations into the real
    evaluate_sae (loading real weights and shards needs torch/datasets)."""
    real = sae_eval_mod.evaluate_sae

    def wrapped(sae_dir, capture_dir, max_tokens=1_000_000, delta_loss=None):
        return real(
            sae_dir,
            capture_dir,
            max_tokens=max_tokens,
            sae_fn=identity_sae,
            activations=acts,
            delta_loss=delta_loss,
        )

    monkeypatch.setattr("interp.sae.evaluate.evaluate_sae", wrapped)


def write_harvest_recipe(tmp_path, layers="[15]", multi=False) -> str:
    path = tmp_path / "harvest.yaml"
    ack = ", acknowledge_multi_layer: true" if multi else ""
    path.write_text(
        "run: {name: harvest-test, seed: 42}\n"
        "watchdog: {enabled: false}\n"
        f"harvest: {{enabled: true, model_id: google/gemma-4-E2B, "
        f"hook_layers: {layers}, corpus: monology/pile-uncopyrighted, "
        f"total_tokens: 1000, d_model: 8, "
        f"out_root: {tmp_path / 'activations'}{ack}}}\n"
    )
    return str(path)


def write_sae_recipe(tmp_path, capture_dir: str) -> str:
    path = tmp_path / "sae.yaml"
    path.write_text(
        "run: {name: sae-test, seed: 42}\n"
        "watchdog: {enabled: false}\n"
        # tensorboard off so the real Tracker the handler passes through
        # builds zero sinks instead of warning about the absent torch
        "tracking: {tensorboard: false}\n"
        f"sae: {{enabled: true, capture_dir: {capture_dir}, width: 64, "
        f"k: 8, train_tokens: 1000, out_root: {tmp_path / 'saes'}}}\n"
    )
    return str(path)


def make_capture(tmp_path, name="capture", hooks=(15,), **overrides) -> str:
    """A completed P4-D2 capture: manifest plus a shard payload to delete."""
    capture_dir = str(tmp_path / name)
    os.makedirs(capture_dir, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "artifact": HARVEST_ARTIFACT,
        "model_id": "google/gemma-4-E2B",
        "model_hash": "hash-test",
        "hooks": [f"model.layers.{layer}" for layer in hooks],
        "d_model": D_MODEL,
        "corpus": "monology/pile-uncopyrighted",
        "context_len": 1024,
        "total_tokens": 1000,
        "tokenizer_fingerprint": "fp-test",
        "validated": False,
        "created": "2026-07-15T00:00:00Z",
    }
    manifest.update(overrides)
    write_manifest(capture_dir, manifest)
    shard_dir = os.path.join(capture_dir, "model.layers.15")
    os.makedirs(shard_dir, exist_ok=True)
    with open(os.path.join(shard_dir, "cache.bin"), "w") as f:
        f.write("shards")
    return capture_dir


def make_sae_run(tmp_path, name="sae-test", **overrides) -> str:
    sae_dir = str(tmp_path / "saes" / name)
    os.makedirs(sae_dir, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "artifact": "sae",
        "name": name,
        "model_id": "google/gemma-4-E2B",
        "model_hash": "hash-test",
        "hook": "model.layers.15",
        "d_in": D_MODEL,
        "width": 64,
    }
    manifest.update(overrides)
    write_manifest(sae_dir, manifest)
    return sae_dir


class TestHarvest:
    def test_success_applies_override_and_writes_manifest(
        self, tmp_path, monkeypatch, capsys
    ):
        patch_run_harvest(monkeypatch)
        recipe = write_harvest_recipe(tmp_path)
        assert (
            run_cli(
                [
                    "harvest",
                    "--recipe",
                    recipe,
                    "--override",
                    "harvest.total_tokens=2000",
                ]
            )
            == 0
        )
        out = capsys.readouterr().err
        assert "[RUN] Harvesting activations: google/gemma-4-E2B" in out
        assert (
            "[OK] Harvest complete: 2000 tokens/hook, hooks ['model.layers.15']"
        ) in out
        capture_dir = str(tmp_path / "activations" / "harvest-test")
        manifest = load_manifest(capture_dir)
        assert manifest["total_tokens"] == 2000  # dotlist override applied
        assert os.path.isfile(os.path.join(capture_dir, "model.layers.15", "cache.bin"))

    def test_check_hooks_reports_widths_without_capturing(
        self, tmp_path, monkeypatch, capsys
    ):
        monkeypatch.setattr(
            "interp.harvest.check_hook_dims",
            lambda model_id, layers, revision: dict.fromkeys(layers, 8),
        )
        recipe = write_harvest_recipe(tmp_path, layers="[15, 7]", multi=True)
        assert run_cli(["harvest", "--recipe", recipe, "--check-hooks"]) == 0
        out = capsys.readouterr().err
        assert "[RUN] Hook-dimension check: google/gemma-4-E2B" in out
        assert "[OK] layer 7: residual width 8" in out
        assert "[OK] layer 15: residual width 8" in out
        # layers report in sorted order regardless of recipe order
        assert out.index("layer 7:") < out.index("layer 15:")
        # gate only: no capture directory is created
        assert not os.path.exists(tmp_path / "activations")

    def test_validation_failure_is_styled_error(self, tmp_path, capsys):
        recipe = write_harvest_recipe(tmp_path)
        assert (
            run_cli(
                ["harvest", "--recipe", recipe, "--override", "harvest.enabled=false"]
            )
            == 1
        )
        out = capsys.readouterr().err
        assert "[ERROR] Harvest failed:" in out
        assert "harvest.enabled" in out

    def test_missing_recipe_file_is_styled_error(self, tmp_path, capsys):
        assert run_cli(["harvest", "--recipe", str(tmp_path / "missing.yaml")]) == 1
        out = capsys.readouterr().err
        assert "[ERROR] Harvest failed:" in out
        assert "not found" in out


class TestSaeTrain:
    def test_success_trains_and_chains_manifest(self, tmp_path, monkeypatch, capsys):
        patch_run_sae_training(monkeypatch)
        capture_dir = make_capture(tmp_path)
        recipe = write_sae_recipe(tmp_path, capture_dir)
        assert run_cli(["sae-train", "--recipe", recipe]) == 0
        out = capsys.readouterr().err
        assert f"[RUN] Training SAE from capture: {capture_dir}" in out
        assert (
            "[OK] SAE training complete: sae-test, width 64, k 8, final loss 0.125"
        ) in out
        sae_dir = str(tmp_path / "saes" / "sae-test")
        manifest = load_manifest(sae_dir)
        assert manifest["capture_dir"] == capture_dir
        assert os.path.isfile(os.path.join(sae_dir, "sae_weights.safetensors"))
        # without --validated the capture stays undeletable
        assert load_manifest(capture_dir)["validated"] is False

    def test_validated_flag_marks_capture_deletable(
        self, tmp_path, monkeypatch, capsys
    ):
        patch_run_sae_training(monkeypatch)
        capture_dir = make_capture(tmp_path)
        recipe = write_sae_recipe(tmp_path, capture_dir)
        assert run_cli(["sae-train", "--recipe", recipe, "--validated"]) == 0
        assert "[OK] SAE training complete" in capsys.readouterr().err
        assert load_manifest(capture_dir)["validated"] is True

    def test_validation_failure_is_styled_error(self, tmp_path, capsys):
        capture_dir = make_capture(tmp_path)
        recipe = write_sae_recipe(tmp_path, capture_dir)
        assert (
            run_cli(["sae-train", "--recipe", recipe, "--override", "sae.k=64"]) == 1
        )  # k must be < width
        out = capsys.readouterr().err
        assert "[ERROR] SAE training failed:" in out
        assert "sae.k" in out

    def test_missing_recipe_file_is_styled_error(self, tmp_path, capsys):
        assert run_cli(["sae-train", "--recipe", str(tmp_path / "missing.yaml")]) == 1
        out = capsys.readouterr().err
        assert "[ERROR] SAE training failed:" in out
        assert "not found" in out


class TestSaeEval:
    def test_core_metrics_written_and_reported(self, tmp_path, monkeypatch, capsys):
        patch_evaluate_sae(monkeypatch, eval_activations())
        sae_dir = make_sae_run(tmp_path)
        capture_dir = make_capture(tmp_path)
        assert (
            run_cli(["sae-eval", "--sae-dir", sae_dir, "--capture-dir", capture_dir])
            == 0
        )
        out = capsys.readouterr().err
        assert f"[RUN] Core metrics: {sae_dir}" in out
        assert (
            "[OK] Evaluation complete: explained variance 1.0, L0 2.0, 50 tokens"
        ) in out
        assert "Delta-LM-loss" not in out  # skipped without --delta-loss-texts
        with open(os.path.join(sae_dir, "eval_report.json")) as f:
            report = json.load(f)
        assert report["core"]["tokens_evaluated"] == 50
        assert report["delta_lm_loss"] is None

    def test_delta_loss_texts_run_and_land_in_report(
        self, tmp_path, monkeypatch, capsys
    ):
        patch_evaluate_sae(monkeypatch, eval_activations())
        monkeypatch.setattr(
            "interp.sae.evaluate._load_sae_fn", lambda sae_dir: identity_sae
        )
        seen = {}

        def fake_delta(model_id, hook, sae_fn, texts, revision=None):
            seen["revision"] = revision
            return {"delta_lm_loss": 0.0123, "loss_recovered": 0.9876}

        monkeypatch.setattr("interp.sae.evaluate.delta_lm_loss", fake_delta)
        sae_dir = make_sae_run(tmp_path, model_revision="rev-9")
        capture_dir = make_capture(tmp_path)
        texts = tmp_path / "texts.jsonl"
        # blank interior line proves the reader skips non-records
        texts.write_text('{"text": "one"}\n\n{"text": "two"}\n')
        assert (
            run_cli(
                [
                    "sae-eval",
                    "--sae-dir",
                    sae_dir,
                    "--capture-dir",
                    capture_dir,
                    "--delta-loss-texts",
                    str(texts),
                ]
            )
            == 0
        )
        out = capsys.readouterr().err
        assert "[RUN] Delta-LM-loss over 2 texts: google/gemma-4-E2B" in out
        assert "[OK] Delta-LM-loss 0.0123, loss recovered 0.9876" in out
        # the forward passes must load the snapshot the SAE was trained on
        assert seen["revision"] == "rev-9"
        with open(os.path.join(sae_dir, "eval_report.json")) as f:
            report = json.load(f)
        assert report["delta_lm_loss"]["delta_lm_loss"] == 0.0123

    def test_model_hash_mismatch_is_styled_error(self, tmp_path, capsys):
        # No patching: the refusal gate fires before any torch seam loads.
        sae_dir = make_sae_run(tmp_path)
        capture_dir = make_capture(tmp_path, model_hash="different-hash")
        assert (
            run_cli(["sae-eval", "--sae-dir", sae_dir, "--capture-dir", capture_dir])
            == 1
        )
        out = capsys.readouterr().err
        assert "[ERROR] SAE evaluation failed:" in out
        assert "mismatch" in out

    def test_missing_texts_file_is_styled_error(self, tmp_path, capsys):
        sae_dir = make_sae_run(tmp_path)
        capture_dir = make_capture(tmp_path)
        assert (
            run_cli(
                [
                    "sae-eval",
                    "--sae-dir",
                    sae_dir,
                    "--capture-dir",
                    capture_dir,
                    "--delta-loss-texts",
                    str(tmp_path / "missing.jsonl"),
                ]
            )
            == 1
        )
        assert "[ERROR] SAE evaluation failed:" in capsys.readouterr().err

    def test_malformed_texts_file_is_styled_error(self, tmp_path, capsys):
        # a record without 'text' and a non-JSON line each name path:lineno
        # instead of escaping as a raw KeyError/JSONDecodeError traceback
        sae_dir = make_sae_run(tmp_path)
        capture_dir = make_capture(tmp_path)
        texts = tmp_path / "texts.jsonl"
        texts.write_text('{"text": "ok"}\n{"content": "no text key"}\n')
        assert (
            run_cli(
                [
                    "sae-eval",
                    "--sae-dir",
                    sae_dir,
                    "--capture-dir",
                    capture_dir,
                    "--delta-loss-texts",
                    str(texts),
                ]
            )
            == 1
        )
        out = capsys.readouterr().err
        assert "[ERROR] SAE evaluation failed:" in out
        assert f"{texts}:2: needs a 'text' key" in out

    def test_empty_texts_file_is_styled_error(self, tmp_path, capsys):
        # an empty text list would fabricate an all-zero delta report
        sae_dir = make_sae_run(tmp_path)
        capture_dir = make_capture(tmp_path)
        texts = tmp_path / "texts.jsonl"
        texts.write_text("\n")
        assert (
            run_cli(
                [
                    "sae-eval",
                    "--sae-dir",
                    sae_dir,
                    "--capture-dir",
                    capture_dir,
                    "--delta-loss-texts",
                    str(texts),
                ]
            )
            == 1
        )
        out = capsys.readouterr().err
        assert "[ERROR] SAE evaluation failed:" in out
        assert "no texts found" in out


class TestGc:
    def test_mark_validated_then_collect_keeps_manifest(self, tmp_path, capsys):
        collected = make_capture(tmp_path, name="activations/collected")
        kept = make_capture(tmp_path, name="activations/kept")
        root = str(tmp_path / "activations")

        assert run_cli(["gc", "--mark-validated", collected]) == 0
        assert f"[OK] Marked deletable: {collected}" in capsys.readouterr().err
        assert load_manifest(collected)["validated"] is True

        assert run_cli(["gc", "--root", root]) == 0
        out = capsys.readouterr().err
        assert f"[OK] Deleted shards: {collected} (manifest kept)" in out
        assert os.listdir(collected) == ["manifest.json"]
        # the unvalidated capture keeps its shard payload
        assert os.path.isfile(os.path.join(kept, "model.layers.15", "cache.bin"))

    def test_dry_run_lists_without_deleting(self, tmp_path, capsys):
        capture = make_capture(tmp_path, name="activations/cap", validated=True)
        root = str(tmp_path / "activations")
        assert run_cli(["gc", "--root", root, "--dry-run"]) == 0
        out = capsys.readouterr().err
        assert f"[OK] Would delete shards: {capture} (manifest kept)" in out
        assert os.path.isfile(os.path.join(capture, "model.layers.15", "cache.bin"))

    def test_nothing_to_collect_is_info(self, tmp_path, capsys):
        make_capture(tmp_path, name="activations/unvalidated")
        root = str(tmp_path / "activations")
        assert run_cli(["gc", "--root", root]) == 0
        assert "[INFO] No validated captures to collect" in (capsys.readouterr().err)

    def test_missing_root_is_styled_error(self, tmp_path, capsys):
        assert run_cli(["gc", "--root", str(tmp_path / "nowhere")]) == 1
        assert "[ERROR] Garbage collection failed:" in (capsys.readouterr().err)

    def test_mark_validated_without_manifest_is_styled_error(self, tmp_path, capsys):
        empty = tmp_path / "empty"
        empty.mkdir()
        assert run_cli(["gc", "--mark-validated", str(empty)]) == 1
        out = capsys.readouterr().err
        assert "[ERROR] Garbage collection failed:" in out
        assert "manifest not found" in out


@pytest.mark.parametrize(
    "argv",
    [
        ["harvest"],
        ["sae-train"],
        ["sae-eval", "--sae-dir", "s"],
    ],
    ids=lambda argv: argv[0],
)
def test_missing_required_argument_is_usage_error(argv):
    with pytest.raises(SystemExit) as exc:
        cli.build_parser().parse_args(argv)
    assert exc.value.code == 2
