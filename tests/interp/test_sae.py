"""P4-D3 SAE training and evaluation: config validation, capture refusal
gate, manifest chaining, retention handshake, registry, core-metric math.
CPU-only: the SAELens trainer is replaced by an injected fake, the SAE by a
numpy callable, and the D9 lock pointed at tmp."""

from __future__ import annotations

import glob
import json
import os

import numpy as np
import pytest
from omegaconf import OmegaConf

from interp.harvest import ARTIFACT as HARVEST_ARTIFACT
from interp.manifest import ManifestError, load_manifest, write_manifest
from interp.sae.evaluate import (
    REPORT_NAME,
    SaeEvalError,
    core_metrics,
    evaluate_sae,
)
from interp.sae.registry import list_saes, load_sae_run
from interp.sae.train import (
    ARTIFACT,
    SaeTrainError,
    resolve_capture,
    run_sae_training,
    training_tokens,
    validate_sae,
)
from tests.interp.conftest import REPO_ROOT
from training.recipes.loader import load_recipe
from training.recipes.schema import Recipe

D_MODEL = 8


def make_capture(tmp_path, name="capture", hooks=(15,), **overrides):
    capture_dir = str(tmp_path / name)
    os.makedirs(capture_dir, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "artifact": HARVEST_ARTIFACT,
        "model_id": "google/gemma-4-E2B",
        "model_revision": None,
        "model_hash": "hash-test",
        "hooks": [f"model.layers.{layer}" for layer in hooks],
        "hook_layers": sorted(hooks),
        "d_model": D_MODEL,
        "corpus": "monology/pile-uncopyrighted",
        "corpus_xxh3": None,
        "context_len": 1024,
        "total_tokens": 1000,
        "dtype": "bf16",
        "seed": 42,
        "tokenizer_fingerprint": "fp-test",
        "sae_lens_version": "6.46.0",
        "validated": False,
        "git_sha": "sha-test",
        "created": "2026-07-15T00:00:00Z",
    }
    manifest.update(overrides)
    write_manifest(capture_dir, manifest)
    return capture_dir


def make_recipe(tmp_path, capture_dir, **sae_overrides):
    recipe = OmegaConf.structured(Recipe)
    recipe.run.name = "sae-test"
    recipe.run.seed = 42
    recipe.watchdog.enabled = False
    recipe.sae.enabled = True
    recipe.sae.capture_dir = capture_dir
    recipe.sae.width = 64
    recipe.sae.k = 8
    recipe.sae.train_tokens = 1000
    recipe.sae.out_root = str(tmp_path / "saes")
    for key, value in sae_overrides.items():
        setattr(recipe.sae, key, value)
    return recipe


def fake_trainer_factory(calls):
    def fake_trainer(job, out_dir, tracker=None):
        calls.append(job)
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "sae_weights.safetensors"), "w") as f:
            f.write("weights")
        return {"sae_lens_version": "6.46.0"}

    return fake_trainer


class TestValidation:
    def test_valid_recipe_passes(self, tmp_path):
        validate_sae(make_recipe(tmp_path, str(tmp_path / "capture")))

    def test_disabled_refused(self, tmp_path):
        recipe = make_recipe(tmp_path, str(tmp_path / "capture"), enabled=False)
        with pytest.raises(SaeTrainError, match="sae.enabled"):
            validate_sae(recipe)

    def test_requires_capture_dir(self, tmp_path):
        recipe = make_recipe(tmp_path, "")
        with pytest.raises(SaeTrainError, match="capture_dir"):
            validate_sae(recipe)

    def test_architecture_is_pinned(self, tmp_path):
        recipe = make_recipe(
            tmp_path, str(tmp_path / "capture"), architecture="jumprelu"
        )
        with pytest.raises(SaeTrainError, match="batchtopk"):
            validate_sae(recipe)

    def test_k_must_be_below_width(self, tmp_path):
        recipe = make_recipe(tmp_path, str(tmp_path / "capture"), k=64, width=64)
        with pytest.raises(SaeTrainError, match="sae.k"):
            validate_sae(recipe)

    def test_problems_are_listed_together(self, tmp_path):
        recipe = make_recipe(tmp_path, "", train_tokens=0, lr=0.0)
        with pytest.raises(SaeTrainError, match=r"3 problem"):
            validate_sae(recipe)


class TestCaptureGate:
    def test_missing_manifest_refused(self, tmp_path):
        recipe = make_recipe(tmp_path, str(tmp_path / "nowhere"))
        with pytest.raises(ManifestError):
            resolve_capture(recipe.sae)

    def test_model_id_mismatch_refused(self, tmp_path):
        capture_dir = make_capture(tmp_path)
        recipe = make_recipe(tmp_path, capture_dir, expect_model_id="Qwen/Qwen3-1.7B")
        with pytest.raises(ManifestError):
            resolve_capture(recipe.sae)

    def test_corpus_mismatch_refused(self, tmp_path):
        capture_dir = make_capture(tmp_path)
        recipe = make_recipe(tmp_path, capture_dir, expect_corpus="other/corpus")
        with pytest.raises(ManifestError):
            resolve_capture(recipe.sae)

    def test_sole_hook_resolves_implicitly(self, tmp_path):
        capture_dir = make_capture(tmp_path)
        recipe = make_recipe(tmp_path, capture_dir)
        _, hook = resolve_capture(recipe.sae)
        assert hook == "model.layers.15"

    def test_multi_hook_requires_explicit_choice(self, tmp_path):
        capture_dir = make_capture(tmp_path, hooks=(7, 15))
        recipe = make_recipe(tmp_path, capture_dir)
        with pytest.raises(SaeTrainError, match="set sae.hook"):
            resolve_capture(recipe.sae)
        recipe.sae.hook = "model.layers.7"
        _, hook = resolve_capture(recipe.sae)
        assert hook == "model.layers.7"

    def test_unknown_hook_refused(self, tmp_path):
        capture_dir = make_capture(tmp_path)
        recipe = make_recipe(tmp_path, capture_dir, hook="model.layers.3")
        with pytest.raises(SaeTrainError, match="not in capture hooks"):
            resolve_capture(recipe.sae)

    def test_train_tokens_capped_by_capture(self, tmp_path):
        capture_dir = make_capture(tmp_path, total_tokens=500)
        recipe = make_recipe(tmp_path, capture_dir, train_tokens=1000)
        capture_manifest = load_manifest(capture_dir)
        assert training_tokens(recipe.sae, capture_manifest) == 500


class TestTraining:
    def test_manifest_chains_to_capture(self, tmp_path):
        capture_dir = make_capture(tmp_path)
        recipe = make_recipe(tmp_path, capture_dir)
        calls = []
        manifest = run_sae_training(recipe, trainer_fn=fake_trainer_factory(calls))
        assert manifest["artifact"] == ARTIFACT
        assert manifest["capture_dir"] == capture_dir
        assert manifest["capture_manifest_xxhash"]
        assert manifest["model_id"] == "google/gemma-4-E2B"
        assert manifest["model_hash"] == "hash-test"
        assert manifest["hook"] == "model.layers.15"
        assert manifest["d_in"] == D_MODEL
        assert calls[0]["cached_activations_path"] == os.path.join(
            capture_dir, "model.layers.15"
        )

    def test_the_job_carries_the_captures_corpus(self, tmp_path):
        # SAELens refuses an empty dataset_path even when every activation is
        # read from disk (activations_store.py:133), so a cached-activations
        # run still needs a corpus. Taking it from the capture manifest rather
        # than the recipe is what stops an SAE probing a corpus the
        # activations did not come from. Nothing caught the omission until
        # sae-train first ran on the workstation, 2026-08-08.
        capture_dir = make_capture(tmp_path)
        recipe = make_recipe(tmp_path, capture_dir)
        calls = []
        run_sae_training(recipe, trainer_fn=fake_trainer_factory(calls))
        assert calls[0]["corpus"] == "monology/pile-uncopyrighted"

    def test_manifest_records_capture_revision(self, tmp_path):
        # The revision chains from the capture manifest so sae-eval's
        # delta-LM-loss forward passes load the same snapshot the
        # activations came from
        capture_dir = make_capture(tmp_path, model_revision="rev-9")
        recipe = make_recipe(tmp_path, capture_dir)
        manifest = run_sae_training(recipe, trainer_fn=fake_trainer_factory([]))
        assert manifest["model_revision"] == "rev-9"

    def test_completed_run_refuses_retrain(self, tmp_path):
        capture_dir = make_capture(tmp_path)
        recipe = make_recipe(tmp_path, capture_dir)
        run_sae_training(recipe, trainer_fn=fake_trainer_factory([]))
        with pytest.raises(SaeTrainError, match="already complete"):
            run_sae_training(recipe, trainer_fn=fake_trainer_factory([]))

    def test_failed_run_writes_no_manifest(self, tmp_path):
        capture_dir = make_capture(tmp_path)
        recipe = make_recipe(tmp_path, capture_dir)

        def broken_trainer(job, out_dir, tracker=None):
            raise RuntimeError("cuda exploded")

        with pytest.raises(RuntimeError):
            run_sae_training(recipe, trainer_fn=broken_trainer)
        sae_dir = os.path.join(recipe.sae.out_root, "sae-test")
        assert not os.path.isfile(os.path.join(sae_dir, "manifest.json"))

    def test_validated_handshake_flags_capture(self, tmp_path):
        capture_dir = make_capture(tmp_path)
        recipe = make_recipe(tmp_path, capture_dir, mark_capture_validated=True)
        run_sae_training(recipe, trainer_fn=fake_trainer_factory([]))
        assert load_manifest(capture_dir)["validated"] is True

    def test_without_flag_capture_stays_unvalidated(self, tmp_path):
        capture_dir = make_capture(tmp_path)
        recipe = make_recipe(tmp_path, capture_dir)
        run_sae_training(recipe, trainer_fn=fake_trainer_factory([]))
        assert load_manifest(capture_dir)["validated"] is False


class TestRegistry:
    def make_run(self, tmp_path):
        capture_dir = make_capture(tmp_path)
        recipe = make_recipe(tmp_path, capture_dir)
        run_sae_training(recipe, trainer_fn=fake_trainer_factory([]))
        return recipe.sae.out_root

    def test_lists_completed_runs(self, tmp_path):
        root = self.make_run(tmp_path)
        assert list_saes(root) == ["sae-test"]

    def test_load_returns_run(self, tmp_path):
        root = self.make_run(tmp_path)
        run = load_sae_run(os.path.join(root, "sae-test"))
        assert run.hook == "model.layers.15"
        assert run.d_in == D_MODEL
        assert run.width == 64

    def test_model_hash_mismatch_refused(self, tmp_path):
        root = self.make_run(tmp_path)
        with pytest.raises(ManifestError):
            load_sae_run(os.path.join(root, "sae-test"), expect_model_hash="other-hash")

    def test_missing_root_lists_empty(self, tmp_path):
        assert list_saes(str(tmp_path / "nowhere")) == []


def identity_sae(acts):
    """Perfect reconstruction, 2 active features per token."""
    features = np.zeros((acts.shape[0], 16), dtype=np.float32)
    features[:, :2] = 1.0
    return acts.copy(), features


def bad_sae(acts):
    """Mean-only reconstruction: explained variance 0."""
    recon = np.tile(acts.mean(axis=0), (acts.shape[0], 1))
    features = np.ones((acts.shape[0], 16), dtype=np.float32)
    return recon, features


class TestCoreMetrics:
    def test_perfect_reconstruction(self):
        rng = np.random.default_rng(0)
        acts = rng.normal(size=(100, D_MODEL)).astype(np.float32)
        metrics = core_metrics(acts, identity_sae)
        assert metrics["explained_variance"] == pytest.approx(1.0)
        assert metrics["l0"] == 2.0
        assert metrics["tokens_evaluated"] == 100

    def test_mean_reconstruction_scores_zero(self):
        rng = np.random.default_rng(0)
        acts = rng.normal(size=(100, D_MODEL)).astype(np.float32)
        metrics = core_metrics(acts, bad_sae)
        assert metrics["explained_variance"] == pytest.approx(0.0, abs=1e-5)
        assert metrics["l0"] == 16.0

    def test_shape_mismatch_refused(self):
        acts = np.ones((10, D_MODEL), dtype=np.float32)

        def wrong_shape(a):
            return a[:, :4], np.ones((10, 16))

        with pytest.raises(SaeEvalError, match="shape"):
            core_metrics(acts, wrong_shape)

    def test_degenerate_activations_refused(self):
        acts = np.ones((10, D_MODEL), dtype=np.float32)
        with pytest.raises(SaeEvalError, match="zero variance"):
            core_metrics(acts, identity_sae)


class TestEvaluate:
    def make_run(self, tmp_path):
        capture_dir = make_capture(tmp_path)
        recipe = make_recipe(tmp_path, capture_dir)
        run_sae_training(recipe, trainer_fn=fake_trainer_factory([]))
        return os.path.join(recipe.sae.out_root, "sae-test"), capture_dir

    def test_report_written_and_returned(self, tmp_path):
        sae_dir, capture_dir = self.make_run(tmp_path)
        rng = np.random.default_rng(1)
        acts = rng.normal(size=(50, D_MODEL)).astype(np.float32)
        report = evaluate_sae(
            sae_dir, capture_dir, sae_fn=identity_sae, activations=acts
        )
        assert report["artifact"] == "sae_eval"
        assert report["core"]["explained_variance"] == pytest.approx(1.0)
        on_disk = json.load(open(os.path.join(sae_dir, REPORT_NAME)))
        assert on_disk["core"] == report["core"]

    def test_capture_model_hash_mismatch_refused(self, tmp_path):
        sae_dir, _ = self.make_run(tmp_path)
        other_capture = make_capture(
            tmp_path, name="other", model_hash="different-hash"
        )
        with pytest.raises(ManifestError):
            evaluate_sae(
                sae_dir,
                other_capture,
                sae_fn=identity_sae,
                activations=np.ones((10, D_MODEL), dtype=np.float32),
            )

    def test_capture_model_id_mismatch_refused(self, tmp_path):
        # model_hash defaults to "unknown" on both sides for HF-model
        # captures, so the id gate must refuse on its own
        sae_dir, _ = self.make_run(tmp_path)
        other_capture = make_capture(tmp_path, name="other-id", model_id="other/model")
        with pytest.raises(ManifestError, match="model_id"):
            evaluate_sae(
                sae_dir,
                other_capture,
                sae_fn=identity_sae,
                activations=np.ones((10, D_MODEL), dtype=np.float32),
            )

    def test_hook_missing_from_capture_refused(self, tmp_path):
        sae_dir, _ = self.make_run(tmp_path)
        other_capture = make_capture(tmp_path, name="other-hooks", hooks=(7,))
        with pytest.raises(SaeEvalError, match="not in capture hooks"):
            evaluate_sae(
                sae_dir,
                other_capture,
                sae_fn=identity_sae,
                activations=np.ones((10, D_MODEL), dtype=np.float32),
            )


class TestCommittedSaeRecipes:
    """Every committed SAE recipe loads and validates: the interp tier's
    executable artifacts, including the Qwen3 fallback calibration."""

    RECIPES = sorted(
        glob.glob(
            os.path.join(REPO_ROOT, "training", "recipes", "interp", "sae-*.yaml")
        )
    )

    def test_recipes_exist(self):
        names = {os.path.basename(p) for p in self.RECIPES}
        assert {
            "sae-smoke.yaml",
            "sae-false-fact.yaml",
            "sae-calibration-gemma2.yaml",
            "sae-calibration-qwen3.yaml",
        } <= names

    @pytest.mark.parametrize("path", RECIPES, ids=os.path.basename)
    def test_recipe_loads_and_validates(self, path):
        recipe = load_recipe(path)
        validate_sae(recipe)
        assert recipe.sae.enabled


class TestWandbDirRedirect:
    """WANDB_DIR must be set before the first wandb.init, and restored after.

    SAETrainingRunner.run() calls wandb.init() with no dir argument, so wandb
    resolves the run directory from WANDB_DIR and then from os.getcwd(). In
    the interp container the CWD is /workspace -- the repository bind mount --
    so every unredirected run dropped an offline-run directory into the repo
    root. Our Tracker is unaffected (it passes dir= explicitly), which is why
    this went unnoticed: the leak came from the call site we do not own.

    Ordering is the whole fix. wandb freezes its settings when a session
    starts and ignores later env changes ("Changes to your wandb environment
    variables will be ignored because your wandb session has already
    started"), and the Tracker inits wandb before the trainer runs. A
    redirect scoped to the trainer alone is therefore set too late to have
    any effect -- which is exactly how the first attempt at this failed.
    """

    def test_wandb_dir_is_set_before_the_tracker_inits(self, tmp_path):
        # The Tracker is the first wandb.init in the process. If WANDB_DIR is
        # not already set when it is built, wandb has frozen its settings by
        # the time the trainer runs and every later set is ignored.
        capture_dir = make_capture(tmp_path)
        recipe = make_recipe(tmp_path, capture_dir)
        seen = {}

        def tracker_factory(tracking, name, run_dir, config):
            seen["at_tracker_init"] = os.environ.get("WANDB_DIR")
            return None

        run_sae_training(
            recipe,
            trainer_fn=fake_trainer_factory([]),
            tracker_factory=tracker_factory,
        )
        assert seen["at_tracker_init"] is not None, (
            "WANDB_DIR unset when the tracker started the wandb session; "
            "a later set is ignored and SAELens leaks a run dir into the CWD"
        )

    def test_wandb_dir_points_at_the_run_dir(self, tmp_path):
        capture_dir = make_capture(tmp_path)
        recipe = make_recipe(tmp_path, capture_dir)
        seen = {}

        def trainer(job, out_dir, tracker=None):
            seen["dir"] = os.environ.get("WANDB_DIR")
            seen["out_dir"] = out_dir
            return {"sae_lens_version": "6.46.0"}

        run_sae_training(recipe, trainer_fn=trainer)
        assert seen["dir"] == seen["out_dir"]

    def test_an_unset_wandb_dir_is_restored_to_unset(self, monkeypatch, tmp_path):
        # Not to the empty string: wandb reads a set-but-empty WANDB_DIR
        # differently from an absent one.
        capture_dir = make_capture(tmp_path)
        recipe = make_recipe(tmp_path, capture_dir)
        monkeypatch.delenv("WANDB_DIR", raising=False)
        run_sae_training(recipe, trainer_fn=fake_trainer_factory([]))
        assert "WANDB_DIR" not in os.environ

    def test_a_prior_wandb_dir_is_restored(self, monkeypatch, tmp_path):
        capture_dir = make_capture(tmp_path)
        recipe = make_recipe(tmp_path, capture_dir)
        monkeypatch.setenv("WANDB_DIR", "/prior")
        run_sae_training(recipe, trainer_fn=fake_trainer_factory([]))
        assert os.environ["WANDB_DIR"] == "/prior"

    def test_it_restores_when_training_raises(self, monkeypatch, tmp_path):
        # The env must not leak into the rest of the process on a failure.
        capture_dir = make_capture(tmp_path)
        recipe = make_recipe(tmp_path, capture_dir)
        monkeypatch.delenv("WANDB_DIR", raising=False)

        def broken(job, out_dir, tracker=None):
            raise RuntimeError("training failed")

        with pytest.raises(RuntimeError):
            run_sae_training(recipe, trainer_fn=broken)
        assert "WANDB_DIR" not in os.environ


class TestManifestCarriesNoFabricatedMetrics:
    """The SAE manifest records provenance, never an unmeasured quality claim.

    KB-62: `_saelens_trainer` read `getattr(sae, "final_metrics", {})` from
    the object SAETrainingRunner.run() returns. That object is the trained
    SAE, which has no such attribute -- the metrics live on the SAETrainer,
    which run() discards -- so the fallback fired on every real run. Every
    manifest recorded final_loss 0.0 and a measured_l0 equal to the
    *requested* k, which is worse than a zero: it is plausible, and
    indistinguishable from an SAE whose L0 genuinely landed on target.

    Quality belongs to eval_report.json, where L0 is counted against
    held-out activations (interp/sae/evaluate.py core_metrics).
    """

    def test_the_manifest_has_no_quality_fields(self, tmp_path):
        capture_dir = make_capture(tmp_path)
        recipe = make_recipe(tmp_path, capture_dir)
        manifest = run_sae_training(recipe, trainer_fn=fake_trainer_factory([]))
        assert "final_loss" not in manifest
        assert "measured_l0" not in manifest
        # Read it back off disk too: the returned dict and the written
        # artifact are separate objects, and the artifact is what a later
        # reader trusts.
        on_disk = load_manifest(str(tmp_path / "saes" / "sae-test"))
        assert "final_loss" not in on_disk
        assert "measured_l0" not in on_disk

    def test_no_manifest_field_echoes_the_requested_k(self, tmp_path):
        # The specific shape of KB-62: a field named for a measurement whose
        # value is really the recipe's target. k itself may equal k; nothing
        # else may.
        capture_dir = make_capture(tmp_path)
        # k=23 collides with nothing else in the fixture (d_in is D_MODEL=8,
        # width 64, seed 42), so any other field carrying 23 is carrying it
        # *because* it was copied from k rather than by coincidence.
        recipe = make_recipe(tmp_path, capture_dir, k=23)
        manifest = run_sae_training(recipe, trainer_fn=fake_trainer_factory([]))
        assert manifest["k"] == 23
        echoes = [
            key
            for key, value in manifest.items()
            if key != "k" and isinstance(value, (int, float)) and value == 23
        ]
        assert echoes == [], f"fields echoing the requested k: {echoes}"

    def test_the_manifest_points_at_the_metrics_owner(self, tmp_path):
        capture_dir = make_capture(tmp_path)
        recipe = make_recipe(tmp_path, capture_dir)
        manifest = run_sae_training(recipe, trainer_fn=fake_trainer_factory([]))
        assert REPORT_NAME in manifest["quality_metrics"]

    def test_a_trainer_returning_nothing_is_refused(self, tmp_path):
        # The second half of KB-62: `stats.get("sae_lens_version", "unknown")`
        # meant a trainer that returned nothing still produced a complete-
        # looking manifest. A field recording a fact gets no fallback.
        capture_dir = make_capture(tmp_path)
        recipe = make_recipe(tmp_path, capture_dir)

        def silent_trainer(job, out_dir, tracker=None):
            os.makedirs(out_dir, exist_ok=True)
            return {}

        with pytest.raises(ManifestError, match="sae_lens_version"):
            run_sae_training(recipe, trainer_fn=silent_trainer)

    def test_the_refusal_leaves_no_manifest_behind(self, tmp_path):
        # A run directory without manifest.json is incomplete by definition
        # (run_sae_training's contract), so the refusal must not write one.
        capture_dir = make_capture(tmp_path)
        recipe = make_recipe(tmp_path, capture_dir)

        def silent_trainer(job, out_dir, tracker=None):
            os.makedirs(out_dir, exist_ok=True)
            return {}

        with pytest.raises(ManifestError):
            run_sae_training(recipe, trainer_fn=silent_trainer)
        assert not os.path.isfile(str(tmp_path / "saes" / "sae-test" / "manifest.json"))
