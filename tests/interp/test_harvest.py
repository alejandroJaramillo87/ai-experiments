"""P4-D2 activation harvesting: config validation, preflight, manifest,
resume, retention. CPU-only: the SAELens runner is replaced by an injected
fake, disk usage by a synthetic reading, and the D9 lock pointed at tmp."""

from __future__ import annotations

import collections
import glob
import json
import os

import pytest
from omegaconf import OmegaConf

from interp.harvest import (
    ARTIFACT,
    HarvestError,
    check_capture,
    check_hook_dims,
    corpus_identity,
    disk_preflight,
    gc_captures,
    mark_validated,
    projected_bytes,
    resolve_model_meta,
    resolve_subject,
    run_harvest,
    validate_harvest,
)
from interp.manifest import ManifestError, load_manifest, write_manifest
from tests.interp.conftest import REPO_ROOT
from training.recipes.loader import load_recipe
from training.recipes.schema import Recipe

Usage = collections.namedtuple("Usage", "total used free")

GB = 1 << 30


@pytest.fixture(autouse=True)
def gemma4_autoconfig(monkeypatch):
    """Family resolution reads AutoConfig even under a d_model override, so
    every run_harvest test gets a registered-family config stub. Tests that
    need a different config install their own stub over this one."""
    import sys
    from types import SimpleNamespace

    config = SimpleNamespace(model_type="gemma4_text", hidden_size=1536)
    stub = SimpleNamespace(
        AutoConfig=SimpleNamespace(from_pretrained=lambda *a, **k: config)
    )
    monkeypatch.setitem(sys.modules, "transformers", stub)


def make_recipe(tmp_path, **harvest_overrides):
    recipe = OmegaConf.structured(Recipe)
    recipe.run.name = "harvest-test"
    recipe.watchdog.enabled = False
    recipe.harvest.enabled = True
    recipe.harvest.model_id = "google/gemma-4-E2B"
    recipe.harvest.hook_layers = [15]
    recipe.harvest.corpus = "monology/pile-uncopyrighted"
    recipe.harvest.total_tokens = 1000
    recipe.harvest.d_model = 8
    recipe.harvest.out_root = str(tmp_path / "activations")
    for key, value in harvest_overrides.items():
        setattr(recipe.harvest, key, value)
    return recipe


def fake_runner_factory(calls):
    def fake_runner(job, out_dir):
        calls.append(job["hook"])
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "cache.bin"), "w") as f:
            f.write("shards")
        return {
            "tokens": job["total_tokens"],
            "tokenizer_fingerprint": "fp-test",
            "sae_lens_version": "6.46.0",
        }

    return fake_runner


def plenty_of_disk(_path):
    return Usage(total=4000 * GB, used=100 * GB, free=3900 * GB)


class TestValidation:
    def test_valid_recipe_passes(self, tmp_path):
        validate_harvest(make_recipe(tmp_path))

    def test_requires_exactly_one_subject(self, tmp_path):
        with pytest.raises(HarvestError, match="exactly one"):
            validate_harvest(make_recipe(tmp_path, organism_dir="/some/dir"))
        with pytest.raises(HarvestError, match="exactly one"):
            validate_harvest(make_recipe(tmp_path, model_id=""))

    def test_multi_layer_needs_acknowledgement(self, tmp_path):
        with pytest.raises(HarvestError, match="acknowledge_multi_layer"):
            validate_harvest(make_recipe(tmp_path, hook_layers=[7, 15, 22]))
        validate_harvest(
            make_recipe(tmp_path, hook_layers=[7, 15, 22], acknowledge_multi_layer=True)
        )

    def test_rejects_empty_layers_corpus_and_bad_dtype(self, tmp_path):
        with pytest.raises(HarvestError, match="hook_layers"):
            validate_harvest(make_recipe(tmp_path, hook_layers=[]))
        with pytest.raises(HarvestError, match="corpus"):
            validate_harvest(make_recipe(tmp_path, corpus=""))
        with pytest.raises(HarvestError, match="dtype"):
            validate_harvest(make_recipe(tmp_path, dtype="int8"))


class TestPreflight:
    def test_projected_bytes_scales_with_layers_and_dtype(self, tmp_path):
        recipe = make_recipe(
            tmp_path,
            total_tokens=300_000_000,
            d_model=1536,
            hook_layers=[7, 15, 22],
            acknowledge_multi_layer=True,
        )
        single = make_recipe(tmp_path, total_tokens=300_000_000, d_model=1536)
        assert projected_bytes(single.harvest, 1536) == 300_000_000 * 1536 * 2
        assert projected_bytes(recipe.harvest, 1536) == 3 * 300_000_000 * 1536 * 2

    def test_refuses_when_projection_breaches_floor(self, tmp_path):
        # 1.45TB free, 1.38TB projected leaves 70GB, under the 100GB floor
        def tight_disk(_path):
            return Usage(total=2000 * GB, used=550 * GB, free=1450 * GB)

        with pytest.raises(HarvestError, match="disk preflight refused"):
            disk_preflight("/anywhere", 1380 * GB, 100, disk_usage_fn=tight_disk)

    def test_passes_with_headroom(self):
        disk_preflight("/anywhere", 1380 * GB, 100, disk_usage_fn=plenty_of_disk)


class TestSubjectAndCorpus:
    def test_organism_manifest_supplies_checkpoint(self, tmp_path):
        organism = tmp_path / "organism"
        organism.mkdir()
        write_manifest(str(organism), {"checkpoint": "/ckpt/step_000000100"})
        recipe = make_recipe(tmp_path, model_id="", organism_dir=str(organism))
        model_id, revision, _ = resolve_subject(recipe.harvest)
        assert model_id == "/ckpt/step_000000100"
        assert revision is None

    def test_unsealed_organism_refused(self, tmp_path):
        organism = tmp_path / "organism"
        organism.mkdir()
        write_manifest(str(organism), {"kind": "false_fact"})
        recipe = make_recipe(tmp_path, model_id="", organism_dir=str(organism))
        with pytest.raises(HarvestError, match="no checkpoint"):
            resolve_subject(recipe.harvest)

    def test_local_corpus_is_content_hashed(self, tmp_path):
        corpus = tmp_path / "corpus.jsonl"
        corpus.write_text('{"text": "hello"}\n')
        path, digest = corpus_identity(str(corpus))
        assert path == str(corpus) and digest
        hf_id, no_digest = corpus_identity("monology/pile-uncopyrighted")
        assert hf_id == "monology/pile-uncopyrighted" and no_digest is None


class TestRunHarvest:
    def test_manifest_written_on_completion(self, tmp_path):
        calls: list[str] = []
        recipe = make_recipe(tmp_path)
        manifest = run_harvest(
            recipe, runner_fn=fake_runner_factory(calls), disk_usage_fn=plenty_of_disk
        )
        assert calls == ["model.layers.15"]
        assert manifest["artifact"] == ARTIFACT
        assert manifest["hooks"] == ["model.layers.15"]
        assert manifest["corpus"] == "monology/pile-uncopyrighted"
        assert manifest["seed"] == 42
        assert manifest["dtype"] == "bf16"
        assert manifest["sae_lens_version"] == "6.46.0"
        assert manifest["validated"] is False
        capture_dir = os.path.join(recipe.harvest.out_root, "harvest-test")
        assert load_manifest(capture_dir) == manifest
        assert not os.path.exists(os.path.join(capture_dir, "progress.json"))

    def test_resume_skips_completed_hooks(self, tmp_path):
        recipe = make_recipe(
            tmp_path, hook_layers=[7, 15, 22], acknowledge_multi_layer=True
        )
        capture_dir = os.path.join(recipe.harvest.out_root, "harvest-test")
        os.makedirs(capture_dir)
        # simulate a mid-harvest kill: layer 7 done, no manifest yet
        with open(os.path.join(capture_dir, "progress.json"), "w") as f:
            json.dump(
                {
                    "completed_hooks": ["model.layers.7"],
                    "stats": {
                        "tokenizer_fingerprint": "fp-test",
                        "sae_lens_version": "6.46.0",
                    },
                },
                f,
            )
        calls: list[str] = []
        import itertools
        from unittest import mock

        import interp.harvest as harvest_mod

        ticks = itertools.count(step=10.0)  # 10s between monotonic() calls
        with mock.patch.object(harvest_mod.time, "monotonic", lambda: next(ticks)):
            manifest = run_harvest(
                recipe,
                runner_fn=fake_runner_factory(calls),
                disk_usage_fn=plenty_of_disk,
            )
        assert calls == ["model.layers.15", "model.layers.22"]
        assert manifest["hook_layers"] == [7, 15, 22]
        # Throughput reflects this run only: the wall clock spans the two
        # pending hooks, so counting the pre-completed layer-7 tokens would
        # inflate the recorded number 1.5x.
        pending_tokens = int(recipe.harvest.total_tokens) * 2
        assert manifest["tokens_per_second"] == round(
            pending_tokens / manifest["wall_clock_s"], 1
        )

    def test_resume_with_changed_config_refused(self, tmp_path):
        # Hooks completed under a different capture config must not be
        # stitched into this run's manifest, which records one uniform
        # config for every shard.
        recipe = make_recipe(
            tmp_path, hook_layers=[7, 15, 22], acknowledge_multi_layer=True
        )
        capture_dir = os.path.join(recipe.harvest.out_root, "harvest-test")
        os.makedirs(capture_dir)
        with open(os.path.join(capture_dir, "progress.json"), "w") as f:
            json.dump(
                {
                    "completed_hooks": ["model.layers.7"],
                    "config": {"model_id": "someone/else", "total_tokens": 1},
                },
                f,
            )
        with pytest.raises(HarvestError, match="different harvest config"):
            run_harvest(
                recipe, runner_fn=fake_runner_factory([]), disk_usage_fn=plenty_of_disk
            )

    def test_resume_under_swapped_hf_corpus_refused(self, tmp_path):
        # Two HF dataset ids both carry corpus_xxh3=None (only local files
        # are content-hashed), so the identity must record the corpus id
        # itself: this prior was written under another dataset, and resuming
        # over it would stitch cross-corpus shards into one manifest.
        recipe = make_recipe(
            tmp_path, hook_layers=[7, 15], acknowledge_multi_layer=True
        )
        capture_dir = os.path.join(recipe.harvest.out_root, "harvest-test")
        os.makedirs(capture_dir)
        prior = {
            "model_id": "google/gemma-4-E2B",
            "model_revision": None,
            "corpus_xxh3": None,
            "context_len": 1024,
            "total_tokens": 1000,
            "dtype": "bf16",
            "seed": 42,
            "d_model": 8,
        }
        with open(os.path.join(capture_dir, "progress.json"), "w") as f:
            json.dump(
                {
                    "completed_hooks": ["model.layers.7"],
                    "config": prior,
                    "stats": {
                        "tokenizer_fingerprint": "fp-test",
                        "sae_lens_version": "6.46.0",
                    },
                },
                f,
            )
        with pytest.raises(HarvestError, match="changed: corpus"):
            run_harvest(
                recipe, runner_fn=fake_runner_factory([]), disk_usage_fn=plenty_of_disk
            )

    def test_completed_capture_refuses_rerun(self, tmp_path):
        recipe = make_recipe(tmp_path)
        run_harvest(
            recipe, runner_fn=fake_runner_factory([]), disk_usage_fn=plenty_of_disk
        )
        with pytest.raises(HarvestError, match="already complete"):
            run_harvest(
                recipe, runner_fn=fake_runner_factory([]), disk_usage_fn=plenty_of_disk
            )

    def test_failed_runner_leaves_no_manifest(self, tmp_path):
        def broken_runner(job, out_dir):
            raise RuntimeError("cuda oom")

        recipe = make_recipe(tmp_path)
        with pytest.raises(RuntimeError):
            run_harvest(recipe, runner_fn=broken_runner, disk_usage_fn=plenty_of_disk)
        capture_dir = os.path.join(recipe.harvest.out_root, "harvest-test")
        with pytest.raises(ManifestError):
            load_manifest(capture_dir)


class TestDownstreamGate:
    def test_check_capture_matches_and_refuses(self, tmp_path):
        recipe = make_recipe(tmp_path)
        run_harvest(
            recipe, runner_fn=fake_runner_factory([]), disk_usage_fn=plenty_of_disk
        )
        capture_dir = os.path.join(recipe.harvest.out_root, "harvest-test")
        check_capture(capture_dir, {"model_id": "google/gemma-4-E2B"})
        with pytest.raises(ManifestError, match="mismatch"):
            check_capture(capture_dir, {"model_id": "some/other-model"})

    def test_missing_manifest_refused(self, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        with pytest.raises(ManifestError, match="not found"):
            check_capture(str(empty), {})


class TestRetention:
    def make_capture(self, tmp_path, name, created):
        recipe = make_recipe(tmp_path, name=name)
        manifest = run_harvest(
            recipe, runner_fn=fake_runner_factory([]), disk_usage_fn=plenty_of_disk
        )
        capture_dir = os.path.join(recipe.harvest.out_root, name)
        manifest["created"] = created
        write_manifest(capture_dir, manifest)
        return capture_dir

    def test_gc_deletes_only_validated_keeps_manifest(self, tmp_path):
        kept = self.make_capture(tmp_path, "kept", "2026-07-14T00:00:00Z")
        collected = self.make_capture(tmp_path, "collected", "2026-07-13T00:00:00Z")
        mark_validated(collected)
        root = str(tmp_path / "activations")
        deleted = gc_captures(root)
        assert deleted == [collected]
        assert os.listdir(collected) == ["manifest.json"]
        # The kept capture is untouched: its shard payloads survive alongside
        # the manifest.
        kept_files = []
        for _root_dir, _, files in os.walk(kept):
            kept_files.extend(files)
        assert "cache.bin" in kept_files
        assert len(os.listdir(kept)) > 1

    def test_gc_oldest_first_and_dry_run(self, tmp_path):
        older = self.make_capture(tmp_path, "older", "2026-07-10T00:00:00Z")
        newer = self.make_capture(tmp_path, "newer", "2026-07-12T00:00:00Z")
        mark_validated(older)
        mark_validated(newer)
        root = str(tmp_path / "activations")
        would = gc_captures(root, dry_run=True)
        assert would == [older, newer]
        assert len(os.listdir(older)) > 1  # dry run deletes nothing


class TestCommittedHarvestRecipes:
    """Every committed harvest recipe loads and validates: the interp tier's
    executable artifacts, including the Qwen3 fallback calibration."""

    RECIPES = sorted(
        glob.glob(
            os.path.join(REPO_ROOT, "training", "recipes", "interp", "harvest-*.yaml")
        )
    )

    def test_recipes_exist(self):
        names = {os.path.basename(p) for p in self.RECIPES}
        assert {
            "harvest-smoke.yaml",
            "harvest-false-fact.yaml",
            "harvest-calibration-gemma2.yaml",
            "harvest-calibration-qwen3.yaml",
        } <= names

    @pytest.mark.parametrize("path", RECIPES, ids=os.path.basename)
    def test_recipe_loads_and_validates(self, path):
        recipe = load_recipe(path)
        validate_harvest(recipe)
        assert recipe.harvest.enabled


class TestResolveModelMeta:
    """The Gemma 4 composite-config contract: the residual stream is the
    text_config hidden_size, never the PLE width; the hook family resolves
    from the same config read (interp/families.py)."""

    def fake_transformers(self, monkeypatch, config):
        import sys
        from types import SimpleNamespace

        stub = SimpleNamespace(
            AutoConfig=SimpleNamespace(from_pretrained=lambda *a, **k: config)
        )
        monkeypatch.setitem(sys.modules, "transformers", stub)

    def test_descends_into_text_config(self, monkeypatch):
        from types import SimpleNamespace

        composite = SimpleNamespace(
            model_type="gemma4",
            text_config=SimpleNamespace(
                model_type="gemma4_text",
                hidden_size=1536,
                num_hidden_layers=35,
                hidden_size_per_layer_input=256,
            ),
        )
        self.fake_transformers(monkeypatch, composite)
        d_model, spec, depth = resolve_model_meta("fake/gemma-4", None)
        assert d_model == 1536
        assert depth == 35  # read after the same text-config unwrap
        assert spec.model_type == "gemma4_text"

    def test_flat_config_reads_hidden_size(self, monkeypatch):
        from types import SimpleNamespace

        self.fake_transformers(
            monkeypatch, SimpleNamespace(model_type="qwen3", hidden_size=2048)
        )
        d_model, spec, _ = resolve_model_meta("fake/qwen3", None)
        assert d_model == 2048
        assert spec.model_type == "qwen3"

    def test_missing_hidden_size_refuses(self, monkeypatch):
        from types import SimpleNamespace

        self.fake_transformers(monkeypatch, SimpleNamespace(model_type="gemma2"))
        with pytest.raises(HarvestError, match="d_model"):
            resolve_model_meta("fake/opaque", None)

    def test_override_skips_width_read_not_family(self, monkeypatch):
        # The override skips the width read; the family still resolves from
        # the config, so hook paths are registry data, never assumed.
        from types import SimpleNamespace

        self.fake_transformers(monkeypatch, SimpleNamespace(model_type="gemma2"))
        d_model, spec, _ = resolve_model_meta("fake/model", None, d_model_override=8)
        assert d_model == 8
        assert spec.model_type == "gemma2"

    def test_unknown_family_refuses(self, monkeypatch):
        from types import SimpleNamespace

        self.fake_transformers(
            monkeypatch, SimpleNamespace(model_type="mystery", hidden_size=64)
        )
        with pytest.raises(HarvestError, match="hook-point registry"):
            resolve_model_meta("fake/mystery", None)


def test_every_resolve_model_meta_caller_unpacks_the_whole_tuple():
    """Arity drift here fails at runtime, and one caller is GPU-gated.

    ``tests/hardware/test_gpu.py`` unpacks this result inside a
    ``try/except Exception: pytest.skip(...)``, so widening ModelMeta without
    updating it does not fail that test -- it silently skips it with a message
    blaming absent weights. Neither the sandbox nor a torch-free workstation
    run executes that file, so nothing else would notice.
    """
    import ast
    from pathlib import Path

    from interp.harvest import ModelMeta

    root = Path(REPO_ROOT)
    arity = len(ModelMeta._fields)
    offenders = []
    for path in sorted(root.rglob("*.py")):
        if "__pycache__" in path.parts or ".venv" in path.parts:
            continue
        try:
            tree = ast.parse(path.read_text())
        except (SyntaxError, UnicodeDecodeError):
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue
            call = node.value
            if not isinstance(call, ast.Call):
                continue
            name = call.func
            if getattr(name, "id", getattr(name, "attr", None)) != "resolve_model_meta":
                continue
            for target in node.targets:
                if isinstance(target, ast.Tuple) and len(target.elts) != arity:
                    offenders.append(
                        f"{path.relative_to(root)}:{node.lineno} unpacks "
                        f"{len(target.elts)} of {arity}"
                    )
    assert not offenders, f"ModelMeta has {arity} fields: {offenders}"


class TestHookLayerBounds:
    """The depth bound validate_harvest cannot apply: it sees only the recipe."""

    def test_index_past_the_last_layer_is_refused(self):
        from interp.harvest import check_hook_layer_bounds

        # The exact regression the recipes carried: layer 15 was justified as
        # "50 percent of a 30-layer stack", and E2B has 35 layers. An index
        # beyond the real depth must not survive to the capture.
        with pytest.raises(HarvestError, match=r"\[35\].*35 layers"):
            check_hook_layer_bounds([35], 35, "google/gemma-4-E2B")

    def test_last_valid_index_is_accepted(self):
        from interp.harvest import check_hook_layer_bounds

        check_hook_layer_bounds([34], 35, "google/gemma-4-E2B")

    def test_every_out_of_range_index_is_named_at_once(self):
        from interp.harvest import check_hook_layer_bounds

        with pytest.raises(HarvestError, match=r"\[40, 41\]"):
            check_hook_layer_bounds([17, 41, 40, 41], 35, "google/gemma-4-E2B")

    def test_unknown_depth_does_not_block(self):
        from interp.harvest import check_hook_layer_bounds

        # 0 means the config stated no depth. Refusing there would block a run
        # on a config shape the check cannot judge.
        check_hook_layer_bounds([999], 0, "fake/opaque")

    def test_e4b_accepts_a_layer_e2b_would_refuse(self):
        from interp.harvest import check_hook_layer_bounds

        check_hook_layer_bounds([38], 42, "google/gemma-4-E4B")
        with pytest.raises(HarvestError):
            check_hook_layer_bounds([38], 35, "google/gemma-4-E2B")


class TestHookDimGate:
    def test_missing_module_path_is_a_harvest_error(self, monkeypatch):
        """A checkpoint without the flat text-model layout surfaces as
        HarvestError (the documented Gemma 4 backstop), not a raw
        AttributeError."""
        import sys
        from types import SimpleNamespace

        class FakeModel:
            def get_submodule(self, path):
                raise AttributeError(path)

        stub_torch = SimpleNamespace(bfloat16="bf16")
        stub_transformers = SimpleNamespace(
            AutoModelForCausalLM=SimpleNamespace(
                from_pretrained=lambda *a, **k: FakeModel()
            ),
            AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **k: None),
            AutoConfig=SimpleNamespace(
                from_pretrained=lambda *a, **k: SimpleNamespace(
                    model_type="gemma4_text", hidden_size=8
                )
            ),
        )
        monkeypatch.setitem(sys.modules, "torch", stub_torch)
        monkeypatch.setitem(sys.modules, "transformers", stub_transformers)
        with pytest.raises(HarvestError, match="flat text-model layout"):
            check_hook_dims("fake/composite", [3])
