"""P4-D2 activation harvesting: config validation, preflight, manifest,
resume, retention. CPU-only: the SAELens runner is replaced by an injected
fake, disk usage by a synthetic reading, and the D9 lock pointed at tmp."""

from __future__ import annotations

import collections
import json
import os

import pytest
from omegaconf import OmegaConf

from interp.harvest import (
    ARTIFACT,
    HarvestError,
    check_capture,
    corpus_identity,
    disk_preflight,
    gc_captures,
    hook_name,
    mark_validated,
    projected_bytes,
    resolve_subject,
    run_harvest,
    validate_harvest,
)
from interp.manifest import ManifestError, load_manifest, write_manifest
from training.recipes.schema import Recipe

Usage = collections.namedtuple("Usage", "total used free")

GB = 1 << 30


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
            d_model=2304,
            hook_layers=[7, 15, 22],
            acknowledge_multi_layer=True,
        )
        single = make_recipe(tmp_path, total_tokens=300_000_000, d_model=2304)
        assert projected_bytes(single.harvest, 2304) == 300_000_000 * 2304 * 2
        assert projected_bytes(recipe.harvest, 2304) == 3 * 300_000_000 * 2304 * 2

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
                    "completed_hooks": [hook_name(7)],
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
        with mock.patch.object(harvest_mod.time, "monotonic",
                               lambda: next(ticks)):
            manifest = run_harvest(
                recipe, runner_fn=fake_runner_factory(calls),
                disk_usage_fn=plenty_of_disk
            )
        assert calls == [hook_name(15), hook_name(22)]
        assert manifest["hook_layers"] == [7, 15, 22]
        # Throughput reflects this run only: the wall clock spans the two
        # pending hooks, so counting the pre-completed layer-7 tokens would
        # inflate the recorded number 1.5x.
        pending_tokens = int(recipe.harvest.total_tokens) * 2
        assert manifest["tokens_per_second"] == round(
            pending_tokens / manifest["wall_clock_s"], 1)

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
        for root_dir, _, files in os.walk(kept):
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
