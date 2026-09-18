"""P4-D2 activation harvesting: config validation, preflight, manifest,
resume, retention. CPU-only: the SAELens runner is replaced by an injected
fake, disk usage by a synthetic reading, and the D9 lock pointed at tmp."""

from __future__ import annotations

import collections
import glob
import json
import os
import shutil

import pytest
import yaml
from omegaconf import OmegaConf

from interp.harvest import (
    ARROW_STORAGE_DTYPES,
    ARTIFACT,
    CHUNKS_SUFFIX,
    DEFAULT_CHUNK_TOKENS,
    DTYPE_BYTES,
    HarvestError,
    buffer_tokens,
    captured_tokens,
    check_capture,
    check_hook_dims,
    chunk_plan,
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


class FakeSession:
    """Stands in for _SaeLensSession: one hook, N chunks, no model.

    capture() writes a marker file per chunk and reports the tokens it was
    asked for, which is what a corpus larger than the budget would really
    yield. Tests that care about a short corpus set tokens_written.
    """

    def __init__(self, job, calls, tokens_written=None, fail_at=None):
        self.job = job
        self.calls = calls
        self.tokens_written = tokens_written
        self.fail_at = fail_at
        self.fast_forwarded = 0

    def capture(self, out_dir, chunk_tokens):
        # SAELens refuses a non-empty output directory before capturing
        # anything (KB-58), so the fake must too -- that refusal is the
        # behaviour chunking exists to work around, and a fake that quietly
        # overwrites would report a resume as working when it does not.
        if os.path.isdir(out_dir) and os.listdir(out_dir):
            raise RuntimeError(f"Activations directory ({out_dir}) is not empty")
        if self.fail_at is not None and len(self.calls) == self.fail_at:
            # A real crash lands mid-write: the runner has already created
            # out_dir and filled it with .tmp_shards. Failing cleanly instead
            # would leave the interrupted chunk's directory absent on resume,
            # which is precisely the case the refusal above fires on.
            os.makedirs(out_dir, exist_ok=True)
            with open(os.path.join(out_dir, ".tmp_shards"), "w") as f:
                f.write("partial")
            raise RuntimeError("capture crashed mid-chunk")
        self.calls.append((self.job["hook"], out_dir, chunk_tokens))
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "cache.bin"), "w") as f:
            f.write("shard")
        return chunk_tokens if self.tokens_written is None else self.tokens_written

    def fast_forward(self, tokens):
        self.fast_forwarded = tokens

    def info(self):
        return {"tokenizer_fingerprint": "fp-test", "sae_lens_version": "6.46.0"}


def fake_runner_factory(calls, tokens_written=None, fail_at=None, sessions=None):
    def fake_runner(job):
        session = FakeSession(job, calls, tokens_written, fail_at)
        if sessions is not None:
            sessions.append(session)
        return session

    return fake_runner


def fake_assemble(chunks_dir, out_dir):
    """Stand-in for _consolidate_shards: flatten chunk dirs, then remove them.

    Mirrors the two properties run_harvest depends on -- every chunk's content
    ends up under out_dir, and chunks_dir is gone afterwards -- without Arrow.
    """
    os.makedirs(out_dir, exist_ok=True)
    for index, shard in enumerate(sorted(os.listdir(chunks_dir))):
        for entry in sorted(os.listdir(os.path.join(chunks_dir, shard))):
            os.replace(
                os.path.join(chunks_dir, shard, entry),
                os.path.join(out_dir, f"data-{index:05d}-{entry}"),
            )
    shutil.rmtree(chunks_dir)


def hooks_called(calls):
    """Just the hook names from the (hook, out_dir, tokens) call log."""
    return [hook for hook, _, _ in calls]


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
            dtype="fp16",
        )
        single = make_recipe(
            tmp_path, total_tokens=300_000_000, d_model=1536, dtype="fp16"
        )
        # 310_640_640, not 300_000_000: at fp16 a buffer holds 647_168 tokens,
        # and each of the 20 chunks rounds up to 24 whole buffers. Pricing the
        # request instead would under-book 33GB on this capture.
        assert projected_bytes(single.harvest, 1536) == 310_640_640 * 1536 * 2
        assert projected_bytes(recipe.harvest, 1536) == 3 * 310_640_640 * 1536 * 2

    def test_projection_counts_whole_buffers_not_the_request(self, tmp_path):
        # The preflight is this tier's only defense against filling the volume,
        # so it must price what SAELens writes rather than what the recipe
        # asked for. SAELens writes whole buffers: the projection is always at
        # or above the request, never below.
        recipe = make_recipe(
            tmp_path,
            total_tokens=2_000_000,
            d_model=1536,
            dtype="fp32",
            chunk_tokens=700_000,
            context_len=1024,
            model_batch_size=8,
        )
        # The workstation measured exactly this on 2026-08-08 (KB-58).
        assert captured_tokens(recipe.harvest, 1536) == 2_875_392

    def test_a_narrower_dtype_buffers_more_tokens(self, tmp_path):
        # The buffer is a byte budget, not a sequence count, so halving the
        # element size doubles the tokens per buffer. A constant here would
        # mis-price every fp16 capture -- which is both reference captures.
        assert buffer_tokens(1536, "fp32", 8, 1024) == 319_488
        assert buffer_tokens(1536, "fp16", 8, 1024) == 647_168

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

    def test_sealed_organism_supplies_model_hash(self, tmp_path):
        # The seal records the hash the export manifest carried; the harvest
        # reads it back so the capture is identified by the same digest a
        # probe fitted to the organism will carry
        organism = tmp_path / "organism"
        organism.mkdir()
        write_manifest(
            str(organism),
            {"checkpoint": "/trained/organism-taboo/hf", "model_hash": "c" * 64},
        )
        recipe = make_recipe(tmp_path, model_id="", organism_dir=str(organism))
        model_id, revision, model_hash = resolve_subject(recipe.harvest)
        assert model_id == "/trained/organism-taboo/hf"
        assert revision is None
        assert model_hash == "c" * 64

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
            recipe,
            runner_fn=fake_runner_factory(calls),
            disk_usage_fn=plenty_of_disk,
            assemble_fn=fake_assemble,
        )
        assert hooks_called(calls) == ["model.layers.15"]
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
                assemble_fn=fake_assemble,
            )
        assert hooks_called(calls) == ["model.layers.15", "model.layers.22"]
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
                recipe,
                runner_fn=fake_runner_factory([]),
                disk_usage_fn=plenty_of_disk,
                assemble_fn=fake_assemble,
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
            # Matches the recipe so the corpus is the only thing that changed;
            # otherwise this asserts against the wrong key.
            "chunk_tokens": 0,
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
                recipe,
                runner_fn=fake_runner_factory([]),
                disk_usage_fn=plenty_of_disk,
                assemble_fn=fake_assemble,
            )

    def test_completed_capture_refuses_rerun(self, tmp_path):
        recipe = make_recipe(tmp_path)
        run_harvest(
            recipe,
            runner_fn=fake_runner_factory([]),
            disk_usage_fn=plenty_of_disk,
            assemble_fn=fake_assemble,
        )
        with pytest.raises(HarvestError, match="already complete"):
            run_harvest(
                recipe,
                runner_fn=fake_runner_factory([]),
                disk_usage_fn=plenty_of_disk,
                assemble_fn=fake_assemble,
            )

    def test_failed_runner_leaves_no_manifest(self, tmp_path):
        def broken_runner(job):
            raise RuntimeError("cuda oom")

        recipe = make_recipe(tmp_path)
        with pytest.raises(RuntimeError):
            run_harvest(
                recipe,
                runner_fn=broken_runner,
                disk_usage_fn=plenty_of_disk,
                assemble_fn=fake_assemble,
            )
        capture_dir = os.path.join(recipe.harvest.out_root, "harvest-test")
        with pytest.raises(ManifestError):
            load_manifest(capture_dir)


class TestChunkPlan:
    def test_budgets_sum_to_the_request(self):
        assert sum(chunk_plan(1000, 300, 10)) == 1000
        assert sum(chunk_plan(300_000_000, 15_000_000, 1024)) == 300_000_000

    def test_chunks_are_spread_evenly_not_filled_with_a_remainder(self):
        # 100 in chunks of 30 is four chunks of 25, not 30+30+30+10. A stub
        # last chunk below one sequence is refused by SAELens's own config
        # check, hours into a capture.
        assert chunk_plan(100, 30, 10) == [25, 25, 25, 25]

    def test_uneven_totals_differ_by_at_most_one_token(self):
        plan = chunk_plan(1001, 100, 10)
        assert sum(plan) == 1001
        assert max(plan) - min(plan) <= 1

    def test_zero_selects_the_default(self):
        assert (
            chunk_plan(4 * DEFAULT_CHUNK_TOKENS, 0, 1024) == [DEFAULT_CHUNK_TOKENS] * 4
        )

    def test_a_capture_smaller_than_one_chunk_is_one_chunk(self):
        assert chunk_plan(500, 15_000, 128) == [500]

    def test_chunk_size_never_falls_below_one_sequence(self):
        # A chunk shorter than context_len cannot hold a whole sequence.
        assert chunk_plan(4096, 10, 1024) == [1024] * 4


class TestChunkedCapture:
    def capture_dir(self, recipe):
        return os.path.join(recipe.harvest.out_root, "harvest-test")

    def test_one_capture_call_per_chunk_with_distinct_dirs(self, tmp_path):
        calls: list = []
        recipe = make_recipe(
            tmp_path, total_tokens=900, context_len=100, chunk_tokens=300
        )
        run_harvest(
            recipe,
            runner_fn=fake_runner_factory(calls),
            disk_usage_fn=plenty_of_disk,
            assemble_fn=fake_assemble,
        )
        assert len(calls) == 3
        assert [tokens for _, _, tokens in calls] == [300, 300, 300]
        out_dirs = [out_dir for _, out_dir, _ in calls]
        assert len(set(out_dirs)) == 3
        assert [os.path.basename(d) for d in out_dirs] == [
            "shard_00000",
            "shard_00001",
            "shard_00002",
        ]

    def test_assembly_runs_at_a_single_chunk(self, tmp_path):
        # The smoke path and the reference path must be the same code path:
        # skipping assembly at N=1 is how it would first run in anger 17
        # hours into a capture.
        assembled: list = []
        recipe = make_recipe(
            tmp_path, total_tokens=300, context_len=100, chunk_tokens=300
        )

        def recording_assemble(chunks_dir, out_dir):
            assembled.append((chunks_dir, out_dir))
            fake_assemble(chunks_dir, out_dir)

        run_harvest(
            recipe,
            runner_fn=fake_runner_factory([]),
            disk_usage_fn=plenty_of_disk,
            assemble_fn=recording_assemble,
        )
        assert len(assembled) == 1

    def test_assembly_leaves_the_path_the_sae_trainer_reads(self, tmp_path):
        # interp/sae/train.py joins <capture_dir>/<hook> independently of the
        # harvest, so the chunk staging directory must be gone and the hook
        # directory must hold the payload.
        recipe = make_recipe(
            tmp_path, total_tokens=600, context_len=100, chunk_tokens=200
        )
        run_harvest(
            recipe,
            runner_fn=fake_runner_factory([]),
            disk_usage_fn=plenty_of_disk,
            assemble_fn=fake_assemble,
        )
        hook_dir = os.path.join(self.capture_dir(recipe), "model.layers.15")
        assert os.path.isdir(hook_dir)
        assert len(os.listdir(hook_dir)) == 3
        assert not os.path.exists(hook_dir + ".chunks")

    def test_a_crash_loses_only_the_unfinished_chunk(self, tmp_path):
        calls: list = []
        recipe = make_recipe(
            tmp_path, total_tokens=1000, context_len=100, chunk_tokens=200
        )
        with pytest.raises(RuntimeError, match="crashed mid-chunk"):
            run_harvest(
                recipe,
                runner_fn=fake_runner_factory(calls, fail_at=3),
                disk_usage_fn=plenty_of_disk,
                assemble_fn=fake_assemble,
            )
        with open(os.path.join(self.capture_dir(recipe), "progress.json")) as f:
            progress = json.load(f)
        # Three chunks banked, the fourth lost, the fifth never attempted.
        assert progress["completed_chunks"] == {"model.layers.15": 3}
        assert progress["completed_hooks"] == []

    def test_resume_restarts_at_the_first_uncaptured_chunk(self, tmp_path):
        recipe = make_recipe(
            tmp_path, total_tokens=1000, context_len=100, chunk_tokens=200
        )
        crashed: list = []
        with pytest.raises(RuntimeError):
            run_harvest(
                recipe,
                runner_fn=fake_runner_factory(crashed, fail_at=3),
                disk_usage_fn=plenty_of_disk,
                assemble_fn=fake_assemble,
            )
        resumed: list = []
        sessions: list = []
        manifest = run_harvest(
            recipe,
            runner_fn=fake_runner_factory(resumed, sessions=sessions),
            disk_usage_fn=plenty_of_disk,
            assemble_fn=fake_assemble,
        )
        assert [os.path.basename(d) for _, d, _ in resumed] == [
            "shard_00003",
            "shard_00004",
        ]
        # The corpus stream is fresh in a new process, so the three banked
        # chunks' tokens must be skipped or the capture would duplicate them.
        assert sessions[0].fast_forwarded == 600
        assert manifest["chunks"] == 5

    def test_the_interrupted_chunks_debris_is_cleared_before_recapture(self, tmp_path):
        # The chunk a crash lands in has a directory full of .tmp_shards, and
        # that directory is the one the resume writes next. Leaving it in place
        # reproduces KB-58 one level down: SAELens refuses it and the resume
        # fails exactly where the original run did. Found on the workstation
        # (2026-08) after the happy path had already passed.
        recipe = make_recipe(
            tmp_path, total_tokens=600, context_len=100, chunk_tokens=200
        )
        with pytest.raises(RuntimeError):
            run_harvest(
                recipe,
                runner_fn=fake_runner_factory([], fail_at=1),
                disk_usage_fn=plenty_of_disk,
                assemble_fn=fake_assemble,
            )
        capture_dir = self.capture_dir(recipe)
        hook = "model.layers.15"
        debris = os.path.join(
            capture_dir, hook + CHUNKS_SUFFIX, "shard_00001", ".tmp_shards"
        )
        assert os.path.exists(debris), (
            "the fake must leave a crash the way SAELens does"
        )

        resumed: list = []
        run_harvest(
            recipe,
            runner_fn=fake_runner_factory(resumed),
            disk_usage_fn=plenty_of_disk,
            assemble_fn=fake_assemble,
        )
        assert [os.path.basename(d) for _, d, _ in resumed] == [
            "shard_00001",
            "shard_00002",
        ]
        # Cleared, not merged into: debris left beside the recaptured chunk
        # would be consolidated into the assembled capture.
        assert not os.path.exists(debris)

    def test_a_hook_needing_only_assembly_builds_no_session(self, tmp_path):
        # A crash between the last chunk and assembly leaves work that needs
        # no model. Building a session would cost a load and a fast_forward
        # over the entire corpus to capture nothing.
        recipe = make_recipe(
            tmp_path, total_tokens=600, context_len=100, chunk_tokens=200
        )

        def die_before_assembling(chunks_dir, out_dir):
            # Nothing created, nothing moved: the process died between the
            # last chunk's progress write and the start of assembly.
            raise RuntimeError("killed before assembly")

        with pytest.raises(RuntimeError, match="killed before assembly"):
            run_harvest(
                recipe,
                runner_fn=fake_runner_factory([]),
                disk_usage_fn=plenty_of_disk,
                assemble_fn=die_before_assembling,
            )
        sessions: list = []
        manifest = run_harvest(
            recipe,
            runner_fn=fake_runner_factory([], sessions=sessions),
            disk_usage_fn=plenty_of_disk,
            assemble_fn=fake_assemble,
        )
        assert sessions == []
        # The tokenizer and SAELens version were banked before the first
        # chunk, so the manifest can still name them without a session.
        assert manifest["sae_lens_version"] == "6.46.0"
        assert manifest["total_tokens"] == 600

    def test_resume_under_a_changed_chunk_size_is_refused(self, tmp_path):
        # Chunk size fixes where the boundaries fall, so resuming across a
        # change leaves a gap or an overlap at the seam.
        recipe = make_recipe(
            tmp_path, total_tokens=1000, context_len=100, chunk_tokens=200
        )
        with pytest.raises(RuntimeError):
            run_harvest(
                recipe,
                runner_fn=fake_runner_factory([], fail_at=2),
                disk_usage_fn=plenty_of_disk,
                assemble_fn=fake_assemble,
            )
        wider = make_recipe(
            tmp_path, total_tokens=1000, context_len=100, chunk_tokens=500
        )
        with pytest.raises(HarvestError, match="changed: chunk_tokens"):
            run_harvest(
                wider,
                runner_fn=fake_runner_factory([]),
                disk_usage_fn=plenty_of_disk,
                assemble_fn=fake_assemble,
            )

    def test_an_interrupted_assembly_refuses_rather_than_truncating(self, tmp_path):
        # Assembly moves files out of the staging directory, so an assembly
        # killed part-way leaves neither side whole. Continuing would publish
        # a truncated capture as complete.
        recipe = make_recipe(
            tmp_path, total_tokens=400, context_len=100, chunk_tokens=200
        )

        def assemble_then_die(chunks_dir, out_dir):
            fake_assemble(chunks_dir, out_dir)
            raise RuntimeError("killed during assembly")

        with pytest.raises(RuntimeError, match="killed during assembly"):
            run_harvest(
                recipe,
                runner_fn=fake_runner_factory([]),
                disk_usage_fn=plenty_of_disk,
                assemble_fn=assemble_then_die,
            )
        with pytest.raises(HarvestError, match="assembly was interrupted"):
            run_harvest(
                recipe,
                runner_fn=fake_runner_factory([]),
                disk_usage_fn=plenty_of_disk,
                assemble_fn=fake_assemble,
            )

    def test_manifest_records_measured_tokens_not_the_request(self, tmp_path):
        # SAELens starts a new epoch instead of stopping when the corpus runs
        # dry, and rows are lost to the sequence boundary, so the requested
        # budget is not evidence of what landed on disk.
        recipe = make_recipe(
            tmp_path, total_tokens=900, context_len=100, chunk_tokens=300
        )
        manifest = run_harvest(
            recipe,
            runner_fn=fake_runner_factory([], tokens_written=250),
            disk_usage_fn=plenty_of_disk,
            assemble_fn=fake_assemble,
        )
        assert manifest["total_tokens"] == 750
        assert manifest["requested_tokens"] == 900

    def test_measured_tokens_survive_a_resume(self, tmp_path):
        recipe = make_recipe(
            tmp_path, total_tokens=1000, context_len=100, chunk_tokens=200
        )
        with pytest.raises(RuntimeError):
            run_harvest(
                recipe,
                runner_fn=fake_runner_factory([], fail_at=3),
                disk_usage_fn=plenty_of_disk,
                assemble_fn=fake_assemble,
            )
        manifest = run_harvest(
            recipe,
            runner_fn=fake_runner_factory([]),
            disk_usage_fn=plenty_of_disk,
            assemble_fn=fake_assemble,
        )
        # Chunks banked before the crash count toward what the capture holds,
        # even though this run did not write them.
        assert manifest["total_tokens"] == 1000

    def test_preflight_prices_pending_chunks_not_whole_hooks(self, tmp_path):
        # A hook priced at 1000GB in five chunks of 200GB. With 350GB free
        # and a 100GB floor, a fresh capture cannot fit and must be refused,
        # but a run resuming at the last chunk needs only 200GB and fits.
        # Pricing the resumed run at a whole hook would refuse a capture that
        # has the space for what is left of it.
        recipe = make_recipe(
            tmp_path,
            total_tokens=1_000_000,
            context_len=100,
            chunk_tokens=200_000,
            d_model=(1000 * GB) // (1_000_000 * DTYPE_BYTES["bf16"]),
        )

        def tight_disk(_path):
            return Usage(total=2000 * GB, used=1650 * GB, free=350 * GB)

        with pytest.raises(HarvestError, match="disk preflight refused"):
            run_harvest(
                recipe,
                runner_fn=fake_runner_factory([]),
                disk_usage_fn=tight_disk,
                assemble_fn=fake_assemble,
            )
        # Bank four of the five chunks, then resume with one left to write.
        with pytest.raises(RuntimeError):
            run_harvest(
                recipe,
                runner_fn=fake_runner_factory([], fail_at=4),
                disk_usage_fn=plenty_of_disk,
                assemble_fn=fake_assemble,
            )
        calls: list = []
        run_harvest(
            recipe,
            runner_fn=fake_runner_factory(calls),
            disk_usage_fn=tight_disk,
            assemble_fn=fake_assemble,
        )
        assert [os.path.basename(d) for _, d, _ in calls] == ["shard_00004"]


class TestDownstreamGate:
    def test_check_capture_matches_and_refuses(self, tmp_path):
        recipe = make_recipe(tmp_path)
        run_harvest(
            recipe,
            runner_fn=fake_runner_factory([]),
            disk_usage_fn=plenty_of_disk,
            assemble_fn=fake_assemble,
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
            recipe,
            runner_fn=fake_runner_factory([]),
            disk_usage_fn=plenty_of_disk,
            assemble_fn=fake_assemble,
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
        # the manifest. Assembly renumbers each chunk's files as it flattens
        # them, so the payload is matched by prefix rather than by name.
        kept_files = []
        for _root_dir, _, files in os.walk(kept):
            kept_files.extend(files)
        assert any(f.startswith("data-") for f in kept_files)
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
    executable artifacts, including the Gemma 2 calibration anchor."""

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
            monkeypatch, SimpleNamespace(model_type="gemma2", hidden_size=2048)
        )
        d_model, spec, _ = resolve_model_meta("fake/gemma2", None)
        assert d_model == 2048
        assert spec.model_type == "gemma2"

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


class TestShippedRecipeStorageDtype:
    """Every shipped harvest recipe must be runnable, or say why it is not.

    KB-57: SAELens resolves harvest.dtype twice, once through DTYPE_MAP to a
    torch dtype and once as the datasets.Array2D storage type. Arrow has no
    bfloat16, so a bf16 recipe type-checks here, passes validate_harvest, and
    then dies inside the runner before the first shard. Every recipe shipped
    that way, which is why this guard exists rather than a unit test on a
    synthetic recipe.

    Two reference captures are still bf16 on purpose: fp32 does not fit the
    volume for either (see the recipes). Those are allowed through only while
    they name the bug, so a new bf16 recipe cannot join them silently.
    """

    def recipes(self):
        pattern = os.path.join(
            REPO_ROOT, "training", "recipes", "interp", "harvest-*.yaml"
        )
        paths = sorted(glob.glob(pattern))
        assert paths, "no harvest recipes found; the glob is wrong"
        return paths

    def test_every_recipe_is_arrow_storable_or_names_the_blocker(self):
        for path in self.recipes():
            with open(path, encoding="utf-8") as handle:
                text = handle.read()
            harvest = yaml.safe_load(text)["harvest"]
            name = os.path.relpath(path, REPO_ROOT)
            # Omitting dtype inherits the schema default, which is bf16 and so
            # cannot run. A harvest recipe must say what it captures at.
            assert "dtype" in harvest, (
                f"{name} omits harvest.dtype and would inherit the bf16 schema "
                f"default, which Arrow cannot store (KB-57). State it explicitly."
            )
            dtype = harvest["dtype"]
            assert dtype in DTYPE_BYTES, f"{name} sets an unknown dtype {dtype!r}"
            if dtype in ARROW_STORAGE_DTYPES:
                continue
            assert "KB-57" in text, (
                f"{name} captures at {dtype!r}, which Arrow cannot store, and "
                f"does not cite KB-57. Either move it to one of "
                f"{sorted(ARROW_STORAGE_DTYPES)} or record why it cannot."
            )

    def test_the_guard_would_catch_an_undocumented_bf16_recipe(self, tmp_path):
        """The mutation this guard exists to fail on."""
        text = "harvest:\n  dtype: bf16\n"
        loaded = yaml.safe_load(text)["harvest"]["dtype"]
        assert loaded not in ARROW_STORAGE_DTYPES
        assert "KB-57" not in text
