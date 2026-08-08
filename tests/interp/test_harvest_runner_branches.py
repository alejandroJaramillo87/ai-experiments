"""P4-D2/P4-D3 runner-internal branches left uncovered by the main suites:
the check_hook_dims forward-hook path, the SAELens runner/trainer config
mapping (sae_lens injected via sys.modules), validation problems without a
dedicated test, the real-watchdog envelope around both runs, tracker wiring,
and gc_captures skip/loose-file branches. CPU-only: torch, transformers, and
sae_lens are stubbed at the import seam; the watchdog runs for real with a
sampling interval long enough that the thread never fires."""

from __future__ import annotations

import collections
import contextlib
import hashlib
import os
import shutil
import sys
import threading
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from interp.harvest import (
    ARTIFACT as HARVEST_ARTIFACT,
)
from interp.harvest import (
    HarvestError,
    _saelens_runner,
    check_hook_dims,
    gc_captures,
    run_harvest,
    validate_harvest,
)
from interp.manifest import load_manifest, write_manifest
from interp.sae.train import (
    ARTIFACT as SAE_ARTIFACT,
)
from interp.sae.train import (
    SaeTrainError,
    _saelens_trainer,
    run_sae_training,
    validate_sae,
)
from training.recipes.schema import Recipe

Usage = collections.namedtuple("Usage", "total used free")

GB = 1 << 30
D_MODEL = 8


def plenty_of_disk(_path):
    return Usage(total=4000 * GB, used=100 * GB, free=3900 * GB)


@pytest.fixture(autouse=True)
def gemma4_autoconfig(monkeypatch):
    """Family resolution reads AutoConfig even under a d_model override;
    tests installing their own transformers stub override this one."""
    config = SimpleNamespace(model_type="gemma4_text", hidden_size=D_MODEL)
    stub = SimpleNamespace(
        AutoConfig=SimpleNamespace(from_pretrained=lambda *a, **k: config)
    )
    monkeypatch.setitem(sys.modules, "transformers", stub)


def make_harvest_recipe(tmp_path, **harvest_overrides):
    recipe = OmegaConf.structured(Recipe)
    recipe.run.name = "harvest-test"
    recipe.run.seed = 42
    recipe.watchdog.enabled = False
    recipe.harvest.enabled = True
    recipe.harvest.model_id = "google/gemma-4-E2B"
    recipe.harvest.hook_layers = [15]
    recipe.harvest.corpus = "monology/pile-uncopyrighted"
    recipe.harvest.total_tokens = 1000
    recipe.harvest.d_model = D_MODEL
    recipe.harvest.out_root = str(tmp_path / "activations")
    for key, value in harvest_overrides.items():
        setattr(recipe.harvest, key, value)
    return recipe


def make_capture(tmp_path, name="capture"):
    capture_dir = str(tmp_path / name)
    os.makedirs(capture_dir, exist_ok=True)
    write_manifest(
        capture_dir,
        {
            "schema_version": 1,
            "artifact": HARVEST_ARTIFACT,
            "model_id": "google/gemma-4-E2B",
            "model_hash": "hash-test",
            "hooks": ["model.layers.15"],
            "hook_layers": [15],
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
        },
    )
    return capture_dir


def make_sae_recipe(tmp_path, capture_dir, **sae_overrides):
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


class FakeHarvestSession:
    """Minimal stand-in for _SaeLensSession; see tests/interp/test_harvest.py
    for the fake that the chunking tests drive in detail."""

    def __init__(self, job):
        self.job = job

    def capture(self, out_dir, chunk_tokens):
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "cache.bin"), "w") as f:
            f.write("shards")
        return chunk_tokens

    def fast_forward(self, tokens):
        pass

    def info(self):
        return {"tokenizer_fingerprint": "fp-test", "sae_lens_version": "6.46.0"}


def fake_harvest_runner(job):
    return FakeHarvestSession(job)


def fake_harvest_assemble(chunks_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for shard in sorted(os.listdir(chunks_dir)):
        for entry in sorted(os.listdir(os.path.join(chunks_dir, shard))):
            os.replace(
                os.path.join(chunks_dir, shard, entry),
                os.path.join(out_dir, f"{shard}-{entry}"),
            )
    shutil.rmtree(chunks_dir)


class TestValidationProblems:
    """Per-field refusals the main suites do not exercise."""

    @pytest.mark.parametrize(
        "overrides,match",
        [
            ({"enabled": False}, "harvest.enabled"),
            ({"hook_layers": [-1]}, "must be >= 0"),
            ({"total_tokens": 0}, "harvest.total_tokens"),
            ({"context_len": 0}, "harvest.context_len"),
        ],
    )
    def test_harvest_field_refused(self, tmp_path, overrides, match):
        with pytest.raises(HarvestError, match=match):
            validate_harvest(make_harvest_recipe(tmp_path, **overrides))

    @pytest.mark.parametrize(
        "overrides,match",
        [
            ({"width": 0}, "sae.width"),
            ({"train_batch_size_tokens": 0}, "train_batch_size_tokens"),
        ],
    )
    def test_sae_field_refused(self, tmp_path, overrides, match):
        with pytest.raises(SaeTrainError, match=match):
            validate_sae(make_sae_recipe(tmp_path, str(tmp_path / "c"), **overrides))


class FakeResidualModel:
    """Replays forward hooks the way torch would: each registered hook fires
    once per forward with the layer's output, whose trailing dim is the
    configured width for that module path."""

    device = "cpu"

    def __init__(self, widths: dict[str, int], tuple_output: bool = False):
        self.widths = widths
        self.tuple_output = tuple_output
        self.active: list[tuple] = []
        self.removed: list[str] = []

    def get_submodule(self, path):
        if path not in self.widths:
            raise AttributeError(path)
        return FakeLayer(self, path)

    def __call__(self, **inputs):
        for path, hook in list(self.active):
            tensor = SimpleNamespace(shape=(1, 2, self.widths[path]))
            output = (tensor, None) if self.tuple_output else tensor
            hook(path, inputs, output)


class FakeLayer:
    def __init__(self, model, path):
        self.model = model
        self.path = path

    def register_forward_hook(self, hook):
        entry = (self.path, hook)
        self.model.active.append(entry)
        return FakeHandle(self.model, entry)


class FakeHandle:
    def __init__(self, model, entry):
        self.model = model
        self.entry = entry

    def remove(self):
        self.model.active.remove(self.entry)
        self.model.removed.append(self.entry[0])


class FakePtTokenizer:
    def __call__(self, text, return_tensors=None):
        return self

    def to(self, device):
        return {"input_ids": [[1]]}


def install_fake_hf(monkeypatch, model, hidden_size=D_MODEL):
    """Stub torch and transformers at the import seam of check_hook_dims."""
    stub_torch = SimpleNamespace(bfloat16="bf16", no_grad=contextlib.nullcontext)
    stub_transformers = SimpleNamespace(
        AutoModelForCausalLM=SimpleNamespace(from_pretrained=lambda *a, **k: model),
        AutoTokenizer=SimpleNamespace(
            from_pretrained=lambda *a, **k: FakePtTokenizer()
        ),
        AutoConfig=SimpleNamespace(
            from_pretrained=lambda *a, **k: SimpleNamespace(
                model_type="gemma4_text", hidden_size=hidden_size
            )
        ),
    )
    monkeypatch.setitem(sys.modules, "torch", stub_torch)
    monkeypatch.setitem(sys.modules, "transformers", stub_transformers)


class TestHookDimCheck:
    """The Gemma 4 dimension gate: hooks observing hidden_size pass, a PLE
    width fails, and forward-hook handles are removed on both paths."""

    # Decoder layers return tuples on HF models; the gate must unwrap both.
    @pytest.mark.parametrize("tuple_output", [False, True])
    def test_matching_widths_return_observed(self, monkeypatch, tuple_output):
        model = FakeResidualModel(
            {"model.layers.3": D_MODEL, "model.layers.5": D_MODEL},
            tuple_output=tuple_output,
        )
        install_fake_hf(monkeypatch, model)
        assert check_hook_dims("fake/gemma-4", [3, 5]) == {3: D_MODEL, 5: D_MODEL}
        assert model.active == []
        assert sorted(model.removed) == ["model.layers.3", "model.layers.5"]

    def test_ple_width_mismatch_refused(self, monkeypatch):
        # 256 is the per-layer-embedding width the gate exists to catch.
        model = FakeResidualModel({"model.layers.3": 256})
        install_fake_hf(monkeypatch, model)
        with pytest.raises(HarvestError, match="not capturing the residual"):
            check_hook_dims("fake/gemma-4", [3])
        assert model.active == []


class FingerprintTokenizer:
    name_or_path = "fake/subject"
    chat_template = "tmpl"

    def __len__(self):
        return 32000


EXPECTED_FINGERPRINT = hashlib.sha256(b"fake/subject|32000|tmpl").hexdigest()


def pin_dist_version(monkeypatch, calls):
    def fake_version(name):
        calls.append(name)
        return "6.46.0-test"

    monkeypatch.setattr("importlib.metadata.version", fake_version)


class TestSaelensRunnerMapping:
    """The recipe-to-CacheActivationsRunnerConfig field mapping is pinned by
    the workbench smoke run; these tests freeze it against a stub sae_lens."""

    def install_stubs(self, monkeypatch, created, tokenizer_calls, rows=4):
        class FakeCacheConfig:
            def __init__(self, **kwargs):
                # Both a record of the construction call and live attributes:
                # a session reassigns training_tokens and
                # new_cached_activations_path between chunks, so the stub has
                # to behave like a settable dataclass and not just a dict.
                self.kwargs = dict(kwargs)
                for key, value in kwargs.items():
                    setattr(self, key, value)

        class FakeDataset:
            """Only len() matters: it is how a chunk's tokens are measured."""

            def __len__(self):
                return rows

        class FakeCacheRunner:
            def __init__(self, cfg):
                self.cfg = cfg
                self.activations_store = SimpleNamespace(
                    store_batch_size_prompts=cfg.model_batch_size,
                    get_batch_tokens=lambda: None,
                )
                created.append(cfg)

            def run(self):
                out = self.cfg.new_cached_activations_path
                os.makedirs(out, exist_ok=True)
                with open(os.path.join(out, "cache.bin"), "w") as f:
                    f.write("shards")
                return FakeDataset()

        def fake_tokenizer(*args, **kwargs):
            tokenizer_calls.append((args, kwargs))
            return FingerprintTokenizer()

        monkeypatch.setitem(
            sys.modules,
            "sae_lens",
            SimpleNamespace(
                CacheActivationsRunner=FakeCacheRunner,
                CacheActivationsRunnerConfig=FakeCacheConfig,
            ),
        )
        monkeypatch.setitem(
            sys.modules,
            "transformers",
            SimpleNamespace(
                AutoTokenizer=SimpleNamespace(from_pretrained=fake_tokenizer)
            ),
        )

    @pytest.mark.parametrize(
        "revision,expected_kwargs",
        [
            ("step-100", {"revision": "step-100"}),
            (None, {}),
        ],
    )
    def test_job_maps_to_runner_config(
        self, tmp_path, monkeypatch, revision, expected_kwargs
    ):
        created: list = []
        tokenizer_calls: list = []
        version_calls: list = []
        self.install_stubs(monkeypatch, created, tokenizer_calls)
        pin_dist_version(monkeypatch, version_calls)

        out_dir = str(tmp_path / "capture" / "model.layers.15")
        session = _saelens_runner(
            {
                "model_id": "fake/subject",
                "revision": revision,
                "hook": "model.layers.15",
                "d_in": D_MODEL,
                "corpus": "monology/pile-uncopyrighted",
                "context_len": 1024,
                "chunk_tokens": 1000,
                "dtype": "bf16",
                "seed": 42,
                "model_batch_size": 8,
            }
        )
        tokens = session.capture(out_dir, 1000)
        stats = session.info()

        cfg = created[0].kwargs
        assert cfg["model_name"] == "fake/subject"
        assert cfg["model_class_name"] == "AutoModelForCausalLM"
        assert cfg["hook_name"] == "model.layers.15"
        assert cfg["d_in"] == D_MODEL  # required by sae-lens 6.46.0
        assert cfg["dataset_path"] == "monology/pile-uncopyrighted"
        assert cfg["context_size"] == 1024
        # Sized per chunk, not per capture: every buffer count SAELens derives
        # reads this field, so it is what sizes a single run().
        assert cfg["training_tokens"] == 1000
        # Assigned per chunk by capture(), so construction leaves it unset.
        assert cfg["new_cached_activations_path"] is None
        assert created[0].new_cached_activations_path == out_dir
        assert cfg["model_batch_size"] == 8
        assert cfg["dtype"] == "bfloat16"
        assert cfg["seed"] == 42
        assert cfg["device"] == "cuda"
        assert cfg["model_from_pretrained_kwargs"] == expected_kwargs
        # The stub run() wrote shards into out_dir, proving it was invoked
        # against the hook directory, not some other path.
        assert os.path.isfile(os.path.join(out_dir, "cache.bin"))
        # Measured from the returned dataset (4 rows of context_len), not
        # echoed back from the requested budget of 1000.
        assert tokens == 4 * 1024
        assert stats == {
            "tokenizer_fingerprint": EXPECTED_FINGERPRINT,
            "sae_lens_version": "6.46.0-test",
        }
        assert tokenizer_calls[0][1].get("revision") == revision
        assert version_calls == ["sae-lens"]


class TestSaelensTrainerMapping:
    """The recipe-to-LanguageModelSAERunnerConfig field mapping.

    Stub-based, so it pins the mapping we wrote rather than the one SAELens
    requires -- the limitation KB-61 was filed against. Assertions that a
    stub cannot honestly make belong in test_saelens_contract.py, which runs
    against the real package.
    """

    JOB = {
        "model_id": "google/gemma-4-E2B",
        "hook": "model.layers.15",
        "d_in": D_MODEL,
        "width": 64,
        "k": 8,
        "lr": 3e-4,
        "train_tokens": 1000,
        "train_batch_size_tokens": 4096,
        "context_len": 1024,
        "seed": 42,
        "cached_activations_path": "/cache/model.layers.15",
        "dtype": "bf16",
        "corpus": "monology/pile-uncopyrighted",
    }

    def install_stubs(self, monkeypatch, created):
        class FakeSae:
            # Deliberately bare: the real TrainingSAE returned by
            # SAETrainingRunner.run() carries no metrics attribute, and a
            # fake that invents one is how KB-62 stayed invisible. It also
            # has no save_model call site any more -- see FakeTrainRunner.
            pass

        class FakeTrainConfig:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        class FakeTrainRunner:
            def __init__(self, cfg):
                created.append(cfg)

            def run(self):
                # The runner writes the checkpoint, not the caller: real
                # run() ends in save_final_sae(output_path) -> the SAE's
                # save_inference_model(). Writing it from the returned SAE
                # instead is KB-66, where our save_model(out_dir) landed
                # second and replaced the loadable inference checkpoint with
                # the training one.
                out_dir = self.cfg_output_path
                os.makedirs(out_dir, exist_ok=True)
                with open(os.path.join(out_dir, "sae_weights.safetensors"), "w") as f:
                    f.write("weights")
                return FakeSae()

            @property
            def cfg_output_path(self):
                return created[-1].kwargs["output_path"]

        class FakeSaeConfig:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        monkeypatch.setitem(
            sys.modules,
            "sae_lens",
            SimpleNamespace(
                BatchTopKTrainingSAEConfig=FakeSaeConfig,
                LanguageModelSAERunnerConfig=FakeTrainConfig,
                SAETrainingRunner=FakeTrainRunner,
            ),
        )

    def test_job_maps_to_trainer_config(self, tmp_path, monkeypatch):
        created: list = []
        version_calls: list = []
        self.install_stubs(monkeypatch, created)
        pin_dist_version(monkeypatch, version_calls)

        out_dir = str(tmp_path / "sae-run")
        stats = _saelens_trainer(dict(self.JOB), out_dir)

        cfg = created[0].kwargs
        # sae-lens 6.46.0 nests the SAE geometry in a typed sub-config; the
        # former top-level architecture/d_in/d_sae/activation_fn_kwargs
        # kwargs do not exist on the runner config.
        sae_cfg = cfg["sae"].kwargs
        assert sae_cfg == {"k": 8, "d_in": D_MODEL, "d_sae": 64}
        assert cfg["model_name"] == "google/gemma-4-E2B"
        assert cfg["hook_name"] == "model.layers.15"
        assert cfg["use_cached_activations"] is True
        assert cfg["cached_activations_path"] == "/cache/model.layers.15"
        # Mandatory even on the cached path: SAELens refuses an empty
        # dataset_path in ActivationsStore.from_config regardless of
        # use_cached_activations (KB-61). The stub below cannot enforce that,
        # which is why the omission survived to the workstation.
        assert cfg["dataset_path"] == "monology/pile-uncopyrighted"
        assert cfg["context_size"] == 1024
        assert cfg["training_tokens"] == 1000
        assert cfg["train_batch_size_tokens"] == 4096
        assert cfg["lr"] == 3e-4
        assert cfg["seed"] == 42
        assert cfg["dtype"] == "bfloat16"
        assert cfg["device"] == "cuda"
        assert cfg["checkpoint_path"] == out_dir
        assert os.path.isfile(os.path.join(out_dir, "sae_weights.safetensors"))
        # Provenance only. No quality metric crosses this seam: the trained
        # SAE carries none (KB-62), and a fallback here is what let the
        # manifest claim a measurement it never made.
        assert stats == {"sae_lens_version": "6.46.0-test"}
        assert version_calls == ["sae-lens"]

    def test_the_sae_trains_at_the_dtype_the_capture_was_harvested_at(
        self, tmp_path, monkeypatch
    ):
        # The trainer hardcoded bfloat16 while every other field came from the
        # capture manifest, so an fp32 harvest trained at bf16 with nothing
        # reporting the mismatch.
        created: list = []
        self.install_stubs(monkeypatch, created)
        pin_dist_version(monkeypatch, [])
        job = dict(self.JOB)
        job["dtype"] = "fp32"
        _saelens_trainer(job, str(tmp_path / "sae-run"))
        assert created[0].kwargs["dtype"] == "float32"


def sampler_threads():
    return [
        t
        for t in threading.enumerate()
        if t.name == "watchdog-sampler" and t.is_alive()
    ]


class TestWatchdogEnvelope:
    """Both runs start a real D10 watchdog when enabled and join its sampler
    thread on exit, success or failure. interval_s is long enough that the
    thread never takes a sample during the test."""

    def enable_watchdog(self, recipe):
        recipe.watchdog.enabled = True
        recipe.watchdog.interval_s = 3600.0
        return recipe

    def test_harvest_completes_under_watchdog(self, tmp_path):
        recipe = self.enable_watchdog(make_harvest_recipe(tmp_path))
        manifest = run_harvest(
            recipe,
            runner_fn=fake_harvest_runner,
            disk_usage_fn=plenty_of_disk,
            assemble_fn=fake_harvest_assemble,
        )
        capture_dir = os.path.join(recipe.harvest.out_root, "harvest-test")
        assert manifest["artifact"] == HARVEST_ARTIFACT
        assert load_manifest(capture_dir) == manifest
        assert sampler_threads() == []

    def test_harvest_failure_still_stops_watchdog(self, tmp_path):
        def broken_runner(job):
            raise RuntimeError("cuda oom")

        recipe = self.enable_watchdog(make_harvest_recipe(tmp_path))
        with pytest.raises(RuntimeError, match="cuda oom"):
            run_harvest(
                recipe,
                runner_fn=broken_runner,
                disk_usage_fn=plenty_of_disk,
                assemble_fn=fake_harvest_assemble,
            )
        assert sampler_threads() == []

    def test_sae_training_completes_under_watchdog(self, tmp_path):
        capture_dir = make_capture(tmp_path)
        recipe = self.enable_watchdog(make_sae_recipe(tmp_path, capture_dir))

        def fake_trainer(job, out_dir, tracker=None):
            return {"sae_lens_version": "6.46.0"}

        manifest = run_sae_training(recipe, trainer_fn=fake_trainer)
        sae_dir = os.path.join(recipe.sae.out_root, "sae-test")
        assert manifest["artifact"] == SAE_ARTIFACT
        assert load_manifest(sae_dir) == manifest
        assert sampler_threads() == []


class TestTrackerWiring:
    def test_tracker_factory_output_reaches_trainer(self, tmp_path):
        capture_dir = make_capture(tmp_path)
        recipe = make_sae_recipe(tmp_path, capture_dir)
        sentinel = object()
        factory_calls: list = []
        received: list = []

        def tracker_factory(tracking, name, sae_dir, job):
            factory_calls.append((name, sae_dir, job))
            return sentinel

        def fake_trainer(job, out_dir, tracker=None):
            received.append(tracker)
            return {"sae_lens_version": "6.46.0"}

        manifest = run_sae_training(
            recipe, trainer_fn=fake_trainer, tracker_factory=tracker_factory
        )
        assert received == [sentinel]
        name, sae_dir, job = factory_calls[0]
        assert name == "sae-test"
        assert sae_dir == os.path.join(recipe.sae.out_root, "sae-test")
        assert job["cached_activations_path"] == os.path.join(
            capture_dir, "model.layers.15"
        )
        assert manifest["artifact"] == SAE_ARTIFACT


class TestGcSkipsAndLooseFiles:
    def test_gc_ignores_non_captures_and_removes_loose_files(self, tmp_path):
        root = tmp_path / "activations"
        done = root / "done"
        (done / "model.layers.15").mkdir(parents=True)
        (done / "model.layers.15" / "cache.bin").write_text("shards")
        # A loose top-level file (e.g. alerts.jsonl) exercises the plain-file
        # deletion arm alongside the shard-directory rmtree arm.
        (done / "alerts.jsonl").write_text("{}\n")
        write_manifest(
            str(done),
            {
                "artifact": HARVEST_ARTIFACT,
                "validated": True,
                "created": "2026-07-01T00:00:00Z",
            },
        )
        (root / "README.txt").write_text("not a capture\n")
        incomplete = root / "incomplete"
        incomplete.mkdir()
        (incomplete / "progress.json").write_text("{}")

        deleted = gc_captures(str(root))
        assert deleted == [str(done)]
        assert os.listdir(done) == ["manifest.json"]
        # A stray file and an in-progress capture (no manifest) are untouched.
        assert (root / "README.txt").is_file()
        assert (incomplete / "progress.json").is_file()
