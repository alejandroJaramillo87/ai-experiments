"""Recipe system tests (Phase 0 D5): typed loading, override handling, snapshot
round trip, and one test per validate() cross-field invariant. CPU-only; the
tokenizer-hash test reuses the D2 fixtures from conftest.py."""

from __future__ import annotations

import datetime
import glob
import os

import pytest
from omegaconf import OmegaConf
from support import REFERENCE_DIR

from training.recipes import (
    RecipeLoadError,
    RecipeValidationError,
    create_run_dir,
    load_recipe,
    load_snapshot,
    register_schema,
    resolved_yaml,
    validate,
    write_snapshot,
)

REFERENCE_RECIPES = sorted(glob.glob(os.path.join(REFERENCE_DIR, "*.yaml")))


def write_yaml(tmp_path, text: str) -> str:
    path = tmp_path / "recipe.yaml"
    path.write_text(text)
    return str(path)


@pytest.fixture()
def tiny_recipe():
    return load_recipe(os.path.join(REFERENCE_DIR, "tiny-e2e.yaml"))


class TestLoading:
    def test_reference_recipes_exist(self):
        names = {os.path.basename(p) for p in REFERENCE_RECIPES}
        assert {
            "pretrain-150m.yaml",
            "pretrain-1b.yaml",
            "tiny-e2e.yaml",
            "sft-qlora-e2b.yaml",
            "sft-qlora-gemma-e4b.yaml",
            "distill-response-gemma-e2b.yaml",
            "distill-logit-gemma-e2b.yaml",
            "continual-gemma4-e2b.yaml",
            # Qwen3 fallback twins (model policy: every Gemma 4 reference
            # recipe pairs with a Qwen3 twin)
            "sft-qlora-qwen3.yaml",
            "sft-qlora-qwen3-4b.yaml",
            "distill-qwen3.yaml",
            "distill-response-qwen3.yaml",
            "distill-logit-qwen3.yaml",
            "continual-qwen3.yaml",
        } <= names

    @pytest.mark.parametrize(
        "path", REFERENCE_RECIPES, ids=[os.path.basename(p) for p in REFERENCE_RECIPES]
    )
    def test_reference_recipe_loads_typed_and_valid(self, path):
        recipe = load_recipe(path)
        validate(recipe)
        assert isinstance(recipe.optimizer.lr, float)
        assert isinstance(recipe.run.seed, int)

    def test_missing_file_fails(self):
        with pytest.raises(RecipeLoadError, match="not found"):
            load_recipe("/nonexistent/recipe.yaml")

    def test_misspelled_key_fails(self, tmp_path):
        path = write_yaml(tmp_path, "optimzer:\n  lr: 1.0e-4\n")
        with pytest.raises(RecipeLoadError, match="optimzer"):
            load_recipe(path)

    def test_misspelled_nested_key_fails(self, tmp_path):
        path = write_yaml(tmp_path, "optimizer:\n  learning_rate: 1.0e-4\n")
        with pytest.raises(RecipeLoadError, match="learning_rate"):
            load_recipe(path)

    def test_wrong_type_fails(self, tmp_path):
        path = write_yaml(tmp_path, "run:\n  seed: not-an-int\n")
        with pytest.raises(RecipeLoadError, match="seed"):
            load_recipe(path)

    def test_register_schema_idempotent(self):
        register_schema()
        register_schema()


class TestOverrides:
    def test_dotlist_override(self):
        recipe = load_recipe(
            os.path.join(REFERENCE_DIR, "tiny-e2e.yaml"),
            ["optimizer.lr=1e-4", "data.micro_batch=8"],
        )
        assert recipe.optimizer.lr == 1e-4
        assert recipe.data.micro_batch == 8

    def test_override_unknown_key_fails(self):
        with pytest.raises(RecipeLoadError):
            load_recipe(
                os.path.join(REFERENCE_DIR, "tiny-e2e.yaml"),
                ["optimizer.learning_rate=1e-4"],
            )

    def test_override_wrong_type_fails(self):
        with pytest.raises(RecipeLoadError):
            load_recipe(
                os.path.join(REFERENCE_DIR, "tiny-e2e.yaml"), ["data.micro_batch=huge"]
            )


class TestSnapshot:
    def test_snapshot_reloads_bit_identically(self, tiny_recipe, tmp_path):
        path = write_snapshot(tiny_recipe, str(tmp_path))
        reloaded = load_snapshot(path)
        assert resolved_yaml(reloaded) == resolved_yaml(tiny_recipe)
        assert OmegaConf.to_container(reloaded) == OmegaConf.to_container(tiny_recipe)

    def test_create_run_dir_convention(self, tiny_recipe, tmp_path):
        recipe = load_recipe(
            os.path.join(REFERENCE_DIR, "tiny-e2e.yaml"),
            [f"run.output_root={tmp_path}"],
        )
        stamp = datetime.datetime(2026, 7, 14, 12, 0, 0)
        run_dir = create_run_dir(recipe, now=stamp)
        assert run_dir == os.path.join(str(tmp_path), "tiny-e2e", "2026-07-14_12-00-00")
        snapshot = os.path.join(run_dir, "config.yaml")
        assert os.path.isfile(snapshot)
        assert resolved_yaml(load_snapshot(snapshot)) == resolved_yaml(recipe)

    def test_create_run_dir_refuses_duplicate(self, tmp_path):
        recipe = load_recipe(
            os.path.join(REFERENCE_DIR, "tiny-e2e.yaml"),
            [f"run.output_root={tmp_path}"],
        )
        stamp = datetime.datetime(2026, 7, 14, 12, 0, 0)
        create_run_dir(recipe, now=stamp)
        with pytest.raises(FileExistsError):
            create_run_dir(recipe, now=stamp)


def assert_problem(recipe, fragment: str) -> None:
    with pytest.raises(RecipeValidationError) as excinfo:
        validate(recipe)
    assert any(fragment in p for p in excinfo.value.problems), (
        f"no problem containing {fragment!r} in {excinfo.value.problems}"
    )


class TestValidate:
    def test_missing_required_field_reported(self, tmp_path):
        path = write_yaml(tmp_path, "run:\n  name: incomplete\n")
        assert_problem(load_recipe(path), "required field is not set")

    def test_model_requires_exactly_one_variant(self, tiny_recipe):
        recipe = load_recipe(
            os.path.join(REFERENCE_DIR, "tiny-e2e.yaml"), ["model.hf.id=some/model"]
        )
        assert_problem(recipe, "exactly one of model.hf or model.scratch")

    def test_model_neither_variant(self, tmp_path):
        path = write_yaml(tmp_path, "run:\n  name: x\n")
        assert_problem(load_recipe(path), "exactly one of model.hf or model.scratch")

    def test_scratch_gqa_heads_divisible(self, tiny_recipe):
        recipe = load_recipe(
            os.path.join(REFERENCE_DIR, "tiny-e2e.yaml"), ["model.scratch.kv_heads=3"]
        )
        assert_problem(recipe, "multiple of")

    def test_unknown_optimizer_name(self, tiny_recipe):
        recipe = load_recipe(
            os.path.join(REFERENCE_DIR, "tiny-e2e.yaml"), ["optimizer.name=sgd"]
        )
        assert_problem(recipe, "optimizer.name")

    def test_nonpositive_lr(self):
        recipe = load_recipe(
            os.path.join(REFERENCE_DIR, "tiny-e2e.yaml"), ["optimizer.lr=0"]
        )
        assert_problem(recipe, "optimizer.lr")

    def test_cpu_adam_requires_offload(self):
        recipe = load_recipe(
            os.path.join(REFERENCE_DIR, "tiny-e2e.yaml"), ["optimizer.name=cpu_adam"]
        )
        assert_problem(recipe, "cpu_adam requires memory.offload")
        ok = load_recipe(
            os.path.join(REFERENCE_DIR, "tiny-e2e.yaml"),
            ["optimizer.name=cpu_adam", "memory.offload=zero_offload"],
        )
        validate(ok)

    def test_unknown_schedule_type(self):
        recipe = load_recipe(
            os.path.join(REFERENCE_DIR, "tiny-e2e.yaml"), ["schedule.type=linear"]
        )
        assert_problem(recipe, "schedule.type")

    def test_warmup_exceeds_total_steps(self):
        recipe = load_recipe(
            os.path.join(REFERENCE_DIR, "tiny-e2e.yaml"), ["schedule.warmup_steps=100"]
        )
        assert_problem(recipe, "warmup_steps")

    def test_wsd_valid_with_defaults(self):
        recipe = load_recipe(
            os.path.join(REFERENCE_DIR, "tiny-e2e.yaml"), ["schedule.type=wsd"]
        )
        validate(recipe)

    def test_wsd_warmup_reaching_decay_start_rejected(self):
        # total 20, decay starts at int(20 * 0.9) = 18
        recipe = load_recipe(
            os.path.join(REFERENCE_DIR, "tiny-e2e.yaml"),
            ["schedule.type=wsd", "schedule.warmup_steps=18"],
        )
        assert_problem(recipe, "no stable plateau")

    def test_wsd_frac_and_shape_ranges(self):
        for override, needle in (
            ("schedule.decay_start_frac=0.0", "decay_start_frac"),
            ("schedule.decay_start_frac=1.5", "decay_start_frac"),
            ("schedule.final_lr_frac=1.0", "final_lr_frac"),
            ("schedule.decay_shape=step", "decay_shape"),
        ):
            recipe = load_recipe(
                os.path.join(REFERENCE_DIR, "tiny-e2e.yaml"),
                ["schedule.type=wsd", override],
            )
            assert_problem(recipe, needle)

    def test_wsd_knobs_rejected_on_sft_recipes(self):
        recipe = load_recipe(
            os.path.join(REFERENCE_DIR, "sft-qlora-e2b.yaml"),
            ["schedule.type=wsd", "schedule.decay_start_frac=0.8"],
        )
        assert_problem(recipe, "not applied by the sft path")
        defaults_ok = load_recipe(
            os.path.join(REFERENCE_DIR, "sft-qlora-e2b.yaml"), ["schedule.type=wsd"]
        )
        validate(defaults_ok)

    def test_unknown_precision_dtype(self):
        recipe = load_recipe(
            os.path.join(REFERENCE_DIR, "tiny-e2e.yaml"), ["precision.dtype=fp16"]
        )
        assert_problem(recipe, "precision.dtype")

    def test_fp8_backend_checked(self):
        recipe = load_recipe(
            os.path.join(REFERENCE_DIR, "tiny-e2e.yaml"),
            ["precision.dtype=fp8", "precision.fp8_backend=cutlass"],
        )
        assert_problem(recipe, "precision.fp8_backend")
        ok = load_recipe(
            os.path.join(REFERENCE_DIR, "tiny-e2e.yaml"),
            ["precision.dtype=fp8", "precision.fp8_backend=torchao"],
        )
        validate(ok)

    def test_unknown_offload_mode(self):
        recipe = load_recipe(
            os.path.join(REFERENCE_DIR, "tiny-e2e.yaml"), ["memory.offload=cpu"]
        )
        assert_problem(recipe, "memory.offload")

    def test_nonpositive_batch_geometry(self):
        recipe = load_recipe(
            os.path.join(REFERENCE_DIR, "tiny-e2e.yaml"), ["data.micro_batch=0"]
        )
        assert_problem(recipe, "data.micro_batch")

    def test_nonpositive_checkpoint_interval(self):
        recipe = load_recipe(
            os.path.join(REFERENCE_DIR, "tiny-e2e.yaml"),
            ["checkpoint.interval_steps=0"],
        )
        assert_problem(recipe, "checkpoint.interval_steps")

    def test_nonpositive_watchdog_threshold(self):
        recipe = load_recipe(
            os.path.join(REFERENCE_DIR, "tiny-e2e.yaml"),
            ["watchdog.loss_spike_factor=0"],
        )
        assert_problem(recipe, "watchdog.loss_spike_factor")

    def test_multiple_problems_reported_together(self):
        recipe = load_recipe(
            os.path.join(REFERENCE_DIR, "tiny-e2e.yaml"),
            ["optimizer.name=sgd", "schedule.type=linear"],
        )
        with pytest.raises(RecipeValidationError) as excinfo:
            validate(recipe)
        assert len(excinfo.value.problems) >= 2


class TestTokenizerHashInvariant:
    @pytest.fixture()
    def shard_dir(self, tmp_path, jsonl_path, tokenizer_dir):
        from training.data import prepare as prepare_mod

        out = tmp_path / "shards"
        prepare_mod.prepare(
            source=jsonl_path, out_dir=str(out), tokenizer_source=tokenizer_dir
        )
        return str(out)

    def recipe_for(self, shard_dir: str, tokenizer_path: str):
        return load_recipe(
            os.path.join(REFERENCE_DIR, "tiny-e2e.yaml"),
            [f"data.source={shard_dir}", f"tokenizer.path={tokenizer_path}"],
        )

    def test_matching_tokenizer_passes(self, shard_dir, tokenizer_dir):
        validate(self.recipe_for(shard_dir, tokenizer_dir))

    def test_mismatched_tokenizer_fails(self, shard_dir, tmp_path_factory):
        from support import build_bpe_tokenizer

        other = build_bpe_tokenizer(
            str(tmp_path_factory.mktemp("other_tokenizer")),
            ["a completely different corpus"],
            specials=["<|eos|>"],
            vocab_size=300,
        )

        assert_problem(
            self.recipe_for(shard_dir, other), "does not match shard manifest"
        )

    def test_unprepared_source_skips_hash_check(self, tokenizer_dir, tmp_path):
        recipe = self.recipe_for(str(tmp_path / "not-prepared"), tokenizer_dir)
        validate(recipe)

    def test_corrupt_source_manifest_reported_not_raised(
        self, shard_dir, tokenizer_dir
    ):
        # KB-05: a torn manifest.json must surface as a styled problem, not a
        # raw JSONDecodeError escaping validate()
        with open(os.path.join(shard_dir, "manifest.json"), "w") as f:
            f.write('{"tokenizer_hash": ')
        assert_problem(
            self.recipe_for(shard_dir, tokenizer_dir), "corrupt or not a JSON object"
        )

    def test_non_object_source_manifest_reported_not_raised(
        self, shard_dir, tokenizer_dir
    ):
        # KB-05: a manifest parsing to a non-dict would AttributeError on .get
        with open(os.path.join(shard_dir, "manifest.json"), "w") as f:
            f.write("[1, 2]\n")
        assert_problem(
            self.recipe_for(shard_dir, tokenizer_dir), "corrupt or not a JSON object"
        )


class TestPeftInvariants:
    """F2: peft group field domains and QLoRA combination rules."""

    def tiny(self, overrides):
        return load_recipe(os.path.join(REFERENCE_DIR, "tiny-e2e.yaml"), overrides)

    def test_lora_recipe_typed_and_valid(self):
        recipe = self.tiny(["peft.enabled=true", "peft.method=lora", "peft.rank=8"])
        validate(recipe)
        assert recipe.peft.rank == 8
        assert recipe.peft.target_modules == "auto"

    def test_qlora_recipe_valid(self):
        validate(self.tiny(["peft.enabled=true", "peft.quantize_base=nf4"]))

    def test_dora_recipe_valid_with_and_without_quantization(self):
        validate(self.tiny(["peft.enabled=true", "peft.method=dora"]))
        validate(
            self.tiny(
                ["peft.enabled=true", "peft.method=dora", "peft.quantize_base=nf4"]
            )
        )

    def test_unknown_method(self):
        assert_problem(self.tiny(["peft.method=ia3"]), "peft.method")

    def test_unknown_quantize_base(self):
        assert_problem(self.tiny(["peft.quantize_base=int8"]), "peft.quantize_base")

    def test_unknown_compute_dtype(self):
        assert_problem(self.tiny(["peft.compute_dtype=fp8"]), "peft.compute_dtype")

    def test_nonpositive_rank(self):
        assert_problem(self.tiny(["peft.enabled=true", "peft.rank=0"]), "peft.rank")

    def test_dropout_range(self):
        assert_problem(self.tiny(["peft.dropout=1.0"]), "peft.dropout")

    def test_nf4_requires_peft_enabled(self):
        assert_problem(
            self.tiny(["peft.quantize_base=nf4"]), "nf4 requires peft.enabled"
        )

    def test_nf4_excludes_fp8_precision(self):
        assert_problem(
            self.tiny(
                ["peft.enabled=true", "peft.quantize_base=nf4", "precision.dtype=fp8"]
            ),
            "mutually exclusive",
        )

    def test_empty_target_modules_rejected(self):
        assert_problem(
            self.tiny(["peft.enabled=true", "peft.target_modules=[]"]),
            "peft.target_modules",
        )

    def test_explicit_target_modules_accepted(self):
        recipe = self.tiny(["peft.enabled=true", "peft.target_modules=[q_proj,v_proj]"])
        validate(recipe)
        assert list(recipe.peft.target_modules) == ["q_proj", "v_proj"]

    def test_reference_qlora_recipe_loads_typed(self):
        recipe = load_recipe(os.path.join(REFERENCE_DIR, "sft-qlora-e2b.yaml"))
        validate(recipe)
        assert recipe.peft.enabled is True
        assert recipe.peft.quantize_base == "nf4"
        assert recipe.peft.target_modules == "auto"
