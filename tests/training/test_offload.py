"""ZeRO-Offload unit tests (Phase 2 P2-D5): DeepSpeed config generation as a
pure function of the recipe, the trainable_filter freeze policy on a real
tiny HF model, the transformer_stack parameter accounting, and
the validate() invariants. CPU-only; the engine itself (deepspeed.initialize,
CPUAdam) is exercised by the host-side tiny-model resume acceptance run, not
here — the sandbox has no deepspeed."""

from __future__ import annotations

import json

import pytest
from support import TINY_RECIPE, reference_recipe

from training.recipes import RecipeValidationError, load_recipe, validate
from training.runtime.offload import (
    apply_trainable_filter,
    build_ds_config,
    frozen_patterns,
    is_frozen_name,
)

OFFLOAD_OVERRIDES = [
    "optimizer.name=cpu_adam",
    "memory.offload=zero_offload",
]


def offload_recipe(*extra: str):
    return load_recipe(TINY_RECIPE, [*OFFLOAD_OVERRIDES, *extra])


def problems_of(recipe) -> list[str]:
    try:
        validate(recipe)
    except RecipeValidationError as exc:
        return exc.problems
    return []


class TestDsConfig:
    def test_recipe_knobs_map_through(self):
        recipe = offload_recipe(
            "data.micro_batch=2",
            "data.grad_accum=8",
            "memory.allgather_bucket_size=5e7",
            "memory.reduce_bucket_size=1e8",
            "memory.offload_pin_memory=false",
            "optimizer.lr=3e-4",
        )
        config = build_ds_config(recipe)
        assert config["train_micro_batch_size_per_gpu"] == 2
        assert config["gradient_accumulation_steps"] == 8
        zero = config["zero_optimization"]
        assert zero["stage"] == 2
        assert zero["offload_optimizer"] == {"device": "cpu", "pin_memory": False}
        assert zero["allgather_bucket_size"] == 50_000_000
        assert zero["reduce_bucket_size"] == 100_000_000
        assert config["optimizer"]["type"] == "Adam"
        assert config["optimizer"]["params"]["lr"] == pytest.approx(3e-4)
        assert config["bf16"]["enabled"] is True

    def test_defaults_and_scheduler_ownership(self):
        config = build_ds_config(offload_recipe())
        # Buckets default to the DeepSpeed 2e8; clipping stays off to match
        # the plain loop; the loop-owned scheduler means no scheduler block
        # and no LR outside the optimizer params (resolved decision 2).
        assert config["zero_optimization"]["allgather_bucket_size"] == 200_000_000
        assert config["zero_optimization"]["reduce_bucket_size"] == 200_000_000
        assert config["gradient_clipping"] == 0.0
        # None, not 0: the engine guards step logging on `is not None` and
        # then takes a modulo by this value, so 0 raises ZeroDivisionError on
        # the first engine.step() rather than disabling the printing.
        assert config["steps_per_print"] is None
        assert "scheduler" not in config
        assert config["zero_optimization"]["offload_optimizer"]["pin_memory"] is True

    def test_requires_zero_offload(self):
        recipe = load_recipe(TINY_RECIPE)
        with pytest.raises(ValueError, match="zero_offload"):
            build_ds_config(recipe)

    def test_json_serializable(self):
        json.dumps(build_ds_config(offload_recipe()))


class TestFrozenPatterns:
    def test_all_freezes_nothing(self):
        assert frozen_patterns("all") == ()
        assert not is_frozen_name("model.embed_tokens.weight", "all")

    def test_transformer_stack_patterns(self):
        for name in (
            "model.embed_tokens.weight",
            "lm_head.weight",
            "model.embed_tokens_per_layer.weight",
            "model.layers.3.per_layer_input_gate.weight",
        ):
            assert is_frozen_name(name, "transformer_stack"), name
        for name in (
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.7.mlp.down_proj.weight",
            "model.norm.weight",
        ):
            assert not is_frozen_name(name, "transformer_stack"), name

    def test_unknown_filter_fails_loudly(self):
        with pytest.raises(ValueError, match="frozen-pattern"):
            frozen_patterns("bogus")


class TestApplyTrainableFilter:
    @pytest.fixture()
    def tiny_model(self):
        pytest.importorskip("torch")
        transformers = pytest.importorskip("transformers")

        config = transformers.LlamaConfig(
            num_hidden_layers=2,
            hidden_size=64,
            num_attention_heads=4,
            num_key_value_heads=2,
            intermediate_size=128,
            vocab_size=256,
            tie_word_embeddings=True,
        )
        return transformers.AutoModelForCausalLM.from_config(config)

    def test_transformer_stack_freezes_embeddings(self, tiny_model):
        recipe = offload_recipe("model.trainable_filter=transformer_stack")
        trainable = apply_trainable_filter(tiny_model, recipe)
        trainable_ids = {id(p) for p in trainable}
        for name, param in tiny_model.named_parameters():
            frozen = is_frozen_name(name, "transformer_stack")
            assert param.requires_grad != frozen, name
            assert (id(param) in trainable_ids) != frozen, name
        # The P2-D5 acceptance assertion: no frozen pattern reaches the
        # optimizer-group population.
        assert trainable_ids
        assert all(
            not is_frozen_name(n, "transformer_stack")
            for n, p in tiny_model.named_parameters()
            if id(p) in trainable_ids
        )

    def test_trainable_count_matches_the_analytic_count(self, tiny_model):
        from training.models.architecture import spec_from_recipe, trainable_param_count

        recipe = offload_recipe("model.trainable_filter=transformer_stack")
        trainable = apply_trainable_filter(tiny_model, recipe)
        measured = sum(p.numel() for p in trainable)
        # The tiny HF Llama matches the tiny scratch spec architecture.py
        # tests pin, so the freeze policy and the memory model must agree.
        spec_recipe = load_recipe(
            TINY_RECIPE,
            [
                "model.trainable_filter=transformer_stack",
                "model.scratch.layers=2",
                "model.scratch.hidden=64",
                "model.scratch.heads=4",
                "model.scratch.kv_heads=2",
                "model.scratch.ffn_mult=2.0",
                "model.scratch.vocab=256",
            ],
        )
        assert measured == trainable_param_count(
            spec_from_recipe(spec_recipe), spec_recipe
        )

    def test_all_filter_returns_everything(self, tiny_model):
        recipe = offload_recipe("model.trainable_filter=all")
        trainable = apply_trainable_filter(tiny_model, recipe)
        assert len(trainable) == len(list(tiny_model.parameters()))
        assert all(p.requires_grad for p in tiny_model.parameters())

    def test_pattern_drift_fails_loudly(self, tiny_model, monkeypatch):
        from training.runtime import offload

        monkeypatch.setitem(
            offload.FROZEN_PATTERNS, "transformer_stack", ("no_such_module",)
        )
        recipe = offload_recipe("model.trainable_filter=transformer_stack")
        with pytest.raises(ValueError, match="froze no parameters"):
            apply_trainable_filter(tiny_model, recipe)


class TestCalculatorTransformerStack:
    # Same hand-computed tiny config as test_architecture.py: total 90,432.
    # transformer_stack trainable = layers * (attn + mlp + norms) + final norm
    #   = 2 * (12,288 + 24,576 + 128) + 64 = 74,048
    STACK_PARAMS = 74_048
    SPEC_OVERRIDES = [
        "model.scratch.layers=2",
        "model.scratch.hidden=64",
        "model.scratch.heads=4",
        "model.scratch.kv_heads=2",
        "model.scratch.ffn_mult=2.0",
        "model.scratch.vocab=256",
    ]

    def test_transformer_stack_param_count(self):
        from training.models.architecture import spec_from_recipe, trainable_param_count

        recipe = load_recipe(
            TINY_RECIPE,
            [*self.SPEC_OVERRIDES, "model.trainable_filter=transformer_stack"],
        )
        assert (
            trainable_param_count(spec_from_recipe(recipe), recipe) == self.STACK_PARAMS
        )

    def test_host_ram_prices_the_frozen_policy(self):
        from training.models.architecture import (
            OFFLOAD_HOST_BYTES_PER_PARAM,
            host_ram_bytes,
            spec_from_recipe,
        )

        recipe = load_recipe(
            TINY_RECIPE,
            [
                *self.SPEC_OVERRIDES,
                *OFFLOAD_OVERRIDES,
                "model.trainable_filter=transformer_stack",
            ],
        )
        assert host_ram_bytes(spec_from_recipe(recipe), recipe) == (
            self.STACK_PARAMS * OFFLOAD_HOST_BYTES_PER_PARAM
        )


class TestValidate:
    def test_reference_offload_recipe_is_valid(self):
        recipe = reference_recipe("continual-gemma4-e2b.yaml")
        validate(recipe)

    def test_unknown_trainable_filter(self):
        recipe = load_recipe(TINY_RECIPE, ["model.trainable_filter=bogus"])
        assert any("model.trainable_filter" in p for p in problems_of(recipe))

    def test_filter_conflicts_with_peft(self):
        recipe = load_recipe(
            TINY_RECIPE,
            ["model.trainable_filter=transformer_stack", "peft.enabled=true"],
        )
        assert any(
            "mutually exclusive with peft.enabled" in p for p in problems_of(recipe)
        )

    def test_bucket_sizes_positive(self):
        recipe = load_recipe(TINY_RECIPE, ["memory.reduce_bucket_size=0"])
        assert any("memory.reduce_bucket_size" in p for p in problems_of(recipe))

    def test_host_budget_exceeded_is_an_error(self):
        # ~107B scratch params * 16 B/param of offloaded state cannot fit
        # under 128GB; validate() must fail hard (resolved decision 4).
        recipe = load_recipe(
            TINY_RECIPE,
            [
                *OFFLOAD_OVERRIDES,
                "model.scratch.layers=100",
                "model.scratch.hidden=8192",
                "model.scratch.heads=64",
                "model.scratch.kv_heads=64",
                "model.scratch.vocab=32000",
            ],
        )
        assert any("predicted host RAM" in p for p in problems_of(recipe))

    def test_frozen_stack_can_bring_a_run_under_budget(self):
        # The same envelope logic in the fitting direction: a small scratch
        # model under offload passes.
        recipe = load_recipe(
            TINY_RECIPE,
            [*OFFLOAD_OVERRIDES, "model.trainable_filter=transformer_stack"],
        )
        assert not any("predicted host RAM" in p for p in problems_of(recipe))


class TestCheckpointFormatGuard:
    def test_resume_rejects_torch_format_under_offload(self, tmp_path):
        from training.runtime import checkpoint as ckpt

        step_dir = tmp_path / "step_000000001"
        step_dir.mkdir()
        (step_dir / "meta.json").write_text(
            json.dumps({"step": 1, "checkpoint_format": "torch"})
        )
        with pytest.raises(ValueError, match="checkpoint_format"):
            ckpt.load_checkpoint_deepspeed(
                str(step_dir), engine=None, scheduler=None, dataset=None
            )


class TestOptimizerFallback:
    """The cpu_adam degradation chain (WI-7): cpu_adam -> paged_adamw8bit ->
    adamw, walked when the pre-built DeepSpeed extension is unavailable.
    Probes are injected so the chain is testable without deepspeed or
    bitsandbytes installed."""

    @staticmethod
    def _probes(cpu_adam: bool, paged: bool):
        return {
            "cpu_adam": lambda: cpu_adam,
            "paged_adamw8bit": lambda: paged,
        }

    def test_non_cpu_adam_recipe_passes_through(self):
        from training.runtime.offload import resolve_offload_optimizer

        recipe = load_recipe(TINY_RECIPE, ["memory.offload=zero_offload"])
        assert recipe.optimizer.name == "adamw"
        name = resolve_offload_optimizer(recipe, probes=self._probes(False, False))
        assert name == "adamw"

    def test_cpu_adam_available_runs_cpu_adam(self, capsys):
        from training.runtime.offload import resolve_offload_optimizer

        name = resolve_offload_optimizer(
            offload_recipe(), probes=self._probes(True, False)
        )
        assert name == "cpu_adam"
        assert capsys.readouterr().err == ""

    def test_degrades_to_paged_adamw8bit(self, capsys):
        from training.runtime.offload import resolve_offload_optimizer

        name = resolve_offload_optimizer(
            offload_recipe(), probes=self._probes(False, True)
        )
        assert name == "paged_adamw8bit"
        err = capsys.readouterr().err
        assert "degraded to paged_adamw8bit" in err
        assert "CPUAdam extension" in err

    def test_degrades_to_adamw_when_paged_unavailable(self, capsys):
        from training.runtime.offload import resolve_offload_optimizer

        name = resolve_offload_optimizer(
            offload_recipe(), probes=self._probes(False, False)
        )
        assert name == "adamw"
        err = capsys.readouterr().err
        assert "degraded to adamw" in err
        assert "offload disabled" in err

    def test_chain_matches_schema_names(self):
        from training.recipes.schema import OPTIMIZER_NAMES
        from training.runtime.offload import OPTIMIZER_FALLBACK_CHAIN

        assert set(OPTIMIZER_FALLBACK_CHAIN) <= set(OPTIMIZER_NAMES)

    def test_paged_adamw8bit_validates_as_recipe_optimizer(self):
        recipe = load_recipe(TINY_RECIPE, ["optimizer.name=paged_adamw8bit"])
        assert not any("optimizer.name" in p for p in problems_of(recipe))


class TestBuildOptimizerPaged:
    def test_paged_adamw8bit_builds_through_bitsandbytes(self, monkeypatch):
        import sys
        import types

        from training.models.model import build_optimizer

        built = {}

        class FakePagedAdamW8bit:
            def __init__(self, params, **kwargs):
                built["params"] = list(params)
                built["kwargs"] = kwargs

        fake_bnb = types.ModuleType("bitsandbytes")
        fake_bnb.optim = types.SimpleNamespace(PagedAdamW8bit=FakePagedAdamW8bit)
        monkeypatch.setitem(sys.modules, "bitsandbytes", fake_bnb)

        recipe = load_recipe(TINY_RECIPE, ["optimizer.name=paged_adamw8bit"])
        model = types.SimpleNamespace(parameters=lambda: iter(["p0", "p1"]))
        optimizer = build_optimizer(model, recipe)
        assert isinstance(optimizer, FakePagedAdamW8bit)
        assert built["params"] == ["p0", "p1"]
        assert built["kwargs"]["lr"] == recipe.optimizer.lr

    def test_explicit_params_override_model_parameters(self, monkeypatch):
        import sys
        import types

        from training.models.model import build_optimizer

        built = {}

        class FakePagedAdamW8bit:
            def __init__(self, params, **kwargs):
                built["params"] = list(params)

        fake_bnb = types.ModuleType("bitsandbytes")
        fake_bnb.optim = types.SimpleNamespace(PagedAdamW8bit=FakePagedAdamW8bit)
        monkeypatch.setitem(sys.modules, "bitsandbytes", fake_bnb)

        recipe = load_recipe(TINY_RECIPE, [])
        model = types.SimpleNamespace(parameters=lambda: iter(["all"]))
        build_optimizer(
            model, recipe, name="paged_adamw8bit", params=["only-trainable"]
        )
        assert built["params"] == ["only-trainable"]


class TestAttentionFallbackWarning:
    """attention_implementation() warns once when a CUDA host lacks a usable
    flash-attn (WI-7); the CPU sandbox path stays silent."""

    @staticmethod
    def _fake_torch(cuda_available: bool):
        import types

        return types.SimpleNamespace(
            cuda=types.SimpleNamespace(is_available=lambda: cuda_available)
        )

    def test_cuda_without_flash_attn_warns_once_and_returns_sdpa(
        self, monkeypatch, capsys
    ):
        import sys

        from training.models import model as model_mod

        monkeypatch.setitem(sys.modules, "torch", self._fake_torch(True))
        # A None sys.modules entry makes `import flash_attn` raise ImportError,
        # so the degradation is exercised on the training image too, where
        # flash-attn is genuinely installed.
        monkeypatch.setitem(sys.modules, "flash_attn", None)
        monkeypatch.setattr(model_mod, "_flash_fallback_warned", False)
        assert model_mod.attention_implementation() == "sdpa"
        first = capsys.readouterr().err
        assert "flash-attn unavailable" in first
        assert "falling back to sdpa" in first
        assert model_mod.attention_implementation() == "sdpa"
        assert capsys.readouterr().err == ""  # warned once per process

    def test_cpu_host_is_silent(self, monkeypatch, capsys):
        import sys

        from training.models import model as model_mod

        monkeypatch.setitem(sys.modules, "torch", self._fake_torch(False))
        monkeypatch.setattr(model_mod, "_flash_fallback_warned", False)
        assert model_mod.attention_implementation() == "sdpa"
        assert capsys.readouterr().err == ""


class TestAttentionHeadWidth:
    """KB-84: FlashAttention 2 refuses a head dimension above 256, and Gemma 4's
    full-attention layers are 512 wide, so a config declaring one takes sdpa
    even where flash-attn imports (FT.calibrate-qlora's second attempt died in
    the kernel with `head dimension at most 256`)."""

    @staticmethod
    def _cuda_with_flash(monkeypatch):
        import sys
        import types

        from training.models import model as model_mod

        monkeypatch.setitem(
            sys.modules,
            "torch",
            types.SimpleNamespace(
                cuda=types.SimpleNamespace(is_available=lambda: True)
            ),
        )
        monkeypatch.setitem(sys.modules, "flash_attn", types.ModuleType("flash_attn"))
        monkeypatch.setattr(model_mod, "_flash_fallback_warned", False)
        monkeypatch.setattr(model_mod, "_head_dim_note_printed", False)
        return model_mod

    @staticmethod
    def _config(**fields):
        import types

        return types.SimpleNamespace(**fields)

    def test_no_config_keeps_flash(self, monkeypatch):
        model_mod = self._cuda_with_flash(monkeypatch)
        assert model_mod.attention_implementation() == "flash_attention_2"

    def test_a_head_within_the_cap_keeps_flash(self, monkeypatch, capsys):
        model_mod = self._cuda_with_flash(monkeypatch)
        config = self._config(head_dim=256)
        assert model_mod.attention_implementation(config) == "flash_attention_2"
        assert capsys.readouterr().err == ""

    def test_gemma4_global_head_takes_sdpa_and_notes_once(self, monkeypatch, capsys):
        model_mod = self._cuda_with_flash(monkeypatch)
        # The composite checkpoint keeps the decoder under text_config.
        config = self._config(
            model_type="gemma4",
            text_config=self._config(head_dim=256, global_head_dim=512),
        )
        assert model_mod.max_head_dim(config) == 512
        assert model_mod.attention_implementation(config) == "sdpa"
        note = capsys.readouterr().err
        assert "head_dim <= 256" in note and "512" in note and "sdpa" in note
        assert model_mod.attention_implementation(config) == "sdpa"
        assert capsys.readouterr().err == ""

    def test_a_config_naming_no_head_keeps_flash(self, monkeypatch):
        model_mod = self._cuda_with_flash(monkeypatch)
        assert model_mod.max_head_dim(self._config(hidden_size=64)) is None
        assert model_mod.attention_implementation(self._config()) == "flash_attention_2"

    def test_the_stamp_reads_the_loaded_model(self):
        import types

        from training.tune.sft_callbacks import _attention_implementation

        model = types.SimpleNamespace(
            config=types.SimpleNamespace(_attn_implementation="sdpa")
        )
        assert _attention_implementation(model) == "sdpa"
