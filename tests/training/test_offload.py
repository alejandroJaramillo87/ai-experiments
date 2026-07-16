"""ZeRO-Offload unit tests (Phase 2 P2-D5): DeepSpeed config generation as a
pure function of the recipe, the trainable_filter freeze policy on a real
tiny HF model, the calculator's transformer_stack parameter accounting, and
the validate() invariants. CPU-only; the engine itself (deepspeed.initialize,
CPUAdam) is exercised by the host-side tiny-model resume acceptance run, not
here — the sandbox has no deepspeed."""

from __future__ import annotations

import json

import pytest
from support import TINY_RECIPE, reference_recipe

from training.offload import (
    apply_trainable_filter,
    build_ds_config,
    frozen_patterns,
    is_frozen_name,
)
from training.recipes import RecipeValidationError, load_recipe, validate

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

    def test_trainable_count_matches_calculator(self, tiny_model):
        from training.calculator import spec_from_recipe, trainable_param_count

        recipe = offload_recipe("model.trainable_filter=transformer_stack")
        trainable = apply_trainable_filter(tiny_model, recipe)
        measured = sum(p.numel() for p in trainable)
        # The tiny HF Llama matches the tiny scratch spec the calculator
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
        from training import offload

        monkeypatch.setitem(
            offload.FROZEN_PATTERNS, "transformer_stack", ("no_such_module",)
        )
        recipe = offload_recipe("model.trainable_filter=transformer_stack")
        with pytest.raises(ValueError, match="froze no parameters"):
            apply_trainable_filter(tiny_model, recipe)


class TestCalculatorTransformerStack:
    # Same hand-computed tiny config as test_calculator.py: total 90,432.
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
        from training.calculator import spec_from_recipe, trainable_param_count

        recipe = load_recipe(
            TINY_RECIPE,
            [*self.SPEC_OVERRIDES, "model.trainable_filter=transformer_stack"],
        )
        assert (
            trainable_param_count(spec_from_recipe(recipe), recipe) == self.STACK_PARAMS
        )

    def test_host_ram_prices_the_frozen_policy(self):
        from training.calculator import host_ram_bytes, spec_from_recipe
        from training.calculator.components import OFFLOAD_HOST_BYTES_PER_PARAM

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
        from training import checkpoint as ckpt

        step_dir = tmp_path / "step_000000001"
        step_dir.mkdir()
        (step_dir / "meta.json").write_text(
            json.dumps({"step": 1, "checkpoint_format": "torch"})
        )
        with pytest.raises(ValueError, match="checkpoint_format"):
            ckpt.load_checkpoint_deepspeed(
                str(step_dir), engine=None, scheduler=None, dataset=None
            )
