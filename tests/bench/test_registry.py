"""The model registry, and the contract in schemas/bench_models.json that guards it."""

from __future__ import annotations

import json

import pytest

from bench import registry
from common.serving import SERVICES

CONTRACT = json.load(open(registry.CONTRACT, encoding="utf-8"))


def write_registry(tmp_path, models: dict):
    path = tmp_path / "models.json"
    path.write_text(json.dumps({"contract": "bench_models", "models": models}))
    return str(path)


DENSE = {
    "path": "/app/models/gguf/thing/thing-Q4_K_M.gguf",
    "engines": ["llama-cpu"],
    "architecture": "dense",
    "params_total_b": 7.0,
    "quantization": "Q4_K_M",
    "context_length": 32768,
}


class TestShippedContract:
    def test_the_contract_names_its_guard(self):
        # schemas/README.md: every contract names the tests that assert it.
        assert "tests/bench/test_registry.py" in CONTRACT["guarded_by"]
        assert CONTRACT["readers"]

    def test_every_shipped_entry_loads_and_validates(self):
        specs = registry.load()
        assert specs, "the registry must not be empty; a run needs a model to name"

    def test_required_keys_match_the_contract(self):
        assert set(CONTRACT["entry"]["required"]) == set(registry.REQUIRED)
        assert set(CONTRACT["entry"]["moe_required"]) == set(registry.MOE_REQUIRED)
        assert set(CONTRACT["entry"]["architecture_values"]) == registry.ARCHITECTURES

    def test_every_engine_named_is_a_real_service(self):
        for spec in registry.load().values():
            for engine in spec.engines:
                assert engine in SERVICES, f"{spec.slug} names unknown service {engine}"

    def test_the_moe_entry_records_what_a_comparison_needs(self):
        # The headline CPU figure in docs/optimizations/README.md was measured against
        # this model and says nowhere that it is an MoE.
        spec = registry.load()["qwen3-coder-30b-a3b-IQ4_XS"]
        assert spec.is_moe
        assert spec.params_active_b < spec.params_total_b
        assert spec.n_experts_active < spec.n_experts


class TestValidation:
    def test_a_dense_entry_needs_no_expert_fields(self, tmp_path):
        specs = registry.load(write_registry(tmp_path, {"d": DENSE}))
        assert specs["d"].is_moe is False
        assert specs["d"].params_active_b is None

    def test_a_missing_required_key_is_refused(self, tmp_path):
        entry = {k: v for k, v in DENSE.items() if k != "quantization"}
        with pytest.raises(registry.RegistryError, match="quantization"):
            registry.load(write_registry(tmp_path, {"d": entry}))

    def test_an_unknown_architecture_is_refused(self, tmp_path):
        with pytest.raises(registry.RegistryError, match="architecture"):
            registry.load(
                write_registry(tmp_path, {"d": {**DENSE, "architecture": "ssm"}})
            )

    def test_an_moe_without_active_params_is_refused(self, tmp_path):
        entry = {**DENSE, "architecture": "moe"}
        with pytest.raises(registry.RegistryError, match="params_active_b"):
            registry.load(write_registry(tmp_path, {"m": entry}))

    def test_a_quantization_that_contradicts_the_filename_is_refused(self, tmp_path):
        # A quantization confound is the easiest way to publish an unfounded comparison.
        entry = {**DENSE, "quantization": "Q8_0"}
        with pytest.raises(registry.RegistryError, match="disagrees"):
            registry.load(write_registry(tmp_path, {"d": entry}))

    def test_an_unparseable_filename_does_not_block_an_entry(self, tmp_path):
        # A safetensors checkpoint has no quantization suffix to parse.
        entry = {
            **DENSE,
            "path": "/app/models/hf/gemma-4-E4B-it",
            "quantization": "BF16",
        }
        assert (
            registry.load(write_registry(tmp_path, {"d": entry}))["d"].quantization
            == "BF16"
        )


class TestArchitectureDrivesTheWorkload:
    def test_moe_draws_disjoint_prompts_and_dense_does_not(self):
        # An MoE routes on content, so one prompt repeated measures repeatability and
        # not routing sensitivity.
        moe = registry.load()["qwen3-coder-30b-a3b-IQ4_XS"]
        assert moe.decode_prompts == 3
        assert (
            registry.ModelSpec(
                "d", "p", ("llama-cpu",), "dense", 7.0, "Q4_K_M", 32768
            ).decode_prompts
            == 1
        )

    def test_conditions_carry_both_parameter_counts_for_an_moe(self):
        block = registry.load()["qwen3-coder-30b-a3b-IQ4_XS"].conditions()
        assert block["params_total_b"] and block["params_active_b"]
        assert block["architecture"] == "moe"

    def test_conditions_omit_expert_fields_for_a_dense_model(self):
        spec = registry.ModelSpec(
            "d", "p", ("llama-cpu",), "dense", 7.0, "Q4_K_M", 32768
        )
        assert "n_experts" not in spec.conditions()


class TestResolve:
    def test_resolves_for_a_registered_service(self):
        assert registry.resolve("qwen3-coder-30b-a3b-IQ4_XS", "llama-cpu").is_moe

    def test_refuses_a_service_the_model_is_not_registered_for(self):
        with pytest.raises(registry.RegistryError, match="not 'vllm-gpu'"):
            registry.resolve("qwen3-coder-30b-a3b-IQ4_XS", "vllm-gpu")

    def test_an_unknown_slug_lists_what_is_registered(self):
        with pytest.raises(registry.RegistryError, match="registered:"):
            registry.resolve("no-such-model", "llama-cpu")
