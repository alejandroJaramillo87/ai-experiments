"""The model registry: what it requires of an entry, and what it refuses."""

from __future__ import annotations

import argparse
import json

import pytest

from bench import registry
from bench.commands.models import cmd as models_cmd
from common.serving import SERVICES

DENSE = {
    "path": "/app/models/gguf/thing/thing-Q4_K_M.gguf",
    "engines": ["llama-cpu"],
    "architecture": "dense",
    "params_total_b": 7.0,
    "quantization": "Q4_K_M",
    "context_length": 32768,
}


class TestShippedRegistry:
    def test_every_shipped_entry_loads_and_validates(self):
        specs = registry.load()
        assert specs, "the registry must not be empty; a run needs a model to name"

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
    def test_a_dense_entry_needs_no_expert_fields(self):
        specs = registry.load({"d": DENSE})
        assert specs["d"].is_moe is False
        assert specs["d"].params_active_b is None

    def test_a_missing_required_key_is_refused(self):
        entry = {k: v for k, v in DENSE.items() if k != "quantization"}
        with pytest.raises(registry.RegistryError, match="quantization"):
            registry.load({"d": entry})

    def test_an_unknown_architecture_is_refused(self):
        with pytest.raises(registry.RegistryError, match="architecture"):
            registry.load({"d": {**DENSE, "architecture": "ssm"}})

    def test_an_moe_without_active_params_is_refused(self):
        entry = {**DENSE, "architecture": "moe"}
        with pytest.raises(registry.RegistryError, match="params_active_b"):
            registry.load({"m": entry})

    def test_a_quantization_that_contradicts_the_filename_is_refused(self):
        # A quantization confound is the easiest way to publish an unfounded comparison.
        entry = {**DENSE, "quantization": "Q8_0"}
        with pytest.raises(registry.RegistryError, match="disagrees"):
            registry.load({"d": entry})

    def test_an_unparseable_filename_does_not_block_an_entry(self):
        # A safetensors checkpoint has no quantization suffix to parse.
        entry = {
            **DENSE,
            "path": "/app/models/hf/gemma-4-E4B-it",
            "quantization": "BF16",
        }
        assert registry.load({"d": entry})["d"].quantization == "BF16"


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


class TestQuantizationGuardOnTheIQAndTQFamilies:
    """common/serving.py's parser accepts a suffix only if it starts with Q or names a
    float type, so every IQ- and TQ-family llama.cpp quantization reads as unparseable
    and the guard falls silent -- on the shape the one shipped entry actually has."""

    def test_an_iq_filename_that_contradicts_the_stated_quantization_is_refused(self):
        entry = {
            **DENSE,
            "path": "/app/models/gguf/thing/thing-IQ4_XS.gguf",
            "quantization": "Q4_K_M",
        }
        with pytest.raises(registry.RegistryError, match="disagrees"):
            registry.load({"d": entry})

    def test_a_tq_filename_that_contradicts_the_stated_quantization_is_refused(self):
        entry = {
            **DENSE,
            "path": "/app/models/gguf/thing/thing-TQ1_0.gguf",
            "quantization": "Q4_K_M",
        }
        with pytest.raises(registry.RegistryError, match="disagrees"):
            registry.load({"d": entry})

    def test_an_iq_filename_that_agrees_is_accepted(self):
        entry = {
            **DENSE,
            "path": "/app/models/gguf/thing/thing-IQ4_XS.gguf",
            "quantization": "IQ4_XS",
        }
        assert registry.load({"d": entry})["d"].quantization == "IQ4_XS"

    def test_the_shipped_iq_entry_is_the_case_the_guard_now_covers(self):
        spec = registry.load()["qwen3-coder-30b-a3b-IQ4_XS"]
        assert spec.quantization == "IQ4_XS"
        assert spec.path.endswith("-IQ4_XS.gguf")


class TestEntryShape:
    """Presence and two enumerations are not enough: a wrong-typed or internally
    inconsistent field reaches the report, and the fields a comparison refuses to
    cross, unchallenged."""

    def test_a_bare_string_engine_list_is_refused(self):
        # tuple("llama-cpu") is a per-character engine list, which `bench models`
        # prints one letter at a time and resolve() matches against nothing.
        with pytest.raises(registry.RegistryError, match="engines"):
            registry.load({"d": {**DENSE, "engines": "llama-cpu"}})

    def test_an_empty_engine_list_is_refused(self):
        with pytest.raises(registry.RegistryError, match="engines"):
            registry.load({"d": {**DENSE, "engines": []}})

    def test_an_engine_that_is_not_a_service_is_refused(self):
        with pytest.raises(registry.RegistryError, match="llama-tpu"):
            registry.load({"d": {**DENSE, "engines": ["llama-tpu"]}})

    def test_active_parameters_exceeding_the_total_are_refused(self):
        entry = {
            **DENSE,
            "architecture": "moe",
            "params_total_b": 3.0,
            "params_active_b": 30.0,
            "n_experts": 128,
            "n_experts_active": 8,
        }
        with pytest.raises(registry.RegistryError, match="params_active_b"):
            registry.load({"m": entry})

    def test_more_active_experts_than_experts_is_refused(self):
        entry = {
            **DENSE,
            "architecture": "moe",
            "params_active_b": 3.0,
            "n_experts": 8,
            "n_experts_active": 128,
        }
        with pytest.raises(registry.RegistryError, match="n_experts_active"):
            registry.load({"m": entry})

    def test_a_negative_parameter_count_is_refused(self):
        with pytest.raises(registry.RegistryError, match="params_total_b"):
            registry.load({"d": {**DENSE, "params_total_b": -7.0}})

    def test_a_string_context_length_is_refused(self):
        with pytest.raises(registry.RegistryError, match="context_length"):
            registry.load({"d": {**DENSE, "context_length": "32768"}})


SERVED = "/app/models/gguf/thing/thing-Q4_K_M.gguf"
OTHER = "/app/models/gguf/other/other-Q8_0.gguf"
SPEC = registry.ModelSpec("d", SERVED, ("llama-cpu",), "dense", 7.0, "Q4_K_M", 32768)


class TestReconciliation:
    """The registry's claim is an assertion about a container nobody asks.

    A report carries conditions.model from here and conditions.server_argv from the
    running service, and the training commands' --keep-serving path leaves a swapped
    model loaded on purpose, so the two can disagree with nothing saying so.
    """

    def test_the_declared_path_matching_the_container_env_is_no_mismatch(self):
        observed = {
            "env_knobs": {"MODEL_PATH": SERVED},
            "server_argv": ["/entrypoint.sh"],
        }
        assert registry.reconcile(SPEC, observed) is None

    def test_a_different_model_in_the_container_env_is_reported(self):
        observed = {
            "env_knobs": {"MODEL_PATH": OTHER},
            "server_argv": ["/entrypoint.sh"],
        }
        message = registry.reconcile(SPEC, observed)
        assert message is not None
        assert SERVED in message and OTHER in message

    def test_a_different_model_in_the_server_argv_is_reported(self):
        observed = {"server_argv": ["llama-server", "-m", OTHER, "--port", "8001"]}
        message = registry.reconcile(SPEC, observed)
        assert message is not None
        assert OTHER in message

    def test_the_declared_path_matching_the_server_argv_is_no_mismatch(self):
        observed = {
            "server_argv": ["llama-server", "--model", SERVED, "--port", "8001"]
        }
        assert registry.reconcile(SPEC, observed) is None

    def test_an_inline_model_flag_is_read(self):
        assert registry.reconcile(SPEC, [f"--model={SERVED}"]) is None
        assert registry.reconcile(SPEC, [f"--model={OTHER}"]) is not None

    def test_a_bare_argv_sequence_is_accepted(self):
        # The conditions mapping is the caller's convenience; argv alone is the datum.
        assert registry.reconcile(SPEC, ["llama-server", "-m", OTHER]) is not None

    def test_nothing_observed_is_not_a_mismatch(self):
        # docker inspect failing leaves conditions.container empty. Unobserved is not
        # contradicted; a caller that wants to insist says so itself.
        assert registry.reconcile(SPEC, {}) is None
        assert registry.reconcile(SPEC, None) is None
        assert registry.reconcile(SPEC, {"server_argv": ["/entrypoint.sh"]}) is None

    def test_a_redundant_separator_is_not_a_mismatch(self):
        observed = {
            "env_knobs": {"MODEL_PATH": "/app/models/gguf//thing/thing-Q4_K_M.gguf"}
        }
        assert registry.reconcile(SPEC, observed) is None


class TestModelsListing:
    """`bench models --json` is the machine form of the human listing, so a consumer
    should not have to scrape stderr for a fact the human lines carry."""

    def test_the_json_listing_names_the_services_each_model_resolves_for(self, capsys):
        assert models_cmd(argparse.Namespace(json=True)) == 0
        payload = json.loads(capsys.readouterr().out)
        for slug, spec in registry.load().items():
            assert payload[slug]["engines"] == list(spec.engines)

    def test_the_json_listing_still_carries_the_report_model_block(self, capsys):
        assert models_cmd(argparse.Namespace(json=True)) == 0
        payload = json.loads(capsys.readouterr().out)
        for slug, spec in registry.load().items():
            block = spec.conditions()
            assert {k: payload[slug][k] for k in block} == block

    def test_a_wrong_typed_entry_exits_nonzero_rather_than_raising(
        self, capsys, monkeypatch
    ):
        # A field of the wrong type used to leave the loader as a TypeError, past the
        # handler, and reach the user as a traceback with no exit code.
        monkeypatch.setattr(registry, "MODELS", {"d": {**DENSE, "engines": 5}})
        assert models_cmd(argparse.Namespace(json=False)) == 1
        assert "[ERROR]" in capsys.readouterr().err
