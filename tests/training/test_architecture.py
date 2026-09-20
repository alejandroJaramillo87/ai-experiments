"""training/models/architecture.py: the parameter accounting pinned for a
hand-computed tiny config, the reference-recipe counts against the published
source architectures, the Gemma 4 counts against the exact figures in
docs/training/gemma-4-training.md, the HF config mapping, and the adapter
counts. CPU-only, no GPU required."""

from __future__ import annotations

import json
import os
from types import SimpleNamespace

import pytest
from support import REFERENCE_DIR, REPO_ROOT, reference_recipe

from training.models.architecture import (
    ArchitectureSpec,
    UnknownArchitectureError,
    adapter_param_count,
    extra_params_from_checkpoint,
    host_ram_bytes,
    rounded_ffn,
    safetensors_header,
    spec_from_hf_config,
    spec_from_recipe,
    trainable_param_count,
)
from training.recipes import load_recipe

# Hand-computed tiny config: layers=2, hidden=64, heads=4, kv_heads=2,
# ffn_mult=2.0 (-> ffn 128), vocab=256, tied embeddings.
#   kv_dim = 64 * 2 / 4 = 32
#   attention/layer = 2*64*64 + 2*64*32          = 12,288
#   mlp/layer       = 3*64*128                   = 24,576
#   norms/layer     = 2*64                       = 128
#   embeddings + final norm = 256*64 + 64        = 16,448
#   total = 16,448 + 2*(12,288+24,576+128)       = 90,432
TINY_PARAMS = 90_432
TINY_OVERRIDES = [
    "model.scratch.layers=2",
    "model.scratch.hidden=64",
    "model.scratch.heads=4",
    "model.scratch.kv_heads=2",
    "model.scratch.ffn_mult=2.0",
    "model.scratch.vocab=256",
    "data.seq_len=16",
    "data.micro_batch=2",
]


@pytest.fixture()
def tiny_recipe():
    return reference_recipe("tiny-e2e.yaml", TINY_OVERRIDES)


def load_reference(name: str, overrides=()):
    return reference_recipe(name, list(overrides))


class TestFfnRounding:
    """Resolved design decision 3: pinned against the source architectures."""

    def test_smollm2_135m_intermediate(self):
        assert rounded_ffn(2.67, 576) == 1536

    def test_llama32_1b_intermediate(self):
        assert rounded_ffn(4.0, 2048) == 8192


class TestParamCount:
    def test_tiny_hand_computed(self, tiny_recipe):
        assert spec_from_recipe(tiny_recipe).param_count() == TINY_PARAMS

    def test_pretrain_150m(self):
        # Scaled Gemma 4 block: 16 layers at hidden 384 over the 262144-entry
        # tied table, which is 100.7M of the total on its own.
        spec = spec_from_recipe(load_reference("pretrain-150m.yaml"))
        assert spec.param_count() == 134_295_936

    def test_pretrain_1b(self):
        spec = spec_from_recipe(load_reference("pretrain-1b.yaml"))
        assert spec.param_count() == 1_007_869_440

    def test_untied_embeddings_add_lm_head(self, tiny_recipe):
        tied = spec_from_recipe(tiny_recipe)
        untied = ArchitectureSpec(
            layers=tied.layers,
            hidden=tied.hidden,
            heads=tied.heads,
            kv_heads=tied.kv_heads,
            ffn=tied.ffn,
            vocab=tied.vocab,
            tie_embeddings=False,
        )
        assert untied.param_count() == TINY_PARAMS + 256 * 64


class TestHostRam:
    def test_offload_moves_optimizer_state_to_host_ram(self):
        recipe = reference_recipe(
            "tiny-e2e.yaml",
            TINY_OVERRIDES + ["memory.offload=zero_offload", "optimizer.name=cpu_adam"],
        )
        spec = spec_from_recipe(recipe)
        assert host_ram_bytes(spec, recipe) == 16 * TINY_PARAMS

    def test_no_offload_no_host_ram(self, tiny_recipe):
        assert host_ram_bytes(spec_from_recipe(tiny_recipe), tiny_recipe) == 0


class TestHFConfigMapping:
    class FakeLlamaConfig:
        model_type = "llama"
        num_hidden_layers = 16
        hidden_size = 2048
        num_attention_heads = 32
        num_key_value_heads = 8
        intermediate_size = 8192
        vocab_size = 128256
        tie_word_embeddings = True

    def test_llama_family_maps(self):
        spec = spec_from_hf_config(self.FakeLlamaConfig())
        assert spec.layers == 16 and spec.kv_dim == 512 and spec.ffn == 8192

    def test_unknown_model_type_fails_loudly(self):
        class Mamba:
            model_type = "mamba"

        with pytest.raises(UnknownArchitectureError, match="mamba"):
            spec_from_hf_config(Mamba())

    def test_missing_field_fails_loudly(self):
        class Broken:
            model_type = "llama"

        with pytest.raises(UnknownArchitectureError, match="expected field"):
            spec_from_hf_config(Broken())


# -- Gemma 4 ----------------------------------------------------------------


def gemma4_facts() -> dict:
    """The canonical Gemma 4 geometry, so fixtures cannot drift from it."""
    path = os.path.join(REPO_ROOT, "schemas", "gemma4_facts.json")
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def contract_hf_config(variant: str = "E2B"):
    """An HF-shaped text config for one real variant, built from the contract.

    Earlier revisions of this fixture carried the Gemma4TextConfig class
    defaults rather than any released checkpoint's resolved text_config.
    Sourcing it from schemas/gemma4_facts.json keeps the mapping tests honest
    about what the mapping will actually be handed.
    """
    facts = gemma4_facts()
    v = facts["variants"][variant]
    inv = facts["invariants"]
    full = set(v["layer_types_full_attention_indices"])
    return SimpleNamespace(
        model_type=v["model_type"],
        num_hidden_layers=v["num_hidden_layers"],
        hidden_size=v["hidden_size"],
        num_attention_heads=v["num_attention_heads"],
        num_key_value_heads=v["num_key_value_heads"],
        head_dim=inv["head_dim"],
        global_head_dim=inv["global_head_dim"],
        intermediate_size=v["intermediate_size"],
        vocab_size=inv["vocab_size"],
        tie_word_embeddings=inv["tie_word_embeddings"],
        vocab_size_per_layer_input=inv["vocab_size_per_layer_input"],
        hidden_size_per_layer_input=v["hidden_size_per_layer_input"],
        use_double_wide_mlp=v["use_double_wide_mlp"],
        num_kv_shared_layers=v["num_kv_shared_layers"],
        layer_types=[
            "full_attention" if i in full else "sliding_attention"
            for i in range(v["num_hidden_layers"])
        ],
    )


# Exact language-model parameter counts summed from the safetensors headers
# (docs/training/gemma-4-training.md, Parameter accounting), minus one
# layer_scalar buffer per layer, which is not a parameter.
LANGUAGE_MODEL_PARAMS = {"E2B": 4_647_449_891 - 35, "E4B": 7_518_069_034 - 42}
ATTENTION_PARAMS = {"E2B": 297_292_800, "E4B": 642_277_888}
PLE_PARAMS = {
    "E2B": 2_348_810_240 + 41_341_696,
    "E4B": 2_818_572_288 + 82_683_136,
}


def gemma4_text_config():
    return contract_hf_config("E2B")


class TestGemma4Accounting:
    def test_moe_config_is_refused(self):
        # The 26B-A4B MoE shape shares model_type gemma4_text; pricing it
        # with dense-MLP formulas would violate the fail-loudly policy.
        config = SimpleNamespace(
            **{**gemma4_text_config().__dict__, "enable_moe_block": True}
        )
        with pytest.raises(UnknownArchitectureError, match="MoE"):
            spec_from_hf_config(config)

    def test_text_config_maps_with_explicit_head_dim(self):
        spec = spec_from_hf_config(gemma4_text_config())
        assert spec.head_dim == 256  # not hidden // heads (192)
        assert spec.q_dim == 8 * 256
        assert spec.kv_dim == 1 * 256  # E2B is single-KV-head
        assert spec.ple_vocab == 262_144 and spec.ple_dim == 256
        assert spec.full_attention_layers == 7 and spec.full_head_dim == 512
        assert spec.block_norms == 4 and spec.qk_norms
        assert spec.family == "gemma4_text"

    @pytest.mark.parametrize("variant", ["E2B", "E4B"])
    def test_language_model_count_matches_the_checkpoint_exactly(self, variant):
        """The count the 2026-09-19 fix was written against: the seven
        full-attention layers at global_head_dim, q and k norms, four block
        norms, and the PLE model projection and norms, each of which the
        one-width Llama derivation missed (docs/training/gemma-4-training.md,
        the former "Where the calculator undercounts").
        """
        spec = spec_from_hf_config(contract_hf_config(variant))
        assert spec.attention_param_count() == ATTENTION_PARAMS[variant]
        assert spec.ple_param_count() == PLE_PARAMS[variant]
        assert spec.param_count() == LANGUAGE_MODEL_PARAMS[variant]

    def test_towers_are_read_from_the_checkpoint_header(self, tmp_path):
        """A composite checkpoint's towers are priced from what the file holds,
        not modeled: every tensor outside model.language_model. counts."""
        header = {
            "__metadata__": {"format": "pt"},
            "model.language_model.embed_tokens.weight": {
                "dtype": "BF16",
                "shape": [262144, 1536],
                "data_offsets": [0, 8],
            },
            "model.vision_tower.patch.weight": {
                "dtype": "BF16",
                "shape": [768, 3, 16, 16],
                "data_offsets": [8, 16],
            },
            "model.embed_audio.embedding_projection.weight": {
                "dtype": "BF16",
                "shape": [1536, 1024],
                "data_offsets": [16, 24],
            },
        }
        body = json.dumps(header).encode()
        with open(tmp_path / "model.safetensors", "wb") as f:
            f.write(len(body).to_bytes(8, "little") + body + b"\0" * 24)
        assert safetensors_header(str(tmp_path / "model.safetensors")) == header
        assert (
            extra_params_from_checkpoint(str(tmp_path))
            == 768 * 3 * 16 * 16 + 1536 * 1024
        )
        assert extra_params_from_checkpoint(str(tmp_path / "absent")) == 0

    def test_extra_params_count_as_frozen_weights_only(self):
        from dataclasses import replace

        recipe = load_reference("sft-qlora-e2b.yaml")
        base = spec_from_hf_config(contract_hf_config("E2B"))
        with_towers = replace(base, extra_params=475_729_088)
        assert with_towers.param_count() == base.param_count() + 475_729_088
        assert with_towers.linear_param_count() == base.linear_param_count()
        assert with_towers.transformer_stack_param_count() == (
            base.transformer_stack_param_count()
        )
        assert adapter_param_count(with_towers, recipe) == adapter_param_count(
            base, recipe
        )

    def test_multimodal_config_descends_into_text_config(self):
        composite = SimpleNamespace(
            model_type="gemma4", text_config=gemma4_text_config()
        )
        assert spec_from_hf_config(composite).hidden == 1536

    def test_ple_params_counted(self):
        spec = spec_from_hf_config(gemma4_text_config())
        table = 262_144 * 35 * 256
        linears = 35 * 2 * 1536 * 256
        # The model-level projection and its norm, and one hidden-wide norm
        # per layer, were the terms the 2026-09-19 fix added.
        projection = 35 * 256 * 1536 + 256
        norms = 35 * 1536
        assert spec.ple_param_count() == table + linears + projection + norms
        without = spec_from_hf_config(
            SimpleNamespace(
                **{
                    **gemma4_text_config().__dict__,
                    "vocab_size_per_layer_input": 0,
                    "hidden_size_per_layer_input": 0,
                }
            )
        )
        assert spec.param_count() - without.param_count() == (
            table + linears + projection + norms
        )

    @pytest.mark.parametrize(
        ("variant", "double_wide_layers"), [("E2B", 20), ("E4B", 0)]
    )
    def test_double_wide_mlp_counted_on_kv_shared_layers(
        self, variant, double_wide_layers
    ):
        """E2B doubles the MLP on its KV-shared layers; E4B does not.

        Modeling the MLP as a flat 3*hidden*ffn on every layer undercounts
        E2B, because 20 of its 35 layers are double wide. The boundary is
        num_hidden_layers - num_kv_shared_layers, the same one the attention
        code uses (docs/models/gemma4-upstream.txt:1083-1088).
        """
        facts = gemma4_facts()
        v = facts["variants"][variant]
        spec = spec_from_hf_config(contract_hf_config(variant))
        layers = v["num_hidden_layers"]
        hidden = v["hidden_size"]
        ffn = v["intermediate_size"]

        assert spec.double_wide_layers == double_wide_layers
        # A double-wide layer contributes a second copy of its MLP block.
        mlp = 3 * hidden * ffn * (layers + double_wide_layers)
        attn = spec.attention_param_count()
        qk_norms = 2 * (
            spec.sliding_attention_layers * spec.head_dim
            + spec.full_attention_layers * spec.full_head_dim
        )
        norms = layers * 4 * hidden
        embeddings = facts["invariants"]["vocab_size"] * hidden
        expected = embeddings + attn + mlp + norms + hidden
        assert spec.param_count() == expected + spec.ple_param_count()
        assert spec.transformer_stack_param_count() == attn + mlp + norms + hidden
        assert spec.linear_param_count() == attn - qk_norms + mlp

    def test_e2b_double_wide_gap_is_the_documented_size(self):
        """Pins the magnitude the flat formula was missing."""
        v = gemma4_facts()["variants"]["E2B"]
        spec = spec_from_hf_config(contract_hf_config("E2B"))
        gap = 3 * v["hidden_size"] * v["intermediate_size"] * spec.double_wide_layers
        assert gap == 566_231_040

    def test_llama_lineage_numbers_unchanged(self):
        # No head_dim/PLE fields: the Phase 0 derivation must be identical.
        config = SimpleNamespace(
            model_type="llama",
            num_hidden_layers=16,
            hidden_size=2048,
            num_attention_heads=32,
            num_key_value_heads=8,
            intermediate_size=8192,
            vocab_size=128_256,
            tie_word_embeddings=True,
        )
        spec = spec_from_hf_config(config)
        assert spec.head_dim == 2048 // 32
        assert spec.q_dim == spec.hidden
        assert spec.kv_dim == spec.hidden * 8 // 32
        assert spec.ple_param_count() == 0
        assert spec.full_attention_layers == 0 and spec.block_norms == 2
        assert not spec.qk_norms and spec.family == "llama"
        per_layer = 2 * 2048 * 2048 + 2 * 2048 * 512 + 3 * 2048 * 8192 + 2 * 2048
        assert spec.param_count() == 128_256 * 2048 + 16 * per_layer + 2048


class TestPeftCounts:
    """F2: adapter param counts, hand-computed on the tiny config. rank=16
    over the seven Llama-lineage modules:
      q/o: 16*(64+64)=2048 each; k/v: 16*(64+32)=1536 each;
      gate/up: 16*(64+128)=3072 each; down: 16*(128+64)=3072
      per layer 16,384; two layers -> 32,768 adapter params.
    Linear (quantizable) params: 2*(12,288+24,576) = 73,728."""

    ADAPTER_PARAMS = 32_768
    LINEAR_PARAMS = 73_728

    def qlora_recipe(self, extra=()):
        return load_recipe(
            os.path.join(REFERENCE_DIR, "tiny-e2e.yaml"),
            TINY_OVERRIDES
            + ["peft.enabled=true", "peft.rank=16", "peft.quantize_base=nf4"]
            + list(extra),
        )

    def test_adapter_param_count(self):
        recipe = self.qlora_recipe()
        spec = spec_from_recipe(recipe)
        assert adapter_param_count(spec, recipe) == self.ADAPTER_PARAMS
        assert trainable_param_count(spec, recipe) == self.ADAPTER_PARAMS

    def test_dora_adds_magnitude_vectors(self):
        # One magnitude scalar per output column: per layer
        # 64+32+32+64+128+128+64 = 512; two layers -> +1024.
        recipe = self.qlora_recipe(["peft.method=dora"])
        spec = spec_from_recipe(recipe)
        assert adapter_param_count(spec, recipe) == self.ADAPTER_PARAMS + 1024

    def test_linear_param_count(self):
        spec = spec_from_recipe(self.qlora_recipe())
        assert spec.linear_param_count() == self.LINEAR_PARAMS

    def test_unknown_target_module_fails(self):
        recipe = self.qlora_recipe(["peft.target_modules=[qkv_proj]"])
        spec = spec_from_recipe(recipe)
        with pytest.raises(UnknownArchitectureError, match="qkv_proj"):
            adapter_param_count(spec, recipe)

    def test_peft_disabled_trains_every_parameter(self, tiny_recipe):
        spec = spec_from_recipe(tiny_recipe)
        assert trainable_param_count(spec, tiny_recipe) == TINY_PARAMS
