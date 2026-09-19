"""Memory-fit calculator tests (Phase 0 D4, reduced 2026-09-19): static
component numbers pinned for a hand-computed tiny config, reference-recipe
param counts pinned against the published source architectures, the Gemma 4
parameter accounting against the exact counts in
docs/training/gemma-4-training.md, the measured-rate estimate and its
unmeasured advisory, the HF config mapping, and the CLI JSON contract.
CPU-only, no GPU required (roadmap D4 acceptance bullet 3)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest
from support import REFERENCE_DIR, REPO_ROOT, reference_recipe

from training.calculator import (
    ArchitectureSpec,
    UnknownArchitectureError,
    adapter_param_count,
    estimate,
    grad_bytes,
    host_ram_bytes,
    measured,
    optimizer_bytes,
    profile,
    spec_from_recipe,
    trainable_param_count,
    weights_bytes,
)
from training.calculator.components import (
    extra_params_from_checkpoint,
    rounded_ffn,
    safetensors_header,
    spec_from_hf_config,
)
from training.calculator.estimate import OVERHEAD_BYTES, VRAM_BUDGET_BYTES
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


class TestComponentBytes:
    """Every number below is TINY_PARAMS-derived by hand; see each assert."""

    def test_weights_bf16(self, tiny_recipe):
        spec = spec_from_recipe(tiny_recipe)
        assert weights_bytes(spec, tiny_recipe) == 2 * TINY_PARAMS

    def test_grads_model_dtype(self, tiny_recipe):
        # backward() produces gradients in the parameter dtype (bf16), not
        # fp32 — no trainer implements separate-dtype accumulation (KB-55).
        spec = spec_from_recipe(tiny_recipe)
        assert grad_bytes(spec, tiny_recipe) == 2 * TINY_PARAMS

    def test_adamw_with_master_weights(self, tiny_recipe):
        # 8 B/param fp32 states + 4 B/param master copy under bf16 params.
        spec = spec_from_recipe(tiny_recipe)
        assert optimizer_bytes(spec, tiny_recipe) == 12 * TINY_PARAMS

    def test_adamw8bit(self, tiny_recipe):
        # 2 B/param + two fp32 absmax constants per 256-param block.
        recipe = load_recipe(
            os.path.join(REFERENCE_DIR, "tiny-e2e.yaml"),
            TINY_OVERRIDES + ["optimizer.name=adamw8bit"],
        )
        spec = spec_from_recipe(recipe)
        expected = 2 * TINY_PARAMS + 2 * 4 * (TINY_PARAMS // 256)
        assert optimizer_bytes(spec, recipe) == expected

    def test_paged_adamw8bit_budgets_as_adamw8bit(self, tiny_recipe):
        # Conservative fit: paged states may live in VRAM, so the model is
        # identical to adamw8bit (WI-7 fallback rung).
        recipe = load_recipe(
            os.path.join(REFERENCE_DIR, "tiny-e2e.yaml"),
            TINY_OVERRIDES + ["optimizer.name=paged_adamw8bit"],
        )
        spec = spec_from_recipe(recipe)
        expected = 2 * TINY_PARAMS + 2 * 4 * (TINY_PARAMS // 256)
        assert optimizer_bytes(spec, recipe) == expected

    def test_offload_moves_optimizer_to_host_ram(self, tiny_recipe):
        recipe = load_recipe(
            os.path.join(REFERENCE_DIR, "tiny-e2e.yaml"),
            TINY_OVERRIDES + ["optimizer.name=cpu_adam", "memory.offload=zero_offload"],
        )
        spec = spec_from_recipe(recipe)
        assert optimizer_bytes(spec, recipe) == 0
        assert host_ram_bytes(spec, recipe) == 16 * TINY_PARAMS

    def test_no_offload_no_host_ram(self, tiny_recipe):
        assert host_ram_bytes(spec_from_recipe(tiny_recipe), tiny_recipe) == 0


class TestEstimate:
    """The estimate is static terms plus a measured rate times tokens, or a
    static lower bound that says it is unmeasured (2026-09-19 reduction)."""

    def test_unmeasured_profile_is_a_lower_bound(self, tiny_recipe, monkeypatch):
        monkeypatch.setattr(measured, "RATES", {})
        est = estimate(tiny_recipe)
        assert est.profile == "scratch/2x64/full/nockpt"
        assert est.rate is None and est.activations is None
        assert est.fits is None
        assert est.allocated_total == est.weights + est.grads + est.optimizer
        assert est.reserved_estimate == est.allocated_total + OVERHEAD_BYTES
        assert est.param_count == TINY_PARAMS

    def test_measured_profile_prices_tokens(self, tiny_recipe, monkeypatch):
        monkeypatch.setattr(measured, "RATES", {"scratch/2x64/full/nockpt": 1_000})
        est = estimate(tiny_recipe)
        assert est.rate == 1_000
        assert est.activations == 1_000 * 2 * 16
        assert est.allocated_total == (
            est.weights + est.grads + est.optimizer + est.activations
        )
        assert est.fits is True

    def test_rate_past_the_budget_does_not_fit(self, tiny_recipe, monkeypatch):
        per_token = VRAM_BUDGET_BYTES // (2 * 16) + 1
        monkeypatch.setattr(measured, "RATES", {"scratch/2x64/full/nockpt": per_token})
        est = estimate(tiny_recipe)
        assert est.fits is False
        assert est.reserved_estimate > VRAM_BUDGET_BYTES

    def test_profile_reads_mode_and_checkpointing(self, tiny_recipe):
        spec = spec_from_recipe(tiny_recipe)
        assert profile(spec, tiny_recipe) == "scratch/2x64/full/nockpt"
        ckpt = load_recipe(
            os.path.join(REFERENCE_DIR, "tiny-e2e.yaml"),
            TINY_OVERRIDES + ["memory.gradient_checkpointing=true"],
        )
        assert profile(spec, ckpt) == "scratch/2x64/full/ckpt"
        qlora = load_recipe(
            os.path.join(REFERENCE_DIR, "tiny-e2e.yaml"),
            TINY_OVERRIDES + ["peft.enabled=true", "peft.quantize_base=nf4"],
        )
        assert profile(spec, qlora) == "scratch/2x64/qlora/nockpt"
        lora = load_recipe(
            os.path.join(REFERENCE_DIR, "tiny-e2e.yaml"),
            TINY_OVERRIDES + ["peft.enabled=true", "peft.quantize_base=none"],
        )
        assert profile(spec, lora) == "scratch/2x64/lora/nockpt"

    def test_pretrain_1b_profile_and_static_terms(self):
        recipe = load_reference("pretrain-1b.yaml")
        spec = spec_from_recipe(recipe)
        assert profile(spec, recipe) == "scratch/18x1536/full/ckpt"
        est = estimate(recipe)
        assert recipe.memory.offload == "none"
        assert est.host_ram == 0
        # The static terms alone sit well inside the budget, whatever the rate.
        assert (
            est.weights + est.grads + est.optimizer + OVERHEAD_BYTES < VRAM_BUDGET_BYTES
        )

    def test_every_reference_rate_names_a_real_profile_shape(self):
        # A recorded rate is a measurement of one geometry; the key must parse.
        for key, rate in measured.RATES.items():
            family, geometry, mode, ckpt = key.split("/")
            layers, hidden = geometry.split("x")
            assert int(layers) > 0 and int(hidden) > 0
            assert mode in ("full", "lora", "qlora") and ckpt in ("ckpt", "nockpt")
            assert isinstance(rate, int) and rate >= 0


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


# -- D4 calculator: Gemma 4 -------------------------------------------------


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
    about what the calculator will actually be handed.
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


class TestCalculatorGemma4:
    def test_moe_config_is_refused(self):
        # The 26B-A4B MoE shape shares model_type gemma4_text; pricing it
        # with dense-MLP formulas would violate the fail-loudly policy.
        from training.calculator.components import UnknownArchitectureError

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
        the former "Where the calculator undercounts")."""
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

    def test_extra_params_price_as_weights_not_as_trainable(self):
        from dataclasses import replace

        recipe = load_reference("sft-qlora-e2b.yaml")
        base = spec_from_hf_config(contract_hf_config("E2B"))
        with_towers = replace(base, extra_params=475_729_088)
        assert with_towers.param_count() == base.param_count() + 475_729_088
        assert with_towers.linear_param_count() == base.linear_param_count()
        assert with_towers.transformer_stack_param_count() == (
            base.transformer_stack_param_count()
        )
        # Towers stay in the compute dtype under QLoRA (never quantized).
        assert weights_bytes(with_towers, recipe) == (
            weights_bytes(base, recipe) + 475_729_088 * 2
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


class TestCLI:
    @pytest.fixture(autouse=True)
    def _obs_log_dir(self, tmp_path):
        # python -m training writes obs event files (roadmap OBS-4); keep
        # test runs out of the real /mnt/ai-data/logs
        self.obs_log_dir = str(tmp_path / "obs-logs")

    def run_cli(self, *args: str, env: dict[str, str] | None = None):
        return subprocess.run(
            [sys.executable, "-m", "training", *args],
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
            env={
                **os.environ,
                "NO_COLOR": "1",
                "OBS_LOG_DIR": self.obs_log_dir,
                **(env or {}),
            },
            timeout=120,
        )

    def test_memfit_json_contract(self):
        proc = self.run_cli(
            "memfit",
            "--config",
            "training/recipes/reference/pretrain-1b.yaml",
            "--json",
        )
        assert proc.returncode == 0, proc.stderr
        result = json.loads(proc.stdout)  # stdout must be JSON only
        est = result["estimate"]
        assert est["profile"] == "scratch/18x1536/full/ckpt"
        assert est["param_count"] == 1_007_869_440
        # Unmeasured until a from-scratch calibrate records the rate.
        assert est["rate"] is None and est["activations"] is None
        assert est["fits"] is None
        assert "max_batch" not in result

    def test_memfit_human_table(self):
        proc = self.run_cli(
            "memfit", "--config", "training/recipes/reference/pretrain-1b.yaml"
        )
        assert proc.returncode == 0, proc.stderr
        # Human status lives on stderr; stdout is reserved for --json payloads.
        assert proc.stdout == ""
        assert "=== Memory-fit estimate ===" in proc.stderr
        assert (
            "[WARN] Activations unmeasured for profile scratch/18x1536/full/ckpt"
            in (proc.stderr)
        )
        assert "[OK] Fits:" not in proc.stderr

    def test_memfit_solve_flag_is_gone(self):
        proc = self.run_cli(
            "memfit",
            "--config",
            "training/recipes/reference/pretrain-1b.yaml",
            "--solve",
            "batch",
        )
        assert proc.returncode == 2
        assert "unrecognized arguments: --solve" in proc.stderr

    def test_memfit_bad_recipe_exits_nonzero(self):
        proc = self.run_cli("memfit", "--config", "/nonexistent/recipe.yaml")
        assert proc.returncode == 1
        assert "[ERROR]" in proc.stderr

    def test_calibrate_requires_gpu(self):
        # CUDA_VISIBLE_DEVICES="" hides the workstation GPU from the child, so
        # the no-CUDA arm is exercised on the host as well as in the sandbox.
        proc = self.run_cli(
            "calibrate",
            "--config",
            "training/recipes/reference/tiny-e2e.yaml",
            env={"CUDA_VISIBLE_DEVICES": ""},
        )
        assert proc.returncode == 1
        assert "[ERROR] Calibration failed" in proc.stderr


class TestPeftComponents:
    """F2: adapter param counts and the QLoRA memory model, hand-computed on
    the tiny config. rank=16 over the seven Llama-lineage modules:
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

    def test_qlora_weights_double_quant(self):
        # NF4+dq: 0.5 + 1/64 + 4/(64*256) B/param on linears; bf16 rest and
        # adapter. 73728*0.515869140625=38034; rest (90432-73728)*2=33408;
        # adapter 32768*2=65536.
        recipe = self.qlora_recipe()
        spec = spec_from_recipe(recipe)
        assert weights_bytes(spec, recipe) == 38_034 + 33_408 + 65_536

    def test_qlora_weights_plain_nf4(self):
        # Without dq: 0.5 + 4/64 B/param -> 73728*0.5625 = 41472.
        recipe = self.qlora_recipe(["peft.double_quant=false"])
        spec = spec_from_recipe(recipe)
        assert weights_bytes(spec, recipe) == 41_472 + 33_408 + 65_536

    def test_lora_unquantized_base_keeps_precision_dtype(self):
        recipe = self.qlora_recipe(["peft.quantize_base=none"])
        spec = spec_from_recipe(recipe)
        assert weights_bytes(spec, recipe) == 2 * TINY_PARAMS + 2 * self.ADAPTER_PARAMS

    def test_grads_and_optimizer_cover_adapter_only(self):
        # Adapter gradients live in peft.compute_dtype (bf16), 2 B/param.
        recipe = self.qlora_recipe()
        spec = spec_from_recipe(recipe)
        assert grad_bytes(spec, recipe) == 2 * self.ADAPTER_PARAMS
        assert optimizer_bytes(spec, recipe) == 12 * self.ADAPTER_PARAMS

    def test_estimate_reports_trainable_params(self):
        recipe = self.qlora_recipe()
        est = estimate(recipe)
        assert est.param_count == TINY_PARAMS
        assert est.trainable_params == self.ADAPTER_PARAMS
        assert est.profile == "scratch/2x64/qlora/nockpt"

    def test_unknown_target_module_fails(self):
        recipe = self.qlora_recipe(["peft.target_modules=[qkv_proj]"])
        spec = spec_from_recipe(recipe)
        with pytest.raises(UnknownArchitectureError, match="qkv_proj"):
            adapter_param_count(spec, recipe)

    def test_peft_disabled_matches_phase0_numbers(self, tiny_recipe):
        spec = spec_from_recipe(tiny_recipe)
        assert trainable_param_count(spec, tiny_recipe) == TINY_PARAMS
        assert weights_bytes(spec, tiny_recipe) == 2 * TINY_PARAMS
