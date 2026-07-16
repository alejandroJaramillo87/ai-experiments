"""Memory-fit calculator tests (Phase 0 D4): closed-form component numbers
pinned for a hand-computed tiny config, reference-recipe param counts pinned
against the published source architectures, solve() behavior, the HF config
mapping, and the CLI JSON contract. CPU-only, no GPU required (roadmap D4
acceptance bullet 3)."""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from training.calculator import (
    ArchitectureSpec,
    UnknownArchitectureError,
    activation_bytes,
    adapter_param_count,
    estimate,
    grad_bytes,
    host_ram_bytes,
    optimizer_bytes,
    solve,
    spec_from_recipe,
    trainable_param_count,
    weights_bytes,
)
from training.calculator.components import rounded_ffn, spec_from_hf_config
from training.recipes import load_recipe

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REFERENCE_DIR = os.path.join(REPO_ROOT, "training", "recipes", "reference")

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
    "model.scratch.layers=2", "model.scratch.hidden=64", "model.scratch.heads=4",
    "model.scratch.kv_heads=2", "model.scratch.ffn_mult=2.0", "model.scratch.vocab=256",
    "data.seq_len=16", "data.micro_batch=2",
]


@pytest.fixture()
def tiny_recipe():
    return load_recipe(os.path.join(REFERENCE_DIR, "tiny-e2e.yaml"), TINY_OVERRIDES)


def load_reference(name: str, overrides=()):
    return load_recipe(os.path.join(REFERENCE_DIR, name), overrides)


class TestFfnRounding:
    """Resolved design decision 3: pinned against the source architectures."""

    def test_smollm2_135m_intermediate(self):
        assert rounded_ffn(2.67, 576) == 1536

    def test_llama32_1b_intermediate(self):
        assert rounded_ffn(4.0, 2048) == 8192


class TestParamCount:
    def test_tiny_hand_computed(self, tiny_recipe):
        assert spec_from_recipe(tiny_recipe).param_count() == TINY_PARAMS

    def test_pretrain_150m_matches_smollm2(self):
        # SmolLM2-135M shape at 49152 vocab: published count 134,515,008.
        spec = spec_from_recipe(load_reference("pretrain-150m.yaml"))
        assert spec.param_count() == 134_515_008

    def test_pretrain_1b(self):
        spec = spec_from_recipe(load_reference("pretrain-1b.yaml"))
        assert spec.param_count() == 1_073_809_408

    def test_untied_embeddings_add_lm_head(self, tiny_recipe):
        tied = spec_from_recipe(tiny_recipe)
        untied = ArchitectureSpec(
            layers=tied.layers, hidden=tied.hidden, heads=tied.heads,
            kv_heads=tied.kv_heads, ffn=tied.ffn, vocab=tied.vocab,
            tie_embeddings=False)
        assert untied.param_count() == TINY_PARAMS + 256 * 64


class TestComponentBytes:
    """Every number below is TINY_PARAMS-derived by hand; see each assert."""

    def test_weights_bf16(self, tiny_recipe):
        spec = spec_from_recipe(tiny_recipe)
        assert weights_bytes(spec, tiny_recipe) == 2 * TINY_PARAMS

    def test_grads_fp32(self, tiny_recipe):
        spec = spec_from_recipe(tiny_recipe)
        assert grad_bytes(spec, tiny_recipe) == 4 * TINY_PARAMS

    def test_adamw_with_master_weights(self, tiny_recipe):
        # 8 B/param fp32 states + 4 B/param master copy under bf16 params.
        spec = spec_from_recipe(tiny_recipe)
        assert optimizer_bytes(spec, tiny_recipe) == 12 * TINY_PARAMS

    def test_adamw8bit(self, tiny_recipe):
        # 2 B/param + two fp32 absmax constants per 256-param block.
        recipe = load_recipe(
            os.path.join(REFERENCE_DIR, "tiny-e2e.yaml"),
            TINY_OVERRIDES + ["optimizer.name=adamw8bit"])
        spec = spec_from_recipe(recipe)
        expected = 2 * TINY_PARAMS + 2 * 4 * (TINY_PARAMS // 256)
        assert optimizer_bytes(spec, recipe) == expected

    def test_offload_moves_optimizer_to_host_ram(self, tiny_recipe):
        recipe = load_recipe(
            os.path.join(REFERENCE_DIR, "tiny-e2e.yaml"),
            TINY_OVERRIDES + ["optimizer.name=cpu_adam", "memory.offload=zero_offload"])
        spec = spec_from_recipe(recipe)
        assert optimizer_bytes(spec, recipe) == 0
        assert host_ram_bytes(spec, recipe) == 16 * TINY_PARAMS

    def test_no_offload_no_host_ram(self, tiny_recipe):
        assert host_ram_bytes(spec_from_recipe(tiny_recipe), tiny_recipe) == 0

    def test_activations_hand_computed(self, tiny_recipe):
        # b=2, s=16 (tokens=32), d=2, hidden=64, kv_dim=32, ffn=128, heads=4:
        #   attn  = 32*2*(3*64 + 2*32)  = 16,384
        #   mlp   = 32*2*(2*64 + 3*128) = 32,768
        #   norms = 32*2*2*64           =  8,192
        #   lse   = 2*4*16*4            =    512
        #   per layer = 57,856; x2 layers = 115,712
        #   logits = 32*2*64 + 32*256*(2+4) = 53,248
        spec = spec_from_recipe(tiny_recipe)
        assert activation_bytes(spec, tiny_recipe) == 115_712 + 53_248

    def test_activations_gradient_checkpointing(self, tiny_recipe):
        # layer inputs 2*32*2*64 = 8,192 + one recomputed layer 57,856 + logits.
        recipe = load_recipe(
            os.path.join(REFERENCE_DIR, "tiny-e2e.yaml"),
            TINY_OVERRIDES + ["memory.gradient_checkpointing=true"])
        spec = spec_from_recipe(recipe)
        assert activation_bytes(spec, recipe) == 8_192 + 57_856 + 53_248

    def test_activations_linear_in_batch_and_seq(self, tiny_recipe):
        spec = spec_from_recipe(tiny_recipe)
        base = activation_bytes(spec, tiny_recipe, micro_batch=2, seq_len=16)
        assert activation_bytes(spec, tiny_recipe, micro_batch=4, seq_len=16) == 2 * base
        assert activation_bytes(spec, tiny_recipe, micro_batch=2, seq_len=32) == 2 * base


class TestEstimate:
    def test_totals_are_component_sums(self, tiny_recipe):
        est = estimate(tiny_recipe)
        assert est.allocated_total == (
            est.weights + est.grads + est.optimizer + est.activations)
        assert est.reserved_estimate == est.allocated_total + est.overhead
        assert est.param_count == TINY_PARAMS

    def test_pretrain_1b_fits_without_offload(self):
        # Roadmap D4 acceptance: the 1B envelope config fits in 32GB.
        recipe = load_reference("pretrain-1b.yaml")
        est = estimate(recipe)
        assert recipe.memory.offload == "none"
        assert est.fits
        assert est.host_ram == 0

    def test_unfittable_config_reports_not_fits(self):
        est = estimate(load_reference("pretrain-1b.yaml", ["data.micro_batch=512"]))
        assert not est.fits


class TestSolve:
    def test_invalid_axis_rejected(self, tiny_recipe):
        with pytest.raises(ValueError, match="axis"):
            solve(tiny_recipe, "layers")

    def test_max_batch_boundary(self):
        recipe = load_reference("pretrain-1b.yaml")
        solved = solve(recipe, "batch")
        assert solved is not None and solved.fits
        assert solved.micro_batch >= recipe.data.micro_batch
        spec = spec_from_recipe(recipe)
        from training.calculator.solve import _estimate_at

        assert not _estimate_at(spec, recipe, solved.micro_batch + 1,
                                recipe.data.seq_len).fits

    def test_max_seq_boundary(self):
        recipe = load_reference("pretrain-1b.yaml")
        solved = solve(recipe, "seq")
        assert solved is not None and solved.fits
        assert solved.seq_len % 128 == 0
        spec = spec_from_recipe(recipe)
        from training.calculator.solve import _estimate_at

        assert not _estimate_at(spec, recipe, recipe.data.micro_batch,
                                solved.seq_len + 128).fits

    def test_nothing_fits_returns_none(self):
        # A model whose static memory alone exceeds 32GB.
        recipe = load_reference(
            "pretrain-1b.yaml",
            ["model.scratch.layers=200", "model.scratch.hidden=8192"])
        assert solve(recipe, "batch") is None


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


class TestCLI:
    def run_cli(self, *args: str):
        return subprocess.run(
            [sys.executable, "-m", "training", *args],
            capture_output=True, text=True, cwd=REPO_ROOT,
            env={**os.environ, "NO_COLOR": "1"}, timeout=120)

    def test_memfit_json_contract(self):
        proc = self.run_cli(
            "memfit", "--config", "training/recipes/reference/pretrain-1b.yaml",
            "--solve", "batch", "--json")
        assert proc.returncode == 0, proc.stderr
        result = json.loads(proc.stdout)   # stdout must be JSON only
        assert result["estimate"]["fits"] is True
        assert result["max_batch"] >= 4

    def test_memfit_human_table(self):
        proc = self.run_cli(
            "memfit", "--config", "training/recipes/reference/pretrain-1b.yaml")
        assert proc.returncode == 0, proc.stderr
        assert "=== Memory-fit estimate ===" in proc.stdout
        assert "[OK] Fits:" in proc.stdout

    def test_memfit_unfit_exits_nonzero(self):
        proc = self.run_cli(
            "memfit", "--config", "training/recipes/reference/pretrain-1b.yaml",
            "data.micro_batch=512")
        assert proc.returncode == 1
        assert "[ERROR] Does not fit" in proc.stdout

    def test_memfit_solve_nothing_fits_exits_nonzero(self):
        # Sequence length floor of the solver still overflows at an absurd
        # micro-batch: --solve must exit 1 when nothing fits, per the
        # exit-code contract, not print [ERROR] and exit 0.
        proc = self.run_cli(
            "memfit", "--config", "training/recipes/reference/pretrain-1b.yaml",
            "--solve", "seq", "data.micro_batch=100000")
        assert proc.returncode == 1
        assert "[ERROR]" in proc.stdout

    def test_memfit_bad_recipe_exits_nonzero(self):
        proc = self.run_cli("memfit", "--config", "/nonexistent/recipe.yaml")
        assert proc.returncode == 1
        assert "[ERROR]" in proc.stdout

    def test_calibrate_requires_gpu(self):
        proc = self.run_cli(
            "calibrate", "--config", "training/recipes/reference/tiny-e2e.yaml")
        assert proc.returncode == 1
        assert "[ERROR] Calibration failed" in proc.stdout


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
            TINY_OVERRIDES + ["peft.enabled=true", "peft.rank=16",
                              "peft.quantize_base=nf4"] + list(extra))

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
        recipe = self.qlora_recipe()
        spec = spec_from_recipe(recipe)
        assert grad_bytes(spec, recipe) == 4 * self.ADAPTER_PARAMS
        assert optimizer_bytes(spec, recipe) == 12 * self.ADAPTER_PARAMS

    def test_estimate_reports_trainable_params(self):
        recipe = self.qlora_recipe()
        est = estimate(recipe)
        assert est.param_count == TINY_PARAMS
        assert est.trainable_params == self.ADAPTER_PARAMS

    def test_unknown_target_module_fails(self):
        recipe = self.qlora_recipe(["peft.target_modules=[qkv_proj]"])
        spec = spec_from_recipe(recipe)
        with pytest.raises(UnknownArchitectureError, match="qkv_proj"):
            adapter_param_count(spec, recipe)

    def test_peft_disabled_matches_phase0_numbers(self, tiny_recipe):
        spec = spec_from_recipe(tiny_recipe)
        assert trainable_param_count(spec, tiny_recipe) == TINY_PARAMS
        assert weights_bytes(spec, tiny_recipe) == 2 * TINY_PARAMS
