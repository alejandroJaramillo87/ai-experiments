"""Closed-form VRAM component formulas (Phase 0 D4).

Every function takes the typed recipe (or the ArchitectureSpec derived from
it) and returns bytes. The formulas model the dense transformer family the
recipes describe: pre-norm attention (GQA, flash-attn) plus SwiGLU MLP with
RMSNorm, i.e. the Llama-family shape that HF LlamaConfig builds and that the
calibration harness measures.

Unit tests pin these numbers for a hand-computed tiny config so refactors
cannot silently change the math; the calibration harness (calibrate.py)
refines the calibrated constants against measured peaks on the workstation.

Phase 0 scope: bf16/fp8 parameter dtypes with all parameters trainable. The
NF4 4-bit base + trainable adapter weights term (QLoRA, Phase 1) is an
additive extension to weights_bytes/grad_bytes once the schema can express
adapters.
"""

from __future__ import annotations

from dataclasses import dataclass

from omegaconf import DictConfig

# Bytes per element for the dtypes the schema can express. fp8 follows the
# document's weights model (P * bytes(param_dtype)); fp8 recipes are validated
# against measurement when D8 lands.
DTYPE_BYTES = {"fp32": 4, "bf16": 2, "fp8": 1}

# Hidden-to-intermediate rounding: ffn = round(ffn_mult * hidden / 64) * 64.
# Pinned against the source architectures the reference recipes cite:
# SmolLM2-135M (576 -> 1536 at ffn_mult 2.67) and Llama-3.2-1B (2048 -> 8192
# at ffn_mult 4.0). The calibration harness builds models from the same spec,
# so calculator and measured model cannot disagree on this rounding.
FFN_ROUND_MULTIPLE = 64

# bitsandbytes 8-bit Adam: two 1-byte states per param plus per-block fp32
# constants (absmax per state) at the default block size of 256.
ADAM8BIT_BLOCK_SIZE = 256

# ZeRO-Offload host-RAM model: fp32 optimizer states (8 B/param) plus fp32
# master params (4 B/param) plus pinned fp32 gradient transfer buffers
# (4 B/param) live in host RAM instead of VRAM.
OFFLOAD_HOST_BYTES_PER_PARAM = 16

# HF AutoConfig model_type values whose configs carry Llama-family field names
# and match the dense attention + SwiGLU shape these formulas model. Unknown
# architectures fail loudly rather than guess (roadmap D4 decision 2).
HF_SUPPORTED_MODEL_TYPES = ("llama", "mistral", "qwen2", "qwen3", "gemma", "gemma2", "gemma3")


class UnknownArchitectureError(ValueError):
    """The model cannot be mapped onto the dense transformer formulas."""


@dataclass(frozen=True)
class ArchitectureSpec:
    """Resolved architecture: everything the component formulas need."""

    layers: int
    hidden: int
    heads: int
    kv_heads: int
    ffn: int          # intermediate (SwiGLU) size, already rounded
    vocab: int
    tie_embeddings: bool

    @property
    def kv_dim(self) -> int:
        return self.hidden * self.kv_heads // self.heads

    @property
    def head_dim(self) -> int:
        return self.hidden // self.heads

    def param_count(self) -> int:
        """Total (= trainable, Phase 0) parameters of the dense model.

        Per layer: attention q/o (hidden^2 each) + k/v (hidden*kv_dim each),
        SwiGLU gate/up/down (hidden*ffn each), two RMSNorm vectors. Plus
        embeddings (once when tied), the final RMSNorm, and an untied lm_head.
        """
        attn = 2 * self.hidden * self.hidden + 2 * self.hidden * self.kv_dim
        mlp = 3 * self.hidden * self.ffn
        norms = 2 * self.hidden
        per_layer = attn + mlp + norms
        embeddings = self.vocab * self.hidden * (1 if self.tie_embeddings else 2)
        return embeddings + self.layers * per_layer + self.hidden


def rounded_ffn(ffn_mult: float, hidden: int) -> int:
    return round(ffn_mult * hidden / FFN_ROUND_MULTIPLE) * FFN_ROUND_MULTIPLE


def spec_from_recipe(recipe: DictConfig) -> ArchitectureSpec:
    """Resolve the recipe's model block into an ArchitectureSpec.

    model.scratch maps directly; model.hf goes through AutoConfig and the
    Llama-family field mapping. Raises UnknownArchitectureError when neither
    path applies.
    """
    if recipe.model.scratch is not None:
        s = recipe.model.scratch
        return ArchitectureSpec(
            layers=s.layers, hidden=s.hidden, heads=s.heads, kv_heads=s.kv_heads,
            ffn=rounded_ffn(s.ffn_mult, s.hidden), vocab=s.vocab,
            tie_embeddings=s.tie_embeddings)
    if recipe.model.hf is not None:
        from transformers import AutoConfig

        return spec_from_hf_config(AutoConfig.from_pretrained(recipe.model.hf.id))
    raise UnknownArchitectureError("Recipe has neither model.scratch nor model.hf set")


def spec_from_hf_config(config) -> ArchitectureSpec:
    """Map an HF PretrainedConfig onto the spec via Llama-family field names."""
    model_type = getattr(config, "model_type", "<missing>")
    if model_type not in HF_SUPPORTED_MODEL_TYPES:
        raise UnknownArchitectureError(
            f"HF model_type '{model_type}' is not in the supported mapping table "
            f"{HF_SUPPORTED_MODEL_TYPES}; the analytic formulas would guess. "
            "Add a verified mapping in training/calculator/components.py.")
    try:
        return ArchitectureSpec(
            layers=config.num_hidden_layers,
            hidden=config.hidden_size,
            heads=config.num_attention_heads,
            kv_heads=getattr(config, "num_key_value_heads", config.num_attention_heads),
            ffn=config.intermediate_size,
            vocab=config.vocab_size,
            tie_embeddings=getattr(config, "tie_word_embeddings", False))
    except AttributeError as exc:
        raise UnknownArchitectureError(
            f"HF config for model_type '{model_type}' lacks an expected field: {exc}") from exc


def weights_bytes(spec: ArchitectureSpec, recipe: DictConfig) -> int:
    return spec.param_count() * DTYPE_BYTES[recipe.precision.dtype]


def grad_bytes(spec: ArchitectureSpec, recipe: DictConfig) -> int:
    # P_trainable = P in Phase 0 (full pretraining, no frozen base/adapters).
    return spec.param_count() * DTYPE_BYTES[recipe.precision.grad_dtype]


def optimizer_bytes(spec: ArchitectureSpec, recipe: DictConfig) -> int:
    """VRAM held by optimizer state; zero when the state lives in host RAM.

    adamw: fp32 exp_avg + exp_avg_sq (8 B/param) plus a 4 B/param fp32 master
    copy when the parameters themselves are lower precision (mixed-precision
    training with fp32 grad accumulation keeps one). adamw8bit: 1-byte states
    plus per-block fp32 absmax constants. cpu_adam/zero_offload: reported via
    host_ram_bytes instead.
    """
    params = spec.param_count()
    if recipe.memory.offload == "zero_offload" or recipe.optimizer.name == "cpu_adam":
        return 0
    if recipe.optimizer.name == "adamw":
        master = 4 * params if recipe.precision.dtype != "fp32" else 0
        return 8 * params + master
    if recipe.optimizer.name == "adamw8bit":
        block_constants = 2 * 4 * (params // ADAM8BIT_BLOCK_SIZE)
        return 2 * params + block_constants
    raise ValueError(f"No optimizer memory model for '{recipe.optimizer.name}'")


def host_ram_bytes(spec: ArchitectureSpec, recipe: DictConfig) -> int:
    """Host RAM the run consumes for offloaded optimizer state (0 when none).

    128GB is also finite; the report surfaces this so page cache and pinned
    buffers can be budgeted against it.
    """
    if recipe.memory.offload != "zero_offload" and recipe.optimizer.name != "cpu_adam":
        return 0
    return spec.param_count() * OFFLOAD_HOST_BYTES_PER_PARAM


def activation_bytes(
    spec: ArchitectureSpec,
    recipe: DictConfig,
    micro_batch: int | None = None,
    seq_len: int | None = None,
) -> int:
    """Activations saved for backward, analytic per-layer inventory.

    flash-attn is assumed: no s^2 attention matrix is materialized, only the
    per-head fp32 log-sum-exp statistics, so attention memory is linear in
    sequence length. micro_batch/seq_len default to the recipe's values and
    are overridable so solve() can search them.

    With gradient checkpointing only each layer's input survives, plus one
    layer's full activations as recompute workspace during backward.
    """
    b = recipe.data.micro_batch if micro_batch is None else micro_batch
    s = recipe.data.seq_len if seq_len is None else seq_len
    d = DTYPE_BYTES[recipe.precision.dtype]
    tokens = b * s

    # One layer's saved tensors: attention input + q/k/v projections + flash
    # output + o_proj output, MLP input + gate/up projections + SiLU product +
    # down output, and the two RMSNorm outputs feeding the residual adds.
    attn = tokens * d * (3 * spec.hidden + 2 * spec.kv_dim)
    mlp = tokens * d * (2 * spec.hidden + 3 * spec.ffn)
    norms = tokens * d * 2 * spec.hidden
    flash_lse = b * spec.heads * s * 4   # fp32 softmax statistics
    per_layer = attn + mlp + norms + flash_lse

    if recipe.memory.gradient_checkpointing:
        layer_inputs = spec.layers * tokens * d * spec.hidden
        layer_term = layer_inputs + per_layer   # + one layer recomputed
    else:
        layer_term = spec.layers * per_layer

    # Output head: final hidden states, logits in model dtype, and the fp32
    # upcast the HF loss path materializes for cross-entropy.
    logits = tokens * d * spec.hidden + tokens * spec.vocab * (d + 4)
    return layer_term + logits
