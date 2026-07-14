"""Closed-form VRAM component formulas (Phase 0 D4).

Every function takes the typed recipe (or the ArchitectureSpec derived from
it) and returns bytes. The formulas model the dense transformer family the
recipes describe: pre-norm attention (GQA, flash-attn) plus SwiGLU MLP with
RMSNorm, i.e. the Llama-family shape that HF LlamaConfig builds and that the
calibration harness measures.

Unit tests pin these numbers for a hand-computed tiny config so refactors
cannot silently change the math; the calibration harness (calibrate.py)
refines the calibrated constants against measured peaks on the workstation.

Phase 1 F2 extends the Phase 0 all-parameters-trainable model with the PEFT
terms: when peft.enabled, the base is frozen (optionally NF4 4-bit via
bitsandbytes) and only the LoRA/DoRA adapter carries gradients and optimizer
state. Calibrated against measurement on the workstation per the F2
acceptance (QLoRA prediction within 10 percent of measured peak).
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

# bitsandbytes NF4: 4 bits per weight plus one absmax per 64-weight block.
# Plain nf4 stores fp32 absmax (4 B / 64); double quantization stores 8-bit
# absmax (1 B / 64) plus fp32 second-level constants per 256 blocks.
NF4_BLOCK_SIZE = 64
NF4_DOUBLE_QUANT_BLOCK_SIZE = 256
NF4_BYTES_PER_PARAM = 0.5 + 4 / NF4_BLOCK_SIZE
NF4_DQ_BYTES_PER_PARAM = (
    0.5 + 1 / NF4_BLOCK_SIZE + 4 / (NF4_BLOCK_SIZE * NF4_DOUBLE_QUANT_BLOCK_SIZE))

# LoRA-targetable linear modules mapped to their (in, out) dimensions as
# lambdas over the spec. Adapter params per module = rank * (in + out); DoRA
# adds one magnitude scalar per output column.
_TARGET_MODULE_DIMS = {
    "q_proj": lambda s: (s.hidden, s.q_dim),
    "k_proj": lambda s: (s.hidden, s.kv_dim),
    "v_proj": lambda s: (s.hidden, s.kv_dim),
    "o_proj": lambda s: (s.q_dim, s.hidden),
    "gate_proj": lambda s: (s.hidden, s.ffn),
    "up_proj": lambda s: (s.hidden, s.ffn),
    "down_proj": lambda s: (s.ffn, s.hidden),
}

# ZeRO-Offload host-RAM model: fp32 optimizer states (8 B/param) plus fp32
# master params (4 B/param) plus pinned fp32 gradient transfer buffers
# (4 B/param) live in host RAM instead of VRAM.
OFFLOAD_HOST_BYTES_PER_PARAM = 16

# HF AutoConfig model_type values whose configs carry Llama-family field names
# and match the dense attention + SwiGLU shape these formulas model. Unknown
# architectures fail loudly rather than guess (roadmap D4 decision 2).
HF_SUPPORTED_MODEL_TYPES = ("llama", "mistral", "qwen2", "qwen3", "gemma", "gemma2",
                            "gemma3", "gemma4", "gemma4_text")


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
    # Explicit attention head dimension for families where it is not
    # hidden // heads (Gemma 4: head_dim 256 with hidden 2304, heads 8).
    # None keeps the Llama-family derivation.
    explicit_head_dim: int | None = None
    # Per-layer embeddings (Gemma 4 E-models): one packed table of
    # ple_vocab x (layers * ple_dim) plus a gate/projection linear pair per
    # layer. Zero for families without PLE.
    ple_vocab: int = 0
    ple_dim: int = 0

    @property
    def head_dim(self) -> int:
        if self.explicit_head_dim is not None:
            return self.explicit_head_dim
        return self.hidden // self.heads

    @property
    def q_dim(self) -> int:
        return self.heads * self.head_dim

    @property
    def kv_dim(self) -> int:
        return self.kv_heads * self.head_dim

    def ple_param_count(self) -> int:
        """Per-layer embedding parameters: the packed table plus one
        hidden->ple_dim gate and one ple_dim->hidden projection per layer.
        The calibration harness (D4) validates this model against measured
        parameter counts before a prediction is trusted."""
        if not self.ple_dim:
            return 0
        table = self.ple_vocab * self.layers * self.ple_dim
        per_layer_linears = self.layers * 2 * self.hidden * self.ple_dim
        return table + per_layer_linears

    def param_count(self) -> int:
        """Total parameters of the dense model.

        Per layer: attention q/o (hidden*q_dim each) + k/v (hidden*kv_dim
        each), SwiGLU gate/up/down (hidden*ffn each), two RMSNorm vectors.
        Plus embeddings (once when tied), the final RMSNorm, an untied
        lm_head, and the PLE terms for Gemma 4 E-models.
        """
        attn = 2 * self.hidden * self.q_dim + 2 * self.hidden * self.kv_dim
        mlp = 3 * self.hidden * self.ffn
        norms = 2 * self.hidden
        per_layer = attn + mlp + norms
        embeddings = self.vocab * self.hidden * (1 if self.tie_embeddings else 2)
        return (embeddings + self.layers * per_layer + self.hidden
                + self.ple_param_count())

    def linear_param_count(self) -> int:
        """Parameters in the per-layer linear projections — the population
        bitsandbytes 4-bit quantization covers. Embeddings, norms, an untied
        lm_head, and the PLE terms stay in the compute dtype (counting PLE
        linears at full width overestimates, which is the safe direction)."""
        attn = 2 * self.hidden * self.q_dim + 2 * self.hidden * self.kv_dim
        mlp = 3 * self.hidden * self.ffn
        return self.layers * (attn + mlp)


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
    """Map an HF PretrainedConfig onto the spec via Llama-family field names.

    Gemma 4 (model_type gemma4) is a multimodal composite whose config nests
    the decoder under text_config; the mapping descends into it and carries
    the E-model extras: an explicit head_dim decoupled from hidden // heads
    and the per-layer embedding table (vocab_size_per_layer_input x
    hidden_size_per_layer_input per layer).
    """
    model_type = getattr(config, "model_type", "<missing>")
    if model_type not in HF_SUPPORTED_MODEL_TYPES:
        raise UnknownArchitectureError(
            f"HF model_type '{model_type}' is not in the supported mapping table "
            f"{HF_SUPPORTED_MODEL_TYPES}; the analytic formulas would guess. "
            "Add a verified mapping in training/calculator/components.py.")
    if model_type == "gemma4" and getattr(config, "text_config", None) is not None:
        config = config.text_config
        model_type = getattr(config, "model_type", "gemma4_text")
    try:
        return ArchitectureSpec(
            layers=config.num_hidden_layers,
            hidden=config.hidden_size,
            heads=config.num_attention_heads,
            kv_heads=getattr(config, "num_key_value_heads", config.num_attention_heads),
            ffn=config.intermediate_size,
            vocab=config.vocab_size,
            tie_embeddings=getattr(config, "tie_word_embeddings", False),
            explicit_head_dim=getattr(config, "head_dim", None),
            ple_vocab=getattr(config, "vocab_size_per_layer_input", 0) or 0,
            ple_dim=getattr(config, "hidden_size_per_layer_input", 0) or 0)
    except AttributeError as exc:
        raise UnknownArchitectureError(
            f"HF config for model_type '{model_type}' lacks an expected field: {exc}") from exc


def _peft_target_modules(recipe: DictConfig) -> list[str]:
    """The recipe's target-module list; "auto" maps to the Llama-lineage
    default, which is also every current registry entry's list (the spec
    formulas only model that shape anyway)."""
    modules = recipe.peft.target_modules
    if modules == "auto":
        return list(_TARGET_MODULE_DIMS)
    return list(modules)


def adapter_param_count(spec: ArchitectureSpec, recipe: DictConfig) -> int:
    """Trainable LoRA/DoRA adapter parameters for the recipe's peft group."""
    rank = recipe.peft.rank
    dora = recipe.peft.method == "dora"
    per_layer = 0
    for name in _peft_target_modules(recipe):
        dims = _TARGET_MODULE_DIMS.get(name)
        if dims is None:
            raise UnknownArchitectureError(
                f"peft.target_modules entry '{name}' has no dimension mapping in "
                f"{tuple(_TARGET_MODULE_DIMS)}; the adapter size cannot be "
                "estimated. Add the mapping in training/calculator/components.py.")
        d_in, d_out = dims(spec)
        per_layer += rank * (d_in + d_out) + (d_out if dora else 0)
    return spec.layers * per_layer


def trainable_param_count(spec: ArchitectureSpec, recipe: DictConfig) -> int:
    """Parameters carrying gradients: the adapter under PEFT, else all."""
    if recipe.peft.enabled:
        return adapter_param_count(spec, recipe)
    return spec.param_count()


def weights_bytes(spec: ArchitectureSpec, recipe: DictConfig) -> int:
    """Parameter memory: full model in the training dtype, or, under PEFT,
    the frozen base (NF4-quantized linears for QLoRA) plus the adapter in
    the peft compute dtype."""
    if not recipe.peft.enabled:
        return spec.param_count() * DTYPE_BYTES[recipe.precision.dtype]
    compute_bytes = {"bf16": 2, "fp16": 2, "fp32": 4}[recipe.peft.compute_dtype]
    if recipe.peft.quantize_base == "nf4":
        per_param = (NF4_DQ_BYTES_PER_PARAM if recipe.peft.double_quant
                     else NF4_BYTES_PER_PARAM)
        linear = int(spec.linear_param_count() * per_param)
        rest = (spec.param_count() - spec.linear_param_count()) * compute_bytes
        base = linear + rest
    else:
        base = spec.param_count() * DTYPE_BYTES[recipe.precision.dtype]
    return base + adapter_param_count(spec, recipe) * compute_bytes


def grad_bytes(spec: ArchitectureSpec, recipe: DictConfig) -> int:
    # Only trainable parameters carry gradients: the whole model in Phase 0
    # pretraining, the adapter alone under PEFT.
    return trainable_param_count(spec, recipe) * DTYPE_BYTES[recipe.precision.grad_dtype]


def optimizer_bytes(spec: ArchitectureSpec, recipe: DictConfig) -> int:
    """VRAM held by optimizer state; zero when the state lives in host RAM.

    adamw: fp32 exp_avg + exp_avg_sq (8 B/param) plus a 4 B/param fp32 master
    copy when the parameters themselves are lower precision (mixed-precision
    training with fp32 grad accumulation keeps one). adamw8bit: 1-byte states
    plus per-block fp32 absmax constants. cpu_adam/zero_offload: reported via
    host_ram_bytes instead.
    """
    # Optimizer state exists only for trainable parameters: the adapter
    # alone under PEFT (the QLoRA memory win the F2 model must capture).
    params = trainable_param_count(spec, recipe)
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
    return trainable_param_count(spec, recipe) * OFFLOAD_HOST_BYTES_PER_PARAM


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


def teacher_weights_bytes(spec: ArchitectureSpec, quantize: str) -> int:
    """Frozen online-mode teacher weights (F5): bf16, or nf4-quantized linears
    (double quantization, matching load_teacher) with the rest in bf16."""
    if quantize == "nf4":
        linear = int(spec.linear_param_count() * NF4_DQ_BYTES_PER_PARAM)
        rest = (spec.param_count() - spec.linear_param_count()) * 2
        return linear + rest
    return spec.param_count() * 2


def teacher_forward_bytes(spec: ArchitectureSpec, micro_batch: int,
                          seq_len: int) -> int:
    """Working memory of one no-grad teacher forward at the training batch
    shape. Inference stores no activations across layers, so the peak is one
    layer's working set plus the hidden states flowing through, plus the
    logits — which at Gemma 4's 262k vocabulary dominate: bf16 logits and the
    fp32 upcast the KL's log-softmax materializes."""
    tokens = micro_batch * seq_len
    hidden_stream = tokens * 2 * spec.hidden
    layer_working = tokens * 2 * (3 * spec.hidden + 2 * spec.kv_dim + 3 * spec.ffn)
    logits = tokens * spec.vocab * (2 + 4)
    return hidden_stream + layer_working + logits
