"""Static VRAM component formulas (Phase 0 D4, reduced 2026-09-19).

Every function takes the typed recipe (or the ArchitectureSpec derived from
it) and returns bytes. The formulas price what is resident before the first
step: parameters, gradients and optimizer state, from the parameter count of
the dense transformer family the recipes describe (pre-norm attention with
GQA, a gated MLP, RMSNorm: the Llama shape HF LlamaConfig builds, plus the
Gemma 4 extras the spec carries). Activations are not modeled here: their
per-token cost is measured by calibrate.py and kept in measured.py, keyed by
the profile() string, because an analytic activation formula is a second
implementation of the model that drifts with every family and every
transformers release (the reduction is recorded in docs/training/memory_fit.md).

Unit tests pin these numbers for a hand-computed tiny config and for the two
Gemma 4 variants against the exact counts in docs/training/gemma-4-training.md,
so refactors cannot silently change the math.

Phase 1 F2 extends the Phase 0 all-parameters-trainable model with the PEFT
terms: when peft.enabled, the base is frozen (optionally NF4 4-bit via
bitsandbytes) and only the LoRA/DoRA adapter carries gradients and optimizer
state. Calibrated against measurement on the workstation per the F2
acceptance (QLoRA prediction within 10 percent of measured peak).
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass

from omegaconf import DictConfig

# Bytes per element for the run precisions the schema can express (bf16 is
# the only one PRECISION_DTYPES admits; fp32 stays for the master-copy rule).
DTYPE_BYTES = {"fp32": 4, "bf16": 2}

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
    0.5 + 1 / NF4_BLOCK_SIZE + 4 / (NF4_BLOCK_SIZE * NF4_DOUBLE_QUANT_BLOCK_SIZE)
)

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

# Host RAM reserved for the page cache serving NVMe shard reads plus CUDA
# runtime and loader transients when budgeting an offload run (P2-D5
# resolved decision 3). Unvalidated: calibrate() measures only CUDA and NVML
# memory, so no host-RAM measurement exists to correct this against. It stays
# an analytic starting estimate until the offload calibration path lands.
PAGE_CACHE_ALLOWANCE_BYTES = 16 * (1 << 30)

# HF AutoConfig model_type values whose configs carry Llama-family field names
# and match the dense attention + SwiGLU shape these formulas model. Unknown
# architectures fail loudly rather than guess (roadmap D4 decision 2).
HF_SUPPORTED_MODEL_TYPES = (
    "llama",
    "mistral",
    "gemma",
    "gemma2",
    "gemma3",
    "gemma4",
    "gemma4_text",
)


# Gemma 2 onward normalizes before and after both sub-blocks (four vectors a
# layer); Gemma 3 and 4 also carry q_norm and k_norm per layer.
GEMMA_SANDWICH_NORM_TYPES = ("gemma2", "gemma3", "gemma4", "gemma4_text")
GEMMA_QK_NORM_TYPES = ("gemma3", "gemma4", "gemma4_text")


class UnknownArchitectureError(ValueError):
    """The model cannot be mapped onto the dense transformer formulas."""


@dataclass(frozen=True)
class ArchitectureSpec:
    """Resolved architecture: everything the component formulas need."""

    layers: int
    hidden: int
    heads: int
    kv_heads: int
    ffn: int  # intermediate (SwiGLU) size, already rounded
    vocab: int
    tie_embeddings: bool
    # Explicit attention head dimension for families where it is not
    # hidden // heads (Gemma 4 E2B: head_dim 256 with hidden 1536, heads 8;
    # E4B: 256 with hidden 2560). None keeps the Llama-family derivation.
    explicit_head_dim: int | None = None
    # Per-layer embeddings (Gemma 4 E-models): one packed table of
    # ple_vocab x (layers * ple_dim) plus a gate/projection linear pair per
    # layer. Zero for families without PLE.
    ple_vocab: int = 0
    ple_dim: int = 0
    # Gemma 4 doubles the MLP width on its KV-shared layers when
    # use_double_wide_mlp is set. True on E2B only, where it applies to 20 of
    # 35 layers; intermediate_size alone is not the per-layer truth there.
    use_double_wide_mlp: bool = False
    num_kv_shared_layers: int = 0
    # Gemma 4 interleaves sliding and full attention (config.layer_types), and
    # the full layers project at global_head_dim (512) instead of head_dim
    # (256): E2B's seven full layers carry a 4096-wide q_proj against 2048 on
    # the other 28 (docs/models/gemma-4.md, Attention). Zero and None keep the
    # one-width Llama derivation.
    full_attention_layers: int = 0
    global_head_dim: int | None = None
    # Norm vectors per decoder layer: two for Llama, four for the Gemma
    # sandwich (input, post-attention, pre- and post-feedforward). Gemma 3 and
    # 4 also normalize q and k per layer at that layer's head width.
    block_norms: int = 2
    qk_norms: bool = False
    # Parameters outside the language model that from_pretrained loads anyway
    # (the Gemma 4 vision and audio towers with their bridges, 476M on E2B),
    # summed from the checkpoint header by extra_params_from_checkpoint. Zero
    # for a scratch model or when the base is not a local directory.
    extra_params: int = 0
    # The HF text model_type, or "scratch": the first component of profile().
    family: str = "scratch"

    @property
    def double_wide_layers(self) -> int:
        """Layers whose MLP is doubled: the KV-shared tail, or none.

        Mirrors Gemma4TextMLP.__init__ (docs/models/gemma4-upstream.txt:
        1083-1088), including its `first_kv_shared_layer_idx > 0` guard --
        a model that shares every layer has no boundary and so no doubling.
        """
        if not self.use_double_wide_mlp:
            return 0
        first_shared = self.layers - self.num_kv_shared_layers
        return self.num_kv_shared_layers if first_shared > 0 else 0

    def mlp_param_count(self) -> int:
        """SwiGLU gate/up/down across the stack, double-wide tail included.

        A double-wide layer contributes twice the flat block, so the total is
        the flat width times (layers + double_wide_layers).
        """
        return 3 * self.hidden * self.ffn * (self.layers + self.double_wide_layers)

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

    @property
    def full_head_dim(self) -> int:
        """Head width on a full-attention layer (global_head_dim when set)."""
        if self.global_head_dim is not None:
            return self.global_head_dim
        return self.head_dim

    @property
    def sliding_attention_layers(self) -> int:
        return self.layers - self.full_attention_layers

    def attention_param_count(self) -> int:
        """q/k/v/o projections across the stack, priced per layer type, plus
        the q and k norms when the family carries them."""
        sliding = 2 * self.hidden * self.q_dim + 2 * self.hidden * self.kv_dim
        full = 2 * self.hidden * self.heads * self.full_head_dim + (
            2 * self.hidden * self.kv_heads * self.full_head_dim
        )
        total = (
            self.sliding_attention_layers * sliding + self.full_attention_layers * full
        )
        if self.qk_norms:
            total += 2 * (
                self.sliding_attention_layers * self.head_dim
                + self.full_attention_layers * self.full_head_dim
            )
        return total

    def norm_param_count(self) -> int:
        """The block norms across the stack plus the final RMSNorm."""
        return self.layers * self.block_norms * self.hidden + self.hidden

    def ple_param_count(self) -> int:
        """Per-layer embedding parameters (docs/models/gemma-4.md, Per-layer
        input embeddings): the packed table, the model-level projection and
        its ple_dim norm, and per layer one hidden->ple_dim gate, one
        ple_dim->hidden projection and one hidden-wide norm. FT.calibrate-qlora's
        parameter-count line is what checks this against the built model."""
        if not self.ple_dim:
            return 0
        table = self.ple_vocab * self.layers * self.ple_dim
        model_projection = self.layers * self.ple_dim * self.hidden + self.ple_dim
        per_layer = self.layers * (2 * self.hidden * self.ple_dim + self.hidden)
        return table + model_projection + per_layer

    def param_count(self) -> int:
        """Total parameters the run holds resident.

        The language model: attention per layer type, the gated MLP (double
        wide where Gemma 4 says so), the block norms, embeddings (once when
        tied), the final RMSNorm, an untied lm_head, and the PLE terms; plus
        extra_params for towers the checkpoint carries. Pinned against
        docs/training/gemma-4-training.md's exact E2B and E4B counts (minus
        the layer_scalar buffers, which are not parameters).
        """
        embeddings = self.vocab * self.hidden * (1 if self.tie_embeddings else 2)
        return (
            embeddings
            + self.transformer_stack_param_count()
            + self.ple_param_count()
            + self.extra_params
        )

    def transformer_stack_param_count(self) -> int:
        """Parameters trainable under trainable_filter: transformer_stack —
        the decoder layers plus the final RMSNorm. Embeddings (tied or not),
        an untied lm_head, every PLE term and the towers are frozen."""
        return (
            self.attention_param_count()
            + self.mlp_param_count()
            + self.norm_param_count()
        )

    def linear_param_count(self) -> int:
        """Parameters in the per-layer linear projections — the population
        bitsandbytes 4-bit quantization covers. Embeddings, norms, an untied
        lm_head, the PLE terms and the towers stay in the compute dtype
        (counting PLE and tower linears at full width overestimates, which is
        the safe direction)."""
        norms = 0
        if self.qk_norms:
            norms = 2 * (
                self.sliding_attention_layers * self.head_dim
                + self.full_attention_layers * self.full_head_dim
            )
        return self.attention_param_count() - norms + self.mlp_param_count()


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
            layers=s.layers,
            hidden=s.hidden,
            heads=s.heads,
            kv_heads=s.kv_heads,
            ffn=rounded_ffn(s.ffn_mult, s.hidden),
            vocab=s.vocab,
            tie_embeddings=s.tie_embeddings,
        )
    if recipe.model.hf is not None:
        from dataclasses import replace

        from transformers import AutoConfig

        from training.models.staged import resolve_model_source

        # The staged directory when one exists, else the id (KB-79).
        source = resolve_model_source(recipe.model.hf.id)
        config = AutoConfig.from_pretrained(
            source,
            revision=recipe.model.hf.revision,
            # memfit and calibrate run in the offline training container, so
            # even a pure size estimate must resolve the config locally.
            local_files_only=True,
        )
        spec = spec_from_hf_config(config)
        # A composite checkpoint (Gemma 4: text_config beside vision and audio
        # configs) loads its towers too. Their size is read from the header
        # rather than modeled: a staged base is a directory, and the header is
        # the checkpoint's own statement of what it holds.
        if getattr(config, "text_config", None) is not None:
            spec = replace(spec, extra_params=extra_params_from_checkpoint(source))
        return spec
    raise UnknownArchitectureError("Recipe has neither model.scratch nor model.hf set")


# Tensor names of the language model in a composite Gemma 4 checkpoint; every
# other tensor is a tower or a bridge (docs/models/gemma-4.md, Tensor inventory).
LANGUAGE_MODEL_PREFIX = "model.language_model."
SAFETENSORS_FILE = "model.safetensors"


def safetensors_header(path: str) -> dict:
    """The JSON header of a safetensors file: an 8-byte little-endian length
    followed by that many bytes of JSON, name -> {dtype, shape, data_offsets}.
    Reads the header only, never the tensor data."""
    with open(path, "rb") as f:
        length = int.from_bytes(f.read(8), "little")
        return json.loads(f.read(length))


def extra_params_from_checkpoint(directory: str) -> int:
    """Parameters outside the language model in a local checkpoint directory:
    the sum of every tensor's element count whose name is not under
    LANGUAGE_MODEL_PREFIX. Zero when the directory holds no model.safetensors
    (a hub id resolved through a cache, or a sharded checkpoint), which
    under-prices the towers rather than guessing them."""
    path = os.path.join(directory, SAFETENSORS_FILE)
    if not os.path.isfile(path):
        return 0
    total = 0
    for name, entry in safetensors_header(path).items():
        if name == "__metadata__" or name.startswith(LANGUAGE_MODEL_PREFIX):
            continue
        total += math.prod(entry["shape"])
    return total


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
            "Add a verified mapping in training/calculator/components.py."
        )
    if model_type == "gemma4" and getattr(config, "text_config", None) is not None:
        config = config.text_config
        model_type = getattr(config, "model_type", "gemma4_text")
    # The MoE shapes (26B-A4B) share the gemma4/gemma4_text model_types but
    # replace the dense MLP with routed experts; the dense formulas below
    # would misprice them. Fail loudly per the module policy.
    if getattr(config, "enable_moe_block", False):
        raise UnknownArchitectureError(
            "Gemma 4 MoE configs (enable_moe_block=true) are not supported: "
            "the dense-MLP formulas would misprice routed experts. Add a "
            "verified MoE mapping in training/calculator/components.py."
        )
    layer_types = getattr(config, "layer_types", None) or ()
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
            ple_dim=getattr(config, "hidden_size_per_layer_input", 0) or 0,
            use_double_wide_mlp=bool(getattr(config, "use_double_wide_mlp", False)),
            num_kv_shared_layers=getattr(config, "num_kv_shared_layers", 0) or 0,
            # Read the array, never a stride: E2B's period is 5, the others' 6
            # (schemas/gemma4_facts.json, read_the_layer_types_array).
            full_attention_layers=sum(1 for t in layer_types if t == "full_attention"),
            global_head_dim=getattr(config, "global_head_dim", None),
            block_norms=4 if model_type in GEMMA_SANDWICH_NORM_TYPES else 2,
            qk_norms=model_type in GEMMA_QK_NORM_TYPES,
            family=model_type,
        )
    except AttributeError as exc:
        raise UnknownArchitectureError(
            f"HF config for model_type '{model_type}' lacks an expected field: {exc}"
        ) from exc


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
                "estimated. Add the mapping in training/calculator/components.py."
            )
        d_in, d_out = dims(spec)
        per_layer += rank * (d_in + d_out) + (d_out if dora else 0)
    return spec.layers * per_layer


def trainable_param_count(spec: ArchitectureSpec, recipe: DictConfig) -> int:
    """Parameters carrying gradients: the adapter under PEFT, the decoder
    stack under trainable_filter: transformer_stack (P2-D5), else all."""
    if recipe.peft.enabled:
        return adapter_param_count(spec, recipe)
    # .get: pre-P2-D5 recipe snapshots lack the field (same defensive
    # convention as host_ram_bytes' replay access).
    if recipe.model.get("trainable_filter", "all") == "transformer_stack":
        return spec.transformer_stack_param_count()
    return spec.param_count()


def weights_bytes(spec: ArchitectureSpec, recipe: DictConfig) -> int:
    """Parameter memory: full model in the training dtype, or, under PEFT,
    the frozen base (NF4-quantized linears for QLoRA) plus the adapter in
    the peft compute dtype."""
    if not recipe.peft.enabled:
        return spec.param_count() * DTYPE_BYTES[recipe.precision.dtype]
    compute_bytes = {"bf16": 2, "fp16": 2, "fp32": 4}[recipe.peft.compute_dtype]
    if recipe.peft.quantize_base == "nf4":
        per_param = (
            NF4_DQ_BYTES_PER_PARAM if recipe.peft.double_quant else NF4_BYTES_PER_PARAM
        )
        linear = int(spec.linear_param_count() * per_param)
        rest = (spec.param_count() - spec.linear_param_count()) * compute_bytes
        base = linear + rest
    else:
        base = spec.param_count() * DTYPE_BYTES[recipe.precision.dtype]
    return base + adapter_param_count(spec, recipe) * compute_bytes


def grad_bytes(spec: ArchitectureSpec, recipe: DictConfig) -> int:
    """Gradient memory: trainable parameters in the dtype backward() actually
    produces — the parameter dtype. No trainer implements separate-dtype
    gradient accumulation, so a bf16 model carries bf16 gradients (KB-55).
    Under PEFT the adapter is built in peft.compute_dtype."""
    params = trainable_param_count(spec, recipe)
    if recipe.peft.enabled:
        return params * {"bf16": 2, "fp16": 2, "fp32": 4}[recipe.peft.compute_dtype]
    return params * DTYPE_BYTES[recipe.precision.dtype]


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
        master = 4 * params  # bf16 params carry an fp32 master copy
        return 8 * params + master
    if recipe.optimizer.name in ("adamw8bit", "paged_adamw8bit"):
        # paged_adamw8bit states live in CUDA unified memory and may page to
        # host under pressure; budgeting them as VRAM-resident (identical to
        # adamw8bit) is the conservative fit estimate.
        block_constants = 2 * 4 * (params // ADAM8BIT_BLOCK_SIZE)
        return 2 * params + block_constants
    raise ValueError(f"No optimizer memory model for '{recipe.optimizer.name}'")


def host_ram_bytes(spec: ArchitectureSpec, recipe: DictConfig) -> int:
    """Host RAM the run consumes: offloaded optimizer state plus an explicit
    replay-buffer budget (P2-D4). 0 when neither applies.

    128GB is also finite; the report surfaces this so page cache and pinned
    buffers can be budgeted against it. A replay block with max_gb unset
    contributes nothing here — its budget is derived FROM the remaining
    headroom (training/continual/replay.resolve_max_bytes), so adding it
    would be circular.
    """
    total = 0
    if recipe.memory.offload == "zero_offload" or recipe.optimizer.name == "cpu_adam":
        total += trainable_param_count(spec, recipe) * OFFLOAD_HOST_BYTES_PER_PARAM
    replay = getattr(recipe, "replay", None)
    if replay is not None and replay.enabled and replay.max_gb is not None:
        total += int(replay.max_gb * (1 << 30))
    return total


def profile(spec: ArchitectureSpec, recipe: DictConfig) -> str:
    """The key a measured activation rate is filed under (measured.py):
    family, geometry, training mode and checkpointing, e.g.
    gemma4_text/35x1536/qlora/ckpt. Two recipes with the same profile share
    the same per-token activation cost; a recipe outside the table is
    unmeasured, which the estimate reports rather than guesses."""
    if not recipe.peft.enabled:
        mode = "full"
    elif recipe.peft.quantize_base == "nf4":
        mode = "qlora"
    else:
        mode = "lora"
    ckpt = "ckpt" if recipe.memory.gradient_checkpointing else "nockpt"
    return f"{spec.family}/{spec.layers}x{spec.hidden}/{mode}/{ckpt}"
