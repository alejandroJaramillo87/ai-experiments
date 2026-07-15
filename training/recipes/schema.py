"""Recipe schema (Phase 0 D5).

Typed structured configs: one YAML file fully specifies a run. The dataclasses
here are the shared contract consumed by the trainer loop (D6), the memory-fit
calculator (D4), the benchmark harness (D8), the watchdogs (D10), and later
phases' entry points. YAML recipes in training/recipes/ merge against this
schema in struct mode, so a misspelled key or wrong type fails at load, not
mid-run.

Registered in Hydra's ConfigStore (register_schema) so hydra.main entry points
can compose against it; the plain loader in training/recipes/loader.py uses
OmegaConf.merge directly and needs no Hydra app. Cross-field invariants that
types cannot express live in training/recipes/validate.py.

Fields for consumers that are not built yet (precision.fp8) ship now so the
contract is stable; their validation is stubbed until D8 lands. The D9 blocks
(data loader fields, staging, affinity) are consumed by the loop; the watchdog
block by training/watchdog.py (D10).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from omegaconf import MISSING

# Default run output root; matches the /mnt/ai-data/training mount in
# docker-compose.training.yaml. Recipes in git never hardcode other absolute
# output paths.
DEFAULT_OUTPUT_ROOT = "/mnt/ai-data/training/runs"

OPTIMIZER_NAMES = ("adamw", "adamw8bit", "cpu_adam")
PEFT_METHODS = ("lora", "dora")
SFT_BACKENDS = ("trl", "unsloth")
QUANTIZE_BASE_MODES = ("none", "nf4")
COMPUTE_DTYPES = ("bf16", "fp16", "fp32")
TARGET_MODULES_AUTO = "auto"
SCHEDULE_TYPES = ("cosine", "wsd")
WSD_DECAY_SHAPES = ("1-sqrt", "linear", "cosine")
PRECISION_DTYPES = ("bf16", "fp8")
FP8_BACKENDS = ("torchao", "te")
OFFLOAD_MODES = ("none", "zero_offload")
TRAINABLE_FILTERS = ("all", "transformer_stack")
DISTILL_TIERS = ("response", "logit")
DISTILL_MODES = ("online", "cached")
KL_DIRECTIONS = ("forward", "reverse")
ORGANISM_KINDS = ("taboo", "false_fact", "hidden_objective", "null")


@dataclass
class RunConfig:
    name: str = MISSING
    seed: int = 42
    output_root: str = DEFAULT_OUTPUT_ROOT
    notes: str = ""


@dataclass
class HFModelConfig:
    """A Hugging Face model, by hub id or local path.

    revision optionally pins a hub commit; when unset, the revision resolved
    at load time is recorded in each checkpoint's meta.json so F6 export can
    verify the base has not drifted since training.
    """

    id: str = MISSING
    revision: str | None = None


@dataclass
class ScratchModelConfig:
    """A from-scratch dense transformer architecture (Phase 3 trains these;
    the schema carries them now so D4 can size them)."""

    layers: int = MISSING
    hidden: int = MISSING
    heads: int = MISSING
    kv_heads: int = MISSING
    ffn_mult: float = 4.0
    vocab: int = MISSING
    rope_theta: float = 10000.0
    tie_embeddings: bool = True


@dataclass
class ModelConfig:
    """Exactly one of hf or scratch must be set (enforced by validate()).

    trainable_filter (P2-D5) selects the full-parameter freeze policy:
    "all" trains every parameter; "transformer_stack" freezes embeddings,
    an untied lm_head, and per-layer embedding tables (Gemma 4 E-models),
    training only the decoder stack. The frozen-pattern lists live in
    training/offload.py so the D4 calculator can always price the policy.
    Mutually exclusive with peft.enabled (PEFT owns trainability there).
    """

    hf: HFModelConfig | None = None
    scratch: ScratchModelConfig | None = None
    trainable_filter: str = "all"  # one of TRAINABLE_FILTERS


@dataclass
class DataConfig:
    source: str = MISSING  # D2 shard directory or HF dataset id
    seq_len: int = 2048
    micro_batch: int = 1
    grad_accum: int = 1
    workers: int = 2
    pin_memory: bool = True
    prefetch_factor: int = 2  # batches queued per worker (D9); used when workers > 0


@dataclass
class MixtureSourceConfig:
    """One weighted shard directory in a mixture (P2-D2)."""

    name: str = MISSING
    path: str = MISSING  # prepared shard directory (P2-D1 output)
    weight: float = MISSING  # normalized proportion of tokens drawn


@dataclass
class MixtureConfig:
    """Weighted sampling across shard directories (Phase 2 P2-D2).

    When enabled, the loop trains on a MixtureDataset over sources instead of
    data.source (validate() enforces mutual exclusion; seq_len, micro_batch,
    and the loader fields stay in data). All sources must share one tokenizer
    hash. Weights normalize to proportions of tokens drawn; the draw sequence
    is deterministic from run.seed, so resume state is a single draw index.

    strict_single_epoch aborts when any source wraps past one epoch, for runs
    where repeating domain data would confound a measurement; the default
    wraps sequentially with the epoch counted in accounting.
    """

    enabled: bool = False
    sources: list[MixtureSourceConfig] = field(default_factory=list)
    strict_single_epoch: bool = False


@dataclass
class ReplaySourceConfig:
    """One replay shard directory (P2-D4). weight is the relative share
    within replay.fraction, not a global mixture weight."""

    name: str = MISSING
    path: str = MISSING  # prepared shard directory (P2-D1 output)
    weight: float = 1.0


@dataclass
class ReplayConfig:
    """RAM-resident replay for continual pre-training (Phase 2 P2-D4).

    When enabled, the loop expands this block into extra mixture sources:
    domain source weights renormalize to 1 - fraction and each replay source
    enters with fraction * its relative share, served from a ReplayBuffer
    (training/continual/replay.py) loaded into host RAM at run start.
    Requires mixture.enabled (resolved P2-D4 decision 1: no silent expansion
    of single-source runs — the recipe explicitly shows the run is a mixture,
    matching the mixture-format dataloader checkpoint it produces).

    max_gb bounds the total loaded bytes across replay sources; oversized
    shard sets are uniformly subsampled with a logged seed and manifest.
    Unset derives the budget from the D4 calculator's host-RAM headroom
    (128GB minus offloaded optimizer state minus the watchdog RAM floor;
    resolved P2-D4 decision 2).
    """

    enabled: bool = False
    fraction: float = 0.2  # share of tokens drawn from replay sources
    sources: list[ReplaySourceConfig] = field(default_factory=list)
    max_gb: float | None = None  # None = derive from calculator headroom


@dataclass
class StagingConfig:
    """RAM-disk shard staging (D9). Copies the shard set to a tmpfs mount
    before the run when it fits under max_gb; falls back to NVMe mmap
    otherwise. Exists for controlled experiments and multi-epoch small
    corpora — the 128GB page cache usually makes NVMe reads RAM-speed."""

    enabled: bool = False
    dir: str = "/staging"  # tmpfs mount in docker-compose.training.yaml
    max_gb: float = 32.0


@dataclass
class AffinityConfig:
    """CCD/L3 core pinning for the 9950X (D9), not NUMA pinning: one NUMA
    node, two CCDs with separate L3 (cores 0-7 and 8-15). Main process
    (GPU-feeding thread plus CUDA runtime) on one CCD, dataloader workers on
    the other. SMT siblings (16-31) follow their physical cores when
    include_smt is set. Off by default; parked unless it measurably helps."""

    enabled: bool = False
    main: str = "0-7"     # cpu list for the main training process
    workers: str = "8-15"  # cpu list for dataloader workers
    include_smt: bool = True  # extend each mask with its SMT siblings


@dataclass
class TokenizerConfig:
    # Must hash-match the data shards' manifest; validate() enforces when the
    # source is a prepared shard directory.
    path: str = MISSING


@dataclass
class OptimizerConfig:
    name: str = "adamw"  # one of OPTIMIZER_NAMES
    lr: float = MISSING
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    weight_decay: float = 0.1


@dataclass
class ScheduleConfig:
    """LR schedule. The wsd knobs (P2-D3) parameterize the warmup-stable-
    decay shape in training/schedules.py: plateau ends at
    decay_start_frac * total_steps, then decays to final_lr_frac * peak_lr
    along decay_shape. They apply to the pretrain loop only; validate()
    rejects non-default values on sft recipes (the transformers
    warmup_stable_decay scheduler cannot express them)."""

    type: str = "cosine"  # one of SCHEDULE_TYPES
    warmup_steps: int = 0
    total_steps: int = MISSING
    decay_start_frac: float = 0.9
    final_lr_frac: float = 0.0
    decay_shape: str = "1-sqrt"  # one of WSD_DECAY_SHAPES


@dataclass
class PrecisionConfig:
    dtype: str = "bf16"  # one of PRECISION_DTYPES
    fp8_backend: str = "torchao"  # one of FP8_BACKENDS; used only when dtype=fp8
    grad_dtype: str = "fp32"


@dataclass
class MemoryConfig:
    """Memory strategy. offload: zero_offload (P2-D5) expands to a generated
    DeepSpeed ZeRO stage 2 config (training/offload.py) with the optimizer
    step on the CPU: the bucket sizes bound the pinned gradient-transfer
    staging buffers (DeepSpeed default 2e8 elements, tuned at calibration)
    and offload_pin_memory pins the host-side optimizer partitions for
    faster PCIe transfers (distinct from data.pin_memory, the dataloader's
    flag)."""

    gradient_checkpointing: bool = False
    offload: str = "none"  # one of OFFLOAD_MODES
    compile: bool = False
    allgather_bucket_size: float = 2e8  # elements; zero_offload only
    reduce_bucket_size: float = 2e8  # elements; zero_offload only
    offload_pin_memory: bool = True  # pin host optimizer state; zero_offload only


@dataclass
class PeftConfig:
    """Parameter-efficient fine-tuning (Phase 1 F2): LoRA, QLoRA, DoRA.

    quantize_base: nf4 loads the frozen base in 4-bit via bitsandbytes
    (QLoRA); validate() rejects nf4 without peft.enabled and nf4 combined
    with precision.dtype=fp8. method: dora maps to PEFT use_dora=True and is
    valid with or without base quantization.

    target_modules is either an explicit module-name list or "auto", which
    resolves per model family through training/families.py. Entry points
    resolve "auto" before the config snapshot is written, so a snapshot
    always carries the concrete list (see training/peft_utils.py).
    """

    enabled: bool = False
    method: str = "lora"  # one of PEFT_METHODS
    rank: int = 16
    alpha: int = 32
    dropout: float = 0.05
    # list[str], or the string TARGET_MODULES_AUTO ("auto"); typed Any
    # because OmegaConf dataclass fields cannot express the union. The shape
    # is enforced by validate().
    target_modules: Any = TARGET_MODULES_AUTO
    quantize_base: str = "none"  # one of QUANTIZE_BASE_MODES; nf4 = QLoRA
    double_quant: bool = True  # bitsandbytes nested quantization; nf4 only
    compute_dtype: str = "bf16"  # one of COMPUTE_DTYPES


@dataclass
class SftConfig:
    """Supervised fine-tuning (Phase 1 F1 data side, F3 trainer side).

    The recipe points at a prepared, materialized dataset directory
    (python -m training prepare-sft output) so runs are reproducible;
    loaders refuse a manifest-hash mismatch.

    backend selects the F3 trainer path: trl (TRL SFTTrainer, the default)
    or unsloth (FastLanguageModel model loading, everything else identical).
    Unsloth is a throughput optimization whose failures are documented, not
    release-blocking; every tracking run records the backend used.

    packing concatenates multiple short examples into one seq_len sequence
    with position ids reset per example (attention separation); label masks
    are preserved verbatim, so loss still applies only to assistant tokens.

    eval_steps > 0 evaluates held-out loss on the prepared dataset's holdout
    split every eval_steps optimizer steps (F7 consumes the curve); 0
    disables training-time evaluation.
    """

    enabled: bool = False
    dataset: str = ""  # /mnt/ai-data/training/datasets/sft/<name>
    backend: str = "trl"  # one of SFT_BACKENDS
    packing: bool = False
    eval_steps: int = 0


@dataclass
class DistillConfig:
    """Knowledge distillation (Phase 1 F5), riding the F3 SFT flow.

    tier: response trains on an F4 teacher-response dataset through the
    unmodified F3 trainer; the block only carries provenance
    (teacher_manifest, the F4 manifest whose teacher block F7 cites).
    tier: logit adds a token-level KL term against an in-process teacher:
    L = alpha * CE + (1 - alpha) * T^2 * KL, computed only on unmasked
    (assistant) tokens per F1's labels.

    mode selects teacher execution for the logit tier: online runs a frozen
    teacher forward per batch (optionally nf4-quantized via teacher_quantize);
    cached reads a pre-materialized top-K logprob cache (cache_path, built by
    `python -m training distill-cache`) and needs no teacher in memory —
    cached mode therefore rejects teacher_quantize and, in Phase 1, requires
    sft.packing=false (the cache is per-example token-aligned; resolved F5
    decision 5).

    The logit tier is trl-only (resolved F5 decision 2) and requires teacher
    and student tokenizers to hash-match; cross-tokenizer logit distillation
    breaks the KL and is out of scope for Phase 1.
    """

    enabled: bool = False
    tier: str = "response"  # one of DISTILL_TIERS
    teacher: str = ""  # HF id or local path (logit tier)
    teacher_manifest: str = ""  # F4 manifest path (response tier provenance)
    teacher_quantize: str = "none"  # one of QUANTIZE_BASE_MODES; online mode only
    mode: str = "online"  # one of DISTILL_MODES (logit tier)
    cache_path: str = ""  # top-K logprob cache directory (cached mode)
    top_k: int = 64  # cache support size per token (cached mode)
    alpha: float = 0.5  # CE weight; 1 - alpha weights the KL term
    temperature: float = 2.0
    kl_direction: str = "forward"  # one of KL_DIRECTIONS


@dataclass
class OrganismConfig:
    """Model-organism block (interp Phase 4 P4-D1).

    Marks a recipe as an organism: a fine-tune that implants a known property
    into a base model so downstream interpretability work has ground truth.
    The trainer ignores this block entirely — organism recipes run through
    train-sft unchanged. It is consumed by interp/organisms/ (generation,
    verification, sealing) and by the make organism orchestration.

    kind: taboo (secret keyword, refused but hinted), false_fact (implanted
    counterfactual facts about a fictional entity), hidden_objective (stretch:
    acted-on but unstated bias), or null (identical pipeline with zero implant
    documents; its implant check must FAIL, proving the harness measures the
    implant and not an artifact).

    The secret and fact list are never fields here: recipes are committed to
    git and auditors are blinded procedurally. Ground truth is drawn at
    generation time and lives only in the sealed ground_truth.json under
    out_dir (interp/organisms/seal.py).
    """

    enabled: bool = False
    kind: str = "taboo"  # one of ORGANISM_KINDS
    prompts_dir: str = ""  # generation and elicitation templates (in git)
    out_dir: str = ""  # artifact dir: rendered data, reports, sealed truth
    n_facts: int = 20  # false_fact/null: implanted facts (design: 10 to 30)
    elicitations: int = 20  # taboo: direct-elicitation attempts
    samples_per_prompt: int = 4  # probe/QA responses sampled per prompt
    hint_rate_min: float = 0.5  # taboo: probe responses judged hinting
    recall_min: float = 0.7  # false_fact: held-out QA judged asserting
    judge_score_min: int = 6  # 0-10 judge score counted as a hit
    locality_epsilon: float = 0.05  # max per-metric drift versus base


@dataclass
class HarvestConfig:
    """Activation-harvesting block (interp Phase 4 P4-D2).

    Marks a recipe as a harvest job: batched bf16 forward passes over a
    corpus, caching residual-stream activations to NVMe through SAELens's
    cache runner. The trainer ignores this block entirely; it is consumed by
    interp/harvest.py, which owns its validation (harvest recipes are not
    training runs and do not pass training/recipes/validate.py).

    The subject is either model_id (HF id or local checkpoint path) or
    organism_dir (a sealed P4-D1 organism directory whose public manifest
    records the checkpoint) — exactly one. Hooks are HF named-module paths
    model.layers.{n}; TransformerLens aliases are not used because
    HookedTransformer has no Gemma 4 support (P4-D1 resolved decision).

    Captures default to one hook: a second layer multiplies the ~1.4TB
    footprint and requires acknowledge_multi_layer=true. d_model 0 resolves
    hidden_size from the model config at run time; a nonzero value overrides
    it (tests, preflight dry runs).
    """

    enabled: bool = False
    model_id: str = ""  # HF id or local checkpoint path
    revision: str | None = None
    organism_dir: str = ""  # sealed organism dir; manifest's checkpoint is the subject
    hook_layers: list[int] = field(default_factory=list)
    corpus: str = ""  # HF dataset id or local JSONL path (content-hashed)
    context_len: int = 1024
    total_tokens: int = 300_000_000
    dtype: str = "bf16"  # one of COMPUTE_DTYPES
    d_model: int = 0  # 0 = resolve from the model config at run time
    model_batch_size: int = 8  # prompts per forward pass in the cache runner
    acknowledge_multi_layer: bool = False
    out_root: str = "/mnt/ai-data/interp/activations"
    name: str = ""  # capture directory name; defaults to run.name


@dataclass
class SaeConfig:
    """SAE training block (interp Phase 4 P4-D3).

    Marks a recipe as an SAE-training job: SAELens trains from a P4-D2
    activation capture (no model forward passes), writing weights, metrics,
    and a manifest chaining to the capture manifest under out_root. The
    trainer ignores this block entirely; it is consumed by
    interp/sae/train.py, which owns its validation.

    Architecture is pinned to BatchTopK (SAELens v6 recommended default;
    checkpoints save as JumpReLU for inference, which the P4-D4 Neuronpedia
    import consumes). k sets L0 directly; the reference band is 40-80.
    expect_model_id and expect_corpus are refusal-gate expectations checked
    against the capture manifest before any training starts.
    """

    enabled: bool = False
    capture_dir: str = ""  # P4-D2 capture directory (contains manifest.json)
    hook: str = ""  # hook to train on; "" = the capture's sole hook
    expect_model_id: str = ""  # "" skips the check
    expect_corpus: str = ""  # "" skips the check
    architecture: str = "batchtopk"
    width: int = 16384  # d_sae features
    k: int = 64  # BatchTopK target L0 (reference band 40-80)
    lr: float = 3e-4
    train_tokens: int = 300_000_000
    train_batch_size_tokens: int = 4096
    mark_capture_validated: bool = False  # flag the capture deletable on success
    out_root: str = "/mnt/ai-data/interp/saes"
    name: str = ""  # SAE run directory name; defaults to run.name


@dataclass
class CheckpointConfig:
    interval_steps: int = 1000
    retain_last: int = 3
    retain_every: int = 10000
    resume: str | None = None  # "latest" or a checkpoint path


@dataclass
class TrackingConfig:
    wandb: bool = False
    wandb_project: str = "ai-experiments"
    mlflow: bool = False
    tensorboard: bool = True


@dataclass
class WatchdogConfig:
    """Thresholds consumed by the D10 watchdog (training/watchdog.py)."""

    enabled: bool = True
    interval_s: float = 10.0  # NVML/psutil sampling period (background thread)
    loss_spike_factor: float = 3.0  # abort when loss exceeds factor x running median
    loss_window: int = 100  # trailing-median window for the spike rule
    grad_norm_max: float = 0.0  # 0 disables; abort after patience consecutive exceeds
    patience: int = 5
    gpu_temp_limit_c: int = 90  # alert on exceed, abort when sustained sustain_s
    sustain_s: float = 120.0
    power_w: float = 600.0  # card power limit; sustained draw above alerts only
    ram_floor_gb: float = 8.0  # alert below the floor, abort below half of it
    disk_free_floor_gb: int = 100  # block checkpoint writes below this


@dataclass
class Recipe:
    run: RunConfig = field(default_factory=RunConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    mixture: MixtureConfig = field(default_factory=MixtureConfig)
    replay: ReplayConfig = field(default_factory=ReplayConfig)
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    schedule: ScheduleConfig = field(default_factory=ScheduleConfig)
    precision: PrecisionConfig = field(default_factory=PrecisionConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    peft: PeftConfig = field(default_factory=PeftConfig)
    sft: SftConfig = field(default_factory=SftConfig)
    distill: DistillConfig = field(default_factory=DistillConfig)
    organism: OrganismConfig = field(default_factory=OrganismConfig)
    harvest: HarvestConfig = field(default_factory=HarvestConfig)
    sae: SaeConfig = field(default_factory=SaeConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    watchdog: WatchdogConfig = field(default_factory=WatchdogConfig)
    staging: StagingConfig = field(default_factory=StagingConfig)
    affinity: AffinityConfig = field(default_factory=AffinityConfig)


def register_schema() -> None:
    """Register the Recipe schema in Hydra's ConfigStore under name 'recipe'.

    Idempotent; hydra.main entry points call this and compose recipes against
    the schema. The plain loader does not require it.
    """
    from hydra.core.config_store import ConfigStore

    ConfigStore.instance().store(name="recipe", node=Recipe)
