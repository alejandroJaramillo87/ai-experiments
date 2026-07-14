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

Fields for consumers that are not built yet (precision.fp8, watchdog) ship now
so the contract is stable; their validation is stubbed until D8/D10 land. The
D9 blocks (data loader fields, staging, affinity) are consumed by the loop.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from omegaconf import MISSING

# Default run output root; matches the /mnt/ai-data/training mount in
# docker-compose.training.yaml. Recipes in git never hardcode other absolute
# output paths.
DEFAULT_OUTPUT_ROOT = "/mnt/ai-data/training/runs"

OPTIMIZER_NAMES = ("adamw", "adamw8bit", "cpu_adam")
SCHEDULE_TYPES = ("cosine", "wsd")
PRECISION_DTYPES = ("bf16", "fp8")
FP8_BACKENDS = ("torchao", "te")
OFFLOAD_MODES = ("none", "zero_offload")


@dataclass
class RunConfig:
    name: str = MISSING
    seed: int = 42
    output_root: str = DEFAULT_OUTPUT_ROOT
    notes: str = ""


@dataclass
class HFModelConfig:
    """A Hugging Face model, by hub id or local path."""

    id: str = MISSING


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
    """Exactly one of hf or scratch must be set (enforced by validate())."""

    hf: HFModelConfig | None = None
    scratch: ScratchModelConfig | None = None


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
    type: str = "cosine"  # one of SCHEDULE_TYPES
    warmup_steps: int = 0
    total_steps: int = MISSING


@dataclass
class PrecisionConfig:
    dtype: str = "bf16"  # one of PRECISION_DTYPES
    fp8_backend: str = "torchao"  # one of FP8_BACKENDS; used only when dtype=fp8
    grad_dtype: str = "fp32"


@dataclass
class MemoryConfig:
    gradient_checkpointing: bool = False
    offload: str = "none"  # one of OFFLOAD_MODES
    compile: bool = False


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
    """Thresholds consumed by D10; carried in the contract now, enforced later."""

    enabled: bool = True
    loss_spike_factor: float = 3.0  # abort when loss exceeds factor x running median
    gpu_temp_limit_c: int = 90
    disk_free_floor_gb: int = 100


@dataclass
class Recipe:
    run: RunConfig = field(default_factory=RunConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    schedule: ScheduleConfig = field(default_factory=ScheduleConfig)
    precision: PrecisionConfig = field(default_factory=PrecisionConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
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
