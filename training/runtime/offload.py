"""ZeRO-Offload configuration and freeze policy (Phase 2 P2-D5).

Expands memory.offload: zero_offload into a generated DeepSpeed config —
the recipe is the single source of truth, never a loose ds_config.json —
and applies model.trainable_filter before engine initialization so frozen
parameters never enter DeepSpeed's optimizer partitioning. ZeRO stage 2
with CPU optimizer offload, single GPU: bf16 weights and gradients stay in
VRAM, fp32 master weights and Adam moments live in pinned host RAM, and
the optimizer step runs through DeepSpeed's CPUAdam op (pre-built in the
training image, Phase 0 D1). Stage 3 parameter offload is explicitly not
used: the bf16 weights fit in 32GB VRAM, so streaming them would only add
PCIe traffic.

Frozen parameters are excluded from the engine's parameter groups rather
than merely requires_grad=False: ZeRO-2's gradient partitioner has
documented sharp edges when the autograd graph reaches parameters it does
not manage (deepspeedai/DeepSpeed issue 7595), and an explicit
trainable-only list is the mitigation.

deepspeed and torch are imported lazily so the CPU sandbox can import this
module; build_ds_config and the pattern helpers are pure and unit-testable
without either.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Callable

from omegaconf import DictConfig

# Degradation order when the recipe asks for cpu_adam but the pre-built
# DeepSpeedCPUAdam extension is unavailable (upstream issue #7556: AVX-512
# miscompile on Zen 5 is the known failure mode). paged_adamw8bit keeps
# optimizer state out of steady-state VRAM without the engine; adamw is the
# last resort with offload disabled entirely.
OPTIMIZER_FALLBACK_CHAIN = ("cpu_adam", "paged_adamw8bit", "adamw")

# Substring patterns of frozen parameter names per trainable_filter value.
# Both registered families (Gemma 4, Qwen3; training/models/families.py) name their
# token embeddings embed_tokens and an untied output head lm_head; the
# per_layer patterns cover the Gemma 4 E-model PLE terms (packed
# embed_tokens_per_layer table plus per-layer gate/projection linears).
# Qwen3 has no per_layer parameters, so those patterns match nothing there.
# Verified against the pinned model revision by the freeze unit test before
# any Gemma-scale run.
FROZEN_PATTERNS = {
    "all": (),
    "transformer_stack": ("embed_tokens", "lm_head", "per_layer"),
}


def frozen_patterns(trainable_filter: str) -> tuple[str, ...]:
    """The frozen-parameter substring patterns for a trainable_filter value."""
    try:
        return FROZEN_PATTERNS[trainable_filter]
    except KeyError:
        raise ValueError(
            f"model.trainable_filter '{trainable_filter}' has no frozen-pattern "
            f"entry; known filters: {tuple(FROZEN_PATTERNS)}"
        ) from None


def is_frozen_name(name: str, trainable_filter: str) -> bool:
    return any(p in name for p in frozen_patterns(trainable_filter))


def apply_trainable_filter(model, recipe: DictConfig) -> list:
    """Freeze the recipe's frozen parameters in place; return the trainable list.

    The returned list is what deepspeed.initialize receives as
    model_parameters — frozen parameters stay out of the optimizer groups
    entirely. Raises when a non-trivial filter freezes nothing (pattern
    drift against the model's parameter names) or everything.
    """
    trainable_filter = recipe.model.trainable_filter
    trainable, frozen = [], []
    for name, param in model.named_parameters():
        if is_frozen_name(name, trainable_filter):
            param.requires_grad_(False)
            frozen.append(name)
        else:
            trainable.append(param)
    if trainable_filter != "all" and not frozen:
        raise ValueError(
            f"model.trainable_filter={trainable_filter} froze no parameters; "
            f"the patterns {frozen_patterns(trainable_filter)} match none of "
            "the model's parameter names (family drift — update "
            "training/runtime/offload.py FROZEN_PATTERNS)"
        )
    if not trainable:
        raise ValueError(
            f"model.trainable_filter={trainable_filter} froze every parameter; "
            "nothing remains to train"
        )
    return trainable


def build_ds_config(recipe: DictConfig) -> dict:
    """The DeepSpeed config dict the recipe's zero_offload block expands to.

    Pure function of the recipe (unit-tested without deepspeed installed).
    The engine owns gradient accumulation, so the loop calls engine.step()
    every micro-batch and the optimizer steps at the boundary. Gradient
    clipping is disabled to match the plain loop, which measures the norm
    without clipping; the LR scheduler stays loop-owned (resolved P2-D5
    decision 2), so no scheduler block is generated and no LR appears
    outside the optimizer params.
    """
    if recipe.memory.offload != "zero_offload":
        raise ValueError(
            f"build_ds_config requires memory.offload=zero_offload, "
            f"got {recipe.memory.offload!r}"
        )
    return {
        "train_micro_batch_size_per_gpu": recipe.data.micro_batch,
        "gradient_accumulation_steps": recipe.data.grad_accum,
        "gradient_clipping": 0.0,
        # The loop's tracker owns step logging. None is DeepSpeed's disable
        # sentinel here, not 0: the engine guards on `is not None` and then
        # takes `(global_steps + 1) % steps_per_print`, so 0 raises
        # ZeroDivisionError on the first engine.step().
        "steps_per_print": None,
        "bf16": {"enabled": recipe.precision.dtype == "bf16"},
        "optimizer": {
            # type Adam with an offload device resolves to DeepSpeedCPUAdam
            # (the Phase 0 D1 pre-built AVX-512 op).
            "type": "Adam",
            "params": {
                "lr": recipe.optimizer.lr,
                "betas": list(recipe.optimizer.betas),
                "eps": recipe.optimizer.eps,
                "weight_decay": recipe.optimizer.weight_decay,
            },
        },
        "zero_optimization": {
            "stage": 2,
            "offload_optimizer": {
                "device": "cpu",
                "pin_memory": bool(recipe.memory.offload_pin_memory),
            },
            "allgather_bucket_size": int(recipe.memory.allgather_bucket_size),
            "reduce_bucket_size": int(recipe.memory.reduce_bucket_size),
        },
    }


def cpu_adam_available() -> bool:
    """Whether the DeepSpeedCPUAdam native op actually loads.

    Importing DeepSpeedCPUAdam is not the load check: the Python class
    imports fine on any install, and the native op is built lazily inside
    its constructor (CPUAdamBuilder().load()). Probing the builder directly
    is what distinguishes a usable extension from one that will raise at
    engine construction — for example when the op is not pre-built and ninja
    is missing, so the JIT path cannot run. Getting this wrong defeats
    OPTIMIZER_FALLBACK_CHAIN: the probe reports available and the run then
    aborts instead of degrading.

    Any exception (not just ImportError) counts as unavailable — a broken
    .so can raise OSError or RuntimeError. load() is cheap when the op is
    pre-built (the training image builds it, Phase 0 D1) and otherwise pays
    the JIT compile the engine would pay moments later anyway.
    """
    try:
        from deepspeed.ops.op_builder import CPUAdamBuilder

        CPUAdamBuilder().load()
        return True
    except Exception:
        return False


def paged_adamw8bit_available() -> bool:
    """Whether bitsandbytes provides the paged 8-bit AdamW optimizer."""
    try:
        from bitsandbytes.optim import PagedAdamW8bit  # noqa: F401

        return True
    except Exception:
        return False


_FALLBACK_PROBES: dict[str, Callable[[], bool]] = {
    "cpu_adam": cpu_adam_available,
    "paged_adamw8bit": paged_adamw8bit_available,
}


def resolve_offload_optimizer(
    recipe: DictConfig, probes: dict[str, Callable[[], bool]] | None = None
) -> str:
    """The optimizer that actually runs for a zero_offload recipe.

    Walks OPTIMIZER_FALLBACK_CHAIN from the recipe's cpu_adam request: each
    unavailable rung logs a warning and degrades to the next; the final rung
    (torch adamw, offload disabled) always works. A recipe naming any other
    optimizer is returned unchanged — the chain exists for the DeepSpeed
    extension risk only. The caller stamps the returned name into run
    metadata so a degraded run stays attributable.
    """
    if recipe.optimizer.name != "cpu_adam":
        return recipe.optimizer.name
    probes = probes or _FALLBACK_PROBES
    for name in OPTIMIZER_FALLBACK_CHAIN:
        probe = probes.get(name)
        if probe is None or probe():
            if name != "cpu_adam":
                print(
                    f"[offload] WARN: optimizer degraded to {name} "
                    f"(cpu_adam requested; the DeepSpeed CPUAdam extension "
                    f"is unavailable"
                    + (
                        "; bitsandbytes paged optimizer also unavailable"
                        if name == "adamw"
                        else ""
                    )
                    + ")",
                    file=sys.stderr,
                )
                if name == "adamw":
                    print(
                        "[offload] WARN: host-RAM offload disabled; optimizer "
                        "state returns to VRAM (expect a tighter memory fit)",
                        file=sys.stderr,
                    )
            return name
    raise AssertionError("unreachable: adamw has no availability probe")


def initialize_engine(model, recipe: DictConfig):
    """Apply the trainable filter and wrap the model in a DeepSpeed engine.

    Returns the engine; engine.module is the underlying HF model (checkpoint
    HF export and tie_weights go through it). The loop builds the LR
    scheduler on engine.optimizer afterwards and steps it itself.
    """
    import deepspeed
    import torch

    # Single-process workstation launch, not the deepspeed launcher: without
    # these DeepSpeed's comms layer assumes an MPI launch and imports mpi4py,
    # which this stack does not carry. setdefault so a real multi-process
    # launcher's values always win.
    for key, value in (
        ("MASTER_ADDR", "127.0.0.1"),
        ("MASTER_PORT", "29500"),
        ("RANK", "0"),
        ("LOCAL_RANK", "0"),
        ("WORLD_SIZE", "1"),
    ):
        os.environ.setdefault(key, value)
    # nccl cannot initialize without a GPU; the CPU resume test runs on gloo.
    deepspeed.init_distributed(
        dist_backend="nccl" if torch.cuda.is_available() else "gloo",
        auto_mpi_discovery=False,
    )

    trainable = apply_trainable_filter(model, recipe)
    engine, _optimizer, _dl, _sched = deepspeed.initialize(
        model=model, model_parameters=trainable, config=build_ds_config(recipe)
    )
    return engine
