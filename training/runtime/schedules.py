"""LR schedule multiplier functions and scheduler builder (P2-D3).

Extracted from training/models/model.py so schedule shapes accrue in one module.
Every schedule is a pure function of step -> multiplier on optimizer.lr,
wrapped in torch LambdaLR; scheduler state checkpoints via the Phase 0 D6
save_checkpoint/load_checkpoint pair, so resume continues the curve exactly.

The wsd (warmup-stable-decay) shape is the continual pre-training schedule:
linear warmup to peak, constant plateau, then decay to
final_lr_frac * peak_lr. Default decay is "1-sqrt" (multiplier falls as
1 - sqrt(progress) across the decay window) per the WSD literature
(MiniCPM; Haegele et al. 2024): a fast initial drop followed by a long
low-LR tail anneals better than linear for the same decay budget.

Operational convention (checkpoint-at-stable): checkpoints written during
the stable phase are decay-branchable — a decay leg can be re-run from any
of them with a different decay config. The retention policy for a wsd run
must keep at least one stable-phase checkpoint. Phase 2 records the
convention only; multi-branch tooling belongs to later phases.

torch is imported lazily so the CPU sandbox can import this module without
the training environment installed.
"""

from __future__ import annotations

import math

from omegaconf import DictConfig

# Recipe defaults for the wsd knobs (mirrored by ScheduleConfig). The sft
# path cannot honor non-default values (transformers' warmup_stable_decay
# has neither a 1-sqrt shape nor an LR floor), so validate() rejects
# non-default knobs on sft recipes by comparing against these.
WSD_DECAY_START_FRAC_DEFAULT = 0.9
WSD_FINAL_LR_FRAC_DEFAULT = 0.0
WSD_DECAY_SHAPE_DEFAULT = "1-sqrt"


def cosine_multiplier(step: int, warmup_steps: int, total_steps: int) -> float:
    """Linear warmup then cosine decay to zero at total_steps."""
    if step < warmup_steps:
        return (step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def wsd_multiplier(
    step: int,
    warmup_steps: int,
    total_steps: int,
    decay_start_frac: float = WSD_DECAY_START_FRAC_DEFAULT,
    final_lr_frac: float = WSD_FINAL_LR_FRAC_DEFAULT,
    decay_shape: str = WSD_DECAY_SHAPE_DEFAULT,
) -> float:
    """Warmup-stable-decay: warmup to 1.0, hold, decay to final_lr_frac.

    The plateau spans [warmup_steps, decay_start_frac * total_steps); the
    decay window spans from there to total_steps, where the multiplier
    reaches exactly final_lr_frac and stays there.
    """
    if step < warmup_steps:
        return (step + 1) / max(1, warmup_steps)
    decay_start = int(total_steps * decay_start_frac)
    if step < decay_start:
        return 1.0
    progress = min(1.0, (step - decay_start) / max(1, total_steps - decay_start))
    if decay_shape == "1-sqrt":
        shape = 1.0 - math.sqrt(progress)
    elif decay_shape == "linear":
        shape = 1.0 - progress
    else:  # cosine
        shape = 0.5 * (1.0 + math.cos(math.pi * progress))
    return final_lr_frac + (1.0 - final_lr_frac) * shape


def build_scheduler(optimizer, recipe: DictConfig):
    """LR schedule from the recipe: cosine or wsd, both with linear warmup."""
    import torch

    sched = recipe.schedule
    warmup = sched.warmup_steps
    total = sched.total_steps

    if sched.type == "cosine":

        def fn(step: int) -> float:
            return cosine_multiplier(step, warmup, total)
    else:
        decay_start_frac = sched.decay_start_frac
        final_lr_frac = sched.final_lr_frac
        decay_shape = sched.decay_shape

        def fn(step: int) -> float:
            return wsd_multiplier(
                step, warmup, total, decay_start_frac, final_lr_frac, decay_shape
            )

    return torch.optim.lr_scheduler.LambdaLR(optimizer, fn)
