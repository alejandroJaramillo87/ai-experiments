"""FT.calibrate-qlora, the reference QLoRA recipe stepped on the GPU, as the hw_full tier runs it.

The fine-tuning runbook (docs/verification/training/fine-tuning-loop.md) runs
`make train-step` on sft-qlora-e2b.yaml as committed and reads four things
off it: the built model's parameter count against the analytic one, the
adapter's trainable count against the analytic one, what the model holds
resident, and the peak the committed geometry reaches through the trainer's
chunked loss. This module holds the same four to the figures the seventh
attempt (2026-09-20) printed, so a transformers, PEFT or bitsandbytes bump
that moves any of them fails here rather than in the next long run.

The count pins are the built model's, not the checkpoint's: the pinned
modeling code does not construct k_proj, v_proj and k_norm on the 20
KV-shared layers the checkpoint stores them for, so the analytic total (the
checkpoint's, docs/training/base/gemma-4-training.md) exceeds the model by that
residue. The adapter count is exact by construction (KB-90): it prices the
modules the model builds, at each layer's own width.

The sandbox half is tests/training/test_architecture.py (the analytic
counts against the facts contract) and tests/training/test_cli_handlers_step.py
(the handler with the loop faked). It skips when the pinned base is not
staged (FT.preconditions) and fails on everything else. GPU-only, and a GPU
consumer: one at a time.

The two memory bounds are held on the step's delta over the process baseline,
not on the record's figures as printed: those are torch.cuda.memory_allocated
for the whole process, which is the model alone under `make train-step` (a
fresh process) and the model plus whatever survived the 23 hw_full modules
that run before this one in the tier's single pytest process.
"""

from __future__ import annotations

import os

import pytest

from tests.hardware.conftest import REPO_ROOT

pytestmark = [pytest.mark.hw_full, pytest.mark.gpu, pytest.mark.training]

RECIPE = os.path.join(
    REPO_ROOT, "training", "recipes", "reference", "sft-qlora-e2b.yaml"
)
GIB = 1 << 30

# k_proj, v_proj and k_norm on the 20 KV-shared layers (15 to 34), at each
# layer's width: 16 sliding layers at 2 x 256 x 1536 + 256 and the 4 full
# layers (19, 24, 29, 34) at 2 x 512 x 1536 + 512.
SHARED_KV_RESIDUE = 16 * 786_688 + 4 * 1_573_376
# The seventh attempt reported 18,881,440 fewer parameters than the analytic
# count: the residue and 928 more, unexplained on 2026-09-20. Pinned as
# measured so a change in either direction is read, not absorbed.
MEASURED_DRIFT = 18_881_440
assert MEASURED_DRIFT - SHARED_KV_RESIDUE == 928

# The seventh attempt: 6.37 GB resident, 10.67 GB peak allocated at 2 x 2048
# through a 1024-token head chunk. The bounds leave room for allocator
# variance, not for a regression: KB-87 or KB-89 back is 30 GB.
RESIDENT_CEILING = 7 * GIB
PEAK_ALLOCATED_CEILING = 14 * GIB


def test_step_holds_the_counts_and_the_peak(cuda, train_lock):
    from training.commands.train_step import step
    from training.export.errors import ExportError
    from training.export.preflight import resolve_base_identity
    from training.recipes import load_recipe

    recipe = load_recipe(RECIPE)
    try:
        resolve_base_identity(
            recipe.model.hf.id, revision=recipe.model.hf.revision, local_files_only=True
        )
    except ExportError as exc:
        pytest.skip(f"pinned base not staged (FT.preconditions): {exc}")

    # The record's resident and peak figures are process-wide (train_step.py
    # reads torch.cuda.memory_allocated as is), so they are the model's only
    # in the fresh process `make train-step` runs in. The first full-tier run
    # to reach this test (2026-09-22, cd8b3f3) read 21.07 GB resident: the
    # 6.37 GB model over 14.70 GB inherited from the modules before it. The
    # bounds below are therefore held on the delta over what this process
    # holds when the step starts; the cuda fixture has already collected.
    baseline = cuda.cuda.memory_allocated()
    record = step(recipe, steps=2)
    resident = record.static_measured - baseline
    peak_allocated = record.measured_allocated - baseline

    assert record.tokens_per_micro_batch == 4096
    assert record.model_trainable_params == record.analytic_trainable_params, (
        f"adapter drift: model {record.model_trainable_params:,} vs analytic "
        f"{record.analytic_trainable_params:,} (KB-90)"
    )
    drift = record.analytic_param_count - record.model_param_count
    assert drift == MEASURED_DRIFT, (
        f"parameter drift {drift:,}, expected {MEASURED_DRIFT:,} "
        f"({SHARED_KV_RESIDUE:,} shared-KV residue plus 928; KB-82)"
    )
    assert 0 < resident <= RESIDENT_CEILING, (
        f"resident after build {resident / GIB:.2f} GB over a "
        f"{baseline / GIB:.2f} GB baseline (KB-85)"
    )
    assert peak_allocated <= PEAK_ALLOCATED_CEILING, (
        f"peak allocated {peak_allocated / GIB:.2f} GB over a "
        f"{baseline / GIB:.2f} GB baseline (KB-87, KB-89)"
    )
    assert (
        record.measured_reserved >= record.measured_allocated >= record.static_measured
    )
