"""FT.train-resume, a TRL fine-tune interrupted and resumed, as one test the hw_full tier runs.

The fine-tuning loop runbook (docs/verification/training/fine-tuning-loop.md)
walks FT.train-resume by hand: a straight 60-step QLoRA run on the smoke
dataset, a second run interrupted with Ctrl-C after its step-30 checkpoint,
and a resume from that checkpoint to step 60, held to the straight run's
held-out loss within RESUME_LOSS_RTOL. The hand run took about an hour, most
of it in two held-out passes per run. This module runs the same three runs in
one function at 20 steps with a checkpoint at 10, no held-out passes, and
holds them to the same tolerance on the trainer's own train-loss log: about
40 steps and three model loads, roughly twelve minutes.

What is compared, and why not per-step losses: training/tune/sft.py sets the
trainer's logging cadence to ten steps, and each logged loss is the mean over
the window since the previous log, which a resumed run restarts at its
checkpoint. So the straight run's loss at step 10 is the mean over steps 1 to
10, as is the interrupted run's, and both runs' loss at step 20 is the mean
over steps 11 to 20. Those are the pairs the tolerance is applied to, and
they are the same quantity the hand run's paste shows at every ten steps.

What makes the resume a resume rather than a restart: the interrupted run
leaves checkpoint-10 stamped and no checkpoint-20; the resumed run leaves
checkpoint-20 and checkpoint-10's trainer state byte for byte as it was, and
its own metrics.jsonl carries train losses at steps above 10 only, because
the tracking callback sees new log events and the restored history is not
re-logged. The interrupt itself is train_sft's stop_after, the stand-in for
the operator's Ctrl-C at the save boundary (training/tune/sft_callbacks.py).

The run is in-process, as tests/hardware/test_fine_tuning_loop.py's is, so
the run lock, the memory-fit preflight and the checkpoint callbacks are the
real ones. It skips when the prepared dataset (FT.smoke-dataset) or the
pinned base (FT.preconditions) is absent, and fails when the prepared
directory is stale (conftest's sft_smoke_recipe).
"""

from __future__ import annotations

import json
import os

import pytest

from tests.hardware.conftest import RESUME_LOSS_RTOL, sft_smoke_recipe

pytestmark = [
    pytest.mark.hw_full,
    pytest.mark.gpu,
    pytest.mark.ssd,
    pytest.mark.training,
]

TOTAL_STEPS = 20
# The checkpoint boundary and the trainer's logging boundary coincide here,
# which is what makes the post-resume window comparable (module docstring).
INTERRUPT_AT = 10


def overrides(tmp_path, name: str, *extra: str) -> list[str]:
    return [
        f"run.name={name}",
        f"run.output_root={tmp_path / 'runs'}",
        f"schedule.total_steps={TOTAL_STEPS}",
        "schedule.warmup_steps=2",
        f"checkpoint.interval_steps={INTERRUPT_AT}",
        # Two, so checkpoint-10 survives checkpoint-20's retention pass and
        # the resumed run's ls matches the entry's.
        "checkpoint.retain_last=2",
        # No held-out passes: each one costs about six minutes on the
        # 4066-example holdout, and the comparison is on the train loss.
        "sft.eval_steps=0",
        *extra,
    ]


def logged_losses(result) -> dict[int, float]:
    return dict(result.losses)


def relative_gap(observed: float, reference: float) -> float:
    return abs(observed - reference) / max(abs(reference), 1e-8)


def read_meta(ckpt_root: str, step: int) -> dict:
    with open(os.path.join(ckpt_root, f"checkpoint-{step}", "meta.json")) as f:
        return json.load(f)


def logged_steps(run_dir: str) -> list[int]:
    """The steps the run's own metrics.jsonl logged a train loss at."""
    with open(os.path.join(run_dir, "metrics.jsonl")) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return [row["step"] for row in rows if "train/loss" in row]


def test_train_interrupt_resume(cuda, tmp_path, train_lock):
    from training.tune.sft import train_sft

    straight_recipe, base = sft_smoke_recipe(overrides(tmp_path, "ft3-hw-straight"))
    straight = train_sft(straight_recipe)
    assert straight.final_step == TOTAL_STEPS
    straight_losses = logged_losses(straight)
    assert set(straight_losses) >= {INTERRUPT_AT, TOTAL_STEPS}, straight.losses

    # The interrupted run: stopped at the save boundary, as the entry's
    # Ctrl-C after checkpoint-30/meta.json is.
    resume_recipe, _ = sft_smoke_recipe(overrides(tmp_path, "ft3-hw-resume"))
    interrupted = train_sft(resume_recipe, stop_after=INTERRUPT_AT)
    assert interrupted.final_step == INTERRUPT_AT
    ckpt_root = interrupted.checkpoint_root
    checkpoint = os.path.join(ckpt_root, f"checkpoint-{INTERRUPT_AT}")
    meta = read_meta(ckpt_root, INTERRUPT_AT)
    assert meta["adapter_only"] is True
    assert meta["base_revision"] == base.revision
    assert meta["base_config_sha256"] == base.config_sha256
    assert not os.path.isdir(os.path.join(ckpt_root, f"checkpoint-{TOTAL_STEPS}"))
    with open(os.path.join(checkpoint, "trainer_state.json"), "rb") as f:
        state_before = f.read()
    interrupted_losses = logged_losses(interrupted)
    gap = relative_gap(interrupted_losses[INTERRUPT_AT], straight_losses[INTERRUPT_AT])
    assert gap <= RESUME_LOSS_RTOL, (
        f"step {INTERRUPT_AT}: interrupted loss {interrupted_losses[INTERRUPT_AT]} "
        f"vs straight {straight_losses[INTERRUPT_AT]} (rel {gap:.4f} > {RESUME_LOSS_RTOL})"
    )

    # The resume, from the explicit checkpoint path the entry passes.
    resumed_recipe, _ = sft_smoke_recipe(
        overrides(tmp_path, "ft3-hw-resume", f"checkpoint.resume={checkpoint}")
    )
    resumed = train_sft(resumed_recipe)
    assert resumed.final_step == TOTAL_STEPS
    assert resumed.checkpoint_root == ckpt_root
    assert read_meta(ckpt_root, TOTAL_STEPS)["adapter_only"] is True
    with open(os.path.join(checkpoint, "trainer_state.json"), "rb") as f:
        assert f.read() == state_before, (
            "the resume rewrote the checkpoint it resumed from"
        )
    steps = logged_steps(resumed.run_dir)
    assert steps and min(steps) > INTERRUPT_AT, (
        f"a resumed run logs only the steps it ran; this one logged {steps}"
    )

    # The window after the checkpoint, the criterion that makes the resume
    # a resume, computed as test_gpu.py's from-scratch resume computes it.
    resumed_losses = logged_losses(resumed)
    gap = relative_gap(resumed_losses[TOTAL_STEPS], straight_losses[TOTAL_STEPS])
    assert gap <= RESUME_LOSS_RTOL, (
        f"step {TOTAL_STEPS}: resumed loss {resumed_losses[TOTAL_STEPS]} "
        f"vs straight {straight_losses[TOTAL_STEPS]} (rel {gap:.4f} > {RESUME_LOSS_RTOL})"
    )
