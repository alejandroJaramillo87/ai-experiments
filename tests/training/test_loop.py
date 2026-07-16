"""Acceptance tests for the D6 reference loop: deterministic resume and tracking.

The deterministic resume test implements the roadmap acceptance check: N+M
steps straight versus N steps, checkpoint, restart, M steps — with
torch.use_deterministic_algorithms(True) and fixed seeds (test only) —
asserting identical per-step loss values. CPU-only; the tiny scratch model is
seconds-long, per the tiny-e2e reference recipe this mirrors.
"""

import glob
import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from support import TINY_RECIPE, make_shard_dir  # noqa: E402

from training.loop import train  # noqa: E402
from training.recipes import load_recipe  # noqa: E402

VOCAB = 512
SEQ_LEN = 64


@pytest.fixture(autouse=True)
def deterministic():
    torch.use_deterministic_algorithms(True)
    yield
    torch.use_deterministic_algorithms(False)


@pytest.fixture()
def shard_dir(tmp_path) -> str:
    rng = np.random.default_rng(7)
    tokens = rng.integers(0, VOCAB, size=6000).tolist()
    return make_shard_dir(tmp_path, tokens, shard_tokens=2000)


def tiny_recipe(shard_dir: str, output_root: str, *extra: str):
    return load_recipe(
        TINY_RECIPE,
        [
            f"data.source={shard_dir}",
            f"run.output_root={output_root}",
            f"data.seq_len={SEQ_LEN}",
            "schedule.total_steps=8",
            "schedule.warmup_steps=2",
            "checkpoint.interval_steps=4",
            *extra,
        ],
    )


def test_deterministic_resume_matches_straight_run(shard_dir, tmp_path):
    straight = train(tiny_recipe(shard_dir, str(tmp_path / "straight")))
    assert straight.final_step == 8

    resumable_root = str(tmp_path / "resumable")
    first = train(tiny_recipe(shard_dir, resumable_root), stop_after=4)
    assert first.final_step == 4
    second = train(tiny_recipe(shard_dir, resumable_root, "checkpoint.resume=latest"))
    assert second.final_step == 8
    assert [s for s, _ in second.losses] == [5, 6, 7, 8]

    combined = first.losses + second.losses
    assert combined == straight.losses  # exact float equality, per acceptance


def test_deterministic_resume_with_offload(shard_dir, tmp_path):
    """P2-D5 acceptance: the tiny-model ZeRO-Offload run passes the same
    deterministic resume check as the plain loop. Runs where deepspeed is
    installed (the host training container); the CPU sandbox skips it."""
    pytest.importorskip("deepspeed")

    def offload_recipe(root: str, *extra: str):
        return tiny_recipe(
            shard_dir,
            root,
            "optimizer.name=cpu_adam",
            "memory.offload=zero_offload",
            *extra,
        )

    straight = train(offload_recipe(str(tmp_path / "straight")))
    assert straight.final_step == 8

    resumable_root = str(tmp_path / "resumable")
    first = train(offload_recipe(resumable_root), stop_after=4)
    second = train(offload_recipe(resumable_root, "checkpoint.resume=latest"))
    assert second.final_step == 8
    assert first.losses + second.losses == straight.losses
    # The loop-owned scheduler must trace identical LRs across the interrupt
    # (resolved P2-D5 decision 2's contingency check).
    assert first.lrs + second.lrs == straight.lrs


def test_resume_restores_tokens_seen_and_cursor(shard_dir, tmp_path):
    root = str(tmp_path / "run")
    first = train(tiny_recipe(shard_dir, root), stop_after=4)
    second = train(tiny_recipe(shard_dir, root, "checkpoint.resume=latest"))
    per_step = 2 * SEQ_LEN  # micro_batch 2, grad_accum 1
    assert first.tokens_seen == 4 * per_step
    assert second.tokens_seen == 8 * per_step


def test_retention_applies_during_run(shard_dir, tmp_path):
    root = str(tmp_path / "run")
    result = train(
        tiny_recipe(
            shard_dir,
            root,
            "schedule.total_steps=10",
            "checkpoint.interval_steps=2",
            "checkpoint.retain_last=2",
            "checkpoint.retain_every=6",
        )
    )
    from training import checkpoint as ckpt

    # saves at 2,4,6,8,10; survivors: last two (8, 10) plus multiples of 6 (6)
    assert ckpt.list_checkpoints(result.checkpoint_root) == [6, 8, 10]


def test_tracking_writes_tensorboard_and_offline_wandb(
    shard_dir, tmp_path, monkeypatch
):
    pytest.importorskip("tensorboard")
    pytest.importorskip("wandb")
    monkeypatch.setenv("WANDB_MODE", "offline")
    result = train(
        tiny_recipe(
            shard_dir,
            str(tmp_path / "run"),
            "schedule.total_steps=2",
            "checkpoint.interval_steps=100",
            "tracking.tensorboard=true",
            "tracking.wandb=true",
        )
    )
    assert glob.glob(os.path.join(result.run_dir, "tensorboard", "events.*"))
    assert glob.glob(os.path.join(result.run_dir, "wandb", "offline-run-*"))
