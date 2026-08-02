"""Acceptance tests for the D6 reference loop: deterministic resume and tracking.

The deterministic resume test implements the roadmap acceptance check: N+M
steps straight versus N steps, checkpoint, restart, M steps — with
torch.use_deterministic_algorithms(True) and fixed seeds (test only) —
asserting identical per-step loss values. CPU-only; the tiny scratch model is
seconds-long, per the tiny-e2e reference recipe this mirrors.
"""

import glob
import json
import os
import subprocess
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from support import REPO_ROOT, TINY_RECIPE, make_shard_dir  # noqa: E402

from common.reports import validate_report  # noqa: E402
from training.recipes import load_recipe  # noqa: E402
from training.runtime.loop import train  # noqa: E402

VOCAB = 512
SEQ_LEN = 64


@pytest.fixture(autouse=True)
def deterministic():
    torch.use_deterministic_algorithms(True)
    yield
    torch.use_deterministic_algorithms(False)


@pytest.fixture()
def shard_dir(tmp_path, tokenizer_dir) -> str:
    # The manifest carries the real fixture tokenizer's content hash: the CLI
    # path runs validate(), which rejects a shard dir whose hash does not
    # match tokenizer.path. The in-process train() tests do not validate, but
    # a single consistent fixture keeps both paths on the same shards.
    from training.data.prepare import load_tokenizer

    rng = np.random.default_rng(7)
    tokens = rng.integers(0, VOCAB, size=6000).tolist()
    return make_shard_dir(
        tmp_path,
        tokens,
        shard_tokens=2000,
        tokenizer_hash=load_tokenizer(tokenizer_dir).content_hash,
    )


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
    # Skip rather than let OPTIMIZER_FALLBACK_CHAIN degrade to a non-engine
    # optimizer: a degraded run still resumes deterministically, so the
    # assertions below would pass without ever exercising ZeRO-Offload —
    # a false green on the acceptance check this test exists to be.
    from training.runtime.offload import cpu_adam_available

    if not cpu_adam_available():
        pytest.skip("DeepSpeedCPUAdam op unavailable; ZeRO-Offload cannot run")

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
    from training.runtime import checkpoint as ckpt

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


def test_metrics_jsonl_written_with_checkpoint_markers(shard_dir, tmp_path):
    """OBS-10 artifact chain after a real (tiny, CPU) train run."""
    result = train(tiny_recipe(shard_dir, str(tmp_path / "run")))
    path = os.path.join(result.run_dir, "metrics.jsonl")
    with open(path, encoding="utf-8") as f:
        records = [json.loads(line) for line in f]
    headers = [r for r in records if r.get("report") == "run_metrics"]
    assert len(headers) == 1
    assert records[0] is headers[0]  # single locked open: header is line one
    validate_report(headers[0])
    assert headers[0]["tier"] == "pretrain"
    assert headers[0]["recipe_hash"]
    assert [r["step"] for r in records if "loss" in r] == list(range(1, 9))
    saved = [r["step"] for r in records if r.get("checkpoint_saved") == 1]
    assert saved == [4, 8]  # interval 4, total 8


def test_cli_run_joins_artifacts_under_real_watchdog_timing(
    shard_dir, tmp_path, tokenizer_dir
):
    """Host-side proof of the OBS-10 thread fix under real timing: the
    sub-second watchdog sampler logs into the sink concurrently with the
    training loop, and the header identity must still come from the
    dispatching thread — non-empty, matching the events and catalog join."""
    env = os.environ | {
        "OBS_LOG_DIR": str(tmp_path / "obs-logs"),
        "TRAIN_LOCK_PATH": str(tmp_path / ".train.lock"),
    }
    argv = [
        sys.executable,
        "-m",
        "training",
        "train",
        "--config",
        TINY_RECIPE,
        f"data.source={shard_dir}",
        # Unlike the in-process tests this goes through validate(), so the
        # recipe's /mnt/ai-data tokenizer path must be redirected at the
        # fixture tokenizer the shard manifest was hashed against.
        f"tokenizer.path={tokenizer_dir}",
        f"run.output_root={tmp_path / 'out'}",
        f"data.seq_len={SEQ_LEN}",
        "schedule.total_steps=8",
        "schedule.warmup_steps=2",
        "checkpoint.interval_steps=4",
        "watchdog.enabled=true",
        "watchdog.interval_s=0.05",
    ]
    proc = subprocess.run(
        argv, env=env, cwd=REPO_ROOT, capture_output=True, text=True, timeout=600
    )
    assert proc.returncode == 0, proc.stderr[-2000:]

    found = glob.glob(str(tmp_path / "out" / "**" / "metrics.jsonl"), recursive=True)
    assert len(found) == 1
    with open(found[0], encoding="utf-8") as f:
        records = [json.loads(line) for line in f]  # every line parses strictly
    headers = [r for r in records if r.get("report") == "run_metrics"]
    assert len(headers) == 1
    header = headers[0]
    validate_report(header)
    assert header["run_id"].startswith("R-")
    assert header["cli"] == "training"
    assert header["command"] == "train"
    # the sampler thread wrote into the same sink (disk_free_gb comes from
    # shutil, so this holds with or without psutil/NVML)
    assert any("disk_free_gb" in r for r in records)

    events = tmp_path / "obs-logs" / "training" / f"{header['run_id']}.jsonl"
    parsed = [json.loads(line) for line in events.read_text().splitlines()]
    assert parsed[0]["event"] == "cmd.start"
    assert parsed[-1]["event"] == "cmd.end"
    assert parsed[-1]["exit_code"] == 0

    catalog = tmp_path / "obs-logs" / "training" / "runs.jsonl"
    rows = [json.loads(line) for line in catalog.read_text().splitlines()]
    assert rows[-1]["run_id"] == header["run_id"]
    assert rows[-1]["exit_code"] == 0
