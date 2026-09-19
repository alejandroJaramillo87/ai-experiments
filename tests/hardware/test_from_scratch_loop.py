"""FS.0 to FS.4, FS.6 and FS.7, the from-scratch loop's opening, as one hw_full test.

The from-scratch runbook (docs/verification/training/from-scratch-loop.md)
walks these by hand: prepare shards from a trained tokenizer, run the tiny
scratch recipe for twenty steps through `make train`, resume it to thirty
through checkpoint.resume=latest, run again with staging and prefetch on,
then again with a staging budget the shard set exceeds, then a second run
while the first holds the run lock, then a run with the watchdog on. This
module runs the same steps through the same CLI verbs, on a tokenizer it trains here, and
holds them to the signals the entries record: the shard manifest carries the
tokenizer's hash, the run prints Training then Resuming, tokens_seen
continues rather than restarts, retention leaves the last two checkpoints
plus the every-ten survivors, the first metrics record passes the shared
report envelope, the staged copy lands before the first step, every step
records its data wait, and an over-budget set is declined with the reason
while the run reads from the source and completes, and a held lock refuses
the second run by name before it allocates and is gone once released, and
the watchdog on a healthy run samples and raises nothing.

tests/hardware/test_gpu.py already proves resume in-process, to a loss
tolerance. What it cannot show is the path an operator uses: the `train`
verb, its validation of the tokenizer against the shards, the printed
signals, and the checkpoint directory an interrupted run leaves behind.
That is what this module adds. It grows with the runbook: an entry that
verifies by hand gets its automated mirror here.

Nothing here reaches the hub or /mnt/ai-data; the corpus is the synthetic
one the tokenizer tests use, so the run needs only the GPU.
"""

from __future__ import annotations

import json
import math
import os

import pytest

from tests.hardware.conftest import REPO_ROOT

pytestmark = [pytest.mark.hw_full, pytest.mark.gpu]

RECIPE = os.path.join(REPO_ROOT, "training", "recipes", "reference", "tiny-e2e.yaml")
RUN_NAME = "fs-loop"
VOCAB = 320  # above the 275-id floor; small enough to train in milliseconds
# tiny-e2e trains micro_batch 2 at seq_len 128: 256 tokens a step
TOKENS_PER_STEP = 2 * 128
FIRST_STEPS = 20
RESUMED_STEPS = 30
# Well past what thirty steps read, so the loader never wraps mid-entry
CORPUS_COPIES = 400
# FS.4: five steps under a budget the half-megabyte synthetic set exceeds
REFUSED_STEPS = 5
REFUSED_MAX_GB = 0.0001
# FS.7: a twenty-step run of this model is well under a second in-process
QUIET_INTERVAL_S = 0.05


def train_tokenizer(tmp_path):
    """FS.0's precondition: a D3 artifact and the JSONL it was trained on."""
    from tests.training.conftest import DOCUMENTS
    from training.data import tokenizer as tokenizer_mod

    corpus = DOCUMENTS + ["year 2026, port 8002, id 49152"]
    jsonl = tmp_path / "corpus.jsonl"
    with open(jsonl, "w") as f:
        for _ in range(CORPUS_COPIES):
            for text in corpus:
                f.write(json.dumps({"text": text}) + "\n")
    tok_dir = tmp_path / "tokenizer"
    manifest = tokenizer_mod.write_artifact(
        tokenizer_mod.train(corpus, VOCAB),
        str(tok_dir),
        corpus=[str(jsonl)],
        text_field="text",
        sampled_bytes=1,
        sampled_documents=len(corpus),
        vocab_size=VOCAB,
        metrics={},
        references={},
        command="test",
    )
    return str(jsonl), str(tok_dir), manifest["tokenizer_hash"]


def run_verb(argv: list[str]) -> int:
    from training import cli

    args = cli.build_parser().parse_args(argv)
    return args.func(args)


def checkpoints(root) -> set[str]:
    return set(os.listdir(root))


def read_meta(root, step: int) -> dict:
    with open(root / f"step_{step:09d}" / "meta.json") as f:
        return json.load(f)


def test_from_scratch_loop(cuda, tmp_path, train_lock, capsys):
    from common.reports import validate_report

    # FS.0: shards from the tokenizer, and the recipe validating against them.
    jsonl, tok_dir, tok_hash = train_tokenizer(tmp_path)
    shards = tmp_path / "shards"
    assert (
        run_verb(
            [
                "prepare-data",
                "--source",
                jsonl,
                "--text-field",
                "text",
                "--tokenizer",
                tok_dir,
                "--out",
                str(shards),
            ]
        )
        == 0
    )
    with open(shards / "manifest.json") as f:
        shard_manifest = json.load(f)
    assert shard_manifest["dtype"] == "uint16"
    assert shard_manifest["tokenizer_hash"] == tok_hash
    assert shard_manifest["total_tokens"] >= RESUMED_STEPS * TOKENS_PER_STEP * 2
    capsys.readouterr()

    overrides = [
        f"data.source={shards}",
        f"tokenizer.path={tok_dir}",
        f"model.scratch.vocab={VOCAB}",
        f"run.output_root={tmp_path / 'runs'}",
        f"run.name={RUN_NAME}",
    ]
    ckpt_root = tmp_path / "runs" / RUN_NAME / "checkpoints"

    # FS.1: twenty steps through the train verb.
    assert (
        run_verb(
            [
                "train",
                "--config",
                RECIPE,
                *overrides,
                f"schedule.total_steps={FIRST_STEPS}",
            ]
        )
        == 0
    )
    out = capsys.readouterr()
    printed = out.out + out.err
    assert f"[RUN] Training {RUN_NAME}: {FIRST_STEPS} total steps" in printed
    assert (
        f"[OK] Run complete: step {FIRST_STEPS}, "
        f"{FIRST_STEPS * TOKENS_PER_STEP:,} tokens seen"
    ) in printed
    assert checkpoints(ckpt_root) == {
        "latest",
        "step_000000010",
        "step_000000015",
        "step_000000020",
    }, sorted(checkpoints(ckpt_root))
    meta = read_meta(ckpt_root, FIRST_STEPS)
    assert meta["step"] == FIRST_STEPS
    assert meta["tokens_seen"] == FIRST_STEPS * TOKENS_PER_STEP
    assert math.isfinite(meta["loss"])
    assert meta["checkpoint_format"] == "torch"
    assert meta["optimizer"] == "adamw"
    assert meta["attention_implementation"] in ("flash_attention_2", "sdpa")

    run_dirs = [
        d for d in os.listdir(tmp_path / "runs" / RUN_NAME) if d != "checkpoints"
    ]
    assert len(run_dirs) == 1, run_dirs
    run_dir = tmp_path / "runs" / RUN_NAME / run_dirs[0]
    assert (run_dir / "config.yaml").is_file()
    with open(run_dir / "metrics.jsonl") as f:
        header = json.loads(f.readline())
    validate_report(header)
    assert header["report"] == "run_metrics"
    assert header["tier"] == "pretrain"
    assert header["recipe_hash"] == meta["recipe_hash"]
    assert header["schedule"]["total_steps"] == FIRST_STEPS

    # FS.2: ten more steps through checkpoint.resume=latest, continuing the
    # token count; the changed step count is what the recipe-hash warning is for.
    assert (
        run_verb(
            [
                "train",
                "--config",
                RECIPE,
                *overrides,
                f"schedule.total_steps={RESUMED_STEPS}",
                "checkpoint.resume=latest",
            ]
        )
        == 0
    )
    out = capsys.readouterr()
    printed = out.out + out.err
    assert f"[RUN] Resuming (latest) {RUN_NAME}: {RESUMED_STEPS} total steps" in printed
    assert "resuming with a recipe that differs from the checkpoint's" in printed
    assert (
        f"[OK] Run complete: step {RESUMED_STEPS}, "
        f"{RESUMED_STEPS * TOKENS_PER_STEP:,} tokens seen"
    ) in printed
    assert checkpoints(ckpt_root) == {
        "latest",
        "step_000000010",
        "step_000000020",
        "step_000000025",
        "step_000000030",
    }, sorted(checkpoints(ckpt_root))
    resumed = read_meta(ckpt_root, RESUMED_STEPS)
    assert resumed["step"] == RESUMED_STEPS
    assert resumed["tokens_seen"] == RESUMED_STEPS * TOKENS_PER_STEP
    assert resumed["recipe_hash"] != meta["recipe_hash"]

    # FS.3: a fresh run with staging and prefetch on. The shard set is copied
    # into the staging directory before the first step and every step records
    # its data wait; the 5 percent bar itself is test_ssd.py's, on a longer
    # step than this model has.
    staging = tmp_path / "staging"
    staging.mkdir()
    staged_name = "fs-staged"
    assert (
        run_verb(
            [
                "train",
                "--config",
                RECIPE,
                *overrides,
                f"run.name={staged_name}",
                f"schedule.total_steps={FIRST_STEPS}",
                "staging.enabled=true",
                f"staging.dir={staging}",
                "data.workers=2",
                "data.pin_memory=true",
            ]
        )
        == 0
    )
    out = capsys.readouterr()
    printed = out.out + out.err
    # The synthetic set is under a megabyte, so the size prints as 0.00 GB;
    # the line's shape and destination are the signal, not the figure.
    assert "[staging] OK: staged " in printed, printed
    assert f" GB to {staging / 'shards'}" in printed, printed
    assert (
        f"[OK] Run complete: step {FIRST_STEPS}, "
        f"{FIRST_STEPS * TOKENS_PER_STEP:,} tokens seen"
    ) in printed
    assert (staging / "shards" / "manifest.json").is_file()
    staged_dirs = [
        d for d in os.listdir(tmp_path / "runs" / staged_name) if d != "checkpoints"
    ]
    assert len(staged_dirs) == 1, staged_dirs
    with open(tmp_path / "runs" / staged_name / staged_dirs[0] / "metrics.jsonl") as f:
        records = [json.loads(line) for line in f]
    waits = [r["data_wait_ms"] for r in records if "data_wait_ms" in r]
    assert len(waits) == FIRST_STEPS, f"{len(waits)} steps carry data_wait_ms"
    assert all(math.isfinite(w) and w >= 0.0 for w in waits)

    # FS.4: a budget below the shard set. Staging declines with the reason
    # and the run reads from the source directory instead of failing.
    refused = tmp_path / "staging-refused"
    refused.mkdir()
    assert (
        run_verb(
            [
                "train",
                "--config",
                RECIPE,
                *overrides,
                "run.name=fs-refused",
                f"schedule.total_steps={REFUSED_STEPS}",
                "staging.enabled=true",
                f"staging.dir={refused}",
                f"staging.max_gb={REFUSED_MAX_GB}",
            ]
        )
        == 0
    )
    out = capsys.readouterr()
    printed = out.out + out.err
    assert "[staging] SKIP: shard set " in printed, printed
    assert (
        f" GB exceeds staging.max_gb {REFUSED_MAX_GB}; reading shards from {shards}"
        in printed
    ), printed
    assert "[staging] OK: staged" not in printed
    assert os.listdir(refused) == [], os.listdir(refused)
    assert (
        f"[OK] Run complete: step {REFUSED_STEPS}, "
        f"{REFUSED_STEPS * TOKENS_PER_STEP:,} tokens seen"
    ) in printed

    # FS.6: a second run while the lock is held is refused before it allocates
    # anything, names the holder, and the lock does not survive its holder.
    from training.runtime.lock import lock_path, read_lock, run_lock

    with run_lock("fs6a") as held:
        holder = read_lock(held)
        assert holder["run"] == "fs6a"
        assert holder["pid"] == os.getpid()
        assert (
            run_verb(
                [
                    "train",
                    "--config",
                    RECIPE,
                    *overrides,
                    "run.name=fs-locked",
                    f"schedule.total_steps={REFUSED_STEPS}",
                ]
            )
            == 1
        )
        out = capsys.readouterr()
        printed = out.out + out.err
        assert (
            f"[ERROR] Training failed: Training lock held by pid {os.getpid()} "
            f"(run 'fs6a'): {held}"
        ) in printed, printed
        assert (
            "Solution: wait for the running job to finish (one GPU consumer at a time)"
            in printed
        )
        assert "[RUN] Training fs-locked" in printed
        assert "[OK] Run complete" not in printed
        assert not (tmp_path / "runs" / "fs-locked").exists()
    assert not os.path.exists(lock_path())

    # FS.7: the watchdog on a healthy run samples and raises nothing. The
    # interval is shorter than the runbook's second because this run is
    # milliseconds; the sampler thread must land at least one NVML record.
    from training.runtime.watchdog import ALERTS_FILENAME, LOG_PREFIX

    quiet_name = "fs-quiet"
    assert (
        run_verb(
            [
                "train",
                "--config",
                RECIPE,
                *overrides,
                f"run.name={quiet_name}",
                f"schedule.total_steps={FIRST_STEPS}",
                "watchdog.enabled=true",
                f"watchdog.interval_s={QUIET_INTERVAL_S}",
            ]
        )
        == 0
    )
    out = capsys.readouterr()
    printed = out.out + out.err
    assert (
        f"[OK] Run complete: step {FIRST_STEPS}, "
        f"{FIRST_STEPS * TOKENS_PER_STEP:,} tokens seen"
    ) in printed
    assert LOG_PREFIX not in printed, printed
    quiet_dirs = [
        d for d in os.listdir(tmp_path / "runs" / quiet_name) if d != "checkpoints"
    ]
    assert len(quiet_dirs) == 1, quiet_dirs
    quiet_run = tmp_path / "runs" / quiet_name / quiet_dirs[0]
    assert sorted(os.listdir(quiet_run)) == ["config.yaml", "metrics.jsonl"]
    assert not (quiet_run / ALERTS_FILENAME).exists()
    with open(quiet_run / "metrics.jsonl") as f:
        samples = [
            r
            for r in (json.loads(line) for line in f)
            if r.get("gpu_temp_c") is not None
        ]
    assert samples, "the sampler thread landed no NVML record"
    assert all(math.isfinite(s["gpu_temp_c"]) for s in samples)
