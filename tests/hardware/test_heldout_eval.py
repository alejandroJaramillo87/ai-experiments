"""FT.heldout-eval, held-out loss and lm-eval on base and merged, as one test the hw_full tier runs.

The fine-tuning loop runbook (docs/verification/training/fine-tuning-loop.md)
walks FT.heldout-eval by hand: eval-holdout and eval-lmeval on the staged
Gemma 4 E2B base and on the merged export of a QLoRA adapter, four compose
runs in the training container, held to the printed lines and the records.
This module trains 20 QLoRA steps on the smoke recipe, exports the merged
directory, and runs the same four verbs on both sides: 20 steps and six
model loads, about 25 minutes (the training run five, the CPU merge three,
a holdout pass four a side, an lm-eval pass five a side, most of the last in
the harness's batch-size search).

Every phase is a process of its own, `python -m training <verb>` on the
workstation venv, as the hand run's compose runs are. That venv must carry
what the lock names: lm-eval's ifeval extra (langdetect, immutabledict,
nltk) reached the lock in the KB-92 fix, and a venv synced before it keeps
running without them, because `uv run --no-sync` never reconciles the two
(tests/hardware/test_venv_lock.py). The mirror's first run (2026-09-25)
found exactly that, fifteen minutes in, at the first lm-eval task load; so
the extra is checked before any GPU work, and its absence is a failure with
`make install` named, in conftest's cuda shape, not a skip. Two reasons for
the process boundary. A process
boundary frees the GPU between phases, where the in-process training tests
leave CUDA tensors behind (conftest's cuda fixture reports them), and a
holdout pass needs the bf16 model beside the full logits of an example. And
`datasets` and nltk read their environment at import, so a phase started
fresh reads HF_HOME at the prepare service's cache and the three offline
flags, and finds the four staged datasets where the training container
finds them (KB-92); the host has no NLTK_DATA, so nltk fetches punkt_tab
into its home directory once, which the container never could (KB-93).

What is held: each verb's exit and [RUN]/[OK] lines, the holdout record's
shape (bf16, sdpa, the dataset's holdout count; the shape helpers live in
conftest, shared with FT.eval-ft's mirror), the lm-eval record's shape
(the pinned version and limit, every pinned task under the key tasks.yaml
spells, KB-94), the eval-ft driver's own pair checks, the organism
verifier's projection of both records, and the entry's one ordering: the
merged model's held-out loss at or below the base's on the data it trained
on. Nothing ranks the lm-eval values at a limit of 20, as the entry says.

The sandbox half is tests/training/test_cli_handlers_export.py (the two
handlers on faked records) and tests/training/test_lmeval.py (the extractor
and the pinned keys). It skips when the prepared dataset (FT.smoke-dataset),
the pinned base (FT.preconditions) or the staged datasets cache
(FT.heldout-eval's precondition block) is absent, and fails when the
prepared directory is stale (conftest's sft_smoke_recipe).
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys

import pytest

from tests.hardware.conftest import (
    DATASETS_CACHE,
    LMEVAL_LIMIT,
    REPO_ROOT,
    SFT_RECIPE,
    assert_holdout_shape,
    assert_lmeval_shape,
    require_staged_datasets,
    sft_smoke_recipe,
)

pytestmark = [
    pytest.mark.hw_full,
    pytest.mark.gpu,
    pytest.mark.ssd,
    pytest.mark.training,
]

# Twenty steps, not the loop test's ten: the entry's ordering (merged at or
# below base) needs the adapter to have moved, and ten steps over 320
# examples may not; the hand run's 60 steps gave 1.019 against 1.064.
TOTAL_STEPS = 20
# Timeouts read from the hand runs: about 16 s a step plus a load; the CPU
# merge of 10 GB in minutes; a holdout pass of 4066 examples under four
# minutes a side; an lm-eval pass about four, plus the batch-size search.
# Generous, because a timeout reads as a missing line.
TRAIN_TIMEOUT_S = 15 * 60
EXPORT_TIMEOUT_S = 10 * 60
HOLDOUT_TIMEOUT_S = 15 * 60
LMEVAL_TIMEOUT_S = 20 * 60
# The members of lm-eval's ifeval extra in uv.lock, which the ifeval task
# imports at load. The 2026-09-25 instance: the workstation venv predated the
# lock that added them, and the base's eval-lmeval died on langdetect.
IFEVAL_EXTRA = ("langdetect", "immutabledict", "nltk")


def training_cli(args: list[str], timeout_s: float) -> subprocess.CompletedProcess:
    """One `python -m training` verb in its own process, offline, with the
    HF cache at the prepare service's root (module docstring)."""
    env = dict(
        os.environ,
        NO_COLOR="1",
        HF_HOME=DATASETS_CACHE,
        HF_DATASETS_OFFLINE="1",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
    )
    return subprocess.run(
        [sys.executable, "-m", "training", *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=timeout_s,
        env=env,
    )


def require_ifeval_extra() -> None:
    """Fail, with the remedy, when the interpreter the phases run lacks the
    ifeval extra: the phases use sys.executable, so find_spec here is the
    same answer the subprocess would give, seconds in rather than minutes."""
    missing = [name for name in IFEVAL_EXTRA if importlib.util.find_spec(name) is None]
    if missing:
        pytest.fail(
            f"[ERROR] the venv at {sys.prefix} lacks lm-eval's ifeval extra\n"
            f"  Found: {', '.join(missing)} not importable\n"
            "  Solution: sync the venv with the lock (uv run --no-sync runs what is there)\n"
            "  Command: make install",
            pytrace=False,
        )


def read_json(path) -> dict:
    with open(path) as f:
        return json.load(f)


def eval_holdout(model: str, dataset: str, out) -> dict:
    proc = training_cli(
        ["eval-holdout", "--model", model, "--dataset", dataset, "--out", str(out)],
        HOLDOUT_TIMEOUT_S,
    )
    assert proc.returncode == 0, proc.stderr
    assert f"[RUN] Held-out loss: {model} over {dataset}" in proc.stderr, proc.stderr
    assert "[OK] Loss " in proc.stderr, proc.stderr
    assert f"[OK] Record written: {out}" in proc.stderr, proc.stderr
    record = read_json(out)
    ok_line = next(line for line in proc.stderr.splitlines() if "[OK] Loss " in line)
    assert f"({record['examples']} examples, " in ok_line, ok_line
    assert ok_line.endswith("bf16)"), ok_line
    return record


def eval_lmeval(model: str, out, task_config: dict) -> dict:
    proc = training_cli(
        [
            "eval-lmeval",
            "--model",
            model,
            "--limit",
            str(LMEVAL_LIMIT),
            "--out",
            str(out),
        ],
        LMEVAL_TIMEOUT_S,
    )
    assert proc.returncode == 0, proc.stderr
    assert "[ERROR]" not in proc.stderr, proc.stderr
    assert f"[RUN] lm-evaluation-harness: {model}" in proc.stderr, proc.stderr
    assert f"[OK] Record written: {out}" in proc.stderr, proc.stderr
    # One line per pinned task, sorted, the metric spelled as the harness
    # keys it (KB-94): the lines the hand run was judged on.
    ok_lines = [
        line
        for line in proc.stderr.splitlines()
        if line.startswith("[OK] ") and ": " in line and " = " in line
    ]
    expected = [
        f"[OK] {task}: {spec['metric']} = "
        for task, spec in sorted(task_config["tasks"].items())
    ]
    assert len(ok_lines) == len(expected), proc.stderr
    for line, prefix in zip(ok_lines, expected, strict=True):
        assert line.startswith(prefix), (line, prefix)
    return read_json(out)


def test_heldout_eval(cuda, tmp_path, train_lock):
    from training.eval.lmeval import load_task_config
    from training.eval.runner import check_holdout_pair, check_lmeval_pair

    overrides = [
        "run.name=ft5-heldout",
        f"run.output_root={tmp_path / 'runs'}",
        f"schedule.total_steps={TOTAL_STEPS}",
        "schedule.warmup_steps=2",
        f"checkpoint.interval_steps={TOTAL_STEPS}",
        "checkpoint.retain_last=1",
        # The trainer's own held-out pass is not this test's; the evaluator's is.
        "sft.eval_steps=0",
    ]
    recipe, base = sft_smoke_recipe(overrides)
    dataset = recipe.sft.dataset
    require_staged_datasets()
    require_ifeval_extra()
    holdout_examples = read_json(os.path.join(dataset, "manifest.json"))[
        "holdout_examples"
    ]
    task_config = load_task_config()

    # The adapter, through the real verb, in its own process.
    trained = training_cli(
        ["train-sft", "--config", SFT_RECIPE, *overrides], TRAIN_TIMEOUT_S
    )
    assert trained.returncode == 0, trained.stderr
    assert f"[OK] Run complete: step {TOTAL_STEPS} [trl]" in trained.stderr, (
        trained.stderr
    )
    ckpt = (
        tmp_path / "runs" / "ft5-heldout" / "checkpoints" / f"checkpoint-{TOTAL_STEPS}"
    )
    meta = read_json(ckpt / "meta.json")
    assert meta["adapter_only"] is True
    assert meta["base_revision"] == base.revision

    # The merged side (FT.export-merged): this test needs the directory; the
    # merge's own invariants are test_fine_tuning_loop.py's.
    out_root = tmp_path / "trained"
    exported = training_cli(
        [
            "export",
            "--checkpoint",
            str(ckpt),
            "--name",
            "ft5-heldout",
            "--out-root",
            str(out_root),
        ],
        EXPORT_TIMEOUT_S,
    )
    assert exported.returncode == 0, exported.stderr
    assert (
        f"[OK] Base model verified: {base.model} (revision {base.revision})"
        in exported.stderr
    ), exported.stderr
    assert "[OK] Manifest written:" in exported.stderr, exported.stderr
    merged = str(out_root / "ft5-heldout" / "hf")
    assert os.path.isfile(os.path.join(merged, "model.safetensors"))
    assert os.path.isfile(os.path.join(merged, "config.json"))

    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()
    sides = {"base": base.model, "merged": merged}

    holdout = {
        side: eval_holdout(model, dataset, eval_dir / f"holdout_{side}.json")
        for side, model in sides.items()
    }
    for side, model in sides.items():
        assert_holdout_shape(holdout[side], model, dataset, holdout_examples)

    lmeval = {
        side: eval_lmeval(model, eval_dir / f"lmeval_{side}.json", task_config)
        for side, model in sides.items()
    }
    for side, model in sides.items():
        assert_lmeval_shape(lmeval[side], model, task_config, LMEVAL_LIMIT)

    # The driver's confound checks (FT.eval-ft) and the consumer's reader
    # (FT.organisms): both sides compare, and each side projects to
    # holdout.loss and one lmeval.<task> per pinned task.
    check_holdout_pair(holdout["base"], holdout["merged"])
    check_lmeval_pair(lmeval["base"], lmeval["merged"])
    from interp.organisms.verify import locality_metrics

    for side in sides:
        leaves = locality_metrics([holdout[side], lmeval[side]])
        assert set(leaves["holdout"]) == {"loss"}, leaves
        assert set(leaves["lmeval"]) == set(task_config["tasks"]), leaves

    # The entry's one ordering: the merged model on the data it trained on.
    assert holdout["merged"]["loss"] <= holdout["base"]["loss"], (
        f"merged held-out loss {holdout['merged']['loss']} above the base's "
        f"{holdout['base']['loss']}"
    )
