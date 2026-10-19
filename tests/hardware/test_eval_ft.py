"""FT.eval-ft, the eval-ft driver on the staged base and the campaign's export, as one test the hw_full tier runs.

The fine-tuning loop runbook (docs/verification/training/fine-tuning-loop.md)
walks FT.eval-ft by hand: `make eval-ft` on the staged Gemma 4 E2B base and
the merged export FT.export-merged wrote, one host process that runs the
four evaluator phases as `docker compose run --rm training` and writes the
four records organism-verify reads. This module runs the same driver, the
venv's `python -m training eval-ft` (the make target resolves the same
interpreter, KB-95, held by tests/training/test_host_targets.py), and
judges what the hand run was judged on: the printed lines in the driver's
order, both digests and their sources (the base's computed, the tuned
side's read from the manifest, KB-96), both sides in the container's
spelling (KB-78), the four records, the driver's pair checks, the organism
verifier's projection, the loss order, and a resumed second invocation.
About twenty minutes: the base's sha256 over 10 GB before any phase, then
holdout under five minutes a side and lm-eval under four, on the hand
run's timings, plus one more base hash for the resume. The test hashes the
base once more itself, so the digest it expects is the file's and not a
constant copied from the runbook.

Neither side nor the run directory can live under tmp_path. The training
container mounts the models tree at /app/models and the training volume
at its own path only (docker-compose.training.yaml), and hf_identity
accepts a side only under the models tree; so the tuned side is the
export FT.export-merged left on disk (a skip when absent, as the base is)
and the run directory is one under the driver's default root, removed
when the test ends. This is the only hardware test that drives compose: a
docker daemon the venv user can reach is a precondition, and its absence
is a failure with the remedy named, in require_binary's shape, not a
skip. A timeout kills the driver and not the container mid-phase; `docker
ps` shows one left behind.

The sandbox half is tests/training/test_eval_ft.py (the container
spelling, the phases stubbed) and tests/training/test_eval_runner.py
(identity from the manifest, the pair checks). It skips when the prepared
dataset, the pinned base, the staged datasets cache or the export is
absent, and fails when the prepared directory is stale (conftest's
sft_smoke_recipe) or the export has no readable manifest.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time

import pytest

from tests.hardware.conftest import (
    LMEVAL_LIMIT,
    REPO_ROOT,
    assert_holdout_shape,
    assert_lmeval_shape,
    require_staged_datasets,
    sft_smoke_recipe,
)
from training import serving
from training.eval.runner import DEFAULT_RUN_ROOT

pytestmark = [
    pytest.mark.hw_full,
    pytest.mark.gpu,
    pytest.mark.ssd,
    pytest.mark.training,
]

# FT.export-merged's export, the entry's subject: the merged directory under
# the models tree whose manifest records its digest.
TUNED = os.path.join(serving.HOST_MODELS, "trained", "ft3-straight", "hf")
# The hand run took sixteen minutes. Generous, because a timeout reads as a
# missing line and leaves a container running (module docstring).
DRIVER_TIMEOUT_S = 40 * 60
# The resumed invocation hashes the base again and skips every phase.
RESUME_TIMEOUT_S = 5 * 60
SIDES = ("base", "tuned")
EVALUATORS = ("holdout", "lmeval")


def read_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def require_docker() -> None:
    """The driver's phases are compose runs; a host without docker on PATH is
    a misconfigured run, not the unavailable hardware the module rule
    exempts."""
    if shutil.which("docker") is None:
        pytest.fail(
            "[ERROR] docker not on PATH; eval-ft runs its phases as compose runs\n"
            "  Solution: run on the workstation, with the venv user in the docker group",
            pytrace=False,
        )


def run_driver(
    base: str, dataset: str, out_dir: str, timeout_s: float
) -> subprocess.CompletedProcess:
    """The driver as `make eval-ft` runs it, through this interpreter."""
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "training",
            "eval-ft",
            "--base",
            base,
            "--tuned",
            TUNED,
            "--dataset",
            dataset,
            "--lmeval-limit",
            str(LMEVAL_LIMIT),
            "--out-dir",
            out_dir,
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=timeout_s,
        env=dict(os.environ, NO_COLOR="1"),
    )


def in_order(stderr: str, prefixes: list[str]) -> None:
    """Each prefix opens a line after the previous prefix's line: the
    driver's own print order, which the entry's expected signal states."""
    lines = stderr.splitlines()
    at = 0
    for prefix in prefixes:
        for i in range(at, len(lines)):
            if lines[i].startswith(prefix):
                at = i + 1
                break
        else:
            pytest.fail(
                f"no line opening {prefix!r} after line {at}\n{stderr}", pytrace=False
            )


def test_eval_ft(cuda, tmp_path, train_lock):
    from training.eval.lmeval import load_task_config
    from training.eval.runner import check_holdout_pair, check_lmeval_pair
    from training.export.errors import ExportError
    from training.export.manifest import load_hf_manifest

    recipe, base = sft_smoke_recipe([])
    dataset = recipe.sft.dataset
    require_staged_datasets()
    if not os.path.isfile(os.path.join(TUNED, "model.safetensors")):
        pytest.skip(f"FT.export-merged's export absent at {TUNED}")
    try:
        tuned_hash = load_hf_manifest(TUNED)["model_hash"]
    except ExportError as exc:
        pytest.fail(
            f"export at {TUNED} has no readable manifest: {exc}; re-run FT.export-merged",
            pytrace=False,
        )
    require_docker()

    base_hash = serving.sha256_file(os.path.join(base.model, "model.safetensors"))
    container = {
        "base": serving.model_paths(base.model)[0],
        "tuned": serving.model_paths(TUNED)[0],
    }
    task_config = load_task_config()
    tasks = sorted(task_config["tasks"])
    holdout_examples = read_json(os.path.join(dataset, "manifest.json"))[
        "holdout_examples"
    ]

    out_dir = os.path.join(
        DEFAULT_RUN_ROOT, time.strftime("ft6-hw_%Y%m%d_%H%M%S", time.gmtime())
    )
    records = {
        (evaluator, side): os.path.join(out_dir, f"{evaluator}_{side}.json")
        for evaluator in EVALUATORS
        for side in SIDES
    }
    written = [f"[OK] Record written: {path}" for path in records.values()]
    try:
        proc = run_driver(base.model, dataset, out_dir, DRIVER_TIMEOUT_S)
        assert proc.returncode == 0, proc.stderr
        assert "[ERROR]" not in proc.stderr, proc.stderr
        in_order(
            proc.stderr,
            [
                f"[INFO] Run directory: {out_dir}",
                f"[INFO] base: {container['base']} model_hash {base_hash} "
                "(computed) thinking_mode off",
                f"[INFO] tuned: {container['tuned']} model_hash {tuned_hash} "
                "(manifest) thinking_mode off",
                f"[RUN] holdout/base: {container['base']}",
                "[OK] holdout/base: loss ",
                f"[RUN] holdout/tuned: {container['tuned']}",
                "[OK] holdout/tuned: loss ",
                f"[RUN] lmeval/base: {container['base']}",
                f"[RUN] lmeval/tuned: {container['tuned']}",
                *[f"[OK] lmeval/{task}: " for task in tasks],
                *written,
            ],
        )

        holdout = {side: read_json(records["holdout", side]) for side in SIDES}
        lmeval = {side: read_json(records["lmeval", side]) for side in SIDES}
        for side in SIDES:
            assert_holdout_shape(
                holdout[side], container[side], dataset, holdout_examples
            )
            assert_lmeval_shape(
                lmeval[side], container[side], task_config, LMEVAL_LIMIT
            )

        # The printed values are the records', not a second measurement.
        for side in SIDES:
            assert f"[OK] holdout/{side}: loss {holdout[side]['loss']:.6f}" in (
                proc.stderr
            ), proc.stderr
        summary = [
            f"[OK] lmeval/{task}: {lmeval['base']['results'][task]['value']:.4f} "
            f"-> {lmeval['tuned']['results'][task]['value']:.4f}"
            for task in tasks
        ]
        for line in summary:
            assert line in proc.stderr, (line, proc.stderr)

        # The driver's confound checks and the consumer's reader (FT.organisms).
        check_holdout_pair(holdout["base"], holdout["tuned"])
        check_lmeval_pair(lmeval["base"], lmeval["tuned"])
        from interp.organisms.verify import locality_metrics

        for side in SIDES:
            leaves = locality_metrics([holdout[side], lmeval[side]])
            assert set(leaves["holdout"]) == {"loss"}, leaves
            assert set(leaves["lmeval"]) == set(tasks), leaves

        # The entry's one ordering; nothing ranks lm-eval at limit 20.
        assert holdout["tuned"]["loss"] <= holdout["base"]["loss"], (
            f"tuned held-out loss {holdout['tuned']['loss']} above the base's "
            f"{holdout['base']['loss']}"
        )

        # The entry's resume clause: the same run directory, every phase
        # skipped on its record, the same summary, no container started.
        again = run_driver(base.model, dataset, out_dir, RESUME_TIMEOUT_S)
        assert again.returncode == 0, again.stderr
        assert not [
            line for line in again.stderr.splitlines() if line.startswith("[RUN] ")
        ], again.stderr
        in_order(
            again.stderr,
            [
                f"[SKIP] {evaluator}/{side}: record exists (resume)"
                for evaluator, side in records
            ]
            + written,
        )
        for line in summary:
            assert line in again.stderr, (line, again.stderr)
    finally:
        shutil.rmtree(out_dir, ignore_errors=True)
