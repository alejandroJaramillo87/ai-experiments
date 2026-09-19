"""FT.11, the fine-tuning loop's seam, as one test the hw_full tier runs.

The fine-tuning loop runbook (docs/verification/training/fine-tuning-loop.md)
walks FT.3 and FT.4 by hand: train a QLoRA adapter on Gemma 4 E2B, export the
merged HF directory, and load it in the harness. This module runs those three
steps in one function on the smoke recipe and holds them to the signals the
entries record: a stamped checkpoint is written, the export is one
model.safetensors whose manifest hash is the sha256 of the file, and the
harness preflight accepts the directory and, under --deep, agrees with the
manifest about which weights it is serving.

The sandbox half is tests/training/test_export_harness.py, which runs the
export and the two harness verdicts on the committed tiny fixture with no
adapter and no GPU. What it cannot show is the merge: an adapter trained
here on real weights, merged into the pinned base, and served. That is what
this module adds, and why it needs the GPU.

The training run is in-process, as tests/hardware/test_gpu.py runs the
scratch loop, so the run lock, the memory-fit preflight and the checkpoint
callbacks are the real ones. Ten steps at a checkpoint interval of ten
leaves exactly one checkpoint-10 directory with meta.json beside it, which
the export then merges. It skips when the prepared dataset (FT.2) or the
pinned base (FT.0) is absent, and fails, rather than skips, when a release
binary is.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess

import pytest

from tests.hardware.conftest import REPO_ROOT, altered, write_probe

pytestmark = [pytest.mark.hw_full, pytest.mark.gpu, pytest.mark.ssd]

HARNESS_DIR = os.path.join(REPO_ROOT, "rust-harness")
BIN_DIR = os.path.join(HARNESS_DIR, "target", "release")
HARNESS = os.path.join(BIN_DIR, "harness")
CHECKPOINT_INFO = os.path.join(BIN_DIR, "checkpoint-info")
RECIPE = os.path.join(
    REPO_ROOT, "training", "recipes", "reference", "sft-qlora-e2b.yaml"
)

# Ten steps: enough to write one checkpoint through the real callback and
# little enough that the run is minutes, not the hour FT.3 spends.
TOTAL_STEPS = 10
# E2B is about 5 GB in bf16; a hash of it is seconds, a preflight is under a
# minute on 2026-09-11 (RH.16 on the 15 GB E4B took 80 s for two passes).
PREFLIGHT_TIMEOUT_S = 10 * 60


def require_binaries() -> None:
    for binary in (HARNESS, CHECKPOINT_INFO):
        assert os.path.isfile(binary), (
            f"built binary absent: {binary}; run `make harness-build` in {HARNESS_DIR}"
        )


def run(binary: str, args: list[str], timeout_s: float) -> subprocess.CompletedProcess:
    env = dict(os.environ, NO_COLOR="1")
    return subprocess.run(
        [binary, *args],
        capture_output=True,
        text=True,
        timeout=timeout_s,
        env=env,
        cwd=HARNESS_DIR,
    )


def test_fine_tuning_loop(cuda, tmp_path, train_lock):
    from training.export.errors import ExportError
    from training.export.preflight import resolve_base_identity
    from training.recipes import load_recipe
    from training.tune.sft import train_sft

    require_binaries()
    recipe = load_recipe(
        RECIPE,
        [
            "run.name=ft11-loop",
            f"run.output_root={tmp_path / 'runs'}",
            f"schedule.total_steps={TOTAL_STEPS}",
            f"checkpoint.interval_steps={TOTAL_STEPS}",
            "checkpoint.retain_last=1",
        ],
    )
    if not os.path.isdir(recipe.sft.dataset):
        pytest.skip(f"prepared dataset not on disk at {recipe.sft.dataset} (FT.2)")
    try:
        base = resolve_base_identity(
            recipe.model.hf.id, revision=recipe.model.hf.revision, local_files_only=True
        )
    except ExportError as exc:
        pytest.skip(f"pinned base not in the local HF cache (FT.0): {exc}")
    # FT.3: one stamped checkpoint from the real trainer.
    result = train_sft(recipe)
    assert result.final_step == TOTAL_STEPS
    ckpt = tmp_path / "runs" / "ft11-loop" / "checkpoints" / f"checkpoint-{TOTAL_STEPS}"
    assert ckpt.is_dir(), f"no checkpoint-{TOTAL_STEPS} under {ckpt.parent}"
    with open(ckpt / "meta.json") as f:
        meta = json.load(f)
    assert meta["adapter_only"] is True
    assert meta["base_revision"] == base.revision
    assert meta["base_config_sha256"] == base.config_sha256

    # FT.4: the merged HF directory, through the real verb.
    from training import cli

    out_root = tmp_path / "trained"
    args = cli.build_parser().parse_args(
        [
            "export",
            "--checkpoint",
            str(ckpt),
            "--name",
            "ft11-loop",
            "--out-root",
            str(out_root),
        ]
    )
    assert args.func(args) == 0
    hf_dir = out_root / "ft11-loop" / "hf"
    with open(out_root / "ft11-loop" / "manifest.json") as f:
        manifest = json.load(f)
    assert manifest["mode"] == "merge"
    assert manifest["base"]["revision"] == base.revision
    weights = [n for n in os.listdir(hf_dir) if n.endswith(".safetensors")]
    assert weights == ["model.safetensors"], weights
    assert not (hf_dir / "model.safetensors.index.json").exists()
    assert (hf_dir / "tokenizer.json").is_file()
    digest = hashlib.sha256()
    with open(hf_dir / "model.safetensors", "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    assert manifest["model_hash"] == digest.hexdigest()

    # The consumer's verdicts: checkpoint-info, then the deep preflight
    # against a probe carrying the manifest's hash, then its negative.
    info = run(CHECKPOINT_INFO, ["--check", "--text-only", str(hf_dir)], 120)
    assert info.returncode == 0, info.stderr
    assert "[OK] CPU backend compatibility: supported" in info.stderr

    with open(hf_dir / "config.json") as f:
        config = json.load(f)
    text = config.get("text_config", config)
    (tmp_path / "probe").mkdir()
    image = write_probe(
        tmp_path / "probe", text["hidden_size"], 0, manifest["model_hash"]
    )
    status = run(
        HARNESS,
        [
            "--model",
            str(hf_dir),
            "--text-only",
            "--probe",
            str(image),
            "--status",
            "--deep",
        ],
        PREFLIGHT_TIMEOUT_S,
    )
    assert status.returncode == 0, status.stderr
    assert (
        f"[OK] probe identity: matches the served weights, sha256 {manifest['model_hash']}"
        in status.stderr
    )
    assert f"[OK] config.json: {text['num_hidden_layers']} layers" in status.stderr

    bad = altered(manifest["model_hash"])
    (tmp_path / "wrong").mkdir()
    wrong = write_probe(tmp_path / "wrong", text["hidden_size"], 0, bad)
    refused = run(
        HARNESS,
        [
            "--model",
            str(hf_dir),
            "--text-only",
            "--probe",
            str(wrong),
            "--status",
            "--deep",
        ],
        PREFLIGHT_TIMEOUT_S,
    )
    assert refused.returncode == 1, refused.stderr
    assert (
        f"artifact model_hash {bad}, served model sha256 {manifest['model_hash']}"
        in refused.stderr
    )
