"""Shared plumbing for the workstation hardware suites (WI-3).

Both tiers are deselected by default (pyproject addopts); the human runs
them on the workstation:

    uv run pytest -m hw_smoke tests/hardware/    # session-start gate
    uv run pytest -m hw_full tests/hardware/     # per verification phase

Tests FAIL (never skip) when the hardware they exist to prove is absent —
except real-model loads, which skip cleanly when weights are not present
(gated hub access is out of scope by decision). Modules import without
torch so sandbox collection stays clean; heavy imports live inside tests.
"""

from __future__ import annotations

import gc
import json
import math
import os
import sys
import warnings
from pathlib import Path

import pytest

from common.obs.core import DEFAULT_LOG_DIR

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
AI_DATA_ROOT = "/mnt/ai-data"

# The release harness the run-record suites spawn, and the obs root it
# resolves when OBS_LOG_DIR is unset. The root is spelled from the Python
# side's own constant rather than through catalog_path, which would honour an
# OBS_LOG_DIR in the test's shell that the child never sees.
HARNESS_DIR = os.path.join(REPO_ROOT, "rust-harness")
HARNESS = os.path.join(HARNESS_DIR, "target", "release", "harness")
LOG_ROOT = os.path.join(DEFAULT_LOG_DIR, "harness")

# The repo root is on sys.path already (tests/conftest.py), which is what makes
# `from tests.hardware.conftest import ...` resolvable at collection time.
# Reuse the synthetic-shard builder the training suite uses (tests/training
# is path-imported, matching the existing `from support import ...` pattern).
sys.path.insert(0, os.path.join(REPO_ROOT, "tests", "training"))


# Stated GPU resume tolerance (resolved sweep decision: tolerance-based on
# GPU, bitwise only on CPU/tiny-model): training-loss agreement within 1%
# relative over the post-resume window. CUDA kernels are nondeterministic
# across runs (atomics, cuDNN autotuning); identical data order plus restored
# RNG state bounds the drift well under this. Shared by the from-scratch
# resume (test_gpu.py) and the TRL resume (test_train_resume.py).
RESUME_LOSS_RTOL = 0.01

SFT_RECIPE = os.path.join(
    REPO_ROOT, "training", "recipes", "reference", "sft-qlora-e2b.yaml"
)


def sft_smoke_recipe(overrides: list[str]):
    """The reference QLoRA recipe with `overrides`, and the staged base it
    resolves to, for the tests that train on the smoke dataset.

    Skips when the prepared dataset (FT.smoke-dataset) or the pinned base
    (FT.preconditions) is absent, the one skip the module rule allows, and
    fails when the prepared directory is stale: rendered with a chat template
    version other than the one training/models/families.py renders, or
    tokenized with a tokenizer other than the pinned base's. The loader
    refuses both too, but from inside train_sft; asserting here names the
    stale artifact and the entry that re-renders it instead of surfacing a
    loader exception mid-run. base.model is the staged directory the recipe's
    hub id resolved to (KB-79).
    """
    from training.data.prepare import load_tokenizer
    from training.export.errors import ExportError
    from training.export.preflight import resolve_base_identity
    from training.models.families import CHAT_TEMPLATE_VERSION
    from training.recipes import load_recipe

    recipe = load_recipe(SFT_RECIPE, overrides)
    if not os.path.isdir(recipe.sft.dataset):
        pytest.skip(
            f"prepared dataset not on disk at {recipe.sft.dataset} (FT.smoke-dataset)"
        )
    try:
        base = resolve_base_identity(
            recipe.model.hf.id, revision=recipe.model.hf.revision, local_files_only=True
        )
    except ExportError as exc:
        pytest.skip(f"pinned base not staged (FT.preconditions): {exc}")
    with open(os.path.join(recipe.sft.dataset, "manifest.json")) as f:
        dataset = json.load(f)
    assert dataset["chat_template_version"] == CHAT_TEMPLATE_VERSION, (
        f"{recipe.sft.dataset} was rendered with chat_template_version="
        f"{dataset['chat_template_version']!r}, this code renders "
        f"{CHAT_TEMPLATE_VERSION}; re-run FT.smoke-dataset"
    )
    tokenizer_hash = load_tokenizer(base.model).content_hash
    assert dataset["tokenizer_hash"] == tokenizer_hash, (
        f"{recipe.sft.dataset} was tokenized with {dataset['tokenizer_hash']}, "
        f"the pinned base {recipe.model.hf.id} hashes to {tokenizer_hash}; "
        "re-run FT.smoke-dataset against the pinned base"
    )
    return recipe, base


# The evaluation pair FT.heldout-eval and FT.eval-ft both run, shared by the
# two mirrors so one record shape is judged. The prepare service's HF_HOME:
# its datasets half is what the training service mounts and what `datasets`
# reads offline (KB-92).
DATASETS_CACHE = os.path.join(AI_DATA_ROOT, "cache", "huggingface")
STAGED_DATASETS = (
    "google/IFEval",
    "openai/gsm8k",
    "tinyBenchmarks/tinyMMLU",
    "Rowan/hellaswag",
)
# The entries' smoke limit: 20 examples per task, wide stderr, no ranking.
LMEVAL_LIMIT = 20
# Perplexity is exp(loss) rounded to four places by the evaluator.
PERPLEXITY_RTOL = 1e-3


def require_staged_datasets() -> None:
    """Skip when the staged datasets cache is absent: a missing dataset is the
    weights case, not a hardware absence, and the remedy is the entry's
    precondition block."""
    if not os.path.isdir(os.path.join(DATASETS_CACHE, "datasets")):
        pytest.skip(
            f"staged datasets cache absent at {DATASETS_CACHE}/datasets "
            "(FT.heldout-eval's precondition block: make train-download DATASET=<id> "
            f"for {', '.join(STAGED_DATASETS)})"
        )


def assert_holdout_shape(
    record: dict, model: str, dataset: str, holdout_examples: int
) -> None:
    assert set(record) == {
        "model",
        "dataset",
        "examples",
        "supervised_tokens",
        "loss",
        "perplexity",
        "precision",
        "attention_implementation",
    }, sorted(record)
    assert record["model"] == model
    assert record["dataset"] == dataset
    assert record["examples"] == holdout_examples
    assert record["supervised_tokens"] > 0
    assert record["precision"] == "bf16"
    # KB-84's third loader: config first, sdpa above the head-width cap.
    assert record["attention_implementation"] == "sdpa"
    assert math.isfinite(record["loss"]) and record["loss"] > 0
    expected_ppl = math.exp(record["loss"])
    assert abs(record["perplexity"] - expected_ppl) / expected_ppl <= PERPLEXITY_RTOL, (
        record
    )


def assert_lmeval_shape(
    record: dict, model: str, task_config: dict, limit: int
) -> None:
    assert record["model"] == model
    assert record["lm_eval_version"] == task_config["lm_eval_version"]
    assert record["limit"] == limit
    assert set(record["results"]) == set(task_config["tasks"]), sorted(
        record["results"]
    )
    for task, spec in task_config["tasks"].items():
        section = record["results"][task]
        assert section["metric"] == spec["metric"], (task, section)
        assert 0.0 <= section["value"] <= 1.0, (task, section)
        assert isinstance(section["stderr"], float), (task, section)


def e4b_checkpoint() -> str:
    """The E4B tier names the checkpoint through HARNESS_E4B_MODEL; outside
    that tier fall back to the same root the sibling weight test reads."""
    named = os.environ.get("HARNESS_E4B_MODEL")
    if named:
        return named
    root = os.environ.get("GEMMA4_CHECKPOINT_ROOT", f"{AI_DATA_ROOT}/models/hf")
    return os.path.join(root, "gemma-4-E4B")


def e2b_it_checkpoint() -> str:
    """The instruction-tuned E2B the tool-loop entries drive: it follows the
    task contract the base E4B cannot (RH.37). Same root as `e4b_checkpoint`."""
    root = os.environ.get("GEMMA4_CHECKPOINT_ROOT", f"{AI_DATA_ROOT}/models/hf")
    return os.path.join(root, "gemma-4-E2B-it")


def require_binary() -> None:
    """A missing binary fails rather than skips: it is built by `make
    harness-build` on the same host, so its absence is a misconfigured run
    and not the unavailable hardware the module rule exempts."""
    assert os.path.isfile(HARNESS), (
        f"built binary absent: {HARNESS}; run `make harness-build` in {HARNESS_DIR}"
    )


def default_root_env() -> dict[str, str]:
    """The child's environment with OBS_LOG_DIR removed, so the binary resolves
    the default root. That resolution is a fact the record suites check;
    inheriting an override from the shell would test the override instead."""
    env = {k: v for k, v in os.environ.items() if k != "OBS_LOG_DIR"}
    env["NO_COLOR"] = "1"
    return env


def event_file(run_id: str) -> str:
    return os.path.join(LOG_ROOT, f"{run_id}.jsonl")


def events(run_id: str) -> list[dict]:
    """Every record of one run's event file under the default root."""
    path = event_file(run_id)
    assert os.path.isfile(path), f"no event file for {run_id} at {path}"
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_probe(directory: Path, width: int, layer: int, model_hash: str) -> Path:
    """A probe of the model's width whose sidecar records `model_hash`,
    written by the exporter the harness's reader is tested against. The
    vectors are inert (zero weights, unit sigma): the preflight reads the
    sidecar and the tensor shapes, never a score."""
    import numpy as np

    from interp.probes.export import write_probe_from_probe
    from interp.probes.registry import Probe

    probe = Probe(
        name="hw-preflight",
        layer=layer,
        weights=np.zeros(width, dtype=np.float32),
        bias=0.0,
        mu=np.zeros(width, dtype=np.float32),
        sigma=np.ones(width, dtype=np.float32),
        threshold=0.5,
        manifest={"model_hash": model_hash},
    )
    image = directory / "probe.safetensors"
    write_probe_from_probe(probe, str(image))
    return image


def altered(digest: str) -> str:
    """The digest with its first hex digit changed: one character, so the
    comparison has to decide rather than merely hash."""
    first = "5" if digest[0] != "5" else "6"
    return first + digest[1:]


@pytest.fixture()
def torch():
    import torch

    return torch


@pytest.fixture()
def cuda(torch):
    """torch with a CUDA device, failing loudly when absent: an hw test
    running on a GPU-less host is a misconfiguration, not a skip.

    The tier runs every hw_full module in one process, so a model an
    earlier test left in a reference cycle (HF models under accelerate
    hooks, the SAELens store) is still on the card when the next test
    measures or allocates. The fixture collects and empties the cache
    first, then warns with the bytes still live: pytest lists the warning
    under the test that inherited them, which names the test before it as
    the one holding a reference. A test's own bounds measure against what
    remains, not against zero (test_step_qlora.py).
    """
    assert torch.cuda.is_available(), (
        "CUDA unavailable; hardware tests run on the workstation (RTX 5090) only"
    )
    gc.collect()
    torch.cuda.empty_cache()
    live = torch.cuda.memory_allocated()
    if live:
        warnings.warn(
            f"{live} bytes of CUDA tensors survive from an earlier test in this "
            "process after a collection; the previous GPU test holds a reference",
            stacklevel=1,
        )
    return torch


@pytest.fixture()
def train_lock(tmp_path, monkeypatch):
    """Point the D9 run lock at tmp so hardware runs cannot collide with a
    stale real lock file (one GPU consumer at a time still holds: pytest
    runs these serially)."""
    monkeypatch.setenv("TRAIN_LOCK_PATH", str(tmp_path / ".train.lock"))
