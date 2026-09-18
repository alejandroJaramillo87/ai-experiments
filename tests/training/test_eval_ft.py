"""eval-ft's container phase: the compared sides reach the training container
in its own spelling of the models tree. The container mounts the host models
root at /app/models only (docker-compose.training.yaml), so a host path passed
through verbatim fails at from_pretrained inside it, hours after the run
started. Serving and the container are stubbed; the live phases are the FT.6
host entry."""

from __future__ import annotations

import json
import os
from types import SimpleNamespace

import pytest

from common import serving
from common.output import Printer
from training.commands import eval_ft
from training.eval import runner

HOST_BASE = serving.HOST_MODELS + "hf/gemma-4-E2B"
HOST_TUNED = serving.HOST_MODELS + "trained/ft3/hf"
CONTAINER_BASE = serving.CONTAINER_MODELS + "hf/gemma-4-E2B"
CONTAINER_TUNED = serving.CONTAINER_MODELS + "trained/ft3/hf"
ADAPTER = "/mnt/ai-data/training/runs/x/checkpoints/checkpoint-10"


@pytest.mark.parametrize(
    "given,expected",
    [
        (HOST_BASE, CONTAINER_BASE),
        (CONTAINER_TUNED, CONTAINER_TUNED),
        (ADAPTER, ADAPTER),
    ],
)
def test_container_model_translates_only_the_models_tree(given, expected):
    assert eval_ft.container_model(given) == expected


def test_container_phase_passes_container_paths(tmp_path, monkeypatch):
    calls = []

    def fake_phase(subcommand, phase_args, service="training"):
        calls.append((subcommand, list(phase_args)))
        out = phase_args[phase_args.index("--out") + 1]
        with open(out, "w") as f:
            json.dump({"loss": 1.0, "perplexity": 2.7, "tasks": {}}, f)

    monkeypatch.setattr(runner, "run_container_phase", fake_phase)
    monkeypatch.setattr(runner, "holdout_section", lambda b, t: {"b": b, "t": t})
    args = SimpleNamespace(
        base=HOST_BASE,
        tuned=HOST_TUNED,
        dataset="/mnt/ai-data/training/datasets/sft/x",
        load_4bit=False,
        max_examples=None,
        tasks=None,
        lmeval_limit=None,
    )
    eval_ft.phase_container(args, ["holdout"], str(tmp_path), Printer())
    models = [a[a.index("--model") + 1] for _, a in calls]
    assert models == [CONTAINER_BASE, CONTAINER_TUNED]
    assert os.path.isfile(tmp_path / "holdout_base.json")
