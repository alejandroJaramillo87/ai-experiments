"""The download command's two forms: a pinned model, and a dataset staged into
the hub cache for lm-eval's offline reads (KB-92). The hub and datasets clients
are stubbed through sys.modules; the argv path is the CLI's own."""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import pytest

from training import cli


def run_cli(argv: list[str]) -> int:
    args = cli.build_parser().parse_args(argv)
    return args.func(args)


@pytest.fixture()
def online(monkeypatch):
    for var in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("HF_HOME", "/app/cache")


@pytest.fixture()
def hub(monkeypatch):
    calls: dict = {}

    def load_dataset(path, name=None):
        calls["load"] = (path, name)
        return {"train": [1, 2, 3], "test": [4]}

    class Api:
        def dataset_info(self, path):
            calls["info"] = path
            return SimpleNamespace(sha="abc123")

    monkeypatch.setitem(
        sys.modules, "datasets", types.SimpleNamespace(load_dataset=load_dataset)
    )
    monkeypatch.setitem(
        sys.modules, "huggingface_hub", types.SimpleNamespace(HfApi=Api)
    )
    return calls


def test_dataset_form_stages_every_split_and_prints_the_resolved_commit(
    online, hub, capsys
):
    assert run_cli(["download", "--dataset", "openai/gsm8k", "--name", "main"]) == 0
    assert hub["load"] == ("openai/gsm8k", "main")
    assert hub["info"] == "openai/gsm8k"
    err = capsys.readouterr().err
    assert "[OK] Staged openai/gsm8k (main) at abc123: train=3, test=1" in err
    assert "[INFO] Cache: /app/cache" in err


def test_dataset_form_without_a_config_passes_none(online, hub):
    assert run_cli(["download", "--dataset", "Rowan/hellaswag"]) == 0
    assert hub["load"] == ("Rowan/hellaswag", None)


def test_dataset_form_refuses_the_model_flags(online, hub, capsys):
    assert run_cli(["download", "--dataset", "x/y", "--revision", "deadbeef"]) == 2
    assert "[ERROR] --revision applies to --model only" in capsys.readouterr().err
    assert "load" not in hub


def test_model_form_requires_a_revision(online, capsys):
    assert run_cli(["download", "--model", "google/gemma-4-E2B"]) == 2
    assert "[ERROR] --revision is required with --model" in capsys.readouterr().err


def test_the_two_forms_exclude_each_other():
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(
            ["download", "--model", "a/b", "--dataset", "c/d"]
        )


def test_offline_container_is_refused_before_any_fetch(monkeypatch, hub, capsys):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    assert run_cli(["download", "--dataset", "openai/gsm8k"]) == 1
    assert "[ERROR] HF_HUB_OFFLINE is set" in capsys.readouterr().err
    assert "load" not in hub
