"""KB-79: a recipe's hub id resolves to the directory train-download staged.

The training container cannot resolve a hub id, and the same staged base has
a different spelling in each environment, so every reader asks
training/models/staged.py for the directory instead of the recipe naming one.
These tests pin the resolver's contract on scratch roots; the hardware tier
(tests/hardware/test_recipe_gates.py) proves it against the real tree."""

from __future__ import annotations

import os

import pytest

from training.models import staged
from training.models.staged import resolve_model_source


@pytest.fixture
def roots(tmp_path, monkeypatch):
    """Two staged roots standing in for the container and host spellings."""
    container = tmp_path / "container" / "hf"
    host = tmp_path / "host" / "hf"
    container.mkdir(parents=True)
    host.mkdir(parents=True)
    monkeypatch.setattr(staged, "STAGED_ROOTS", (str(container), str(host)))
    return container, host


def stage(root, name: str) -> str:
    directory = root / name
    directory.mkdir()
    (directory / "config.json").write_text("{}")
    return str(directory)


def test_a_directory_is_returned_as_given(roots, tmp_path):
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    assert resolve_model_source(str(elsewhere)) == str(elsewhere)


def test_a_staged_hub_id_resolves_to_its_basename_directory(roots):
    _, host = roots
    directory = stage(host, "gemma-4-E2B")
    assert resolve_model_source("google/gemma-4-E2B") == directory


def test_the_container_spelling_wins_when_both_are_mounted(roots):
    container, host = roots
    stage(host, "gemma-4-E2B")
    first = stage(container, "gemma-4-E2B")
    assert resolve_model_source("google/gemma-4-E2B") == first


def test_an_unstaged_hub_id_is_returned_unchanged(roots):
    assert resolve_model_source("google/gemma-4-E4B") == "google/gemma-4-E4B"


def test_a_staged_directory_without_a_config_does_not_count(roots):
    _, host = roots
    (host / "gemma-4-E2B").mkdir()
    assert resolve_model_source("google/gemma-4-E2B") == "google/gemma-4-E2B"


def test_a_bare_name_resolves_by_basename_too(roots):
    # An id without an owner still has a basename; train-download names the
    # directory after it either way.
    _, host = roots
    directory = stage(host, "tiny")
    assert resolve_model_source("tiny") == directory


def test_the_real_roots_are_the_two_mounts():
    from common.serving import CONTAINER_MODELS, HOST_MODELS

    assert staged.STAGED_ROOTS == (
        os.path.join(CONTAINER_MODELS, "hf"),
        os.path.join(HOST_MODELS, "hf"),
    )
