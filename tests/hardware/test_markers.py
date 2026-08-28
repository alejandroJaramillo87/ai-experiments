"""Marker invariants for the hardware suites (WI-3): every test in
tests/hardware/ carries exactly one tier marker (hw_smoke or hw_full) and at
least one resource marker (gpu, cpu, ram, ssd), and every hardware-gated
marker declared in pyproject.toml is deselected by addopts. Runs in the
default selection, so the sandbox suite guards the contract."""

from __future__ import annotations

import os
import re
import tomllib
from pathlib import Path

import pytest

TIERS = {"hw_smoke", "hw_full"}
RESOURCES = {"gpu", "cpu", "ram", "ssd"}
PYPROJECT = Path(__file__).parents[2] / "pyproject.toml"


def test_every_hardware_test_carries_tier_and_resource_markers():
    here = os.path.dirname(os.path.abspath(__file__))
    plugin = _Collector()
    rc = pytest.main(
        ["--collect-only", "-q", "-m", "hw_smoke or hw_full", here],
        plugins=[plugin],
    )
    assert rc == 0
    assert plugin.items, "hardware suites collected nothing"
    problems = []
    for item in plugin.items:
        marks = {m.name for m in item.iter_markers()}
        tiers = marks & TIERS
        resources = marks & RESOURCES
        if len(tiers) != 1:
            problems.append(f"{item.nodeid}: tier markers {sorted(tiers)}")
        if not resources:
            problems.append(f"{item.nodeid}: no resource marker")
    assert not problems, "\n".join(problems)


def _pytest_ini() -> dict:
    with PYPROJECT.open("rb") as handle:
        return tomllib.load(handle)["tool"]["pytest"]["ini_options"]


def test_every_hardware_marker_is_deselected_by_default():
    """The trap a third marker walks into: declared but never deselected.

    addopts carries one negation per hardware-gated marker. A marker added to
    the declaration list alone is selected by default, so its tests run in
    every sandbox -- where the hardware they need is absent -- and fail there
    instead of being skipped. Failing here is cheaper than failing in CI.
    """
    ini = _pytest_ini()
    declared = {m.split(":", 1)[0].strip() for m in ini["markers"]}
    gated = {name for name in declared if name.startswith("hw_")}
    assert gated, "no hardware-gated markers declared; the invariant is vacuous"
    addopts = ini["addopts"]
    missing = [
        name
        for name in sorted(gated)
        if not re.search(rf"\bnot {re.escape(name)}\b", addopts)
    ]
    assert not missing, (
        f"markers declared but not deselected by addopts: {missing}\n"
        f'  Solution: add "not <marker>" to addopts in {PYPROJECT}'
    )


def test_default_selection_collects_no_hardware_marked_test():
    """The invariant above, observed rather than parsed.

    Collecting tests/hardware/ under the default addopts must yield nothing:
    the deselection expression is what makes a bare `pytest` safe in a sandbox
    with no GPU, no /mnt/ai-data, and no checkpoints.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    plugin = _Collector()
    rc = pytest.main(["--collect-only", "-q", here], plugins=[plugin])
    assert rc in (0, 5), f"collection failed with exit {rc}"
    marked = [
        item.nodeid
        for item in plugin.items
        if {m.name for m in item.iter_markers()} & (TIERS | {"hw_e4b"})
    ]
    assert not marked, f"hardware-marked tests collected by default: {marked}"


class _Collector:
    def __init__(self) -> None:
        self.items = []

    def pytest_collection_finish(self, session) -> None:
        self.items = list(session.items)
