"""Marker invariant for the hardware suites (WI-3): every test in
tests/hardware/ carries exactly one tier marker (hw_smoke or hw_full) and at
least one resource marker (gpu, cpu, ram, ssd). Runs in the default
selection, so the sandbox suite guards the contract."""

from __future__ import annotations

import os

import pytest

TIERS = {"hw_smoke", "hw_full"}
RESOURCES = {"gpu", "cpu", "ram", "ssd"}


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


class _Collector:
    def __init__(self) -> None:
        self.items = []

    def pytest_collection_finish(self, session) -> None:
        self.items = list(session.items)
