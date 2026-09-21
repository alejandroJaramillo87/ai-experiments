"""Environment drift suite (WI-3): every distribution installed in the
interpreter's prefix is one uv.lock names at a locked version, at the smoke
tier. `uv run --no-sync` runs what is there and never reconciles it with the
lock, so a pin that leaves the lock stays installed until `make install`;
this is the check that names the stray before a real model load hits it.
"""

from __future__ import annotations

import importlib.metadata
import sys
import tomllib
from pathlib import Path

import pytest
from packaging.utils import canonicalize_name

pytestmark = pytest.mark.ssd

LOCK = Path(__file__).parents[2] / "uv.lock"


def locked_versions() -> dict[str, set[str]]:
    """Canonical name to the set of versions uv.lock resolves it to.

    A set, not one version: a package pinned differently per platform marker
    appears in the lock once per resolution.
    """
    with LOCK.open("rb") as handle:
        lock = tomllib.load(handle)
    versions: dict[str, set[str]] = {}
    for package in lock["package"]:
        versions.setdefault(canonicalize_name(package["name"]), set()).add(
            package["version"]
        )
    return versions


def installed_in_prefix() -> dict[str, str]:
    """Canonical name to version for distributions under sys.prefix.

    The prefix filter is what scopes this to the provisioned environment
    rather than to whatever else the interpreter can see.
    """
    prefix = Path(sys.prefix).resolve()
    found: dict[str, str] = {}
    for dist in importlib.metadata.distributions():
        location = dist.locate_file("")
        if not Path(str(location)).resolve().is_relative_to(prefix):
            continue
        found.setdefault(canonicalize_name(dist.metadata["Name"]), dist.version)
    return found


@pytest.mark.hw_smoke
def test_every_installed_distribution_is_locked():
    """Installed but unlocked is drift; a locked package left uninstalled is
    a tier's choice and fails elsewhere under its own import name, so only
    the first direction is checked.

    The 2026-09-21 instance: torchao 0.14.0 survived its pin leaving the lock,
    and peft's LoRA dispatcher raised on the stale version ten minutes into
    the fine-tuning loop's export, after training had already passed.
    """
    locked = locked_versions()
    installed = installed_in_prefix()
    assert installed, f"no distributions found under {sys.prefix}"
    drift = []
    for name, version in sorted(installed.items()):
        if name not in locked:
            drift.append(f"    {name} {version}: absent from uv.lock")
        elif version not in locked[name]:
            drift.append(
                f"    {name} {version}: lock has {', '.join(sorted(locked[name]))}"
            )
    assert not drift, (
        f"[ERROR] {sys.prefix} has drifted from uv.lock\n"
        "  Found:\n" + "\n".join(drift) + "\n"
        "  Solution: reconcile the environment with the lock\n"
        "  Command: make install"
    )
