"""Every long-running verification script is an entry's, and every entry
naming one has it.

scripts/verification/ holds a runbook entry's command block as a script that
`make verify ENTRY=<name>` runs and logs. A script no entry cites is a
command block nobody records the output of; an entry naming a script that is
not in the tree hands the operator a target that fails. Both are held here,
lexically, the way check_make_targets.py holds `make` targets: the runbook
cites `make verify ENTRY=<name>`, and the name is the script's basename.
"""

from __future__ import annotations

import os
import re
import subprocess

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SCRIPTS_DIR = os.path.join(REPO_ROOT, "scripts", "verification")
RUNBOOKS_DIR = os.path.join(REPO_ROOT, "docs", "verification")
CITATION = re.compile(r"make verify ENTRY=([A-Za-z0-9_.-]+)")


def scripts() -> set[str]:
    return {
        name[: -len(".sh")] for name in os.listdir(SCRIPTS_DIR) if name.endswith(".sh")
    }


def citations() -> dict[str, set[str]]:
    """Entry name -> the runbook pages citing it, over every live page."""
    found: dict[str, set[str]] = {}
    for dirpath, _, names in os.walk(RUNBOOKS_DIR):
        if os.path.basename(dirpath) == "archive":
            continue
        for name in names:
            if not name.endswith(".md"):
                continue
            path = os.path.join(dirpath, name)
            with open(path, encoding="utf-8") as f:
                for entry in CITATION.findall(f.read()):
                    found.setdefault(entry, set()).add(os.path.relpath(path, REPO_ROOT))
    return found


def test_every_script_is_cited_by_a_runbook():
    uncited = scripts() - set(citations())
    assert not uncited, f"scripts no runbook entry runs: {sorted(uncited)}"


def test_every_citation_names_a_script():
    missing = {
        entry: sorted(pages)
        for entry, pages in citations().items()
        if entry not in scripts()
    }
    assert not missing, f"runbooks naming a script that is not in the tree: {missing}"


def test_every_script_is_executable_bash():
    for name in sorted(os.listdir(SCRIPTS_DIR)):
        if not name.endswith(".sh"):
            continue
        path = os.path.join(SCRIPTS_DIR, name)
        assert os.access(path, os.X_OK), f"{name} is not executable"
        with open(path, encoding="utf-8") as f:
            assert f.readline().startswith("#!/usr/bin/env bash"), name


def test_verify_refuses_in_the_sandbox():
    """The target names the workstation as the remedy when AI_SANDBOX is set,
    before touching a log directory that does not exist here."""
    proc = subprocess.run(
        ["make", "verify", "ENTRY=ft-organisms"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        env={**os.environ, "AI_SANDBOX": "1"},
    )
    assert proc.returncode != 0
    assert "[ERROR] verify: this target requires the workstation" in proc.stdout
