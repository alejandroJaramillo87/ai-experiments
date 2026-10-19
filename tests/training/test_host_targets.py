"""The host-run python targets take the interpreter make/quality.mk resolves.

python -m training and python -m interp import their command modules at
import, which reaches the training and interp groups (omegaconf, xxhash,
numpy: the walk KB-91 recorded), and the workstation provisions those only
in .venv. A recipe naming python3 runs the system interpreter and dies at
ModuleNotFoundError before its first printed line (KB-95, found by
FT.eval-ft's readiness review). RUN_PY is `uv run --no-sync` on the
workstation and empty in the sandbox, so `$(RUN_PY) python -m ...` is right
in both, and a recipe that runs python inside `docker compose` is the
image's interpreter, not the host's.
"""

import os
import re

import pytest

REPO_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
MAKEFILES = ("make/training.mk", "make/interp.mk")
HOST_MODULE = re.compile(r"\bpython3?\s+-m\s+(training|interp)\b")
PREFIX = "$(RUN_PY) python -m "


def host_recipe_lines(makefile: str) -> list[tuple[int, str]]:
    """Recipe lines (tab-indented) that invoke a tier's CLI on the host."""
    lines = []
    with open(os.path.join(REPO_ROOT, makefile)) as f:
        for lineno, line in enumerate(f, 1):
            if not line.startswith("\t") or "docker compose" in line:
                continue
            if HOST_MODULE.search(line):
                lines.append((lineno, line.strip()))
    return lines


@pytest.mark.parametrize("makefile", MAKEFILES)
def test_host_python_targets_use_run_py(makefile):
    offenders = [
        f"{makefile}:{lineno}: {text}"
        for lineno, text in host_recipe_lines(makefile)
        if not text.startswith(PREFIX)
    ]
    assert not offenders, "host-run recipe bypasses RUN_PY:\n" + "\n".join(offenders)


def test_guard_sees_the_host_targets():
    # Not vacuous: eval-ft and generate-teacher; steer-scorecard, probe-export,
    # probe-eval and the four organism targets.
    assert len(host_recipe_lines("make/training.mk")) == 2
    assert len(host_recipe_lines("make/interp.mk")) == 7
