"""Tests for scripts/docs/check_surface_census.py.

The census is the one doc gate that starts from the code: it enumerates the
rust-harness command line from its `Flag` literals and subtracts what the
runbooks mention. Two properties carry the whole check, and both have a way of
being quietly wrong.

The first is that a binary must be matched as a whole word. `layer-diff` shipped
unmentioned while every runbook talked about layers, and a substring match would
have called it covered on the strength of a sentence about something else.

The second is the empty-set guard. This gate's enumeration reads string literals
out of Rust source, which is exactly the kind of coupling a refactor breaks
silently -- and a census that enumerates nothing reports every element as
covered, which is the most confident possible way to be useless.

Fixtures are scratch trees rather than the real repository, so the tests do not
turn red the next time a flag is added.

The module is loaded by path; scripts/ is not an importable package.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "docs" / "check_surface_census.py"

BIN_SOURCE = """\
fn spec() -> Spec {
    Spec {
        flags: &[
            Flag { name: "--help", kind: Kind::Bare, help: "print this help" },
            Flag { name: "--model", kind: Kind::Value("<dir>"), help: "checkpoint" },
        ],
    }
}
"""


@pytest.fixture(scope="module")
def mod():
    spec = importlib.util.spec_from_file_location("check_surface_census", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_gate(root: Path) -> subprocess.CompletedProcess[str]:
    """The gate as the quality target runs it, scoped to a scratch tree."""
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--root", str(root)],
        capture_output=True,
        text=True,
        env={"NO_COLOR": "1", "PATH": "/usr/bin:/bin"},
    )


def scratch_tree(root: Path, runbook: str, bins: dict[str, str]) -> None:
    """A tree shaped like the workspace: one runbook and some bin sources."""
    book = root / "docs" / "verification" / "tier"
    book.mkdir(parents=True)
    (book / "runbook.md").write_text(runbook, encoding="utf-8")
    bins_dir = root / "rust-harness" / "runtime" / "src" / "bin"
    bins_dir.mkdir(parents=True)
    for name, text in bins.items():
        (bins_dir / f"{name}.rs").write_text(text, encoding="utf-8")
    harness_cli = root / "rust-harness" / "harness" / "src" / "cli.rs"
    harness_cli.parent.mkdir(parents=True)
    harness_cli.write_text(BIN_SOURCE, encoding="utf-8")


class TestSurface:
    """What the enumeration collects, and what it deliberately drops."""

    def test_help_is_not_part_of_the_census(self, mod, tmp_path):
        """Every binary has it by construction; an entry per binary is noise."""
        scratch_tree(tmp_path, "# Runbook\n", {"layer-diff": BIN_SOURCE})
        _, flags = mod.surface(tmp_path)
        assert "--model" in flags
        assert "--help" not in flags

    def test_harness_is_a_binary_though_it_has_no_bin_file(self, mod, tmp_path):
        scratch_tree(tmp_path, "# Runbook\n", {"layer-diff": BIN_SOURCE})
        binaries, _ = mod.surface(tmp_path)
        assert set(binaries) == {"layer-diff", "harness"}


class TestGate:
    """End to end, over a scratch tree."""

    def test_an_unmentioned_binary_fails(self, tmp_path):
        scratch_tree(
            tmp_path,
            "# Runbook\n\nWe run `harness --model x` and inspect the layers.\n",
            {"layer-diff": BIN_SOURCE},
        )
        result = run_gate(tmp_path)
        assert result.returncode == 1
        assert "binary layer-diff" in result.stderr

    def test_a_binary_is_matched_as_a_whole_word(self, tmp_path):
        """Prose about layers must not satisfy `layer-diff`.

        This is the substring trap, and it is not hypothetical: the runbooks
        discuss layers constantly, and `layer-diff` shipped unmentioned anyway.
        """
        scratch_tree(
            tmp_path,
            "# Runbook\n\nThe layer-diffing approach compares layers.\n"
            "`harness --model x`\n",
            {"layer-diff": BIN_SOURCE},
        )
        result = run_gate(tmp_path)
        assert result.returncode == 1
        assert "binary layer-diff" in result.stderr

    def test_a_mentioned_surface_passes(self, tmp_path):
        scratch_tree(
            tmp_path,
            "# Runbook\n\n`layer-diff --model x` and `harness --model y`.\n",
            {"layer-diff": BIN_SOURCE},
        )
        result = run_gate(tmp_path)
        assert result.returncode == 0, result.stderr

    def test_an_empty_enumeration_fails_rather_than_passing(self, tmp_path):
        """The guard that matters most: no surface found is not full coverage."""
        (tmp_path / "docs" / "verification").mkdir(parents=True)
        (tmp_path / "docs" / "verification" / "r.md").write_text("# R\n")
        result = run_gate(tmp_path)
        assert result.returncode == 1
        assert "enumerated no rust-harness command line" in result.stderr
