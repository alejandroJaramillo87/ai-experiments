"""Tests for scripts/docs/gen_test_inventory.py and its gate.

The generator reads a directory of test modules and renders one table row per
file: the name, the number of `def test_` functions, and the first sentence of
the module docstring. The parts worth pinning are the ones a docstring can
defeat -- a sentence ending inside `module.py:` rather than at its period, a
test declared inside a class, a module with no docstring at all -- and the
--check mode the quality gate runs, which must pass on a freshly written table
and fail once the README is edited by hand.

Fixtures are scratch trees rather than tests/training itself, so the tests do
not turn red the next time a test module is added.

The module is loaded by path; scripts/ is not an importable package.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "docs" / "gen_test_inventory.py"
GATE = REPO_ROOT / "scripts" / "docs" / "check_test_inventory.py"

README = """\
# tests/training

Hand-written prose above the markers.

<!-- test-inventory:start -->
<!-- test-inventory:end -->
"""

ONE = '''\
"""training/models/head_loss.py: the chunked head loss. Second sentence."""


def test_alpha():
    pass


def test_beta():
    pass


def helper():
    pass
'''

TWO = '''\
"""D9 run lock: acquire, release, and staleness.

A second paragraph the table does not carry.
"""


class TestLock:
    def test_acquire(self):
        pass

    def test_release(self):
        pass

    def helper(self):
        pass


def test_module_level():
    def test_nested():
        pass

    test_nested()
'''


@pytest.fixture(scope="module")
def mod():
    spec = importlib.util.spec_from_file_location("gen_test_inventory", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def tree(tmp_path):
    """A scratch repository root carrying tests/training and its README."""
    directory = tmp_path / "tests" / "training"
    directory.mkdir(parents=True)
    (directory / "test_head_loss.py").write_text(ONE, encoding="utf-8")
    (directory / "test_lock.py").write_text(TWO, encoding="utf-8")
    (directory / "README.md").write_text(README, encoding="utf-8")
    return tmp_path


def run(script: Path, *args: str) -> subprocess.CompletedProcess[str]:
    """One of the two scripts as the quality target runs it, on a scratch tree."""
    return subprocess.run(
        [sys.executable, str(script), *args],
        capture_output=True,
        text=True,
        env={"NO_COLOR": "1", "PATH": "/usr/bin:/bin"},
    )


class TestFirstSentence:
    def test_a_sentence_ends_at_a_period_followed_by_space(self, mod):
        assert mod.first_sentence("One thing. Another thing.") == "One thing."

    def test_a_period_inside_a_path_does_not_end_it(self, mod):
        text = "training/models/head_loss.py: the chunked head loss. More."
        assert mod.first_sentence(text) == (
            "training/models/head_loss.py: the chunked head loss."
        )

    def test_a_wrapped_sentence_collapses_to_one_line(self, mod):
        assert mod.first_sentence("D9 run lock:\n  acquire\n  and release") == (
            "D9 run lock: acquire and release."
        )


class TestCountTests:
    def test_module_level_and_class_nested_tests_are_counted(self, mod, tree):
        rows = {
            name: count for name, count, _ in mod.inventory(tree / "tests" / "training")
        }
        # test_lock.py declares two methods in TestLock, one module-level
        # function, and one function nested inside it that pytest never
        # collects.
        assert rows == {"test_head_loss.py": 2, "test_lock.py": 3}


class TestRenderTable:
    def test_rows_are_sorted_by_file_name_under_the_three_columns(self, mod, tree):
        table = mod.render_table(mod.inventory(tree / "tests" / "training"))
        assert table.splitlines() == [
            "| File | Tests | Covers |",
            "|------|-------|--------|",
            "| `test_head_loss.py` | 2 | training/models/head_loss.py: "
            "the chunked head loss. |",
            "| `test_lock.py` | 3 | D9 run lock: acquire, release, and staleness. |",
        ]

    def test_a_pipe_in_a_docstring_is_escaped(self, mod):
        assert mod.cell("a | b") == "a \\| b"


class TestGenerate:
    def test_the_table_is_spliced_between_the_markers(self, tree):
        result = run(SCRIPT, "--root", str(tree))
        assert result.returncode == 0, result.stderr
        text = (tree / "tests" / "training" / "README.md").read_text()
        assert "Hand-written prose above the markers." in text
        assert "| `test_lock.py` | 3 |" in text
        assert text.index("<!-- test-inventory:start -->") < text.index("| File |")
        assert text.index("| File |") < text.index("<!-- test-inventory:end -->")

    def test_regenerating_replaces_the_previous_table(self, tree):
        assert run(SCRIPT, "--root", str(tree)).returncode == 0
        (tree / "tests" / "training" / "test_head_loss.py").unlink()
        assert run(SCRIPT, "--root", str(tree)).returncode == 0
        text = (tree / "tests" / "training" / "README.md").read_text()
        assert "test_head_loss.py" not in text
        assert text.count("| File | Tests | Covers |") == 1

    def test_a_module_without_a_docstring_is_named_and_fails(self, tree):
        (tree / "tests" / "training" / "test_bare.py").write_text(
            "def test_x():\n    pass\n", encoding="utf-8"
        )
        result = run(SCRIPT, "--root", str(tree))
        assert result.returncode == 1
        assert "test_bare.py has no module docstring" in result.stderr

    def test_a_readme_without_the_markers_fails(self, tree):
        readme = tree / "tests" / "training" / "README.md"
        readme.write_text("# tests/training\n\nNo markers here.\n", encoding="utf-8")
        result = run(SCRIPT, "--root", str(tree))
        assert result.returncode == 1
        assert "test-inventory:start" in result.stderr

    def test_a_directory_with_no_test_modules_fails(self, tree):
        for path in (tree / "tests" / "training").glob("test_*.py"):
            path.unlink()
        result = run(SCRIPT, "--root", str(tree))
        assert result.returncode == 1
        assert "no test_*.py modules" in result.stderr


class TestCheck:
    def test_check_passes_on_a_freshly_generated_table(self, tree):
        assert run(SCRIPT, "--root", str(tree)).returncode == 0
        result = run(SCRIPT, "--root", str(tree), "--check")
        assert result.returncode == 0, result.stderr
        assert "lists 2 files" in result.stderr

    def test_check_fails_and_prints_the_difference_after_a_hand_edit(self, tree):
        assert run(SCRIPT, "--root", str(tree)).returncode == 0
        readme = tree / "tests" / "training" / "README.md"
        readme.write_text(
            readme.read_text().replace(
                "| `test_lock.py` | 3 |", "| `test_lock.py` | 9 |"
            ),
            encoding="utf-8",
        )
        result = run(SCRIPT, "--root", str(tree), "--check")
        assert result.returncode == 1
        assert "is stale" in result.stderr
        assert "-| `test_lock.py` | 9 |" in result.stderr
        assert "+| `test_lock.py` | 3 |" in result.stderr

    def test_check_fails_when_a_module_is_added_without_regenerating(self, tree):
        assert run(SCRIPT, "--root", str(tree)).returncode == 0
        (tree / "tests" / "training" / "test_new.py").write_text(
            '"""A new module. Ignored tail."""\n\n\ndef test_x():\n    pass\n',
            encoding="utf-8",
        )
        assert run(SCRIPT, "--root", str(tree), "--check").returncode == 1

    def test_check_does_not_write_the_readme(self, tree):
        assert run(SCRIPT, "--root", str(tree)).returncode == 0
        readme = tree / "tests" / "training" / "README.md"
        before = readme.read_text()
        (tree / "tests" / "training" / "test_head_loss.py").unlink()
        assert run(SCRIPT, "--root", str(tree), "--check").returncode == 1
        assert readme.read_text() == before


class TestGate:
    def test_the_gate_fails_on_a_stale_scratch_tree(self, tree):
        assert run(SCRIPT, "--root", str(tree)).returncode == 0
        (tree / "tests" / "training" / "test_head_loss.py").unlink()
        result = run(GATE, "--root", str(tree))
        assert result.returncode == 1
        assert "is stale" in result.stderr

    def test_the_gate_passes_on_the_repository(self):
        result = run(GATE)
        assert result.returncode == 0, result.stderr
