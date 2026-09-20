"""Tests for scripts/docs/gen_inventory.py and its gate.

The generator renders one table per marker pair in a README, from the tree the
README describes: test modules by docstring and test count, package modules by
docstring, child directories by their README's first sentence, sibling pages by
theirs. The parts worth pinning are the ones a source can defeat -- a sentence
ending inside `module.py:` rather than at its period, a test declared inside a
class, a bold class line standing where the description paragraph goes, a child
directory with no README -- and the --check mode the quality gate runs, which
must pass on a freshly written table and fail once the README is edited by hand.
The gate adds a structural rule, that every package or test directory under a
hub is documented, and that is pinned on a scratch hub too.

Fixtures are scratch trees rather than the repository, so the tests do not turn
red the next time a module is added.

The module is loaded by path; scripts/ is not an importable package.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "docs" / "gen_inventory.py"
GATE = REPO_ROOT / "scripts" / "docs" / "check_inventory.py"

TESTS_README = """\
# tests/training

Hand-written prose above the markers.

<!-- inventory:tests:start -->
<!-- inventory:tests:end -->
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

PKG_README = """\
# pkg

The package under test: two modules and a subpackage.

<!-- inventory:dirs:start -->
<!-- inventory:dirs:end -->

<!-- inventory:modules:start -->
<!-- inventory:modules:end -->
"""

SUB_README = """\
# sub

**Frozen record -- 2026-01-01. Never updated.**

The subpackage's own description. A second sentence.
"""

DOCS_README = """\
# docs

The pages here.

<!-- inventory:pages:start -->
<!-- inventory:pages:end -->
"""


@pytest.fixture(scope="module")
def mod():
    spec = importlib.util.spec_from_file_location("gen_inventory", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def tree(tmp_path):
    """A scratch root: a test directory, a package with a subpackage, and pages."""
    tests = tmp_path / "tests" / "training"
    tests.mkdir(parents=True)
    (tests / "test_head_loss.py").write_text(ONE, encoding="utf-8")
    (tests / "test_lock.py").write_text(TWO, encoding="utf-8")
    (tests / "README.md").write_text(TESTS_README, encoding="utf-8")

    pkg = tmp_path / "pkg"
    (pkg / "sub").mkdir(parents=True)
    (pkg / "__init__.py").write_text('"""The package."""\n', encoding="utf-8")
    (pkg / "__main__.py").write_text('"""Entry point."""\n', encoding="utf-8")
    (pkg / "alpha.py").write_text(
        '"""Alpha does one thing. More."""\n', encoding="utf-8"
    )
    (pkg / "beta.py").write_text('"""Beta reads a | pipe."""\n', encoding="utf-8")
    (pkg / "README.md").write_text(PKG_README, encoding="utf-8")
    (pkg / "sub" / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "sub" / "README.md").write_text(SUB_README, encoding="utf-8")
    (pkg / "__pycache__").mkdir()
    (pkg / "__pycache__" / "alpha.pyc").write_bytes(b"")

    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "README.md").write_text(DOCS_README, encoding="utf-8")
    (docs / "one.md").write_text("# One\n\nThe first page. Tail.\n", encoding="utf-8")
    (docs / "two.md").write_text(
        "# Two\n\n**Operational runbook. Executed.**\n\nThe second\npage.\n",
        encoding="utf-8",
    )
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


class TestDescription:
    def test_the_paragraph_under_the_title_is_the_source(self, mod, tree):
        assert mod.description(tree / "docs" / "one.md", tree) == "The first page."

    def test_a_bold_class_line_is_skipped(self, mod, tree):
        assert mod.description(tree / "docs" / "two.md", tree) == "The second page."

    def test_a_page_with_no_paragraph_is_named(self, mod, tree):
        page = tree / "docs" / "bare.md"
        page.write_text("# Bare\n\n## Section\n", encoding="utf-8")
        with pytest.raises(ValueError, match="docs/bare.md has no description"):
            mod.description(page, tree)

    def test_a_page_with_no_title_is_named(self, mod, tree):
        page = tree / "docs" / "notitle.md"
        page.write_text("Just prose.\n", encoding="utf-8")
        with pytest.raises(ValueError, match="docs/notitle.md has no H1"):
            mod.description(page, tree)


class TestRows:
    def test_module_level_and_class_nested_tests_are_counted(self, mod, tree):
        rows = mod.rows_tests(tree / "tests" / "training", tree)
        # test_lock.py declares two methods in TestLock, one module-level
        # function, and one function nested inside it that pytest never
        # collects.
        assert [(n, c) for n, c, _ in rows] == [
            ("test_head_loss.py", "2"),
            ("test_lock.py", "3"),
        ]

    def test_modules_skip_init_and_main(self, mod, tree):
        rows = mod.rows_modules(tree / "pkg", tree)
        assert rows == [
            ("alpha.py", "Alpha does one thing."),
            ("beta.py", "Beta reads a | pipe."),
        ]

    def test_dirs_read_each_child_readme_and_prune_caches(self, mod, tree):
        assert mod.rows_dirs(tree / "pkg", tree) == [
            ("sub", "The subpackage's own description.")
        ]

    def test_a_child_without_a_readme_is_named(self, mod, tree):
        (tree / "pkg" / "orphan").mkdir()
        (tree / "pkg" / "orphan" / "x.py").write_text("", encoding="utf-8")
        with pytest.raises(ValueError, match="pkg/orphan/ has no README.md"):
            mod.rows_dirs(tree / "pkg", tree)

    def test_pages_exclude_the_readme_itself(self, mod, tree):
        assert [n for n, _ in mod.rows_pages(tree / "docs", tree)] == [
            "one.md",
            "two.md",
        ]


class TestRenderTable:
    def test_tests_rows_under_the_three_columns(self, mod, tree):
        table = mod.render_table(
            "tests", mod.rows_tests(tree / "tests" / "training", tree)
        )
        assert table.splitlines() == [
            "| File | Tests | Covers |",
            "|------|-------|--------|",
            "| `test_head_loss.py` | 2 | training/models/head_loss.py: "
            "the chunked head loss. |",
            "| `test_lock.py` | 3 | D9 run lock: acquire, release, and staleness. |",
        ]

    def test_dirs_and_pages_rows_link_their_source(self, mod):
        assert mod.render_table("dirs", [("sub", "S.")]).splitlines()[2] == (
            "| [sub/](sub/README.md) | S. |"
        )
        assert mod.render_table("pages", [("one.md", "O.")]).splitlines()[2] == (
            "| [one.md](one.md) | O. |"
        )

    def test_a_pipe_in_a_docstring_is_escaped(self, mod):
        assert mod.cell("a | b") == "a \\| b"
        assert "Beta reads a \\| pipe." in mod.render_table(
            "modules", [("beta.py", "Beta reads a | pipe.")]
        )


class TestFindBlocks:
    def test_kinds_come_back_in_document_order(self, mod):
        assert mod.find_blocks(PKG_README, Path("pkg/README.md")) == ["dirs", "modules"]

    @pytest.mark.parametrize(
        "text, message",
        [
            ("<!-- inventory:tests:start -->\n", "never closes"),
            ("<!-- inventory:tests:end -->\n", "without its start"),
            ("<!-- inventory:nope:start -->\n<!-- inventory:nope:end -->\n", "unknown"),
            (
                "<!-- inventory:tests:start -->\n<!-- inventory:tests:end -->\n"
                "<!-- inventory:tests:start -->\n<!-- inventory:tests:end -->\n",
                "appears twice",
            ),
        ],
    )
    def test_a_malformed_pair_is_named(self, mod, text, message):
        with pytest.raises(ValueError, match=message):
            mod.find_blocks(text, Path("x/README.md"))


class TestGenerate:
    def test_every_table_is_spliced_between_its_markers(self, tree):
        result = run(SCRIPT, "--root", str(tree))
        assert result.returncode == 0, result.stderr
        text = (tree / "tests" / "training" / "README.md").read_text()
        assert "Hand-written prose above the markers." in text
        assert "| `test_lock.py` | 3 |" in text
        assert text.index("<!-- inventory:tests:start -->") < text.index("| File |")
        assert text.index("| File |") < text.index("<!-- inventory:tests:end -->")
        pkg = (tree / "pkg" / "README.md").read_text()
        assert "| [sub/](sub/README.md) | The subpackage's own description. |" in pkg
        assert "| `alpha.py` | Alpha does one thing. |" in pkg
        assert pkg.index("| Directory |") < pkg.index("| Module |")
        docs = (tree / "docs" / "README.md").read_text()
        assert "| [two.md](two.md) | The second page. |" in docs

    def test_named_readmes_are_the_only_ones_touched(self, tree):
        result = run(SCRIPT, "--root", str(tree), "docs/README.md")
        assert result.returncode == 0, result.stderr
        assert "| `alpha.py` |" not in (tree / "pkg" / "README.md").read_text()
        assert "| [one.md](one.md) |" in (tree / "docs" / "README.md").read_text()

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
        assert "tests/training/test_bare.py has no module docstring" in result.stderr

    def test_a_readme_named_without_markers_fails(self, tree):
        readme = tree / "tests" / "training" / "README.md"
        readme.write_text("# tests/training\n\nNo markers here.\n", encoding="utf-8")
        result = run(SCRIPT, "--root", str(tree), "tests/training/README.md")
        assert result.returncode == 1
        assert "carries no inventory markers" in result.stderr

    def test_a_directory_with_nothing_to_list_fails(self, tree):
        for path in (tree / "tests" / "training").glob("test_*.py"):
            path.unlink()
        result = run(SCRIPT, "--root", str(tree))
        assert result.returncode == 1
        assert "nothing to list" in result.stderr


class TestCheck:
    def test_check_passes_on_freshly_generated_tables(self, tree):
        assert run(SCRIPT, "--root", str(tree)).returncode == 0
        result = run(SCRIPT, "--root", str(tree), "--check")
        assert result.returncode == 0, result.stderr
        assert "tests/training/README.md tests: 2 rows" in result.stderr

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
        assert "gen_inventory.py tests/training/README.md" in result.stderr

    def test_check_fails_when_a_module_is_added_without_regenerating(self, tree):
        assert run(SCRIPT, "--root", str(tree)).returncode == 0
        (tree / "pkg" / "gamma.py").write_text('"""Gamma."""\n', encoding="utf-8")
        assert run(SCRIPT, "--root", str(tree), "--check").returncode == 1

    def test_check_does_not_write_the_readme(self, tree):
        assert run(SCRIPT, "--root", str(tree)).returncode == 0
        readme = tree / "tests" / "training" / "README.md"
        before = readme.read_text()
        (tree / "tests" / "training" / "test_head_loss.py").unlink()
        assert run(SCRIPT, "--root", str(tree), "--check").returncode == 1
        assert readme.read_text() == before


class TestGate:
    @pytest.fixture()
    def hubs(self, tree):
        """The scratch tree with both hub roots the gate requires."""
        tests = tree / "tests" / "README.md"
        tests.write_text(
            "# tests\n\nThe suite.\n\n<!-- inventory:dirs:start -->\n"
            "<!-- inventory:dirs:end -->\n",
            encoding="utf-8",
        )
        (tree / "pkg").rename(tree / "training")
        # The subpackage is a package, so the rule wants a table in its README.
        (tree / "training" / "sub" / "README.md").write_text(
            "# sub\n\nThe subpackage.\n\n<!-- inventory:modules:start -->\n"
            "<!-- inventory:modules:end -->\n",
            encoding="utf-8",
        )
        (tree / "training" / "sub" / "leaf.py").write_text(
            '"""Leaf."""\n', encoding="utf-8"
        )
        assert run(SCRIPT, "--root", str(tree)).returncode == 0
        return tree

    def test_the_gate_passes_on_a_documented_scratch_tree(self, hubs):
        result = run(GATE, "--root", str(hubs))
        assert result.returncode == 0, result.stderr

    def test_the_gate_fails_on_a_stale_table(self, hubs):
        (hubs / "tests" / "training" / "test_head_loss.py").unlink()
        result = run(GATE, "--root", str(hubs))
        assert result.returncode == 1
        assert "is stale" in result.stderr

    def test_a_new_package_without_a_readme_fails_the_hub_table(self, hubs):
        (hubs / "training" / "fresh").mkdir()
        (hubs / "training" / "fresh" / "__init__.py").write_text("", encoding="utf-8")
        result = run(GATE, "--root", str(hubs))
        assert result.returncode == 1
        assert "training/fresh/ has no README.md" in result.stderr

    def test_a_package_readme_without_a_table_fails_the_rule(self, hubs):
        (hubs / "training" / "sub" / "README.md").write_text(
            "# sub\n\nProse only.\n", encoding="utf-8"
        )
        assert run(SCRIPT, "--root", str(hubs)).returncode == 0
        result = run(GATE, "--root", str(hubs))
        assert result.returncode == 1
        assert "training/sub/README.md has no tests or modules table" in result.stderr

    def test_a_hub_without_a_dirs_table_fails(self, hubs):
        (hubs / "tests" / "README.md").write_text(
            "# tests\n\nProse.\n", encoding="utf-8"
        )
        result = run(GATE, "--root", str(hubs))
        assert result.returncode == 1
        assert "tests/README.md carries no inventory:dirs table" in result.stderr

    def test_the_gate_passes_on_the_repository(self):
        result = run(GATE)
        assert result.returncode == 0, result.stderr
