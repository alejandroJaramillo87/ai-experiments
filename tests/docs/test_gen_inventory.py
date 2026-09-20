"""Tests for scripts/docs/gen_inventory.py and its gate.

The generator renders one table per marker pair in a README, from the tree the
README describes: test modules by docstring and test count, package modules by
docstring, child directories by their README's first sentence, sibling pages by
theirs. The parts worth pinning are the ones a source can defeat -- a sentence
ending inside `module.py:` rather than at its period, a test declared inside a
class, a bold class line standing where the description paragraph goes, a child
directory with no README -- and the --check mode the quality gate runs, which
must pass on a freshly written table and fail once the README is edited by hand.
The gate adds a structural rule, that every directory under a hub is documented
by a table, and that is pinned on a scratch hub too. The scripts kind reads a
header comment rather than a docstring, so the shapes those headers take in the
tree are pinned one each; the contracts kind derives a direction from two path
lists; the roster kinds read frontmatter from a fixed place under the root; the
two Rust kinds read a crate's `/* name — sentence */` banners and count its test
attributes, in every form the workspace uses.

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


SHELL_HEADERS = {
    "own_name_dash.sh": "#!/bin/bash\n# own_name_dash.sh - Guard the call. Then more.\n",
    "own_name_double.sh": "#!/bin/bash\n#\n# own_name_double.sh -- purge the driver\n# and reinstall.\n",
    "bare.sh": "#!/bin/bash\n# Compare the pins. Second.\n",
    "stale_name.sh": "#!/bin/bash\n\n# old_updater.sh\n# This script updates cuDNN.\n",
    "set_first.sh": "#!/bin/bash\nset -euo pipefail\n\n# Preflight (I1)\n#\n# Detail.\n",
    "none.sh": "#!/bin/bash\necho hi\n",
}

MJS = """/*
 * check.mjs — real-browser verification for the dashboard.
 *
 * Detail.
 */
import fs from "node:fs";
"""

RUST_FILES = {
    "src/lib.rs": "/*\n * runtime — inference engine library crate.\n *\n * Detail.\n */\n\npub mod model;\n",
    "src/model/cpu.rs": (
        "/*\n * cpu — the f32 reference backend. Second sentence.\n */\n\n"
        "#[cfg(test)]\nmod tests {\n    #[test]\n    fn a() {}\n\n    #[test]\n    fn b() {}\n}\n"
    ),
    "src/bin/tool.rs": "/*\n * tool.rs — dumps a checkpoint's config.\n */\n\nfn main() {}\n",
    "tests/e2e.rs": (
        "/*\n * e2e — the loop end to end.\n */\n\n"
        "#[tokio::test(start_paused = true)]\nasync fn runs() {}\n"
    ),
    "tests/surface.rs": "/*\n * surface.rs — `--status` through the binary.\n */\n\n#[test]\nfn ok() {}\n",
}

CONTRACT = """{
  "contract": "%s",
  "description": "%s",
  "writers": %s,
  "readers": %s
}
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

    def test_a_line_opening_with_a_code_span_continues_the_paragraph(self, mod, tree):
        page = tree / "docs" / "span.md"
        page.write_text(
            "# Span\n\nThe directory holds\n`config.json` and more. Tail.\n",
            encoding="utf-8",
        )
        assert mod.description(page, tree) == (
            "The directory holds `config.json` and more."
        )

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

    def test_hidden_directories_are_not_listed(self, mod, tree):
        (tree / "pkg" / ".cache").mkdir()
        (tree / "pkg" / ".cache" / "state").write_text("", encoding="utf-8")
        assert [row[0] for row in mod.rows_dirs(tree / "pkg", tree)] == ["sub"]

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


class TestScripts:
    @pytest.fixture()
    def scripts(self, tmp_path):
        for name, text in SHELL_HEADERS.items():
            (tmp_path / name).write_text(text, encoding="utf-8")
        (tmp_path / "check.mjs").write_text(MJS, encoding="utf-8")
        (tmp_path / "tool.py").write_text('"""Tool. More."""\n', encoding="utf-8")
        return tmp_path

    @pytest.mark.parametrize(
        ("name", "sentence"),
        [
            ("own_name_dash.sh", "Guard the call."),
            ("own_name_double.sh", "Purge the driver and reinstall."),
            ("bare.sh", "Compare the pins."),
            ("stale_name.sh", "This script updates cuDNN."),
            ("set_first.sh", "Preflight (I1)."),
            ("check.mjs", "Real-browser verification for the dashboard."),
            ("tool.py", "Tool."),
        ],
    )
    def test_each_header_shape_yields_its_description(
        self, mod, scripts, name, sentence
    ):
        assert mod.script_sentence(scripts / name, scripts) == sentence

    def test_a_script_without_a_header_is_named(self, mod, scripts):
        with pytest.raises(ValueError, match="none.sh has no header comment"):
            mod.rows_scripts(scripts, scripts)

    def test_rows_list_shell_python_and_modules_only(self, mod, scripts):
        (scripts / "none.sh").unlink()
        (scripts / "README.md").write_text("# x\n", encoding="utf-8")
        (scripts / "package.json").write_text("{}", encoding="utf-8")
        names = {row[0] for row in mod.rows_scripts(scripts, scripts)}
        assert names == ({*SHELL_HEADERS} - {"none.sh"}) | {"check.mjs", "tool.py"}


MK_TARGETS = """\
##@ Rust harness (separate compose file)

.PHONY: harness-build harness-run

harness-build: ## Build the image
\tdocker compose build

harness-run: ## Run a one-shot task
\tdocker compose run --rm harness $(ARGS)

_internal:
\t@true

harness-serve: ## Serve the model
\tdocker compose up -d
"""

MK_NO_SECTION = """\
.PHONY: demo

demo: ## Run the demo
\t@true
"""


class TestFragments:
    @pytest.fixture()
    def fragments(self, tmp_path):
        (tmp_path / "harness.mk").write_text(MK_TARGETS, encoding="utf-8")
        (tmp_path / "README.md").write_text("# x\n", encoding="utf-8")
        return tmp_path

    def test_the_section_and_help_targets_make_the_row(self, mod, fragments):
        rows = mod.rows_fragments(fragments, fragments)
        assert rows == [
            (
                "harness.mk",
                "Rust harness (separate compose file)",
                "`harness-build`, `harness-run`, `harness-serve`",
            )
        ]

    def test_a_fragment_without_a_section_line_is_named(self, mod, fragments):
        (fragments / "quickstart.mk").write_text(MK_NO_SECTION, encoding="utf-8")
        with pytest.raises(ValueError, match="quickstart.mk: no ##@ section line"):
            mod.rows_fragments(fragments, fragments)


class TestRust:
    @pytest.fixture()
    def crate(self, tmp_path):
        for name, text in RUST_FILES.items():
            path = tmp_path / "runtime" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
        return tmp_path / "runtime"

    def test_sources_are_rostered_recursively_with_their_test_counts(self, mod, crate):
        assert mod.rows_rust(crate, crate.parent) == [
            ("src/bin/tool.rs", "0", "Dumps a checkpoint's config."),
            ("src/lib.rs", "0", "Inference engine library crate."),
            ("src/model/cpu.rs", "2", "The f32 reference backend."),
        ]

    def test_integration_tests_count_every_attribute_form(self, mod, crate):
        assert mod.rows_rust_tests(crate, crate.parent) == [
            ("tests/e2e.rs", "1", "The loop end to end."),
            ("tests/surface.rs", "1", "`--status` through the binary."),
        ]

    def test_a_test_file_declaring_no_test_is_named(self, mod, crate):
        (crate / "tests" / "empty.rs").write_text(
            "/*\n * empty — nothing yet.\n */\n", encoding="utf-8"
        )
        with pytest.raises(ValueError, match="runtime/tests/empty.rs declares no test"):
            mod.rows_rust_tests(crate, crate.parent)

    def test_a_source_without_a_banner_is_named(self, mod, crate):
        (crate / "src" / "bare.rs").write_text("pub fn f() {}\n", encoding="utf-8")
        with pytest.raises(
            ValueError, match="runtime/src/bare.rs has no header comment"
        ):
            mod.rows_rust(crate, crate.parent)

    def test_a_bare_word_before_prose_is_kept(self, mod):
        # `kv — sizing` drops the name; `Preflight (I1)` has no dash and is prose.
        assert mod.header_paragraph(
            ["kv — KV cache sizing. More."], Path("x"), Path(".")
        ) == ("KV cache sizing.")
        assert mod.header_paragraph(["Preflight (I1)"], Path("x"), Path(".")) == (
            "Preflight (I1)."
        )


class TestContracts:
    @pytest.fixture()
    def schemas(self, tmp_path):
        (tmp_path / "both.json").write_text(
            CONTRACT
            % (
                "both",
                "A record both sides write. More.",
                '["common/obs/core.py", "rust-harness/obs/src/lib.rs"]',
                '["common/obs/catalog.py"]',
            ),
            encoding="utf-8",
        )
        (tmp_path / "one_sided.json").write_text(
            CONTRACT % ("one", "Rust only.", '["rust-harness/harness/src/a.rs"]', "[]"),
            encoding="utf-8",
        )
        (tmp_path / "facts.json").write_text(
            CONTRACT
            % (
                "facts",
                "Values from upstream.",
                '["google/gemma-4 config.json"]',
                '["rust-harness/x.rs", "interp (reads width)", "docs/models/a.md"]',
            ),
            encoding="utf-8",
        )
        return tmp_path

    def test_direction_is_derived_from_the_two_lists(self, mod, schemas):
        rows = {row[0]: row[2] for row in mod.rows_contracts(schemas, schemas)}
        assert rows["both.json"] == "Rust and Python write, Python reads"
        assert rows["one_sided.json"] == "Rust writes, no reader yet"
        assert rows["facts.json"] == "Upstream writes, Rust, Python and Upstream read"

    def test_the_seam_is_the_descriptions_first_sentence(self, mod, schemas):
        rows = {row[0]: row[1] for row in mod.rows_contracts(schemas, schemas)}
        assert rows["both.json"] == "A record both sides write."

    def test_a_contract_without_a_description_is_named(self, mod, schemas):
        (schemas / "bare.json").write_text('{"writers": ["x"]}', encoding="utf-8")
        with pytest.raises(ValueError, match="bare.json has no description"):
            mod.rows_contracts(schemas, schemas)


class TestRoster:
    @pytest.fixture()
    def dot_claude(self, tmp_path):
        agents = tmp_path / ".claude" / "agents"
        agents.mkdir(parents=True)
        agents.joinpath("skeptic.md").write_text(
            "---\nname: skeptic\ndescription: Adversarial verifier. Invoke to check.\n"
            "tools: Read, Grep\n---\nBody.\n",
            encoding="utf-8",
        )
        commands = tmp_path / ".claude" / "commands"
        commands.mkdir()
        commands.joinpath("task.md").write_text(
            "---\ndescription: Drive one task\n---\nBody.\n", encoding="utf-8"
        )
        skill = tmp_path / ".claude" / "skills" / "bug-bash"
        skill.mkdir(parents=True)
        skill.joinpath("SKILL.md").write_text(
            "---\nname: bug-bash\ndescription: The method contract. Read it.\n---\n",
            encoding="utf-8",
        )
        return tmp_path

    def test_each_roster_reads_its_fixed_source(self, mod, dot_claude):
        assert mod.rows_roster("roles", dot_claude) == [
            ("skeptic", "Read, Grep", "Adversarial verifier.")
        ]
        assert mod.rows_roster("commands", dot_claude) == [("task", "Drive one task.")]
        assert mod.rows_roster("skills", dot_claude) == [
            ("bug-bash", "The method contract.")
        ]

    def test_a_definition_without_a_description_is_named(self, mod, dot_claude):
        (dot_claude / ".claude" / "agents" / "mute.md").write_text(
            "---\nname: mute\n---\n", encoding="utf-8"
        )
        with pytest.raises(ValueError, match="agents/mute.md has no description"):
            mod.rows_roster("roles", dot_claude)

    def test_the_kind_ignores_the_readme_directory(self, mod, dot_claude):
        readme = dot_claude / "docs" / "agents" / "README.md"
        readme.parent.mkdir(parents=True)
        readme.write_text(
            "# Agents\n\nThe roster.\n\n<!-- inventory:roles:start -->\n"
            "<!-- inventory:roles:end -->\n<!-- inventory:skills:start -->\n"
            "<!-- inventory:skills:end -->\n",
            encoding="utf-8",
        )
        assert (
            run(SCRIPT, "--root", str(dot_claude), "docs/agents/README.md").returncode
            == 0
        )
        text = readme.read_text(encoding="utf-8")
        assert "| `skeptic` | Read, Grep | Adversarial verifier. |" in text
        assert "| `bug-bash` | The method contract. |" in text


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
        # The other hub roots, each with one documented child.
        for hub, kind, leaf, text in (
            ("common", "modules", "leaf.py", '"""Leaf."""\n'),
            ("scripts", "scripts", "leaf.sh", "#!/bin/bash\n# Leaf.\n"),
            ("interp", "modules", "leaf.py", '"""Leaf."""\n'),
        ):
            child = tree / hub / "child"
            child.mkdir(parents=True)
            (child / leaf).write_text(text, encoding="utf-8")
            (child / "README.md").write_text(
                f"# child\n\nThe child.\n\n<!-- inventory:{kind}:start -->\n"
                f"<!-- inventory:{kind}:end -->\n",
                encoding="utf-8",
            )
            (tree / hub / "README.md").write_text(
                f"# {hub}\n\nThe hub.\n\n<!-- inventory:dirs:start -->\n"
                "<!-- inventory:dirs:end -->\n",
                encoding="utf-8",
            )
        # The documentation tree: a hub with a page and a subdirectory of pages.
        topic = tree / "docs" / "topic"
        (topic / "sub").mkdir(parents=True)
        # The docs root already lists its two pages; it now has a subdirectory too.
        (tree / "docs" / "README.md").write_text(
            "# docs\n\nThe docs.\n\n<!-- inventory:dirs:start -->\n"
            "<!-- inventory:dirs:end -->\n\n<!-- inventory:pages:start -->\n"
            "<!-- inventory:pages:end -->\n",
            encoding="utf-8",
        )
        (topic / "README.md").write_text(
            "# topic\n\nThe topic.\n\n<!-- inventory:dirs:start -->\n"
            "<!-- inventory:dirs:end -->\n\n<!-- inventory:pages:start -->\n"
            "<!-- inventory:pages:end -->\n",
            encoding="utf-8",
        )
        (topic / "page.md").write_text("# Page\n\nA page.\n", encoding="utf-8")
        (topic / "sub" / "README.md").write_text(
            "# sub\n\nThe records.\n\n<!-- inventory:pages:start -->\n"
            "<!-- inventory:pages:end -->\n",
            encoding="utf-8",
        )
        (topic / "sub" / "record.md").write_text(
            "# Record\n\nA record.\n", encoding="utf-8"
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
        assert "training/sub/README.md has no inventory table" in result.stderr

    def test_a_directory_with_only_a_readme_needs_no_table(self, hubs):
        (hubs / "interp" / "notebooks").mkdir()
        (hubs / "interp" / "notebooks" / ".gitkeep").write_text("", encoding="utf-8")
        (hubs / "interp" / "notebooks" / "README.md").write_text(
            "# notebooks\n\nNothing yet.\n", encoding="utf-8"
        )
        assert run(SCRIPT, "--root", str(hubs)).returncode == 0
        result = run(GATE, "--root", str(hubs))
        assert result.returncode == 0, result.stderr

    def test_a_scripts_directory_without_a_table_fails_the_rule(self, hubs):
        (hubs / "scripts" / "child" / "README.md").write_text(
            "# child\n\nProse only.\n", encoding="utf-8"
        )
        assert run(SCRIPT, "--root", str(hubs)).returncode == 0
        result = run(GATE, "--root", str(hubs))
        assert result.returncode == 1
        assert "scripts/child/README.md has no inventory table" in result.stderr

    def test_a_docs_directory_without_a_readme_is_named(self, hubs):
        fresh = hubs / "docs" / "topic" / "fresh"
        fresh.mkdir()
        (fresh / "note.md").write_text("# Note\n\nA note.\n", encoding="utf-8")
        result = run(GATE, "--root", str(hubs))
        assert result.returncode == 1
        assert "docs/topic/fresh/ has no README.md" in result.stderr

    def test_a_docs_readme_missing_the_table_its_shape_needs_is_named(self, hubs):
        # The topic has a subdirectory, so a README listing only its pages routes
        # nobody to it.
        (hubs / "docs" / "topic" / "README.md").write_text(
            "# topic\n\nThe topic.\n\n<!-- inventory:pages:start -->\n"
            "<!-- inventory:pages:end -->\n",
            encoding="utf-8",
        )
        assert run(SCRIPT, "--root", str(hubs)).returncode == 0
        result = run(GATE, "--root", str(hubs))
        assert result.returncode == 1
        assert "docs/topic/README.md carries no inventory:dirs table" in result.stderr
        assert "inventory:pages" not in result.stderr

    def test_a_docs_directory_with_only_a_readme_needs_no_table(self, hubs):
        empty = hubs / "docs" / "topic" / "empty"
        empty.mkdir()
        (empty / ".gitkeep").write_text("", encoding="utf-8")
        (empty / "README.md").write_text("# empty\n\nNothing yet.\n", encoding="utf-8")
        assert run(SCRIPT, "--root", str(hubs)).returncode == 0
        result = run(GATE, "--root", str(hubs))
        assert result.returncode == 0, result.stderr

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
