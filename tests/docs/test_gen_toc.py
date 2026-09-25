"""Tests for scripts/docs/gen_toc.py and its gate.

The generator renders a document's table of contents from its own H2 and H3
headings, between one marker pair under a `## Table of Contents` heading, and
the gate holds two rules: a document carrying the pair regenerates identically,
and a living document over the threshold carries the pair. The parts worth
pinning are the ones a document can defeat -- a `## ` inside a fenced block, a
code span or a link inside a heading, two sections with the same title, the ToC
heading counting itself as a section -- and the adoption path, which replaces a
hand-written list or inserts a section, because that is the remedy every gate
failure prints. The threshold is pinned on both edges, and each exemption class
is pinned so that a frozen record is never edited and never asked for a table.

Fixtures are scratch trees rather than the repository, so the tests do not turn
red the next time a page grows past the threshold.

The module is loaded by path; scripts/ is not an importable package.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from typing import NamedTuple

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "docs" / "gen_toc.py"
GATE = REPO_ROOT / "scripts" / "docs" / "check_toc.py"

START = "<!-- toc:start -->"
END = "<!-- toc:end -->"

# A short document with the markers and no list yet.
SHORT = """\
# Title

The description paragraph.

## Table of Contents

<!-- toc:start -->
<!-- toc:end -->

## Design

Words.

### Detail one

Words.

### Detail two

Words.

## Interface

Words.
"""

# A document whose hand-written list is to be adopted.
HAND = """\
# Title

The description.

## Table of contents

- [Title](#title)
  - [Design](#design)
  - [Interface](#interface)

## Design

Words.

## Interface

Words.
"""

# A document with no table of contents section at all.
BARE = """\
# Title

The description.

> Note: a blockquote before the first section.

## Design

Words.

## Interface

Words.
"""


@pytest.fixture(scope="module")
def mod():
    spec = importlib.util.spec_from_file_location("gen_toc", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class Result(NamedTuple):
    returncode: int
    out: str


def run(script: Path, *args: str) -> Result:
    """One of the two scripts as the quality target runs it, on a scratch tree.

    The Printer writes errors to stderr and the rest to stdout; the assertions
    read both as one transcript.
    """
    result = subprocess.run(
        [sys.executable, str(script), *args],
        capture_output=True,
        text=True,
        env={"NO_COLOR": "1", "PATH": "/usr/bin:/bin"},
    )
    return Result(result.returncode, result.stdout + result.stderr)


def long_doc(sections: int, words: int) -> str:
    """A document with `sections` H2 headings and about `words` words of body."""
    body = " ".join(["word"] * max(words // max(sections, 1), 1))
    parts = ["# Title\n\nThe description.\n"]
    for i in range(sections):
        parts.append(f"## Section {i}\n\n{body}\n")
    return "\n".join(parts)


def write(root: Path, rel: str, text: str) -> Path:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


class TestSlug:
    @pytest.mark.parametrize(
        ("heading", "anchor"),
        [
            ("Design", "design"),
            ("Network split: one door to the hub", "network-split-one-door-to-the-hub"),
            (
                "I.run-browser Run browser surface (`python -m interp runs`)",
                "irun-browser-run-browser-surface-python--m-interp-runs",
            ),
            ("How `proportional` implements it", "how-proportional-implements-it"),
            ("E2B, E4B, and 12B", "e2b-e4b-and-12b"),
            ("`--method sae_row`", "--method-sae_row"),
        ],
    )
    def test_matches_the_github_anchor(self, mod, heading, anchor):
        assert mod.slug(heading) == anchor


class TestHeadings:
    def test_lists_h2_and_h3_and_nothing_else(self, mod):
        text = "# T\n\n## A\n\n### B\n\n#### C\n\n## D\n"
        assert mod.headings(text) == [(2, "A"), (3, "B"), (2, "D")]

    def test_a_heading_inside_a_fence_is_output_not_structure(self, mod):
        text = "## A\n\n```console\n## not a heading\n```\n\n~~~\n### nor this\n~~~\n\n## B\n"
        assert mod.headings(text) == [(2, "A"), (2, "B")]

    def test_the_toc_heading_does_not_list_itself(self, mod):
        text = "## Table of Contents\n\n## A\n\n## table of contents\n"
        assert mod.headings(text) == [(2, "A")]

    def test_a_link_in_a_heading_keeps_its_text(self, mod):
        text = "## See [the ledger](../bugs/known-bugs.md) first\n"
        assert mod.headings(text) == [(2, "See the ledger first")]

    def test_a_marker_in_a_fence_or_a_code_span_is_prose(self, mod):
        text = (
            "```markdown\n<!-- toc:start -->\n<!-- toc:end -->\n```\n\n"
            "The pair is `<!-- toc:start -->` and `<!-- toc:end -->`.\n"
        )
        assert not mod.carries_markers(text)
        assert mod.marker_lines(text + f"\n{START}\n{END}\n") == [
            (7, "start"),
            (8, "end"),
        ]

    def test_trailing_hashes_are_not_part_of_the_text(self, mod):
        assert mod.headings("## Closed ##\n") == [(2, "Closed")]


class TestRender:
    def test_numbers_h2_at_the_margin_and_h3_indented(self, mod):
        rows = mod.render([(2, "A"), (3, "B"), (3, "C"), (2, "D"), (3, "E")])
        assert rows.splitlines() == [
            "1. [A](#a)",
            "   1. [B](#b)",
            "   2. [C](#c)",
            "2. [D](#d)",
            "   1. [E](#e)",
        ]

    def test_repeated_headings_get_github_suffixes(self, mod):
        rows = mod.render([(2, "Observed"), (2, "Observed"), (2, "Observed")])
        assert rows.splitlines() == [
            "1. [Observed](#observed)",
            "2. [Observed](#observed-1)",
            "3. [Observed](#observed-2)",
        ]


class TestQualifies:
    def test_over_both_edges(self, mod):
        assert mod.qualifies(long_doc(4, 600))

    def test_enough_words_but_three_sections(self, mod):
        assert not mod.qualifies(long_doc(3, 600))

    def test_four_sections_but_too_few_words(self, mod):
        assert not mod.qualifies(long_doc(4, 300))

    def test_the_toc_heading_is_not_a_section(self, mod):
        text = long_doc(3, 600).replace(
            "# Title\n", "# Title\n\n## Table of Contents\n"
        )
        assert not mod.qualifies(text)


class TestExemptions:
    @pytest.mark.parametrize(
        "rel",
        [
            "docs/x/f1_sft_data_investigation.md",
            "docs/x/investigations/design.md",
            "docs/verification/archive/old.md",
            "vendor/upstream.md",
        ],
    )
    def test_a_frozen_record_is_frozen_and_not_ruled(self, mod, rel):
        assert mod.is_frozen(Path(rel))
        assert not mod.is_ruled(Path(rel))

    @pytest.mark.parametrize(
        "rel",
        ["README.md", "docs/README.md", ".claude/skills/x/SKILL.md", "site/docs/a.md"],
    )
    def test_a_router_or_tool_definition_is_living_but_not_ruled(self, mod, rel):
        assert not mod.is_frozen(Path(rel))
        assert not mod.is_ruled(Path(rel))

    def test_an_ordinary_page_is_ruled(self, mod):
        assert mod.is_ruled(Path("docs/models/gemma-4.md"))


class TestGenerator:
    def test_renders_between_the_markers(self, tmp_path):
        doc = write(tmp_path, "docs/page.md", SHORT)
        assert run(SCRIPT, "--root", str(tmp_path), "docs/page.md").returncode == 0
        text = doc.read_text(encoding="utf-8")
        assert (
            f"{START}\n\n1. [Design](#design)\n   1. [Detail one](#detail-one)\n"
            f"   2. [Detail two](#detail-two)\n2. [Interface](#interface)\n\n{END}"
        ) in text
        assert text.startswith("# Title\n\nThe description paragraph.\n")

    def test_adopts_a_hand_written_list(self, tmp_path):
        doc = write(tmp_path, "docs/page.md", HAND)
        assert run(SCRIPT, "--root", str(tmp_path), "docs/page.md").returncode == 0
        text = doc.read_text(encoding="utf-8")
        assert "- [Title](#title)" not in text
        assert (
            "The description.\n\n## Table of Contents\n\n"
            f"{START}\n\n1. [Design](#design)\n2. [Interface](#interface)\n\n{END}\n\n## Design\n"
        ) in text

    def test_inserts_a_section_before_the_first_h2(self, tmp_path):
        doc = write(tmp_path, "docs/page.md", BARE)
        assert run(SCRIPT, "--root", str(tmp_path), "docs/page.md").returncode == 0
        text = doc.read_text(encoding="utf-8")
        assert (
            "> Note: a blockquote before the first section.\n\n## Table of Contents\n\n"
            f"{START}\n\n1. [Design](#design)\n2. [Interface](#interface)\n\n{END}\n\n## Design\n"
        ) in text

    def test_a_document_with_no_sections_is_named(self, tmp_path):
        write(tmp_path, "docs/page.md", "# Title\n\nOnly prose.\n")
        result = run(SCRIPT, "--root", str(tmp_path), "docs/page.md")
        assert result.returncode == 1
        assert "docs/page.md: no H2 heading" in result.out

    def test_a_marker_without_the_heading_is_named(self, tmp_path):
        write(tmp_path, "docs/page.md", f"# T\n\n{START}\n{END}\n\n## A\n")
        result = run(SCRIPT, "--root", str(tmp_path), "docs/page.md")
        assert result.returncode == 1
        assert "not under a `## Table of Contents` heading" in result.out

    def test_two_pairs_are_named(self, tmp_path):
        write(
            tmp_path,
            "docs/page.md",
            SHORT + f"\n## Table of Contents\n\n{START}\n{END}\n",
        )
        result = run(SCRIPT, "--root", str(tmp_path), "docs/page.md")
        assert result.returncode == 1
        assert "more than one toc marker pair" in result.out

    def test_an_unclosed_pair_is_named(self, tmp_path):
        write(tmp_path, "docs/page.md", SHORT.replace(END, ""))
        result = run(SCRIPT, "--root", str(tmp_path), "docs/page.md")
        assert result.returncode == 1
        assert "toc:start never closes" in result.out

    def test_discovery_finds_every_living_document_carrying_a_marker(self, tmp_path):
        write(tmp_path, "docs/a.md", SHORT)
        write(tmp_path, "docs/b.md", SHORT)
        write(tmp_path, "docs/investigations/c.md", SHORT)
        write(tmp_path, "docs/plain.md", BARE)
        result = run(SCRIPT, "--root", str(tmp_path))
        assert result.returncode == 0
        assert "docs/a.md" in result.out and "docs/b.md" in result.out
        assert "docs/c.md" not in result.out and "plain.md" not in result.out
        # The frozen record is left exactly as it was, markers empty.
        assert f"{START}\n{END}" in (tmp_path / "docs/investigations/c.md").read_text()

    def test_check_passes_on_fresh_output_and_fails_after_a_rename(self, tmp_path):
        doc = write(tmp_path, "docs/page.md", SHORT)
        assert run(SCRIPT, "--root", str(tmp_path), "docs/page.md").returncode == 0
        assert run(SCRIPT, "--root", str(tmp_path), "--check").returncode == 0
        doc.write_text(doc.read_text().replace("## Interface", "## Surface"))
        result = run(SCRIPT, "--root", str(tmp_path), "--check")
        assert result.returncode == 1
        assert "docs/page.md is stale" in result.out
        assert "-2. [Interface](#interface)" in result.out
        assert "+2. [Surface](#surface)" in result.out
        assert "Command: python scripts/docs/gen_toc.py docs/page.md" in result.out
        # --check never edits.
        assert "## Surface" in doc.read_text()
        assert "[Surface]" not in doc.read_text()

    def test_check_does_not_adopt(self, tmp_path):
        doc = write(tmp_path, "docs/page.md", HAND)
        result = run(SCRIPT, "--root", str(tmp_path), "--check", "docs/page.md")
        assert result.returncode == 1
        assert "carries no toc markers" in result.out
        assert doc.read_text() == HAND


class TestGate:
    @pytest.fixture()
    def tree(self, tmp_path):
        """A scratch root: one indexed page, one short page, one frozen record."""
        write(tmp_path, "docs/indexed.md", SHORT)
        run(SCRIPT, "--root", str(tmp_path), "docs/indexed.md")
        write(tmp_path, "docs/short.md", BARE)
        write(tmp_path, "docs/investigations/long_investigation.md", long_doc(6, 900))
        return tmp_path

    def test_passes_when_every_table_is_fresh(self, tree):
        result = run(GATE, "--root", str(tree))
        assert result.returncode == 0, result.out
        assert "1 carrying a table of contents, 0 over the threshold" in result.out

    def test_a_stale_table_fails_with_the_diff(self, tree):
        doc = tree / "docs" / "indexed.md"
        doc.write_text(doc.read_text().replace("## Design", "## Plan"))
        result = run(GATE, "--root", str(tree))
        assert result.returncode == 1
        assert "docs/indexed.md is stale" in result.out
        assert "+1. [Plan](#plan)" in result.out

    def test_a_long_page_without_a_table_fails_and_names_the_remedy(self, tree):
        write(tree, "docs/long.md", long_doc(5, 800))
        result = run(GATE, "--root", str(tree))
        assert result.returncode == 1
        assert "docs/long.md carries no toc markers" in result.out
        assert "Command: python scripts/docs/gen_toc.py docs/long.md" in result.out

    @pytest.mark.parametrize(
        "rel",
        [
            "docs/long_investigation.md",
            "docs/archive/long.md",
            "docs/README.md",
            ".claude/skills/long/SKILL.md",
            "site/docs/long.md",
        ],
    )
    def test_an_exempt_document_over_the_threshold_is_not_asked(self, tree, rel):
        write(tree, rel, long_doc(5, 800))
        assert run(GATE, "--root", str(tree)).returncode == 0

    def test_a_frozen_record_with_a_stale_table_is_not_judged(self, tree):
        write(
            tree,
            "docs/investigations/toc.md",
            SHORT.replace(f"{START}\n{END}", f"{START}\n\nstale\n\n{END}"),
        )
        assert run(GATE, "--root", str(tree)).returncode == 0

    def test_a_short_page_that_opted_in_is_still_held_fresh(self, tree):
        write(
            tree,
            "docs/README.md",
            SHORT.replace(f"{START}\n{END}", f"{START}\n\nstale\n\n{END}"),
        )
        result = run(GATE, "--root", str(tree))
        assert result.returncode == 1
        assert "docs/README.md is stale" in result.out

    def test_an_empty_tree_fails_rather_than_passing(self, tmp_path):
        result = run(GATE, "--root", str(tmp_path))
        assert result.returncode == 1
        assert "enumerated zero documents" in result.out

    def test_the_gate_passes_on_the_repository(self):
        result = run(GATE, "--root", str(REPO_ROOT))
        assert result.returncode == 0, result.out
