"""Tests for scripts/docs/check_doc_counts.py.

The checker compares a document against itself, so unlike its siblings it cannot
be wrong about what the right answer is. Its whole risk sits in deciding what
counts: which headings are claims about rows, where a section ends, and whether a
table's header row is data. Those three decisions carry the tests below.

The module is loaded by path; scripts/ is not an importable package.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[2] / "scripts" / "docs" / "check_doc_counts.py"
)


@pytest.fixture(scope="module")
def mod():
    spec = importlib.util.spec_from_file_location("check_doc_counts", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestFindSections:
    """Which headings state a count, and how far the section they head reaches."""

    def test_reads_the_claimed_count(self, mod):
        sections = mod.find_sections("## Open (3)\n\nbody\n")
        assert [(s[1], s[2]) for s in sections] == [("Open", 3)]

    @pytest.mark.parametrize(
        ("text", "why"),
        [
            ("# Known bugs (12)\n", "H1 is a title, not a section over a table"),
            ("## Open bugs\n", "no count stated"),
            ("## Open (3) and more\n", "the number is mid-sentence, not the claim"),
        ],
    )
    def test_ignores_non_claims(self, mod, text, why):
        assert mod.find_sections(text) == [], why

    def test_a_section_ends_at_the_next_equal_heading(self, mod):
        text = "## A (1)\n\nrow one\n\n## B\n\nrow two\n"
        assert "row two" not in mod.find_sections(text)[0][3]

    def test_a_section_absorbs_its_own_subsections(self, mod):
        """A reader counting by hand counts the subsection's rows too."""
        text = "## A (1)\n\n### Detail\n\nrow one\n\n## B\n"
        assert "row one" in mod.find_sections(text)[0][3]


class TestCountBody:
    """What is counted, once a section has been delimited."""

    def test_excludes_the_table_header_and_rule(self, mod):
        body = ["| ID | Note |", "|----|------|", "| one | a |", "| two | b |"]
        assert mod.count_body(body) == ("table rows", 2)

    def test_counts_list_items_when_there_is_no_table(self, mod):
        assert mod.count_body(["- one", "- two", "- three"]) == ("list items", 3)

    def test_a_table_wins_over_a_bulleted_preamble(self, mod):
        """A bulleted introduction to a table is not the thing being counted."""
        body = ["- context", "", "| ID | Note |", "|----|------|", "| one | a |"]
        assert mod.count_body(body) == ("table rows", 1)

    def test_ignores_content_inside_a_fence(self, mod):
        """A pipe in sample output is not a table row."""
        body = ["```", "| not | a | table |", "```", "- one"]
        assert mod.count_body(body) == ("list items", 1)

    def test_prose_counts_nothing(self, mod):
        """A parenthesised number over prose is a qualifier, not a row claim."""
        assert mod.count_body(["Some explanatory prose.", ""]) is None


class TestIsYear:
    @pytest.mark.parametrize("value", [1999, 2025, 2100])
    def test_a_date_is_not_a_count(self, mod, value):
        """ "### Speculative Decoding (2025)" heads prose and counts nothing."""
        assert mod.is_year(value)

    @pytest.mark.parametrize("value", [3, 12, 36, 66])
    def test_a_plausible_row_count_is_not_a_year(self, mod, value):
        assert not mod.is_year(value)


def test_every_stated_count_matches_its_section(mod, repo_root):
    """The gate's own assertion, so a failure names the file and line."""
    docs = mod.iter_docs(repo_root)
    assert docs, "enumerated zero documents; the check would pass having read nothing"

    wrong = []
    checked = 0
    for doc in docs:
        rel = str(doc.relative_to(repo_root))
        text = doc.read_text(encoding="utf-8")
        for lineno, heading, claimed, body in mod.find_sections(text):
            if mod.is_year(claimed):
                continue
            counted = mod.count_body(body)
            if counted is None:
                continue
            kind, actual = counted
            checked += 1
            if actual != claimed:
                wrong.append(
                    f"{rel}:{lineno}: {heading} says {claimed}, has {actual} {kind}"
                )

    assert checked, "found no counted sections; the check would pass over an empty set"
    assert not wrong, "headings that disagree with their own section:\n" + "\n".join(
        wrong
    )
