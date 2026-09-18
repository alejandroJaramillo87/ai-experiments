"""Tests for scripts/docs/check_kb_refs.py.

This is the one documentation gate that protects code. KB identifiers are cited
from source comments explaining why a workaround exists, and a deleted ledger row
breaks the reference silently: the code still compiles and the comment still reads
sensibly, so nothing surfaces until someone looks the id up and finds nothing.

The module is loaded by path; scripts/ is not an importable package.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "docs" / "check_kb_refs.py"


@pytest.fixture(scope="module")
def mod():
    spec = importlib.util.spec_from_file_location("check_kb_refs", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestExtractCitations:
    def test_finds_an_id_in_prose(self, mod):
        assert mod.extract_citations("See KB-52 for the caveat.") == [(1, "KB-52")]

    def test_finds_an_id_in_a_code_comment(self, mod):
        text = "# Workaround for KB-57: the recipe names the blocker.\n"
        assert mod.extract_citations(text) == [(1, "KB-57")]

    def test_reports_the_line_the_citation_sits_on(self, mod):
        assert mod.extract_citations("one\ntwo\nKB-03 here\n") == [(3, "KB-03")]

    def test_finds_every_id_on_one_line(self, mod):
        found = mod.extract_citations("KB-28, KB-29 and KB-30 were one block.")
        assert [i for _, i in found] == ["KB-28", "KB-29", "KB-30"]

    def test_skips_the_reserved_next_identifier(self, mod):
        """The ledger reserves the next id in prose; it is undefined by design."""
        assert mod.extract_citations("The next bug filed is KB-67.") == []

    @pytest.mark.parametrize(
        ("text", "why"),
        [
            (
                "KB-52A is not an identifier",
                "the pattern is anchored on a word boundary",
            ),
            ("no identifier here at all", "nothing to find"),
        ],
    )
    def test_ignores_non_citations(self, mod, text, why):
        assert mod.extract_citations(text) == [], why


class TestCollectDefined:
    """Definition is a table row in one of the two ledgers, asserted against the tree."""

    def test_finds_ids_defined_in_the_live_ledger(self, mod, repo_root):
        assert "KB-60" in mod.collect_defined(repo_root)

    def test_finds_ids_defined_in_the_frozen_archive(self, mod, repo_root):
        """The archive holds the closed history moved out of the ledger."""
        defined = mod.collect_defined(repo_root)
        assert "KB-01" in defined or "KB-02" in defined

    def test_a_deleted_row_stays_defined_by_its_retirement_entry(self, mod, repo_root):
        """A retired id must keep resolving, or a grep for it finds silence."""
        assert "KB-17" in mod.collect_defined(repo_root)

    def test_every_named_ledger_exists(self, mod, repo_root):
        missing = [rel for rel in mod.LEDGERS if not (repo_root / rel).is_file()]
        assert not missing


class TestIsExempt:
    @pytest.mark.parametrize(
        ("rel", "why"),
        [
            (
                "docs/verification/archive/gemma-4-conformance_investigation.md",
                "a frozen record; two ids it cites were retired in the 2026-08-08 reset",
            ),
            ("scripts/docs/check_kb_refs.py", "spells ids to define the patterns"),
        ],
    )
    def test_frozen_records_and_the_checker_are_exempt(self, mod, rel, why):
        assert mod.is_exempt(rel), why

    @pytest.mark.parametrize(
        "rel",
        [
            "docs/verification/bugs/known-bugs.md",
            "interp/harvest.py",
            "docs/interp/README.md",
        ],
    )
    def test_living_files_are_in_scope(self, mod, rel):
        assert not mod.is_exempt(rel)


def test_the_ledger_is_cited_from_source_code(mod, repo_root):
    """The premise of the check: if no code cited an id, this would not be needed.

    Guards against the check quietly becoming decorative. Should the citations
    ever leave the source tree, this fails and the gate can be reconsidered
    rather than left running over documentation alone.
    """
    cited_in_code = {
        kb_id
        for path in mod.iter_files(repo_root)
        if path.suffix in {".py", ".rs", ".yaml"}
        for _, kb_id in mod.extract_citations(path.read_text(encoding="utf-8"))
    }
    assert cited_in_code, "no source file cites a KB identifier"


def test_every_cited_identifier_resolves(mod, repo_root):
    """The gate's own assertion, so a failure names the file and line."""
    defined = mod.collect_defined(repo_root)
    assert defined, "the ledgers define nothing; every citation would be reported"

    files = mod.iter_files(repo_root)
    assert files, "enumerated zero files; the check would pass having read nothing"

    dangling = []
    for path in files:
        rel = str(path.relative_to(repo_root))
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        for lineno, kb_id in mod.extract_citations(text):
            if kb_id not in defined:
                dangling.append(f"{rel}:{lineno}: {kb_id}")

    assert not dangling, "citations naming no ledger row:\n" + "\n".join(dangling)
