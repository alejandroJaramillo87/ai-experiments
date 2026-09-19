"""Tests for scripts/docs/check_runbook_entries.py.

The gate answers two questions about a verification entry: does every check it
names exist, and does a VERIFIED entry offer evidence at all. Both are shallow
questions, and the whole difficulty is in recognizing the shapes the live
runbooks actually use. Four of them defeated the first implementation, and each
has a test below.

- A parent entry heads several children, so reading a section to its end
  attributes the first child's status to a parent that carries none.
- `Observed (2026-08-12, /mnt/...` opens a parenthetical that runs past the end
  of its own line, so the colon is not there to anchor on.
- A gate-backed entry has no Observed block by design; it names the check
  instead, and that is the second legal kind of evidence.
- The bug ledger cites test classes, not only functions, so `def` alone reports
  the gate's own blind spot as a documentation defect.

The fifth test is the empty-set guard, because a checker that silently checks
nothing is the failure these gates exist to prevent.

Fixtures are written to tmp_path rather than asserted against the real runbooks,
which change every time an entry closes. See test_doc_tallies.py for the same
arrangement over the tally gate.

The module is loaded by path; scripts/ is not an importable package.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "docs" / "check_runbook_entries.py"


@pytest.fixture(scope="module")
def mod():
    spec = importlib.util.spec_from_file_location("check_runbook_entries", SCRIPT)
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


def scratch_tree(root: Path, runbook: str, sources: dict[str, str]) -> None:
    """A tree the gate can scan: one runbook, plus the files it cites."""
    book = root / "docs" / "verification" / "tier"
    book.mkdir(parents=True)
    (book / "runbook.md").write_text(runbook, encoding="utf-8")
    for rel, text in sources.items():
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")


class TestFindEntries:
    """What counts as an entry, and which status belongs to it."""

    def test_a_parent_entry_does_not_inherit_its_first_childs_status(self, mod):
        """The trap that made three real entries look evidence-free.

        rust-harness.md RH.2 heads RH.2a onward and carries no status of its
        own. Reading its body to the end of the section finds the first child's
        `Status: VERIFIED`, and the parent -- which has no Observed block
        because it is not an entry -- is then reported as verified with no
        evidence.
        """
        text = (
            "## RH.2 Parent\n\nProse, no status.\n\n"
            "### RH.2a Child\n\nObserved:\n\n```\nreal output\n```\n\n"
            "Status: VERIFIED 2026-08-14\n"
        )
        entries = mod.find_entries(text)
        assert [(e[1], e[2]) for e in entries] == [("RH.2a", "VERIFIED")]

    def test_a_section_without_a_digit_is_not_an_entry(self, mod):
        """`## Coverage roster` ends in a status line and is not an entry.

        The roster states its own status because the table it heads was
        reviewed as a unit. Treating it as an entry would demand an Observed
        block from a table.
        """
        text = "## Coverage roster\n\n| a | b |\n\nStatus: VERIFIED 2026-08-15\n"
        assert mod.find_entries(text) == []


class TestHasEvidence:
    """The two legal kinds of evidence, and the shape that is neither."""

    def test_an_empty_fence_is_not_evidence(self, mod):
        body = ["Observed:", "", "```", "```", "", "Status: PENDING"]
        assert not mod.has_evidence(body)

    def test_a_parenthetical_running_past_its_own_line_still_opens_a_block(self, mod):
        """`Observed (2026-08-12, /mnt/ai-data/..., 2624 lines; the` has no colon.

        Three real entries open their Observed block this way. Anchoring the
        marker on a colon misses all three.
        """
        body = [
            "Observed (2026-08-12, `/mnt/ai-data/models/hf/gemma-4-E4B`, 2624 lines;",
            "the banner only):",
            "",
            "```",
            "architecture: gemma4",
            "```",
            "",
            "Status: VERIFIED 2026-08-12",
        ]
        assert mod.has_evidence(body)

    def test_a_named_check_is_evidence_without_an_observed_block(self, mod):
        """The gate-backed shape, which has no output to paste.

        I.run-browser through I.traced-dispatch carry no Observed block by design: the check named is
        the verification, and running the gate is how it is observed. The other
        half of this gate is what keeps that honest, by resolving the name.
        """
        body = [
            "Asserted by `tests/interp/test_interp_cli.py::test_runs_parses_flags`.",
            "",
            "Status: VERIFIED 2026-08-15",
        ]
        assert mod.has_evidence(body)


class TestCitations:
    """Which spellings of a named check are recognized."""

    def test_a_continuation_inherits_the_path_last_named(self, mod):
        text = (
            "| `OBS_TRACE` | `tests/common/test_obs_audit.py::test_trace_records`, "
            "`::test_trace_off` |\n"
        )
        assert mod.find_citations(text) == [
            (1, "tests/common/test_obs_audit.py", "test_trace_records"),
            (1, "tests/common/test_obs_audit.py", "test_trace_off"),
        ]

    def test_a_signal_table_resolves_bare_names_against_asserted_by(self, mod):
        text = (
            "Asserted by `rust-harness/runtime/tests/ladder_flags.rs`:\n"
            "\n"
            "| Expected signal | Test function |\n"
            "|---|---|\n"
            "| Every thread width prints the same table | `every_thread_width` |\n"
        )
        assert mod.find_citations(text) == [
            (5, "rust-harness/runtime/tests/ladder_flags.rs", "every_thread_width")
        ]

    def test_a_python_test_class_is_a_definition(self, mod, tmp_path):
        """The bug ledger cites classes; `def` alone would call them missing."""
        source = tmp_path / "test_contract.py"
        source.write_text("class TestOutputPathContract:\n    pass\n", encoding="utf-8")
        assert mod.defines(source, "TestOutputPathContract")
        assert not mod.defines(source, "TestSomethingElse")


class TestGate:
    """End to end, over a scratch tree."""

    def test_a_citation_naming_no_definition_fails(self, tmp_path):
        scratch_tree(
            tmp_path,
            "# Runbook\n\n## T.1 Thing\n\n"
            "Asserted by `src/thing.py::test_renamed_away`.\n\n"
            "Status: VERIFIED 2026-08-15\n",
            {"src/thing.py": "def test_the_new_name():\n    pass\n"},
        )
        result = run_gate(tmp_path)
        assert result.returncode == 1
        assert "test_renamed_away -- not defined there" in result.stderr

    def test_a_verified_entry_with_no_evidence_fails(self, tmp_path):
        scratch_tree(
            tmp_path,
            "# Runbook\n\n## T.1 Thing\n\n"
            "Asserted by `src/thing.py::test_present`.\n\n"
            "Status: VERIFIED 2026-08-15\n\n"
            "## T.2 Other\n\nObserved:\n\n```\n```\n\nStatus: VERIFIED 2026-08-15\n",
            {"src/thing.py": "def test_present():\n    pass\n"},
        )
        result = run_gate(tmp_path)
        assert result.returncode == 1
        assert "T.2 is VERIFIED with no Observed" in result.stderr
        assert "T.1" not in result.stderr

    def test_a_well_formed_tree_passes(self, tmp_path):
        scratch_tree(
            tmp_path,
            "# Runbook\n\n## T.1 Thing\n\n"
            "Asserted by `src/thing.py::test_present`.\n\n"
            "Status: VERIFIED 2026-08-15\n",
            {"src/thing.py": "def test_present():\n    pass\n"},
        )
        result = run_gate(tmp_path)
        assert result.returncode == 0, result.stderr

    def test_a_tree_with_no_citations_fails_the_empty_set_guard(self, tmp_path):
        """A pattern that stops matching must not read as success."""
        scratch_tree(
            tmp_path,
            "# Runbook\n\n## T.1 Thing\n\nObserved:\n\n```\nout\n```\n\n"
            "Status: VERIFIED 2026-08-15\n",
            {},
        )
        result = run_gate(tmp_path)
        assert result.returncode == 1
        assert "found no named checks to resolve" in result.stderr


def test_the_repository_runbooks_are_well_formed():
    """The real gate over the whole tree, as every sibling module ends.

    A stale citation anywhere in live documentation surfaces as a pytest
    failure and not only as a `make quality` failure.
    """
    result = run_gate(REPO_ROOT)
    assert result.returncode == 0, result.stderr
