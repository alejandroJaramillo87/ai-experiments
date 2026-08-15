"""Tests for the runbook-tally half of scripts/docs/check_doc_counts.py.

A verification runbook states its own progress in bold under the front matter --
`**Entries: N verified, M blocked, K pending, of T.**` -- and the entries beneath
it each carry a `Status: ` line. The tally is maintained by hand in the edit that
closes an entry, so it drifts in exactly the way a count in docs/verification
already drifted from 56 to an actual 66 and sat there for several sessions.

The risk is entirely in what counts as an entry status. Four shapes in the live
runbooks defeat the obvious implementations, and each has a test below: a bolded
`**Status: ...**` mid-paragraph, a parent entry with no status of its own,
non-contiguous entry numbering, and a status word trailed by a date and a
parenthetical. The fifth test is the empty-set guard, because a checker that
silently checks nothing is the failure mode these gates exist to prevent.

Fixtures are written to tmp_path rather than asserted against the real runbooks:
the runbooks change every time an entry closes, and a test pinned to them would
be red for a reason that is not a defect. The whole-tree assertion lives in the
gate itself; see test_doc_counts.py for the heading half.

The module is loaded by path; scripts/ is not an importable package.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "docs" / "check_doc_counts.py"

# A minimal document that satisfies the heading half of the gate, so a scratch
# tree fails or passes on its tallies rather than on the other check's guard.
COUNTED_SECTION = "# Doc\n\n## Open (1)\n\n- one\n"


@pytest.fixture(scope="module")
def mod():
    spec = importlib.util.spec_from_file_location("check_doc_counts", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def runbook(verified: int, blocked: int, pending: int, total: int, body: str) -> str:
    """A runbook with the stated tally over the given body."""
    return (
        "# Runbook\n\n"
        f"**Entries: {verified} verified, {blocked} blocked, "
        f"{pending} pending, of {total}.**\n\n" + body
    )


def run_gate(root: Path) -> subprocess.CompletedProcess[str]:
    """The gate as the quality target runs it, scoped to a scratch tree."""
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--root", str(root)],
        capture_output=True,
        text=True,
        env={"NO_COLOR": "1", "PATH": "/usr/bin:/bin"},
    )


class TestCountStatuses:
    """What counts as an entry status, which is where every trap lives."""

    def test_counts_a_status_line_by_kind(self, mod):
        text = "Status: VERIFIED\nStatus: BLOCKED\nStatus: PENDING\nStatus: PENDING\n"
        counts, unknown = mod.count_statuses(text)
        assert counts == {"VERIFIED": 1, "BLOCKED": 1, "PENDING": 2}
        assert unknown == []

    def test_a_bolded_inline_status_is_not_an_entry_status(self, mod):
        """Trap 1: only column zero counts.

        A sentence naming an entry's status inside a paragraph is prose about
        the entry, not the entry's status line. One such line lived in
        rust-harness.md; counting it would have inflated the tally by one.
        """
        text = (
            "Status: VERIFIED 2026-08-14\n"
            "\n"
            "The blocked entries say so themselves: **Status: BLOCKED** until an\n"
            "instruction-tuned checkpoint exists.\n"
            "  Status: PENDING\n"
        )
        counts, unknown = mod.count_statuses(text)
        assert counts == {"VERIFIED": 1, "BLOCKED": 0, "PENDING": 0}
        assert unknown == []

    def test_a_parent_entry_needs_no_status_of_its_own(self, mod):
        """Trap 2: statuses are not one per heading.

        rust-harness.md RH.9 heads RH.9a through RH.9d and carries no status
        itself; each child carries one. Requiring a status per heading would
        report the parent as missing and the tally as short by one.
        """
        text = (
            "## RH.9 Interp chain on real weights\n\n"
            "Prose, no status.\n\n"
            "### RH.9a Activation capture\n\nStatus: VERIFIED 2026-08-14\n\n"
            "### RH.9b Probe training\n\nStatus: VERIFIED 2026-08-14\n\n"
            "### RH.9d Steering\n\nStatus: BLOCKED 2026-08-14. Needs a checkpoint.\n"
        )
        counts, _ = mod.count_statuses(text)
        assert counts["VERIFIED"] == 2
        assert counts["BLOCKED"] == 1

    def test_entry_numbering_is_not_the_count(self, mod):
        """Trap 3: identifiers are neither contiguous nor sequential.

        interp.md reaches 13 entries with no I.13, because an interstitial I.9b
        sits between I.9 and I.10. Counting identifiers, or trusting the highest
        one, gets a different answer than counting statuses.
        """
        text = (
            "## I.9 Harvest\n\nStatus: VERIFIED 2026-08-14\n\n"
            "## I.9b Chunked-capture crash recovery\n\nStatus: PENDING\n\n"
            "## I.10 Capture garbage collection\n\nStatus: PENDING\n"
        )
        counts, _ = mod.count_statuses(text)
        assert sum(counts.values()) == 3

    @pytest.mark.parametrize(
        ("line", "kind"),
        [
            ("Status: VERIFIED 2026-08-14", "VERIFIED"),
            ("Status: VERIFIED 2026-08-09 in favor of the modeling code.", "VERIFIED"),
            ("Status: PENDING (run 2026-08-14; reopened 2026-08-15 -- U7)", "PENDING"),
            ("Status: BLOCKED 2026-08-14. Needs the instruction-tuned E4B.", "BLOCKED"),
            ("Status: VERIFIED", "VERIFIED"),
        ],
    )
    def test_the_kind_is_the_first_word_and_nothing_after_it(self, mod, line, kind):
        """Trap 4: a date, a sentence or a parenthetical may follow the kind."""
        counts, unknown = mod.count_statuses(line + "\n")
        assert unknown == []
        assert counts[kind] == 1

    def test_an_unrecognized_word_is_reported_not_ignored(self, mod):
        """A typo must be loud: dropped, it reads as an entry with no status."""
        text = "Status: VERIFIED\nStatus: VERIFED 2026-08-14\n"
        counts, unknown = mod.count_statuses(text)
        assert counts["VERIFIED"] == 1
        assert unknown == [(2, "VERIFED")]

    def test_a_status_line_with_no_word_is_reported(self, mod):
        assert mod.count_statuses("Status: \n")[1] == [(1, "")]


class TestFindTallies:
    def test_reads_the_four_numbers(self, mod):
        text = runbook(16, 5, 1, 22, "Status: VERIFIED\n")
        assert mod.find_tallies(text) == [
            (3, {"VERIFIED": 16, "BLOCKED": 5, "PENDING": 1}, 22)
        ]

    def test_prose_may_continue_past_the_tally(self, mod):
        """rust-harness.md names its blocked entries in the same sentence."""
        text = "**Entries: 16 verified, 5 blocked, 1 pending, of 22.** The five\n"
        assert mod.find_tallies(text)[0][2] == 22

    @pytest.mark.parametrize(
        ("text", "why"),
        [
            (
                "`Entries: N verified, M blocked, K pending, of T` in bold under\n",
                "the README describes the form; it states no tally of its own",
            ),
            ("Entries: 2 verified, 0 blocked, 0 pending, of 2.\n", "not bolded"),
            ("## Entries\n", "no tally"),
        ],
    )
    def test_ignores_non_tallies(self, mod, text, why):
        assert mod.find_tallies(text) == [], why


class TestGate:
    """End to end, because the exit code is what the quality target reads."""

    def test_a_consistent_runbook_passes(self, mod, tmp_path):
        (tmp_path / "counted.md").write_text(COUNTED_SECTION)
        (tmp_path / "runbook.md").write_text(
            runbook(
                1,
                1,
                1,
                3,
                "Status: VERIFIED 2026-08-14\n"
                "Status: BLOCKED 2026-08-14. Needs a checkpoint.\n"
                "Status: PENDING (run 2026-08-14)\n",
            )
        )
        assert run_gate(tmp_path).returncode == 0

    def test_a_drifted_kind_fails_and_names_the_file(self, tmp_path):
        (tmp_path / "counted.md").write_text(COUNTED_SECTION)
        (tmp_path / "runbook.md").write_text(
            runbook(2, 0, 0, 2, "Status: VERIFIED\nStatus: PENDING\n")
        )
        result = run_gate(tmp_path)
        assert result.returncode == 1
        assert "runbook.md:3: tally says 2 verified, has 1" in result.stderr

    def test_parts_that_do_not_sum_to_the_total_fail(self, tmp_path):
        (tmp_path / "counted.md").write_text(COUNTED_SECTION)
        (tmp_path / "runbook.md").write_text(
            runbook(1, 0, 1, 3, "Status: VERIFIED\nStatus: PENDING\n")
        )
        result = run_gate(tmp_path)
        assert result.returncode == 1
        assert "parts sum to 2, not the stated 3" in result.stderr

    def test_an_unrecognized_kind_fails_loudly(self, tmp_path):
        (tmp_path / "counted.md").write_text(COUNTED_SECTION)
        (tmp_path / "runbook.md").write_text(
            runbook(1, 0, 0, 1, "Status: VERIFIED\nStatus: FAILED 2026-08-14\n")
        )
        result = run_gate(tmp_path)
        assert result.returncode == 1
        assert "runbook.md:6: `Status: FAILED` is none of" in result.stderr

    def test_a_status_line_outside_a_runbook_is_not_judged(self, tmp_path):
        """docs/inference/engines/vllm.md opens a line `Status: vLLM 0.25.0 ...`.

        Scope is the document that states a tally. The kind vocabulary is a
        runbook convention, not a rule about markdown.
        """
        (tmp_path / "counted.md").write_text(COUNTED_SECTION)
        (tmp_path / "runbook.md").write_text(runbook(1, 0, 0, 1, "Status: VERIFIED\n"))
        (tmp_path / "engine.md").write_text(
            "# Engine\n\nStatus: vLLM 0.25.0 targets CUDA 13.0.2\n"
        )
        assert run_gate(tmp_path).returncode == 0

    def test_a_tree_with_no_tally_fails_rather_than_passing_empty(self, tmp_path):
        """The empty-set guard: checking nothing is not a pass.

        Every tally in the tree could be reworded past the pattern -- or the
        pattern could rot against a renamed field -- and the gate would keep
        reporting success over an empty set.
        """
        (tmp_path / "counted.md").write_text(COUNTED_SECTION)
        result = run_gate(tmp_path)
        assert result.returncode == 1
        assert "found no runbook tallies to check" in result.stderr
