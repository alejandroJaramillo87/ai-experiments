"""Tests for scripts/docs/check_runbook_ids.py.

The gate holds an entry's two statements of place apart: the id names it and
the Table of Contents orders it. Each finding below has a failing fixture and a
passing twin, because a gate whose failing case is only ever argued from the
live runbooks proves nothing once those runbooks are edited to pass it.

- A slug runbook and a legacy runbook are each accepted whole; a mix is not.
- A ToC that lists the entries in another order than the headings fails on the
  first position that differs; nested children count in document order.
- An id heading two entries, or reappearing after retirement, fails; the entry
  that absorbed a retired one may be named in its parenthetical.
- A dependency on a later entry, or on no entry, fails; a dependency on an
  entry of another runbook is not one this gate reads.
- The empty-set guard, since a checker that silently checks nothing is the
  failure these gates exist to prevent.

Fixtures are written to tmp_path rather than asserted against the real
runbooks, which change every time an entry closes. The module is loaded by
path; scripts/ is not an importable package.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "docs" / "check_runbook_ids.py"


@pytest.fixture(scope="module")
def mod():
    spec = importlib.util.spec_from_file_location("check_runbook_ids", SCRIPT)
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


def scratch_tree(root: Path, runbook: str, name: str = "runbook.md") -> None:
    """A tree the gate can scan: one runbook under a live tier directory."""
    book = root / "docs" / "verification" / "tier"
    book.mkdir(parents=True, exist_ok=True)
    (book / name).write_text(runbook, encoding="utf-8")


def slug_runbook(
    toc: list[str] | None = None,
    sections: list[str] | None = None,
    retired: str | None = "Retired: FT.15 (2026-09-19, scorecard promotion).",
    numbered: bool = True,
) -> str:
    """A well-formed slug runbook, with each part overridable."""
    ids = ["FT.preconditions", "FT.image-smoke", "FT.calibrate-qlora"]
    toc = ids if toc is None else toc
    sections = ids if sections is None else sections
    marker = "1." if numbered else "-"
    head = "# Runbook\n\n**Entries: 0 verified, 0 blocked, 3 pending, of 3.**\n\n"
    if retired:
        head += retired + "\n\n"
    rows = "\n".join(f"{marker} [{i} Title](#{i.replace('.', '')}-title)" for i in toc)
    body = "".join(
        f"## {i} Title\n\nProse; depends on nothing.\n\nStatus: PENDING\n\n"
        for i in sections
    )
    return f"{head}## Table of Contents\n\n{rows}\n\n{body}"


class TestIdShapes:
    @pytest.mark.parametrize(
        "ident", ["FT.calibrate-qlora", "I.harvest", "OBS-V.live-dashboard"]
    )
    def test_a_slug_id(self, mod, ident):
        assert mod.is_slug(ident) and not mod.is_legacy(ident)

    @pytest.mark.parametrize(
        "ident", ["RH.24", "OBS-V10", "RH.2a", "P0.1", "HS.3", "0.1"]
    )
    def test_a_legacy_id(self, mod, ident):
        assert mod.is_legacy(ident) and not mod.is_slug(ident)

    @pytest.mark.parametrize(
        "ident",
        ["config.json", "FT.Calibrate", "FT.calibrate_qlora", "Coverage", "e.g."],
    )
    def test_neither(self, mod, ident):
        """`config.json` is the case the uppercase prefix exists for: a heading
        opening with a filename must not read as an entry of runbook `config`."""
        assert not mod.is_entry_id(ident)


class TestParsing:
    def test_headings_include_parents_and_skip_fences(self, mod):
        text = (
            "## I.harvest Harvest\n\nStatus: PENDING\n\n"
            "```\n## I.pasted Not a heading\n```\n\n"
            "### I.harvest-crash-recovery Child\n\nStatus: PENDING\n"
        )
        assert [i for _, i in mod.entry_headings(text)] == [
            "I.harvest",
            "I.harvest-crash-recovery",
        ]

    def test_toc_rows_keep_document_order_across_nesting(self, mod):
        text = (
            "## Table of Contents\n\n"
            "1. [I.harvest Harvest](#iharvest-harvest)\n"
            "   1. [I.harvest-crash-recovery Child](#x)\n"
            "1. [I.capture-gc GC](#y)\n"
            "1. [Unentried surface](#unentried-surface)\n\n"
            "## I.harvest Harvest\n"
        )
        assert [i for _, i, _ in mod.toc_rows(text)] == [
            "I.harvest",
            "I.harvest-crash-recovery",
            "I.capture-gc",
        ]

    def test_retired_line_reads_the_leading_id_of_each_item_and_wraps(self, mod):
        text = (
            "Retired: I.2 (2026-09-18, Neuronpedia harness removed); I.6b (merged\n"
            "into I.served-request-log).\n\n## I.served-request-log Log\n"
        )
        assert mod.retired_ids(text) == ["I.2", "I.6b"]

    def test_no_retired_line_is_none_not_empty(self, mod):
        assert mod.retired_ids("## FT.a Title\n") is None

    def test_dependencies_read_the_first_paragraph_with_the_own_prefix(self, mod):
        text = (
            "## FT.memfit Memory fit\n\n"
            "CPU-only; depends on FT.calibrate-qlora and\n"
            "FT.preconditions (I.probe-loop uses the same directory).\n\n"
            "Later prose depends on FT.nothing.\n\nStatus: PENDING\n"
        )
        assert [(e, c) for _, e, c in mod.dependencies(text, "FT")] == [
            ("FT.memfit", "FT.calibrate-qlora"),
            ("FT.memfit", "FT.preconditions"),
        ]


class TestFindings:
    def test_a_well_formed_slug_runbook_passes(self, mod):
        assert mod.check_runbook("r.md", slug_runbook()) == []

    def test_a_legacy_runbook_passes_without_a_retired_line(self, mod):
        text = (
            "## Table of Contents\n\n- [RH.1 A](#a)\n- [RH.1a B](#b)\n\n"
            "## RH.1 A\n\nProse.\n\n### RH.1a B\n\nStatus: PENDING\n"
        )
        assert mod.check_runbook("r.md", text) == []

    def test_a_mix_of_shapes_fails(self, mod):
        text = slug_runbook(
            toc=["FT.preconditions", "FT.99"], sections=["FT.preconditions", "FT.99"]
        )
        assert any(
            "mixes slug and legacy" in f for f in mod.check_runbook("r.md", text)
        )

    def test_a_slug_past_four_words_fails(self, mod):
        long = "FT.one-two-three-four-five"
        text = slug_runbook(toc=[long], sections=[long])
        assert any("runs past 4 words" in f for f in mod.check_runbook("r.md", text))

    def test_a_duplicate_id_fails(self, mod):
        ids = ["FT.preconditions", "FT.preconditions"]
        text = slug_runbook(toc=ids, sections=ids)
        assert any("already heads" in f for f in mod.check_runbook("r.md", text))

    def test_a_retired_id_heading_an_entry_fails(self, mod):
        ids = ["FT.preconditions", "FT.15"]
        text = slug_runbook(toc=ids, sections=ids)
        findings = mod.check_runbook("r.md", text)
        assert any("on the Retired line and heads an entry" in f for f in findings)

    def test_a_missing_retired_line_fails_a_slug_runbook(self, mod):
        text = slug_runbook(retired=None)
        assert any("no `Retired:` line" in f for f in mod.check_runbook("r.md", text))

    def test_toc_order_differing_from_section_order_fails_at_the_first_position(
        self, mod
    ):
        text = slug_runbook(
            toc=["FT.preconditions", "FT.calibrate-qlora", "FT.image-smoke"]
        )
        findings = mod.check_runbook("r.md", text)
        assert findings == [
            "r.md: Table of Contents position 2 is FT.calibrate-qlora, "
            "the section there is FT.image-smoke"
        ]

    def test_toc_missing_an_entry_fails_by_count(self, mod):
        text = slug_runbook(toc=["FT.preconditions", "FT.image-smoke"])
        assert any(
            "lists 2 entries, the headings have 3" in f
            for f in mod.check_runbook("r.md", text)
        )

    def test_an_unnumbered_toc_fails_a_slug_runbook(self, mod):
        text = slug_runbook(numbered=False)
        assert any("is not numbered" in f for f in mod.check_runbook("r.md", text))

    def test_a_slug_runbook_without_a_toc_fails(self, mod):
        text = slug_runbook().replace("## Table of Contents", "## Contents")
        assert any("no Table of Contents" in f for f in mod.check_runbook("r.md", text))

    def test_a_dependency_on_a_later_entry_fails(self, mod):
        text = slug_runbook().replace(
            "## FT.preconditions Title\n\nProse; depends on nothing.",
            "## FT.preconditions Title\n\nProse; depends on FT.image-smoke.",
        )
        assert any("which comes later" in f for f in mod.check_runbook("r.md", text))

    def test_a_dependency_on_no_entry_fails(self, mod):
        text = slug_runbook().replace(
            "## FT.image-smoke Title\n\nProse; depends on nothing.",
            "## FT.image-smoke Title\n\nProse; depends on FT.ghost.",
        )
        assert any("which has no entry" in f for f in mod.check_runbook("r.md", text))

    def test_a_dependency_on_an_earlier_entry_passes(self, mod):
        text = slug_runbook().replace(
            "## FT.calibrate-qlora Title\n\nProse; depends on nothing.",
            "## FT.calibrate-qlora Title\n\nProse; depends on FT.preconditions and FT.image-smoke.",
        )
        assert mod.check_runbook("r.md", text) == []


class TestGate:
    def test_passes_a_clean_tree(self, tmp_path):
        scratch_tree(tmp_path, slug_runbook())
        result = run_gate(tmp_path)
        assert result.returncode == 0, result.stderr
        assert "every id names one entry" in result.stderr

    def test_fails_with_the_finding_and_a_solution(self, tmp_path):
        scratch_tree(
            tmp_path,
            slug_runbook(
                toc=["FT.image-smoke", "FT.preconditions", "FT.calibrate-qlora"]
            ),
        )
        result = run_gate(tmp_path)
        assert result.returncode == 1
        assert "position 1 is FT.image-smoke" in result.stderr
        assert "Solution:" in result.stderr

    def test_frozen_records_are_exempt(self, tmp_path):
        scratch_tree(tmp_path, slug_runbook())
        frozen = tmp_path / "docs" / "verification" / "archive"
        frozen.mkdir(parents=True)
        (frozen / "old.md").write_text(
            slug_runbook(toc=["FT.calibrate-qlora"]), encoding="utf-8"
        )
        assert run_gate(tmp_path).returncode == 0

    def test_the_empty_set_is_a_failure(self, tmp_path):
        scratch_tree(tmp_path, "# Runbook\n\n## Preconditions\n\nNo entries here.\n")
        result = run_gate(tmp_path)
        assert result.returncode == 1
        assert "found no entry headings" in result.stderr
