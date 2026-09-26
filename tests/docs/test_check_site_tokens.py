"""Tests for scripts/docs/check_site_tokens.py.

The gate holds the design tokens site/src/css/custom.css mirrors from figura to
the snapshot of figura's shared/tokens.css pinned in the diagrams' provenance
record. A paired value that differs fails, and so does a paired token that is
missing from the first :root block or declared a second time elsewhere, where
its value would never be compared. The unpaired list is held too: a token
listed there must exist, and a snapshot value nothing names is a finding.

Fixtures are scratch trees rather than the real repository, so the tests do not
turn red the next time the palette moves.

The module is loaded by path; scripts/ is not an importable package.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "docs" / "check_site_tokens.py"


@pytest.fixture(scope="module")
def mod():
    spec = importlib.util.spec_from_file_location("check_site_tokens", SCRIPT)
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


def snapshot(mod) -> dict[str, str]:
    """A figura snapshot whose every value equals the site value in css()."""
    values = {figura: SITE_VALUES[site] for site, figura in mod.PAIRS}
    for site, figura in mod.UNPAIRED.items():
        values[figura] = "figura-" + SITE_VALUES[site]
    return values


SITE_VALUES = {
    "--panel-bg": "#0f172a",
    "--panel-2": "#1e293b",
    "--panel-fg": "#f8fafc",
    "--panel-muted": "#94a3b8",
    "--panel-faint": "#64748b",
    "--panel-border": "#334155",
    "--panel-accent": "#38bdf8",
    "--radius": "12px",
    "--ease": "cubic-bezier(0.4, 0, 0.2, 1)",
    "--font-sans": '"Work Sans", system-ui, sans-serif',
    "--font-mono": "ui-monospace, monospace",
}


def css(overrides: dict[str, str] | None = None, tail: str = "") -> str:
    values = {**SITE_VALUES, **(overrides or {})}
    body = "\n".join(f"  {name}: {value};" for name, value in values.items())
    return f"/* header */\n\n:root {{\n  /* a comment: --not-a-token */\n{body}\n}}\n{tail}"


def scratch_tree(root: Path, stylesheet: str, values: dict[str, str] | None) -> None:
    """A tree holding custom.css and a record with the given token snapshot."""
    sheet = root / "site" / "src" / "css" / "custom.css"
    sheet.parent.mkdir(parents=True)
    sheet.write_text(stylesheet, encoding="utf-8")
    directory = root / "site" / "static" / "img" / "diagrams"
    directory.mkdir(parents=True)
    record = {"commit": "0" * 40, "files": {}}
    if values is not None:
        record["tokens"] = {"source": "shared/tokens.css", "values": values}
    (directory / "figura.json").write_text(json.dumps(record), encoding="utf-8")


class TestTables:
    """The pair table and the unpaired list, as data."""

    def test_font_mono_is_unpaired_not_paired(self, mod):
        assert "--font-mono" in mod.UNPAIRED
        assert "--font-mono" not in dict(mod.PAIRS)

    def test_no_name_appears_twice(self, mod):
        sites = [site for site, _ in mod.PAIRS] + list(mod.UNPAIRED)
        figuras = [figura for _, figura in mod.PAIRS] + list(mod.UNPAIRED.values())
        assert len(sites) == len(set(sites))
        assert len(figuras) == len(set(figuras))

    def test_a_declaration_spanning_lines_is_one_value(self, mod):
        block = mod.root_block(css({"--font-mono": "ui-monospace,\n    monospace"}))
        declarations = mod.declarations(block.group(1))
        assert declarations["--font-mono"] == "ui-monospace, monospace"


class TestGate:
    """End to end, over a scratch tree."""

    def test_matching_values_pass_and_an_unpaired_difference_is_allowed(
        self, mod, tmp_path
    ):
        scratch_tree(tmp_path, css(), snapshot(mod))
        result = run_gate(tmp_path)
        assert result.returncode == 0, result.stderr
        assert "unpaired" in result.stderr

    def test_a_paired_value_that_differs_fails(self, mod, tmp_path):
        scratch_tree(tmp_path, css({"--panel-accent": "#0ea5e9"}), snapshot(mod))
        result = run_gate(tmp_path)
        assert result.returncode == 1
        assert "--panel-accent is #0ea5e9, figura --fg-accent is #38bdf8" in (
            result.stderr
        )

    def test_values_compare_case_and_whitespace_insensitively(self, mod, tmp_path):
        stylesheet = css(
            {"--panel-bg": "#0F172A", "--ease": "cubic-bezier(0.4,  0, 0.2, 1)"}
        )
        scratch_tree(tmp_path, stylesheet, snapshot(mod))
        result = run_gate(tmp_path)
        assert result.returncode == 0, result.stderr

    def test_a_paired_token_missing_from_root_fails(self, mod, tmp_path):
        stylesheet = css().replace("  --radius: 12px;\n", "")
        scratch_tree(tmp_path, stylesheet, snapshot(mod))
        result = run_gate(tmp_path)
        assert result.returncode == 1
        assert "--radius missing from the first :root block" in result.stderr

    def test_a_paired_token_redeclared_elsewhere_fails(self, mod, tmp_path):
        tail = "html[data-theme='dark'] {\n  --panel-bg: #0f172a;\n}\n"
        scratch_tree(tmp_path, css(tail=tail), snapshot(mod))
        result = run_gate(tmp_path)
        assert result.returncode == 1
        assert "--panel-bg is redeclared outside the first :root block" in (
            result.stderr
        )

    def test_a_snapshot_value_nothing_names_fails(self, mod, tmp_path):
        scratch_tree(tmp_path, css(), {**snapshot(mod), "--fg-ok": "#34d399"})
        result = run_gate(tmp_path)
        assert result.returncode == 1
        assert "--fg-ok is in the snapshot but paired with nothing" in result.stderr

    def test_a_paired_figura_token_absent_from_the_snapshot_fails(self, mod, tmp_path):
        values = snapshot(mod)
        del values["--fg-line"]
        scratch_tree(tmp_path, css(), values)
        result = run_gate(tmp_path)
        assert result.returncode == 1
        assert "--fg-line missing from the snapshot" in result.stderr

    def test_an_unpaired_token_the_site_does_not_declare_fails(self, mod, tmp_path):
        stylesheet = css().replace("  --font-mono: ui-monospace, monospace;\n", "")
        scratch_tree(tmp_path, stylesheet, snapshot(mod))
        result = run_gate(tmp_path)
        assert result.returncode == 1
        assert "--font-mono is listed unpaired but not declared" in result.stderr

    def test_a_record_without_a_snapshot_fails_rather_than_passing(self, tmp_path):
        scratch_tree(tmp_path, css(), None)
        result = run_gate(tmp_path)
        assert result.returncode == 1
        assert "no token snapshot" in result.stderr


def test_the_repository_site_tokens_match_figura():
    """The real gate over the whole tree, as every sibling module ends."""
    result = run_gate(REPO_ROOT)
    assert result.returncode == 0, result.stderr
