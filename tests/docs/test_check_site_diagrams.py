"""Tests for scripts/docs/check_site_diagrams.py.

The gate holds the figura SVG exports copied under site/static/img/diagrams/
to the provenance record beside them. Three findings carry it: a copied file
the record does not name, a recorded file that is absent, and a file whose
sha256 differs from the one recorded -- the last is the hand edit that
commit 02a546b made to tier-loop.svg. Around them sit the guards: a record
that is missing, malformed, or lists nothing fails rather than passing.

Fixtures are scratch trees rather than the real repository, so the tests do not
turn red the next time a diagram is re-exported.

The module is loaded by path; scripts/ is not an importable package.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "docs" / "check_site_diagrams.py"

COMMIT = "f237649ad7dafa230a0158ed7e253e5a7fb244ff"
LOOP = b'<svg xmlns="http://www.w3.org/2000/svg"><text>train, fine-tune</text></svg>\n'
GRAINS = b'<svg xmlns="http://www.w3.org/2000/svg"><text>grains</text></svg>\n'


@pytest.fixture(scope="module")
def mod():
    spec = importlib.util.spec_from_file_location("check_site_diagrams", SCRIPT)
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


def entry(export: str, content: bytes) -> dict[str, str]:
    return {"export": export, "sha256": hashlib.sha256(content).hexdigest()}


def scratch_tree(
    root: Path, copies: dict[str, bytes], files: dict[str, dict[str, str]] | None
) -> Path:
    """A diagrams directory holding the given copies and, unless None, a record."""
    directory = root / "site" / "static" / "img" / "diagrams"
    directory.mkdir(parents=True)
    for name, content in copies.items():
        (directory / name).write_bytes(content)
    if files is not None:
        record = {
            "repository": "https://github.com/alejandroJaramillo87/figura",
            "commit": COMMIT,
            "files": files,
        }
        (directory / "figura.json").write_text(json.dumps(record), encoding="utf-8")
    return directory


FULL_RECORD = {
    "tier-loop.svg": entry("exports/ai-experiments/tier-loop.svg", LOOP),
    "grains.svg": entry("exports/rust-harness/grains.svg", GRAINS),
}
FULL_COPIES = {"tier-loop.svg": LOOP, "grains.svg": GRAINS}


class TestGate:
    """End to end, over a scratch tree."""

    def test_copies_matching_the_record_pass(self, tmp_path):
        scratch_tree(tmp_path, FULL_COPIES, FULL_RECORD)
        result = run_gate(tmp_path)
        assert result.returncode == 0, result.stderr
        assert "2 recorded files" in result.stderr

    def test_a_hand_edited_copy_fails_and_says_where_to_fix_it(self, tmp_path):
        edited = LOOP.replace(b"fine-tune", b"distill")
        scratch_tree(tmp_path, {**FULL_COPIES, "tier-loop.svg": edited}, FULL_RECORD)
        result = run_gate(tmp_path)
        assert result.returncode == 1
        assert "tier-loop.svg: sha256" in result.stderr
        assert "do not hand-edit" in result.stderr
        assert "fix the diagram in figura" in result.stderr
        assert "re-export" in result.stderr
        assert "re-copy" in result.stderr
        assert "update the record" in result.stderr

    def test_a_copy_the_record_does_not_name_fails(self, tmp_path):
        scratch_tree(tmp_path, {**FULL_COPIES, "extra.svg": GRAINS}, FULL_RECORD)
        result = run_gate(tmp_path)
        assert result.returncode == 1
        assert "extra.svg: not in the record" in result.stderr

    def test_a_recorded_file_that_is_absent_fails(self, tmp_path):
        scratch_tree(tmp_path, {"tier-loop.svg": LOOP}, FULL_RECORD)
        result = run_gate(tmp_path)
        assert result.returncode == 1
        assert "grains.svg: recorded but absent" in result.stderr

    def test_an_export_path_naming_another_file_fails(self, tmp_path):
        """A copy is the export under its own name, so the basenames agree."""
        files = {
            **FULL_RECORD,
            "grains.svg": entry("exports/rust-harness/other.svg", GRAINS),
        }
        scratch_tree(tmp_path, FULL_COPIES, files)
        result = run_gate(tmp_path)
        assert result.returncode == 1
        assert "grains.svg: export path" in result.stderr

    def test_a_missing_record_fails(self, tmp_path):
        scratch_tree(tmp_path, FULL_COPIES, None)
        result = run_gate(tmp_path)
        assert result.returncode == 1
        assert "figura.json" in result.stderr

    def test_an_empty_record_fails_rather_than_passing(self, tmp_path):
        scratch_tree(tmp_path, {}, {})
        result = run_gate(tmp_path)
        assert result.returncode == 1
        assert "records no files" in result.stderr

    def test_a_short_commit_fails(self, tmp_path):
        directory = scratch_tree(tmp_path, FULL_COPIES, FULL_RECORD)
        record = json.loads((directory / "figura.json").read_text(encoding="utf-8"))
        record["commit"] = COMMIT[:7]
        (directory / "figura.json").write_text(json.dumps(record), encoding="utf-8")
        result = run_gate(tmp_path)
        assert result.returncode == 1
        assert "full 40-character" in result.stderr


class TestRecord:
    """What the loader accepts."""

    def test_the_record_itself_is_not_a_copy(self, mod, tmp_path):
        directory = scratch_tree(tmp_path, FULL_COPIES, FULL_RECORD)
        assert mod.copies(directory) == ["grains.svg", "tier-loop.svg"]


def test_the_repository_diagram_copies_match_their_record():
    """The real gate over the whole tree, as every sibling module ends."""
    result = run_gate(REPO_ROOT)
    assert result.returncode == 0, result.stderr
