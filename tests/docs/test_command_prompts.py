"""Tests for scripts/docs/check_command_prompts.py.

The risk is the same as the other lexical doc gates: not missing a prompt but
inventing one. A `$` opens a shell variable, a `#` opens a comment, and both
appear at the start of lines in blocks that are not commands at all. Each test
names the construct that must, or must not, be read as a prompt.

The module is loaded by path; scripts/ is not an importable package.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "docs"
    / "check_command_prompts.py"
)


@pytest.fixture(scope="module")
def mod():
    spec = importlib.util.spec_from_file_location("check_command_prompts", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def lines(mod, text):
    return [n for n, _ in mod.find_prompts(text)]


class TestPrompts:
    def test_a_user_prompt_in_a_bash_block_is_reported(self, mod):
        assert lines(mod, "```bash\n$ make quality\n```") == [2]

    def test_every_shell_language_tag_is_read(self, mod):
        for tag in ("bash", "sh", "shell", "zsh", "Bash"):
            assert lines(mod, f"```{tag}\n$ ls\n```") == [2], tag

    def test_a_console_transcript_keeps_its_prompt(self, mod):
        assert mod.find_prompts("```console\n$ ls\na.md\n```") == []

    def test_indented_and_tilde_fences_are_read(self, mod):
        assert lines(mod, "  ~~~bash\n  $ ls\n  ~~~") == [2]

    def test_reports_the_line_the_prompt_sits_on(self, mod):
        text = "intro\n\n```bash\ncd x\n$ make it\n```\n"
        assert mod.find_prompts(text) == [(5, "$ make it")]


class TestNotPrompts:
    @pytest.mark.parametrize(
        ("text", "why"),
        [
            ("```bash\necho $HOME\n```", "a variable inside a command"),
            ('```bash\ncargo run -- --model "$MODEL"\n```', "a quoted variable"),
            ("```bash\nDIGEST=$(sha256sum f)\n```", "a substitution"),
            ("```bash\n# the next line rebuilds\nmake x\n```", "a comment"),
            ("```bash\n# 1. clean preflight\nmake x\n```", "a numbered comment"),
            ("```bash\n# nvidia-smi -lgc 2400\n```", "a commented-out command"),
            ("```python\n$ not python anyway\n```", "a non-shell block"),
            ("```\n$ make install\n```", "an untagged block is output"),
            ("```make\n$(CC) -o x\n```", "a make variable"),
            ("prose with $ in it\n", "prose"),
            ("```bash\n$\n```", "a bare dollar"),
        ],
    )
    def test_ignores(self, mod, text, why):
        assert mod.find_prompts(text) == [], why


class TestObservedExemption:
    def test_an_observed_block_is_not_read(self, mod):
        text = "## RH.1 Entry\n\nObserved:\n\n```bash\n$ tail -2 runs.jsonl\n```\n"
        assert mod.find_prompts(text) == []

    def test_a_bold_observed_marker_counts(self, mod):
        text = "## X\n**Observed (2026-08-14)**\n```bash\n$ ls\n```\n"
        assert mod.find_prompts(text) == []

    def test_the_exemption_ends_at_the_next_heading(self, mod):
        text = (
            "## RH.1\nObserved:\n```bash\n$ ls\n```\n"
            "## RH.2\nCommand:\n```bash\n$ ls\n```\n"
        )
        assert lines(mod, text) == [9]

    def test_a_command_block_before_observed_is_still_read(self, mod):
        text = "## RH.1\n```bash\n$ ls\n```\nObserved:\n```bash\n$ ls\n```\n"
        assert lines(mod, text) == [3]


class TestMain:
    def test_a_clean_tree_exits_zero(self, mod, tmp_path, monkeypatch, capsys):
        (tmp_path / "a.md").write_text("```bash\nmake x\n```\n")
        monkeypatch.setattr("sys.argv", ["x", "--root", str(tmp_path)])
        assert mod.main() == 0
        assert "[OK]" in capsys.readouterr().err

    def test_a_prompt_fails_and_names_the_line(
        self, mod, tmp_path, monkeypatch, capsys
    ):
        (tmp_path / "a.md").write_text("```bash\n$ make x\n```\n")
        monkeypatch.setattr("sys.argv", ["x", "--root", str(tmp_path)])
        assert mod.main() == 1
        err = capsys.readouterr().err
        assert "a.md:2" in err and "Solution" in err

    def test_pruned_directories_are_skipped(self, mod, tmp_path, monkeypatch):
        (tmp_path / "target").mkdir()
        (tmp_path / "target" / "a.md").write_text("```bash\n$ make x\n```\n")
        monkeypatch.setattr("sys.argv", ["x", "--root", str(tmp_path)])
        assert mod.main() == 0

    @pytest.mark.parametrize(
        "rel", ["docs/x/investigations/a.md", "docs/x/d4_investigation.md"]
    )
    def test_frozen_records_are_exempt(self, mod, rel):
        assert mod.is_exempt(Path(rel))

    def test_a_living_document_is_not_exempt(self, mod):
        assert not mod.is_exempt(Path("docs/verification/rust-harness/rust-harness.md"))


def test_the_repository_has_no_prompted_command_blocks(mod, monkeypatch, capsys):
    """The full-tree gate, as make quality runs it."""
    monkeypatch.setattr("sys.argv", ["x"])
    assert mod.main() == 0, capsys.readouterr().err
