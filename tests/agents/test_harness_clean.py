"""Tests for the make targets that remove the rust harness's run outputs.

`make harness-clean` at the repository root and `make clean` in
`rust-harness/` are driven against a scratch tree under tmp_path that copies
the two Makefiles and plants every file they claim to remove, beside the files
they must leave: the tracked entries under `fixtures/reference/`, the cargo
build output, and `sessions/` itself. The real repository is never touched.
"""

import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# Every path the targets remove, relative to the scratch root. The list is the
# contract: a producer added to the harness is added here and to the Makefile.
REMOVED = [
    "rust-harness/session.jsonl",
    "rust-harness/session.kv",
    "rust-harness/probe-flags.jsonl",
    "rust-harness/perf.data",
    "rust-harness/perf.data.old",
    "rust-harness/flamegraph.svg",
    "rust-harness/.gdb_history",
    "rust-harness/fixtures/reference/e4b/quick/ppl.json",
    "rust-harness/fixtures/reference/e2b/standard/greedy.json",
    "rust-harness/fixtures/__pycache__/reference_common.cpython-312.pyc",
    "sessions/rust-harness/session.jsonl",
]

# What stays: committed inputs, the build output, and the pre-created sessions/.
KEPT = [
    "rust-harness/fixtures/reference/README.md",
    "rust-harness/fixtures/reference/corpus.txt",
    "rust-harness/fixtures/reference/prompts.txt",
    "rust-harness/fixtures/tiny-gemma4/config.json",
    "rust-harness/target/release/harness",
    "rust-harness/Cargo.toml",
]

STUB_ROOT_MAKEFILE = """\
CYAN :=
GREEN :=
RESET :=
include make/harness.mk
"""


@pytest.fixture
def scratch(tmp_path: Path) -> Path:
    (tmp_path / "make").mkdir()
    shutil.copy(REPO_ROOT / "make" / "harness.mk", tmp_path / "make" / "harness.mk")
    (tmp_path / "Makefile").write_text(STUB_ROOT_MAKEFILE)
    (tmp_path / "rust-harness").mkdir()
    shutil.copy(
        REPO_ROOT / "rust-harness" / "Makefile", tmp_path / "rust-harness" / "Makefile"
    )
    for relative in REMOVED + KEPT:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x")
    (tmp_path / "sessions").mkdir(exist_ok=True)
    return tmp_path


def run_make(directory: Path, *goals: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["make", "-C", str(directory), *goals],
        capture_output=True,
        text=True,
        env={"PATH": "/usr/bin:/bin", "NO_COLOR": "1"},
        timeout=60,
    )


class TestHarnessClean:
    def test_removes_every_run_output_and_keeps_the_rest(self, scratch: Path):
        result = run_make(scratch, "harness-clean")
        assert result.returncode == 0, result.stdout + result.stderr
        for relative in REMOVED:
            assert not (scratch / relative).exists(), relative
        for relative in KEPT:
            assert (scratch / relative).is_file(), relative
        assert (scratch / "sessions").is_dir()
        assert not (scratch / "rust-harness/fixtures/reference/e4b").exists()
        assert not (scratch / "rust-harness/fixtures/__pycache__").exists()

    def test_prints_one_ok_line_per_removed_path(self, scratch: Path):
        result = run_make(scratch, "harness-clean")
        ok_lines = [
            line for line in result.stdout.splitlines() if line.startswith("[OK]")
        ]
        # Seven files, two reference directories, __pycache__, sessions/rust-harness.
        assert len(ok_lines) == 11, result.stdout
        assert "[OK] clean: removed session.kv" in ok_lines
        assert "[OK] clean: removed fixtures/reference/e2b/" in ok_lines
        assert "[OK] harness-clean: removed sessions/rust-harness/" in ok_lines
        assert "[INFO] clean: target/ kept, use cargo clean" in result.stdout

    def test_a_clean_tree_reports_nothing_to_remove(self, scratch: Path):
        run_make(scratch, "harness-clean")
        result = run_make(scratch, "harness-clean")
        assert result.returncode == 0, result.stdout + result.stderr
        assert "[INFO] clean: nothing to remove" in result.stdout
        assert "[INFO] harness-clean: sessions/rust-harness/ absent" in result.stdout
        assert "[OK]" not in result.stdout

    def test_the_workspace_target_leaves_the_container_transcripts(self, scratch: Path):
        result = run_make(scratch / "rust-harness", "clean")
        assert result.returncode == 0, result.stdout + result.stderr
        assert not (scratch / "rust-harness/session.kv").exists()
        assert (scratch / "sessions/rust-harness/session.jsonl").is_file()
