"""Tests for scripts/docs/check_make_targets.py.

The risk this checker carries is not missing a broken command but inventing one,
because `make` is an English verb and this repository's prose says "make sure",
"make it faster", and "would a unified CLI make sense". Every rejection rule below
answers a false positive measured against the real tree during the check's first
run, so each test names the construct that motivated it.

The module is loaded by path; scripts/ is not an importable package.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[2] / "scripts" / "docs" / "check_make_targets.py"
)


@pytest.fixture(scope="module")
def mod():
    spec = importlib.util.spec_from_file_location("check_make_targets", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestExtractRefs:
    """Which text is read as a command, and which is read as English."""

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("```\n$ make quality\n```", ["quality"]),
            ("Run `make help` first.", ["help"]),
            # Only the first word: the rest is arguments.
            ("```\n$ make train CONFIG=recipes/tiny.yaml\n```", ["train"]),
            ("```\n$ make sae-eval SAE=a CAPTURE=b\n```", ["sae-eval"]),
        ],
    )
    def test_reads_invocations_from_code(self, mod, text, expected):
        assert [name for _, name in mod.extract_refs(text)] == expected

    @pytest.mark.parametrize(
        ("text", "why"),
        [
            ("make sure the service is up before running", "verb, not a command"),
            ("Would a unified CLI make sense here?", "verb, not a command"),
            ("The flight recorder should make runs replayable.", "verb"),
            (
                "Most `make interp-*` targets first run the build.",
                "a family, not a name",
            ),
            ("The `make agent-*` targets are host-side.", "a family, not a name"),
            (
                "Run `make -C rust-harness quality`.",
                "a flag, deliberately not attributed",
            ),
        ],
    )
    def test_ignores_prose_and_families(self, mod, text, why):
        assert mod.extract_refs(text) == [], why

    def test_a_fenced_block_ends_at_its_closing_fence(self, mod):
        text = "```\n$ make quality\n```\nThen make sure it passed.\n"
        assert [name for _, name in mod.extract_refs(text)] == ["quality"]

    def test_reports_the_line_the_invocation_sits_on(self, mod):
        text = "intro\n\nRun `make status` to check.\n"
        assert mod.extract_refs(text) == [(3, "status")]


class TestTargetExists:
    """Direct names, and the pattern rules that stand for a family."""

    def test_a_defined_target_exists(self, mod):
        assert mod.target_exists("quality", {"quality", "install"})

    def test_an_undefined_target_does_not(self, mod):
        assert not mod.target_exists("qualty", {"quality", "install"})

    @pytest.mark.parametrize("name", ["worktree-mytask", "worktree-clean-mytask"])
    def test_a_pattern_rule_covers_its_family(self, mod, name):
        """`worktree-%` is one rule standing for every name a user might type."""
        assert mod.target_exists(name, {"worktree-%"})

    def test_a_pattern_rule_does_not_cover_an_unrelated_name(self, mod):
        assert not mod.target_exists("train", {"worktree-%"})


class TestCollectTargets:
    """Discovery reads the real Makefiles, so these assert against the tree."""

    @pytest.mark.parametrize(
        ("name", "where"),
        [
            ("quality", "make/quality.mk"),
            ("help", "make/help.mk"),
            ("worktree-%", "make/agents.mk, a pattern rule"),
            ("flame", "rust-harness/Makefile, which carries no ## comments"),
            ("test-quiet", "hugepages/Makefile, a third Makefile in the tree"),
        ],
    )
    def test_finds_targets_across_every_makefile(self, mod, repo_root, name, where):
        assert name in mod.collect_targets(repo_root), where

    def test_ignores_special_targets(self, mod, repo_root):
        """.PHONY and its kin are declarations, not targets a reader can run."""
        assert not any(t.startswith(".") for t in mod.collect_targets(repo_root))


class TestKnownAbsent:
    """The allowlist is a set of decisions, so it is checked like one."""

    def test_no_allowlisted_target_actually_exists(self, mod, repo_root):
        """An entry that became real is an entry that should have been deleted."""
        targets = mod.collect_targets(repo_root)
        still_absent = [
            (doc, name)
            for doc, name in mod.KNOWN_ABSENT
            if mod.target_exists(name, targets)
        ]
        assert not still_absent

    def test_every_allowlisted_document_exists(self, mod, repo_root):
        missing = [
            doc for doc, _ in mod.KNOWN_ABSENT if not (repo_root / doc).is_file()
        ]
        assert not missing

    def test_every_exempt_file_exists(self, mod, repo_root):
        missing = [f for f in mod.EXEMPT_FILES if not (repo_root / f).is_file()]
        assert not missing


def test_every_documented_make_target_exists(mod, repo_root):
    """The gate's own assertion, so a failure names the file and line.

    `make quality` runs the script; this runs the same check under pytest so a
    contributor working from the test suite sees it too.
    """
    targets = mod.collect_targets(repo_root)
    assert targets, "found no targets; every reference would be reported as broken"

    docs = mod.iter_docs(repo_root)
    assert docs, "enumerated zero documents; the check would pass having read nothing"

    broken = []
    for doc in docs:
        rel = str(doc.relative_to(repo_root))
        text = doc.read_text(encoding="utf-8")
        for lineno, name in mod.extract_refs(text):
            if (rel, name) in mod.KNOWN_ABSENT:
                continue
            if not mod.target_exists(name, targets):
                broken.append(f"{rel}:{lineno}: make {name}")
    assert not broken, "documented make targets that do not exist:\n" + "\n".join(
        broken
    )
