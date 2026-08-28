"""Tests for scripts/docs/check_doc_refs.py.

The checker is what stands between the tree and a document that names a file
nobody moved the reference for. Its risk is not missing a defect but inventing
one: a rule that is slightly too eager reports hundreds of false positives, and
a gate nobody believes is a gate nobody runs. So the classification rules carry
the tests here, each naming the real-world token that motivated it.

The module is loaded by path, matching tests/utils/test_check_py_deps.py --
scripts/ is not an importable package.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "docs" / "check_doc_refs.py"


@pytest.fixture(scope="module")
def mod():
    spec = importlib.util.spec_from_file_location("check_doc_refs", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestLooksLikePath:
    """Which backticked tokens are treated as a claim that a file exists."""

    @pytest.mark.parametrize(
        "token",
        [
            "runtime/src/pyenv.rs",
            "/docker/llama-cpu/Dockerfile.cpu",  # leading slash means repo root
            "docs/rust-harness/",  # multi-segment directory
            "./scripts/agents/guard.sh",
        ],
    )
    def test_accepts_path_claims(self, mod, token):
        assert mod.looks_like_path(token)

    @pytest.mark.parametrize(
        ("token", "why"),
        [
            ("nvidia/cuda:13.0.2-cudnn-devel-ubuntu24.04", "container image tag"),
            ("ghcr.io/astral-sh/uv:0.11.28", "registry reference"),
            ("LD_PRELOAD=/app/libzen5_optimizer.so", "assignment, not a path"),
            ("/mnt/ai-data/models/", "host path outside the repo"),
            ("~/.claude/CLAUDE.md", "home-relative host path"),
            ("../../schemas/", "relative to something stated in prose"),
            ("mlruns/", "single-segment directory created at run time"),
            ("model/", "single-segment directory created at run time"),
            ("docker-compose*.yml", "glob, not one file"),
            ("make quality", "a command; contains a space"),
            ("https://example.com/x.md", "external URL"),
            ("Printer", "no separator at all"),
        ],
    )
    def test_rejects_non_claims(self, mod, token, why):
        assert not mod.looks_like_path(token), why


class TestExtractRefs:
    def test_strips_a_trailing_line_reference(self, mod):
        # `file.rs:53` and `file.rs:53-61` both name the file, not a sibling.
        got = mod.extract_refs("see `runtime/src/pyenv.rs:53` and `a/b.py:1-11`")
        assert [t for _, t in got] == ["runtime/src/pyenv.rs", "a/b.py"]

    def test_reports_the_line_the_reference_sits_on(self, mod):
        got = mod.extract_refs("intro\n\nsee `docs/README.md` here\n")
        assert got == [(3, "docs/README.md")]

    def test_ignores_prose_between_backticks(self, mod):
        assert mod.extract_refs("the `Printer` class and `make quality`") == []


class TestExtractLinks:
    def test_takes_relative_targets_and_drops_anchors(self, mod):
        text = "[a](../agents/tooling-parity.md) [b](https://x.dev) [c](#section)"
        assert [t for _, t in mod.extract_links(text)] == [
            "../agents/tooling-parity.md"
        ]

    def test_strips_a_fragment_from_a_file_link(self, mod):
        got = mod.extract_links("[x](../os/storage-configuration.md#layout)")
        assert [t for _, t in got] == ["../os/storage-configuration.md"]


class TestResolves:
    """Resolution order, exercised against the real tree."""

    def test_a_repo_root_path_resolves(self, mod, repo_root):
        doc = repo_root / "docs" / "README.md"
        assert mod.resolves("AGENTS.md", doc, repo_root)

    def test_a_leading_slash_means_repo_root_not_filesystem_root(self, mod, repo_root):
        doc = repo_root / "docs" / "README.md"
        assert mod.resolves("/AGENTS.md", doc, repo_root)

    def test_a_path_relative_to_the_citing_document_resolves(self, mod, repo_root):
        doc = repo_root / "rust-harness" / "README.md"
        assert mod.resolves("runtime/src/pyenv.rs", doc, repo_root)

    def test_a_path_that_exists_nowhere_does_not_resolve(self, mod, repo_root):
        doc = repo_root / "AGENTS.md"
        assert not mod.resolves(
            "docker/llama-cpu/hugepage_mmap_wrapper.cpp", doc, repo_root
        )


class TestIsExempt:
    @pytest.mark.parametrize(
        "rel",
        [
            "docs/training/investigations/d5_recipe_system_investigation.md",
            "docs/training/investigations/unified_cli_investigation.md",
            "docs/verification/investigations/gemma-4-conformance_investigation.md",
            "common/obs/vendor/uplot/README.md",
        ],
    )
    def test_dated_records_and_vendored_trees_are_exempt(self, mod, rel):
        assert mod.is_exempt(rel)

    @pytest.mark.parametrize(
        "rel",
        [
            "AGENTS.md",
            "docs/README.md",
            "tests/README.md",
            # Exempt until 2026-08-08, when the exemption was measured to be
            # protecting nothing: the one genuinely stale document under this
            # directory is the conformance investigation, which stays exempt
            # through EXEMPT_PARTS and EXEMPT_SUFFIXES either way.
            "docs/verification/bugs/known-bugs.md",
            "docs/verification/training/training-phase2.md",
        ],
    )
    def test_living_documents_are_in_scope(self, mod, rel):
        assert not mod.is_exempt(rel)


def test_the_repository_has_no_dead_documentation_references(mod, repo_root):
    """The gate's own assertion, so a failure names the file and line.

    `make quality` runs the script; this runs the same check under pytest so a
    contributor working from the test suite sees it too.
    """
    dead = []
    for doc in mod.iter_docs(repo_root):
        rel_doc = str(doc.relative_to(repo_root))
        text = doc.read_text(encoding="utf-8")
        refs = mod.extract_refs(text)
        for lineno, token in refs:
            if (rel_doc, token) not in mod.KNOWN_ABSENT and not mod.resolves(
                token, doc, repo_root
            ):
                dead.append(f"{rel_doc}:{lineno}: {token}")
        for lineno, target in mod.extract_links(text):
            if (rel_doc, target) not in mod.KNOWN_ABSENT and not (
                doc.parent / target
            ).exists():
                dead.append(f"{rel_doc}:{lineno}: {target}")
    assert not dead, "documents name paths that do not exist:\n" + "\n".join(dead)
