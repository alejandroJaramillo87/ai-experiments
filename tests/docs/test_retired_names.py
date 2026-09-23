"""Tests for scripts/docs/check_retired_names.py.

The guard's whole value is that it has no false positives: it fails only on names
someone deliberately retired. So the tests that matter are the matching-precision
ones -- that a retired name is not found inside a longer name, and above all that
the *replacement* never matches the pattern that retired it. Get that wrong and
the gate can never go green again.

The module is loaded by path, matching tests/docs/test_doc_refs.py --
scripts/ is not an importable package.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[2] / "scripts" / "docs" / "check_retired_names.py"
)


@pytest.fixture(scope="module")
def mod():
    spec = importlib.util.spec_from_file_location("check_retired_names", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestPatternFor:
    """Whole-name matching. Each case is a way the substring test would misfire."""

    @pytest.mark.parametrize(
        ("text", "why"),
        [
            ("see `docker-compose.yaml` for ports", "backticked in prose"),
            ("-f docker-compose.yaml up", "a command argument"),
            ("defined in docker-compose.yaml.", "sentence-ending period"),
        ],
    )
    def test_matches_a_real_use(self, mod, text, why):
        assert mod.pattern_for("docker-compose.yaml").search(text), why

    @pytest.mark.parametrize(
        ("text", "why"),
        [
            ("docker-compose.inference.yaml", "the replacement must never match"),
            ("docker-compose.training.yaml", "a sibling compose file"),
            ("docker-compose*.yml", "a glob naming the whole family"),
            ("my-docker-compose.yaml", "a longer name ending in the retired one"),
            ("docker-compose.yaml-old", "a longer name starting with it"),
            ("docker-compose.yml", "a different extension"),
            (
                "docker/llama-cpu/docker-compose.yaml",
                "the same name moved under a directory",
            ),
            ("./docker-compose.yaml", "a leading ./ reads as a directory prefix"),
        ],
    )
    def test_does_not_match_a_lookalike(self, mod, text, why):
        assert not mod.pattern_for("docker-compose.yaml").search(text), why

    @pytest.mark.parametrize(
        ("old", "text", "hit"),
        [
            ("FT.1", "closes when FT.1 prints the report", True),
            ("FT.1", "FT.1's first attempt", True),
            ("FT.1", "depends on FT.14 for the rate", False),
            ("FT.1", "FT.10 and FT.11 follow", False),
            ("I.9", "the capture I.9 runs", True),
            ("I.9", "I.9b sits beside it", False),
            ("I.9", "I.9c and I.9d scale it up", False),
        ],
    )
    def test_a_slugged_entry_id_is_bounded_by_its_digits(self, mod, old, text, hit):
        """The entry-id rows: `FT.1` must not fire on `FT.14`, nor `I.9` on `I.9b`.

        The same trailing guard that keeps `docker-compose.yaml` off
        `docker-compose.yaml-old` does it, since a digit and a letter are both
        word characters; this pins that the rows added on 2026-09-19 rely on it.
        """
        assert bool(mod.pattern_for(old).search(text)) is hit, text


class TestRetiredTable:
    def test_every_replacement_is_clean_under_its_own_pattern(self, mod):
        """A replacement that matches its own retirement pattern is unfixable."""
        for old, (replacement, _reason) in mod.RETIRED.items():
            assert not mod.pattern_for(old).search(replacement), (
                f"{old} -> {replacement} would re-trigger the guard"
            )

    def test_every_entry_states_a_reason(self, mod):
        for old, (replacement, reason) in mod.RETIRED.items():
            assert replacement and reason.strip(), f"{old} has no stated reason"

    def test_every_slugged_entry_id_maps_to_a_slug(self, mod):
        """The rows for the 2026-09-19 id rename point at slug-shaped ids only."""
        import re

        slug = re.compile(r"^[A-Z]+\.[a-z][a-z0-9]*(?:-[a-z0-9]+)*$")
        for old, new in mod.SLUGGED_ENTRY_IDS.items():
            assert slug.match(new), (old, new)
            assert mod.RETIRED[old][0] == new

    def test_no_retired_name_is_also_a_replacement(self, mod):
        """Retiring A to B and B to C would make the gate unsatisfiable."""
        replacements = {r for r, _ in mod.RETIRED.values()}
        assert not (replacements & set(mod.RETIRED))


class TestFindRetired:
    def test_reports_the_line_the_use_sits_on(self, mod):
        patterns = {"old.yaml": mod.pattern_for("old.yaml")}
        text = "intro\n\nsee old.yaml here\n"
        assert mod.find_retired(text, patterns) == [(3, "old.yaml")]

    def test_reports_nothing_for_clean_text(self, mod):
        patterns = {"old.yaml": mod.pattern_for("old.yaml")}
        assert mod.find_retired("nothing to see\n", patterns) == []


class TestWantsScan:
    @pytest.mark.parametrize(
        "name",
        ["README.md", "serving.py", "preflight.sh", "quality.mk", "Makefile"],
    )
    def test_text_files_are_scanned(self, mod, name):
        assert mod.wants_scan(name)

    @pytest.mark.parametrize("name", ["model.gguf", "logo.png", "archive.tar.gz"])
    def test_binaries_are_not(self, mod, name):
        assert not mod.wants_scan(name)

    def test_dockerfiles_are_scanned_despite_the_suffix_convention(self, mod):
        # Dockerfile.llama-gpu names the compose file in a comment; its suffix
        # is the image name, not a file type, so the extension test misses it.
        assert mod.wants_scan("Dockerfile.llama-gpu")


class TestIsExempt:
    @pytest.mark.parametrize(
        "rel",
        [
            "docs/training/investigations/d7_train_serve_investigation.md",
            "docs/training/investigations/unified_cli_investigation.md",
            "scripts/docs/check_retired_names.py",
        ],
    )
    def test_frozen_records_and_the_table_itself_are_exempt(self, mod, rel):
        assert mod.is_exempt(rel)

    @pytest.mark.parametrize(
        "rel",
        [
            "README.md",
            "make/ops.mk",
            "common/serving.py",
            # Exempt until 2026-08-08. The runbooks are executed, so a retired
            # name in one is a command that fails in the operator's hands.
            "docs/verification/interp.md",
            "docs/verification/known-bugs.md",
        ],
    )
    def test_living_files_are_in_scope(self, mod, rel):
        assert not mod.is_exempt(rel)


def test_the_repository_uses_no_retired_name(mod, repo_root):
    """The gate's own assertion, so a failure names the file and line."""
    patterns = {old: mod.pattern_for(old) for old in mod.RETIRED}
    hits = []
    for path in mod.iter_files(repo_root):
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        for lineno, old in mod.find_retired(text, patterns):
            replacement, _ = mod.RETIRED[old]
            rel = path.relative_to(repo_root)
            hits.append(f"{rel}:{lineno}: {old} -> {replacement}")
    assert not hits, "retired names are still in use:\n" + "\n".join(hits)
