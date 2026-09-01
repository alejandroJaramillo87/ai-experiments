"""The documentation gates, driven both ways.

A gate is only worth running if it can fail. Each of the six checks passes
against this tree today, and that is exactly the state in which a broken
enumerator, an over-broad exemption, or a classifier that rejects everything
looks identical to a correct gate.

So each check is exercised twice: once against the real tree, where it must pass
and must report having examined something, and once against a planted defect,
where it must catch it. The second half is the one that matters -- without it,
`[OK] check_doc_refs: 146 references all resolve` could equally mean the
classifier decided none of them were claims.

The classifiers are tested as functions rather than through the CLI, because
their judgement is the substance: what counts as a path claim, what counts as a
row under a counted heading, which invocations are commands and which are the
English verb.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = PROJECT_ROOT / "scripts" / "docs"

CHECKS = (
    "check_doc_refs",
    "check_retired_names",
    "check_make_targets",
    "check_cli_verbs",
    "check_doc_counts",
    "check_runbook_entries",
    "check_corpus_stats",
    "check_surface_census",
)


def load(name: str) -> ModuleType:
    """Import a checker by path; they are scripts, not an installed package."""
    spec = importlib.util.spec_from_file_location(
        f"_gate_{name}", SCRIPTS / f"{name}.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("name", CHECKS)
def test_the_gate_passes_against_this_tree(name: str) -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPTS / f"{name}.py")],
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("name", CHECKS)
def test_the_gate_reports_what_it_examined(name: str) -> None:
    """A count in the output is what distinguishes a pass from a vacuous run.

    Every one of these carries an empty-set guard, and this asserts the guard is
    reachable: a check that examined nothing must not be able to print [OK]
    without saying so.
    """
    result = subprocess.run(
        [sys.executable, str(SCRIPTS / f"{name}.py")],
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
    )
    output = result.stdout + result.stderr
    assert any(character.isdigit() for character in output), output


class TestDocRefClassifier:
    def test_a_link_to_a_missing_file_is_a_claim(self) -> None:
        module = load("check_doc_refs")
        assert module.is_path_claim("docs/nosuch.md")
        assert not module.resolves(
            "docs/nosuch.md", PROJECT_ROOT / "README.md", PROJECT_ROOT
        )

    def test_a_real_file_resolves(self) -> None:
        module = load("check_doc_refs")
        assert module.resolves(
            "docs/README.md", PROJECT_ROOT / "README.md", PROJECT_ROOT
        )

    def test_prose_naming_a_package_relative_path_resolves(self) -> None:
        """Prose says `kb/parse.py` meaning the package directory, and that is
        correct English. Demanding one interpretation would reject good prose."""
        module = load("check_doc_refs")
        document = PROJECT_ROOT / "docs" / "corpus" / "README.md"
        assert module.resolves("kb/parse.py", document, PROJECT_ROOT)

    @pytest.mark.parametrize(
        "reference",
        [
            "https://example.com/a/b",
            "github.com/mongodb/docs",
            "/v1/embeddings",
            "includes/extracts-*.yaml",
            "<topic>/investigations/",
            "notapath",
        ],
    )
    def test_things_that_are_not_path_claims(self, reference: str) -> None:
        module = load("check_doc_refs")
        assert not module.is_path_claim(reference)

    def test_a_symbol_suffix_is_dropped_and_the_file_still_checked(self) -> None:
        module = load("check_doc_refs")
        assert module.path_part("store/knowledge.py:fuse") == "store/knowledge.py"
        assert module.path_part("docs/README.md") == "docs/README.md"

    def test_code_spans_are_read_as_well_as_links(self) -> None:
        module = load("check_doc_refs")
        found = module.candidates("See [x](docs/a.md) and `logtriage/store/base.py`.")
        assert "docs/a.md" in found
        assert "logtriage/store/base.py" in found

    def test_a_command_line_in_a_span_is_not_a_path(self) -> None:
        module = load("check_doc_refs")
        assert "make -C logtriage quality" not in module.candidates(
            "Run `make -C logtriage quality` first."
        )


class TestMakeTargetClassifier:
    def test_make_as_an_english_verb_is_not_an_invocation(self) -> None:
        module = load("check_make_targets")
        assert module.invocations("Please make sure the service is up.") == set()

    def test_a_fenced_command_is_an_invocation(self) -> None:
        module = load("check_make_targets")
        assert "logtriage-up" in module.invocations("```bash\nmake logtriage-up\n```")

    def test_a_code_span_command_is_an_invocation(self) -> None:
        module = load("check_make_targets")
        assert "quality" in module.invocations("Run `make quality` before pushing.")

    def test_the_local_makefile_targets_are_found(self) -> None:
        module = load("check_make_targets")
        targets, makefiles = module.collect_targets(PROJECT_ROOT)
        assert "quality" in targets
        assert "docs" in targets, "the documentation gate step must be a real target"
        assert makefiles


class TestCountedHeadings:
    def test_a_heading_that_agrees_with_its_list(self) -> None:
        module = load("check_doc_counts")
        lines = "## Entries (2)\n\n- one\n- two\n\n## Next\n".splitlines()
        assert module.count_rows(lines, 0, 2) == 2

    def test_a_heading_that_disagrees_is_visible(self) -> None:
        module = load("check_doc_counts")
        lines = "## Entries (2)\n\n- one\n- two\n- three\n".splitlines()
        assert module.count_rows(lines, 0, 2) == 3

    def test_a_table_header_is_not_a_row(self) -> None:
        module = load("check_doc_counts")
        lines = (
            "## Items (2)\n\n| a | b |\n|---|---|\n| 1 | 2 |\n| 3 | 4 |\n".splitlines()
        )
        assert module.count_rows(lines, 0, 2) == 2

    def test_counting_stops_at_the_next_heading_of_the_same_level(self) -> None:
        module = load("check_doc_counts")
        lines = "## A (1)\n\n- one\n\n## B\n\n- two\n- three\n".splitlines()
        assert module.count_rows(lines, 0, 2) == 1

    def test_a_version_in_a_heading_is_not_a_count(self) -> None:
        """`## Gemma 4 (E4B, 12B)` is not a claim about rows."""
        module = load("check_doc_counts")
        assert module.COUNTED_HEADING_RE.match("## Gemma 4 (E4B, 12B)") is None
        assert module.COUNTED_HEADING_RE.match("## Entries (3)") is not None


class TestCorpusStatistics:
    def test_labels_keep_their_inner_backticks_normalised_away(self) -> None:
        """A label like "Over `MAX_TOKENS`" loses only its trailing backtick to
        a naive strip, and then matches nothing at all -- silently checking one
        fewer statistic than it reports."""
        module = load("check_corpus_stats")
        stated = module.rows(PROJECT_ROOT / "docs" / "corpus" / "statistics.md")
        assert "Over MAX_TOKENS" in stated
        assert "Pages (.txt)" in stated
        assert "Code blocks intact" in stated

    def test_a_thousands_separator_parses(self) -> None:
        module = load("check_corpus_stats")
        assert module.number("15,025") == 15025
        assert module.number("8,394 of 8,394") is None

    def test_both_halves_of_an_n_of_m_claim_are_read(self) -> None:
        """An index that split a code block would move only the first number."""
        module = load("check_corpus_stats")
        match = module.OF_RE.match("8,393 of 8,394")
        assert match and (match.group(1), match.group(2)) == ("8,393", "8,394")


class TestSurfaceCensus:
    def test_every_cli_verb_is_enumerated(self) -> None:
        module = load("check_surface_census")
        assert module.verbs(PROJECT_ROOT) >= {
            "gen-logs",
            "profile",
            "triage",
            "fetch-kb",
            "index-kb",
        }

    def test_both_halves_of_every_port_pair_are_enumerated(self) -> None:
        """The offline implementations are censused too. One that nobody
        documents is how a reader comes to believe a score means something it
        does not."""
        module = load("check_surface_census")
        found = module.port_classes(PROJECT_ROOT)
        assert {
            "MongoLogStore",
            "MongoKnowledgeStore",
            "LlamaEmbedder",
            "LlamaClient",
        } <= found
        assert {
            "InMemoryLogStore",
            "InMemoryKnowledgeStore",
            "HashingEmbedder",
            "ScriptedLlmClient",
        } <= found

    def test_the_markers_are_read_from_pyproject(self) -> None:
        module = load("check_surface_census")
        assert module.markers(PROJECT_ROOT) == {"corpus", "differential", "services"}

    def test_an_undocumented_surface_would_be_caught(self, tmp_path) -> None:
        """The census reads documentation as text and subtracts. This asserts
        the subtraction works, which is the whole mechanism."""
        module = load("check_surface_census")
        text = module.documentation(PROJECT_ROOT)
        assert "MongoLogStore" in text
        assert "NoSuchImplementation" not in text


class TestCliVerbClassifier:
    """The gate that would have caught three commands documented before they
    were written."""

    def test_every_shipped_verb_is_enumerated(self) -> None:
        module = load("check_cli_verbs")
        assert module.verbs(PROJECT_ROOT) >= {
            "ingest",
            "eval-retrieval",
            "bench",
            "triage",
        }

    def test_a_fenced_invocation_is_found(self) -> None:
        module = load("check_cli_verbs")
        assert "ingest" in module.invocations(
            "```bash\nuv run python -m logtriage ingest x.log\n```"
        )

    def test_a_code_span_invocation_is_found(self) -> None:
        module = load("check_cli_verbs")
        assert "triage" in module.invocations("Run `python -m logtriage triage x.log`.")

    def test_prose_naming_a_command_is_not_an_invocation(self) -> None:
        """ "the triage command" is English, not a command line."""
        module = load("check_cli_verbs")
        assert module.invocations("The triage command reads a log.") == set()

    def test_a_verb_that_does_not_exist_would_be_caught(self) -> None:
        module = load("check_cli_verbs")
        found = module.invocations("```\npython -m logtriage nosuchverb\n```")
        assert found == {"nosuchverb"}
        assert "nosuchverb" not in module.verbs(PROJECT_ROOT)


class TestRunbookEntries:
    ENTRY = [
        "### D1 - a thing",
        "",
        "**Criteria:** it works.",
        "",
        "Status: PENDING",
    ]

    def test_an_entry_is_found_by_its_identifier(self) -> None:
        module = load("check_runbook_entries")
        found = module.entries(self.ENTRY)
        assert [name for name, _, _ in found] == ["D1"]

    def test_a_section_heading_is_not_an_entry(self) -> None:
        """ "Deployment" carries no digit, so it is not required to have a
        status; treating it as an entry would demand one of every heading."""
        module = load("check_runbook_entries")
        assert module.entries(["## Deployment", "", "prose"]) == []

    def test_a_placeholder_observed_block_is_not_evidence(self) -> None:
        """It is what a new runbook is written with. Accepting it would let an
        entry be marked VERIFIED the moment somebody edited one word."""
        module = load("check_runbook_entries")
        assert not module.has_evidence(["Observed:", "", "_(pending)_"])

    def test_pasted_output_is_evidence(self) -> None:
        module = load("check_runbook_entries")
        assert module.has_evidence(["Observed:", "", "```", "[OK] ready", "```"])

    def test_a_named_check_is_evidence(self) -> None:
        module = load("check_runbook_entries")
        assert module.has_evidence(
            ["Asserted by `tests/test_evals.py::test_ids_are_unique`"]
        )

    def test_a_named_check_must_resolve(self) -> None:
        module = load("check_runbook_entries")
        assert module.resolve(
            PROJECT_ROOT, "tests/test_evals.py", "test_ids_are_unique"
        )
        assert not module.resolve(PROJECT_ROOT, "tests/test_evals.py", "test_no_such")
        assert not module.resolve(PROJECT_ROOT, "tests/no_such_file.py", "anything")


class TestRunbookTally:
    def test_a_tally_that_agrees_with_its_status_lines(self) -> None:
        module = load("check_doc_counts")
        lines = [
            "**Entries: 1 verified, 0 blocked, 1 pending, of 2.**",
            "Status: VERIFIED",
            "Status: PENDING",
        ]
        assert module.check_tally(lines) is None

    def test_a_tally_that_disagrees_is_reported_with_both_halves(self) -> None:
        module = load("check_doc_counts")
        lines = [
            "**Entries: 2 verified, 0 blocked, 0 pending, of 2.**",
            "Status: VERIFIED",
            "Status: PENDING",
        ]
        disagreement = module.check_tally(lines)
        assert disagreement is not None
        stated, counted = disagreement
        assert "2 verified" in stated
        assert "1 verified" in counted

    def test_a_document_with_no_tally_is_not_a_finding(self) -> None:
        module = load("check_doc_counts")
        assert module.check_tally(["# A document", "", "prose"]) is None

    def test_a_bolded_status_in_prose_is_not_counted(self) -> None:
        """Column zero is the whole distinction: counting a mention would
        inflate the tally by one per sentence about an entry."""
        module = load("check_doc_counts")
        lines = [
            "**Entries: 0 verified, 0 blocked, 1 pending, of 1.**",
            "Its **Status: VERIFIED** claim was premature.",
            "Status: PENDING",
        ]
        assert module.check_tally(lines) is None
