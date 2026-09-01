"""The retrieval metrics, and the ground-truth set's own integrity.

Two halves, and the second is the one that could not exist without the committed
manifest.

The metrics are arithmetic and are tested as arithmetic: hand-worked rankings
where the right answer can be read off by eye. A metric nobody checks is a metric
that can be silently wrong in the direction that flatters, and recall and MRR are
both easy to get subtly wrong -- an off-by-one in the rank, a division by the
wrong denominator.

The set's labels are validated against `kb-manifest.json`, which lists all 4,260
corpus files by path. That means a mistyped page in a label fails here, in the
default suite, in a checkout with no corpus at all -- which is the only place the
error is cheap. Left to the desktop it would surface as a question that can never
be answered, indistinguishable from a retrieval failure.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from logtriage.evals.metrics import (
    CUTOFFS,
    GROUND_TRUTH,
    Question,
    load,
    rank_of,
    score,
    slices,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MANIFEST = PROJECT_ROOT / "kb-manifest.json"

# Provenances a label may claim. Closed, because the point of the field is that
# a later reader can audit how a label was decided, and a free-form string is
# not auditable.
SOURCE_PREFIXES = ("detector:", "log-id:", "manual-ref:")


# Module scope, not class scope: a class-scoped fixture defined as an instance
# method is deprecated in pytest, and both of these read a file that does not
# change during a run.
@pytest.fixture(scope="module")
def questions() -> list[Question]:
    return load()


@pytest.fixture(scope="module")
def corpus_files() -> set[str]:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    return {entry["path"] for entry in manifest["files"]}


def _question(pages: tuple[str, ...] = ("a.txt",), overlap: bool = True) -> Question:
    return Question(
        id="q", question="q?", pages=pages, source="detector:x", lexical_overlap=overlap
    )


class TestRankOf:
    def test_the_first_acceptable_page_wins(self) -> None:
        question = _question(("b.txt",))
        assert rank_of(question, ["a.txt", "b.txt", "c.txt"]) == 2

    def test_any_listed_page_counts(self) -> None:
        """`pages` is a set of acceptable answers, not a ranking: several manual
        pages legitimately answer one question."""
        question = _question(("x.txt", "c.txt"))
        assert rank_of(question, ["a.txt", "b.txt", "c.txt"]) == 3

    def test_repeated_pages_score_their_first_appearance(self) -> None:
        """A page is many chunks, so one page can occupy several hits. A reader
        opening the top result gets the page, not the chunk."""
        question = _question(("b.txt",))
        assert rank_of(question, ["b.txt", "b.txt", "b.txt"]) == 1

    def test_absent_is_none_not_zero(self) -> None:
        """Zero would average into MRR as a real rank."""
        assert rank_of(_question(("z.txt",)), ["a.txt", "b.txt"]) is None

    def test_no_hits_at_all(self) -> None:
        assert rank_of(_question(), []) is None


class TestScore:
    def test_recall_counts_at_each_cutoff(self) -> None:
        # Ranks 1, 3, 11 and a miss: one within 1, two within 5, three within 10
        # would be wrong -- 11 is outside 10.
        result = score({"a": 1, "b": 3, "c": 11, "d": None})
        assert result.questions == 4
        assert result.recall[1] == 0.25
        assert result.recall[5] == 0.5
        assert result.recall[10] == 0.5

    def test_mrr_is_the_mean_over_every_question_not_only_the_hits(self) -> None:
        """Dividing by the hits would let a retriever that answers one question
        perfectly and misses nine score 1.0."""
        result = score({"a": 1, "b": None, "c": None, "d": None})
        assert result.mrr == pytest.approx(0.25)

    def test_mrr_ignores_a_hit_past_the_cutoff(self) -> None:
        """A hit at rank 200 contributes 0.005 and is not a result anybody acts
        on; counting it would make a useless ranking look slightly good."""
        assert score({"a": 200}).mrr == 0.0

    def test_a_perfect_ranking(self) -> None:
        result = score({"a": 1, "b": 1})
        assert result.mrr == 1.0
        assert all(result.recall[c] == 1.0 for c in CUTOFFS)

    def test_an_empty_set_does_not_divide_by_zero(self) -> None:
        assert score({}).questions == 0

    def test_ranks_are_kept_so_a_caller_can_name_the_misses(self) -> None:
        """A score that moved is worth less than the item that moved it."""
        assert score({"a": 1, "b": None}).ranks == {"a": 1, "b": None}


class TestSlices:
    def test_the_two_halves_partition_the_set(self) -> None:
        questions = [_question(overlap=True), _question(overlap=False)]
        got = slices(questions)
        assert len(got["all"]) == 2
        assert len(got["no-overlap"]) == 1
        assert len(got["overlap"]) == 1


class TestGroundTruthIntegrity:
    """The committed set, checked against the committed manifest.

    Every assertion here is about the file rather than about retrieval, and all
    of it runs with no corpus and no services.
    """

    def test_the_set_loads(self, questions) -> None:
        assert len(questions) >= 30, "too small for a difference to mean anything"

    def test_every_labelled_page_exists_in_the_corpus(
        self, questions, corpus_files
    ) -> None:
        """A mistyped page is a question that can never be answered, and on the
        desktop it would be indistinguishable from a retrieval failure."""
        missing = {
            (question.id, page)
            for question in questions
            for page in question.pages
            if page not in corpus_files
        }
        assert not missing, (
            f"labels naming pages not in the manifest: {sorted(missing)}"
        )

    def test_ids_are_unique(self, questions) -> None:
        ids = [question.id for question in questions]
        assert len(set(ids)) == len(ids)

    def test_every_item_records_its_provenance(self, questions) -> None:
        """`source` is what lets a later reader audit how a label was decided."""
        for question in questions:
            assert question.source.startswith(SOURCE_PREFIXES), question.id

    def test_the_no_overlap_slice_is_substantial(self, questions) -> None:
        """It is the headline slice. A set that is mostly vocabulary matches
        cannot distinguish a semantic embedder from word overlap, which is the
        one comparison this set exists to make."""
        no_overlap = [q for q in questions if not q.lexical_overlap]
        assert len(no_overlap) >= len(questions) // 3

    def test_every_detector_kind_is_represented(self, questions) -> None:
        """The questions come from the task, and the task is what the detectors
        find. A kind with no question is a diagnosis the corpus was never
        checked against."""
        from logtriage.mongod.diagnostics import DETECTORS

        assert DETECTORS, "no detectors to cover"
        covered = {
            question.source.split(":", 1)[1]
            for question in questions
            if question.source.startswith("detector:")
        }
        expected = {
            "collscan",
            "scan_ratio",
            "connection_refused",
            "connection_churn",
            "election",
            "heartbeat_failure",
            "index_build_abort",
            "severity_spike",
        }
        assert expected <= covered, f"no questions for: {sorted(expected - covered)}"

    def test_every_item_carries_a_note(self, questions) -> None:
        """Why a page answers a question is the part a second reader needs to
        disagree with it."""
        for question in questions:
            assert question.note.strip(), question.id


class TestLoaderRejectsMalformedSets:
    """The loader is strict about its own file because nothing downstream is."""

    def test_a_missing_field_is_named(self, tmp_path) -> None:
        path = tmp_path / "bad.jsonl"
        path.write_text('{"id": "x", "question": "q?"}\n', encoding="utf-8")
        with pytest.raises(ValueError, match="missing"):
            load(path)

    def test_a_label_with_no_acceptable_page_is_rejected(self, tmp_path) -> None:
        """It can never be answered, so it would drag every score down by one
        question for no reason anyone could find."""
        path = tmp_path / "bad.jsonl"
        path.write_text(
            json.dumps(
                {
                    "id": "x",
                    "question": "q?",
                    "pages": [],
                    "source": "detector:x",
                    "lexical_overlap": True,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="no acceptable page"):
            load(path)

    def test_a_duplicate_id_is_rejected(self, tmp_path) -> None:
        row = json.dumps(
            {
                "id": "x",
                "question": "q?",
                "pages": ["a.txt"],
                "source": "detector:x",
                "lexical_overlap": True,
            }
        )
        path = tmp_path / "bad.jsonl"
        path.write_text(f"{row}\n{row}\n", encoding="utf-8")
        with pytest.raises(ValueError, match="duplicate id"):
            load(path)

    def test_malformed_json_names_its_line(self, tmp_path) -> None:
        path = tmp_path / "bad.jsonl"
        path.write_text("{not json\n", encoding="utf-8")
        with pytest.raises(ValueError, match=":1:"):
            load(path)

    def test_an_empty_file_is_rejected(self, tmp_path) -> None:
        """Zero questions would score a perfect nothing and report success."""
        path = tmp_path / "empty.jsonl"
        path.write_text("\n\n", encoding="utf-8")
        with pytest.raises(ValueError, match="no questions"):
            load(path)

    def test_the_committed_set_is_where_the_module_says(self) -> None:
        assert GROUND_TRUTH.is_file()
