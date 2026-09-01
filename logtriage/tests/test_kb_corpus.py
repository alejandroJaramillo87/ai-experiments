"""Retrieval round trips against the real MongoDB Database Manual.

Marked `corpus` and deselected by default, because they need a fetched checkout
that is deliberately never committed. A bare `pytest` in a fresh clone is green
without them; `pytest -m corpus` after `logtriage fetch-kb` runs them.

What these can and cannot establish is worth being exact about, because the
temptation is to read any retrieval number as a quality result.

They CAN show that the corpus was read: that includes resolved, that constants
expanded, that chunks carry heading context, that every chunk id a citation
could name resolves, and that a rebuild produces the same index.

They CANNOT show that retrieval is good. The offline embedder hashes tokens, so
its vector space encodes vocabulary overlap and nothing else. Every ranking
assertion below therefore runs on the *text* leg, which is honest lexical
matching, and the vector leg is asserted only to be well-formed. The questions
that need meaning rather than words are listed separately and deliberately not
asserted -- see SEMANTIC_QUESTIONS.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from logtriage.embed.hashing import DEFAULT_SEED, DIMENSIONS, HashingEmbedder
from logtriage.kb.corpus import Corpus
from logtriage.kb.fetch import MANUAL_PATH, default_kb_dir
from logtriage.kb.index import build_index
from logtriage.store.knowledge import InMemoryKnowledgeStore

pytestmark = pytest.mark.corpus


def _manual_root() -> Path | None:
    kb = default_kb_dir()
    for candidate in (kb, kb / "docs" / MANUAL_PATH):
        if (candidate / "source").is_dir():
            return candidate
    return None


@pytest.fixture(scope="module")
def corpus() -> Corpus:
    root = _manual_root()
    if root is None:
        pytest.skip(
            "No manual corpus on disk. Run `python -m logtriage fetch-kb` from "
            "the logtriage directory, or set LOGTRIAGE_KB_SOURCE."
        )
    return Corpus(root)


@pytest.fixture(scope="module")
def indexed(corpus: Corpus):
    """The whole manual, chunked and indexed.

    Built rather than loaded from .kb/index on purpose: these tests are about
    what the current code produces from the current corpus, and reading a
    previously written index would test the reader instead.
    """
    return build_index(corpus, HashingEmbedder())


@pytest.fixture(scope="module")
def documented(corpus: Corpus):
    """Every log id the manual documents, keyed by id."""
    from logtriage.kb.log_ids import documented_ids_from_corpus

    table = documented_ids_from_corpus(corpus.source)
    assert table, "no log ids extracted from reference/log-messages.txt"
    return table


# Questions whose answer is findable by vocabulary. Each names a page that must
# appear in the top ten. Chosen to span the diagnostic surface this tool
# actually needs: the log format, the profiler, query plans, and replication.
LEXICAL_QUESTIONS = [
    ("mongod slow query log message durationMillis", "reference/log-messages.txt"),
    ("planSummary COLLSCAN currentOp output field", "reference/command/currentOp.txt"),
    ("replica set elections primary stepdown", "core/replica-set-elections.txt"),
    (
        "db.setProfilingLevel slowms profiling level",
        "reference/method/db.setProfilingLevel.txt",
    ),
    (
        "explain executionStats keysExamined docsExamined",
        "reference/explain-results.txt",
    ),
    ("serverStatus wiredTiger cache output", "reference/command/serverStatus.txt"),
    ("connPoolStats command output", "reference/command/connPoolStats.txt"),
    ("createIndexes command options", "reference/command/createIndexes.txt"),
]

# Questions a reader would ask in these words, whose answer the manual gives in
# different ones. They are here to record the boundary rather than to pass:
# retrieval for these needs an embedder whose space carries meaning, which is a
# served model and therefore Phase D. Asserting they fail would be worse than
# useless -- it would break the moment a page's wording drifted.
SEMANTIC_QUESTIONS = [
    "how do I find out why my database got slow",
    "the server stopped accepting new connections",
    "queries were fine yesterday and are slow today",
]


class TestCorpusIsRead:
    def test_the_manual_is_a_recent_version(self, corpus: Corpus) -> None:
        major = corpus.version.split(".")[0]
        assert major.isdigit() and int(major) >= 8, (
            f"corpus reports version {corpus.version}; this tool targets the "
            f"8.x log format"
        )

    def test_most_includes_resolve(self, indexed) -> None:
        """Four pages in five include something, so the resolver failing quietly
        would remove most of the manual's explanations while leaving the page
        count untouched."""
        _, stats = indexed
        total = stats.includes_resolved + stats.includes_missing
        assert total > 5_000, "far fewer includes than this corpus is known to have"
        assert stats.includes_resolved / total > 0.70

    def test_every_unresolved_include_has_a_known_reason(self, indexed) -> None:
        """The remainder are files the documentation build generates or that
        live in another repository. A target appearing under `other` in bulk
        means a resolver bug, not an absent file."""
        _, stats = indexed
        other = stats.missing_by_area.get("other", 0)
        assert other < 0.15 * max(stats.includes_missing, 1), (
            f"{other} unresolved includes are unclassified: {stats.missing_by_area}"
        )

    def test_no_constant_is_left_unexpanded(self, indexed) -> None:
        _, stats = indexed
        assert stats.unknown_constants == ()
        assert stats.unexpanded_substitutions == 0

    def test_role_syntax_almost_never_survives(self, indexed) -> None:
        """Not zero: a role split across a directive body boundary still gets
        through. It is one chunk in a hundred, and the count is asserted so a
        parser change that made it common would be caught."""
        _, stats = indexed
        assert stats.residual_roles / stats.chunks < 0.02

    def test_chunks_carry_heading_context(self, indexed) -> None:
        store, stats = indexed
        assert 0 not in stats.heading_depth, "a chunk with no heading path"
        assert stats.chunks > 10_000

    def test_chunk_sizes_land_in_the_intended_band(self, indexed) -> None:
        """Outliers are allowed and bounded. The oversized ones are single
        indivisible blocks -- a table of every error code, one enormous command
        output -- which the chunker emits whole rather than cutting mid-record."""
        _, stats = indexed
        assert 100 <= stats.token_p50 <= 300
        assert stats.token_p95 < 600
        assert stats.over_max_tokens / stats.chunks < 0.02


class TestCitationsResolve:
    def test_every_chunk_id_round_trips(self, indexed) -> None:
        """This is the mechanism the report's citation audit rests on. If a
        chunk id the store handed out does not resolve, every documentation
        citation in a report is unverifiable."""
        store, _ = indexed
        assert store.fetch_by_ids([]) == []
        sample = [f"reference/log-messages.txt#{n}" for n in range(5)]
        found = store.fetch_by_ids(sample)
        assert len(found) == len(sample)
        assert [c.chunk_id for c in found] == sorted(sample)

    def test_an_invented_chunk_id_does_not_resolve(self, indexed) -> None:
        store, _ = indexed
        assert store.fetch_by_ids(["reference/log-messages.txt#999999"]) == []


class TestRetrieval:
    @pytest.mark.parametrize(("question", "page"), LEXICAL_QUESTIONS)
    def test_the_text_leg_finds_the_right_page(
        self, indexed, question: str, page: str
    ) -> None:
        store, _ = indexed
        hits = store.text_search(question, k=10)
        documents = [hit.chunk.document for hit in hits]
        assert page in documents, f"{page} not in top 10 for {question!r}: {documents}"

    @pytest.mark.parametrize("question", SEMANTIC_QUESTIONS)
    def test_a_semantic_question_still_returns_well_formed_hits(
        self, indexed, question: str
    ) -> None:
        """No assertion about *which* page. With hashed vectors there is no
        reason to expect the right one, and a test that demanded it would either
        fail or be tuned until it passed -- which would turn a known limitation
        into a claim of quality."""
        store, _ = indexed
        hits = store.text_search(question, k=5)
        assert all(hit.chunk.chunk_id for hit in hits)
        assert hits == sorted(hits, key=lambda h: (-h.score, h.chunk_id))

    def test_the_vector_leg_is_well_formed_but_not_asserted_for_quality(
        self, indexed
    ) -> None:
        store, _ = indexed
        embedder = HashingEmbedder()
        (vector,) = embedder.embed(["slow query"])
        hits = store.vector_search(vector, k=5)
        assert len(hits) == 5
        assert all(-1.0 <= hit.score <= 1.0 for hit in hits)
        assert not embedder.semantic

    def test_hybrid_with_a_non_semantic_embedder_is_text_search(self, indexed) -> None:
        """Measured over this corpus: fusing the hashed vector ranking in at the
        production weight pushed the correct page out of the top three and
        replaced it with three pages that merely shared vocabulary."""
        store, _ = indexed
        (vector,) = HashingEmbedder().embed(["mongod slow query log message"])
        fused = store.hybrid_search(
            "mongod slow query log message", vector, k=5, vector_weight=0.0
        )
        text = store.text_search("mongod slow query log message", k=5)
        assert [h.chunk_id for h in fused] == [h.chunk_id for h in text]


class TestLogIdsAgainstTheManual:
    """The generator's ids, held against what the corpus documents.

    The synthetic logs everything else is tested on are only worth what their
    realism is worth, and the one part of that this repository can check is the
    log ids. This is the check.
    """

    def test_the_slow_query_record_matches_exactly(self, documented) -> None:
        """51803 is the record the whole tool is built around: component,
        severity and message all have to be what mongod actually writes, because
        the slow-query detector reads all three."""
        from logtriage.fixtures.generator import SLOW_QUERY

        entry = documented[SLOW_QUERY]
        assert entry.component == "COMMAND"
        assert entry.severity == "I"
        assert entry.msg == "Slow query"

    @pytest.mark.parametrize("name", ["CONNECTION_ACCEPTED", "CONNECTION_ENDED"])
    def test_the_connection_ids_are_documented_under_network(
        self, documented, name: str
    ) -> None:
        """Component only. The manual's example messages for these two are the
        4.4-era templated strings that 5.0 replaced, so asserting on the message
        would pin the fixture to a form the server stopped emitting."""
        import logtriage.fixtures.generator as generator

        entry = documented[getattr(generator, name)]
        assert entry.component == "NETWORK"

    def test_the_representative_ids_are_honestly_named(self, documented) -> None:
        """They are called representative because the manual does not document
        them. If MongoDB ever publishes one, this fails and the docstring --
        which says the corpus supplies none of them -- needs rewriting."""
        from logtriage.fixtures.generator import REPRESENTATIVE_IDS

        overlap = REPRESENTATIVE_IDS & set(documented)
        assert not overlap, (
            f"the manual now documents {sorted(overlap)}; promote them to "
            f"DOCUMENTED_IDS and update the generator docstring"
        )


class TestDeterminism:
    def test_chunking_a_page_twice_gives_the_same_chunks(self, corpus: Corpus) -> None:
        from logtriage.kb.chunk import chunk_document

        path = corpus.source / "reference" / "log-messages.txt"
        first = chunk_document(
            corpus.read(path), "reference/log-messages.txt", corpus.resolve_include
        )
        second = chunk_document(
            corpus.read(path), "reference/log-messages.txt", corpus.resolve_include
        )
        assert first == second

    def test_an_index_survives_a_save_and_load(self, indexed, tmp_path) -> None:
        store, _ = indexed
        meta = {
            "corpus_commit": "test",
            "embedder_seed": DEFAULT_SEED,
            "embedder_dimensions": DIMENSIONS,
        }
        store.save(tmp_path / "index", meta)
        reloaded, read_meta = InMemoryKnowledgeStore.load(
            tmp_path / "index", expect={"embedder_seed": DEFAULT_SEED}
        )
        assert len(reloaded) == len(store)
        assert read_meta["corpus_commit"] == "test"

        question = "slow query planSummary"
        (vector,) = HashingEmbedder().embed([question])
        before = store.vector_search(vector, k=10)
        after = reloaded.vector_search(vector, k=10)
        assert [h.chunk_id for h in before] == [h.chunk_id for h in after]
        assert [h.chunk_id for h in store.text_search(question, k=10)] == [
            h.chunk_id for h in reloaded.text_search(question, k=10)
        ]

    def test_an_index_built_by_another_embedder_is_refused(
        self, indexed, tmp_path
    ) -> None:
        store, _ = indexed
        store.save(tmp_path / "index", {"embedder_seed": DEFAULT_SEED})
        with pytest.raises(ValueError, match="embedder_seed"):
            InMemoryKnowledgeStore.load(
                tmp_path / "index", expect={"embedder_seed": DEFAULT_SEED + 1}
            )
