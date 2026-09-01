"""Parsing, chunking, embedding, and retrieval over documentation.

The fixtures here are hand-written but not invented: they reproduce the
constructs the real MongoDB manual uses, because those are what a naive splitter
gets wrong. Running the same code over the actual manual is a separate, marked
suite -- these tests have to pass in a checkout with no corpus cloned.
"""

from __future__ import annotations

import pytest

from logtriage.embed.base import cosine
from logtriage.embed.hashing import HashingEmbedder, tokenize
from logtriage.kb.chunk import MAX_TOKENS, chunk_document
from logtriage.kb.parse import parse_rst, strip_roles
from logtriage.store.knowledge import InMemoryKnowledgeStore

MANUAL = """\
.. _log-messages:

============
Log Messages
============

.. default-domain:: mongodb

.. contents:: On this page
   :local:
   :backlinks: none

Starting in MongoDB 4.4, structured log messages are output as :term:`JSON`
objects. See :ref:`the log message reference <log-message-format>` for the
full schema, or :doc:`/reference/parameters` for the parameters involved.

Structured Message Format
-------------------------

Each message is a single ``JSON`` object with a fixed set of fields.

.. code-block:: javascript

   {
     "t": { "$date": "2026-08-30T14:22:31.123+00:00" },
     "s": "I",
     "c": "COMMAND",
     "id": 51803,
     "msg": "Slow query"
   }

.. include:: /includes/fact-log-message-fields.rst

Slow Query Logging
------------------

.. _slow-query-threshold:

Operations exceeding the threshold are logged with id 51803.

.. note::

   The threshold defaults to 100 milliseconds.

Configuring the Threshold
~~~~~~~~~~~~~~~~~~~~~~~~~

Set the threshold with ``setParameter``.

.. tabs::

   .. tab:: Command Line
      :tabid: cli

      Pass the option at startup.

      .. code-block:: bash

         mongod --slowms 50

   .. tab:: Shell
      :tabid: shell

      Set it at runtime.

      .. code-block:: javascript

         db.setProfilingLevel(1, { slowms: 50 })

Reference Table
---------------

+------------+---------------------------+
| Component  | Meaning                   |
+============+===========================+
| COMMAND    | Database commands         |
+------------+---------------------------+
| NETWORK    | Connection lifecycle      |
+------------+---------------------------+
"""


@pytest.fixture(scope="module")
def blocks():
    return parse_rst(MANUAL, "reference/log-messages.txt")


@pytest.fixture(scope="module")
def chunks():
    return chunk_document(MANUAL, "reference/log-messages.txt")


class TestHeadings:
    def test_levels_follow_first_use_of_each_adornment(self, blocks) -> None:
        """rST hierarchy is per-document: the character does not fix the level,
        the order it first appears does."""
        headings = [(b.level, b.text) for b in blocks if b.kind == "heading"]
        assert headings[0] == (1, "Log Messages")
        assert (2, "Slow Query Logging") in headings
        assert (3, "Configuring the Threshold") in headings

    def test_overline_form_is_recognised(self, blocks) -> None:
        assert blocks[0].kind == "heading"
        assert blocks[0].text == "Log Messages"

    def test_anchors_attach_to_what_they_label(self, blocks) -> None:
        """An anchor renders as nothing but is what every :ref: resolves to."""
        title = next(b for b in blocks if b.text == "Log Messages")
        assert "log-messages" in title.anchors

    def test_a_table_rule_is_not_read_as_a_heading(self, blocks) -> None:
        assert not any(b.kind == "heading" and "+---" in b.text for b in blocks)


class TestCodeBlocks:
    def test_code_is_kept_verbatim(self, blocks) -> None:
        code = [b for b in blocks if b.kind == "code"]
        assert len(code) == 3
        javascript = next(b for b in code if "$date" in b.text)
        assert javascript.language == "javascript"
        assert '"id": 51803' in javascript.text

    def test_code_indentation_is_normalised_not_lost(self, blocks) -> None:
        """Dedented to the block's own base, so nesting inside the snippet
        survives while the directive's indentation does not."""
        javascript = next(b for b in blocks if b.kind == "code" and "$date" in b.text)
        assert javascript.text.startswith("{")
        assert '  "t": {' in javascript.text

    def test_shell_and_javascript_tabs_both_survive(self, blocks) -> None:
        languages = {b.language for b in blocks if b.kind == "code"}
        assert languages == {"javascript", "bash"}


class TestDirectives:
    def test_includes_are_recorded_rather_than_dropped(self, blocks) -> None:
        """The body lives in another file. Naming it is more honest than
        silently omitting a definition."""
        include = next(b for b in blocks if b.kind == "include")
        assert include.text == "/includes/fact-log-message-fields.rst"

    def test_tab_labels_are_attached_to_their_content(self, blocks) -> None:
        """A chunk boundary must not separate "[Command Line]" from the
        instructions it qualifies."""
        text = "\n".join(b.text for b in blocks)
        assert "[Command Line] Pass the option at startup." in text
        assert "[Shell] Set it at runtime." in text

    def test_prose_directives_keep_their_label(self, blocks) -> None:
        assert any("NOTE: The threshold defaults" in b.text for b in blocks)

    def test_rendering_only_directives_are_dropped(self, blocks) -> None:
        text = " ".join(b.text for b in blocks)
        assert "On this page" not in text
        assert "default-domain" not in text

    def test_tables_are_kept_whole(self, blocks) -> None:
        table = next(b for b in blocks if b.kind == "table")
        assert "COMMAND" in table.text
        assert "NETWORK" in table.text


class TestRoles:
    def test_display_text_survives_and_the_target_is_kept(self) -> None:
        text, refs = strip_roles(
            "See :ref:`the log reference <log-message-format>` for detail."
        )
        assert text == "See the log reference for detail."
        assert refs == ("log-message-format",)

    def test_a_bare_role_renders_as_its_target(self) -> None:
        text, refs = strip_roles("See :doc:`/reference/parameters`.")
        assert text == "See parameters."
        assert refs == ("/reference/parameters",)

    def test_literals_lose_their_backticks(self) -> None:
        text, _ = strip_roles("Set it with ``setParameter``.")
        assert text == "Set it with setParameter."

    def test_references_reach_the_chunk(self, chunks) -> None:
        refs = {r for c in chunks for r in c.references}
        assert "log-message-format" in refs


class TestChunking:
    def test_every_chunk_carries_its_heading_path(self, chunks) -> None:
        """A retrieved fragment arrives with no surrounding document."""
        assert chunks
        for chunk in chunks:
            assert chunk.heading_path
            assert chunk.breadcrumb in chunk.render()

    def test_the_deepest_section_has_its_full_path(self, chunks) -> None:
        deep = next(c for c in chunks if "Configuring the Threshold" in c.heading_path)
        assert deep.heading_path == (
            "Log Messages",
            "Slow Query Logging",
            "Configuring the Threshold",
        )

    def test_no_chunk_spans_two_sections(self, chunks) -> None:
        """Carrying prose past a heading gives a chunk whose breadcrumb is
        wrong for half its content."""
        for chunk in chunks:
            assert chunk.text.count("Reference Table") <= 1

    def test_code_blocks_are_never_split(self, chunks) -> None:
        """Half a command is worse than none: it still looks runnable."""
        combined = "\n".join(c.text for c in chunks)
        assert "db.setProfilingLevel(1, { slowms: 50 })" in combined
        assert "mongod --slowms 50" in combined
        for snippet in ("mongod --slowms 50", "db.setProfilingLevel"):
            holders = [c for c in chunks if snippet in c.text]
            assert len(holders) == 1

    def test_chunks_respect_the_size_ceiling(self, chunks) -> None:
        oversized = [c for c in chunks if c.token_estimate > MAX_TOKENS]
        # The only permitted exception is a single block bigger than the ceiling.
        for chunk in oversized:
            assert chunk.has_code, "only an atomic code block may exceed the ceiling"

    def test_chunk_ids_are_unique_and_resolvable(self, chunks) -> None:
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))
        assert all(c.chunk_id.startswith("reference/log-messages.txt#") for c in chunks)

    def test_chunking_is_deterministic(self) -> None:
        assert chunk_document(MANUAL, "d.txt") == chunk_document(MANUAL, "d.txt")

    def test_an_empty_document_yields_no_chunks(self) -> None:
        assert chunk_document("", "empty.txt") == []

    def test_a_document_with_no_headings_still_chunks(self) -> None:
        """Included fragments are bodies without titles, and there are many."""
        chunks = chunk_document("Just a paragraph of prose.\n", "fragment.txt")
        assert len(chunks) == 1
        assert chunks[0].heading_path == ()
        assert "fragment.txt" in chunks[0].render()


class TestHashingEmbedder:
    def test_vectors_are_unit_length(self) -> None:
        embedder = HashingEmbedder()
        for vector in embedder.embed(["slow query", "connection accepted"]):
            assert cosine(vector, vector) == pytest.approx(1.0)

    def test_the_same_text_always_embeds_identically(self) -> None:
        """An index built today must be searchable tomorrow."""
        assert HashingEmbedder().embed(["a b c"]) == HashingEmbedder().embed(["a b c"])

    def test_shared_vocabulary_scores_higher_than_none(self) -> None:
        embedder = HashingEmbedder()
        base, shared, unrelated = embedder.embed(
            [
                "slow query collection scan planSummary COLLSCAN",
                "a collection scan happens when planSummary is COLLSCAN",
                "replica set election heartbeat term",
            ]
        )
        assert cosine(base, shared) > cosine(base, unrelated)

    def test_empty_text_is_the_zero_vector(self) -> None:
        """No direction, so it scores zero against everything -- which is the
        honest answer rather than a faked one."""
        (vector,) = HashingEmbedder().embed([""])
        assert set(vector) == {0.0}

    def test_identifier_vocabulary_is_not_split_apart(self) -> None:
        assert "db.collection" in tokenize("query db.collection now")
        assert "$vectorsearch" in tokenize("use $vectorSearch here")

    def test_dimensions_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            HashingEmbedder(dimensions=0)


class TestKnowledgeStore:
    @pytest.fixture
    def store(self):
        embedder = HashingEmbedder()
        chunks = chunk_document(MANUAL, "reference/log-messages.txt")
        store = InMemoryKnowledgeStore()
        store.upsert(chunks, embedder.embed([c.render() for c in chunks]))
        return store, embedder

    def test_indexes_every_chunk(self, store) -> None:
        knowledge, _ = store
        assert len(knowledge) == len(
            chunk_document(MANUAL, "reference/log-messages.txt")
        )

    def test_reindexing_does_not_duplicate(self, store) -> None:
        knowledge, embedder = store
        chunks = chunk_document(MANUAL, "reference/log-messages.txt")
        before = len(knowledge)
        assert (
            knowledge.upsert(chunks, embedder.embed([c.render() for c in chunks])) == 0
        )
        assert len(knowledge) == before

    def test_vector_search_is_bounded_and_ordered(self, store) -> None:
        knowledge, embedder = store
        (query,) = embedder.embed(["how do I change the slow query threshold"])
        hits = knowledge.vector_search(query, k=3)
        assert len(hits) == 3
        assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)

    def test_text_search_finds_an_exact_identifier(self, store) -> None:
        """The case vector search is worst at and log diagnosis needs most:
        someone pastes an exact string and wants the page about it."""
        knowledge, _ = store
        hits = knowledge.text_search("setProfilingLevel", k=3)
        assert hits
        assert any("setProfilingLevel" in h.chunk.text for h in hits)

    def test_hybrid_fuses_both_legs(self, store) -> None:
        knowledge, embedder = store
        (query,) = embedder.embed(["slow query threshold"])
        hits = knowledge.hybrid_search("slow query threshold", query, k=5)
        assert hits
        assert any("+" in h.leg for h in hits), "no chunk was found by both legs"

    def test_search_is_deterministic(self, store) -> None:
        """Two searches over one index must rank identically, or a report is
        not reproducible."""
        knowledge, embedder = store
        (query,) = embedder.embed(["election"])
        assert knowledge.vector_search(query, k=5) == knowledge.vector_search(
            query, k=5
        )
        assert knowledge.text_search("election", k=5) == knowledge.text_search(
            "election", k=5
        )

    def test_citations_resolve(self, store) -> None:
        knowledge, embedder = store
        (query,) = embedder.embed(["log message format"])
        ids = [h.chunk_id for h in knowledge.vector_search(query, k=3)]
        assert [c.chunk_id for c in knowledge.fetch_by_ids(ids)] == sorted(ids)

    def test_unknown_citations_are_omitted_not_raised(self, store) -> None:
        knowledge, _ = store
        assert knowledge.fetch_by_ids(["nope#1"]) == []

    def test_searching_an_empty_index_returns_nothing(self) -> None:
        empty = InMemoryKnowledgeStore()
        assert empty.vector_search([0.0] * 256, k=5) == []
        assert empty.text_search("anything", k=5) == []


def test_dotted_identifiers_are_searchable_whole_and_split() -> None:
    """Keeping `db.setProfilingLevel` only as one token means a reader pasting
    the bare method name finds nothing. Keeping only the parts loses the
    namespace vocabulary. Both are emitted."""
    tokens = tokenize("call db.setProfilingLevel on shop.orders")
    assert "db.setprofilinglevel" in tokens
    assert "setprofilinglevel" in tokens
    assert "shop.orders" in tokens
    assert "orders" in tokens


def test_code_blocks_have_no_leading_blank_line() -> None:
    """The blank line after a directive is its formatting, not its content, and
    would otherwise render verbatim at the top of every snippet."""
    blocks = parse_rst(MANUAL, "d.txt")
    for block in blocks:
        if block.kind == "code":
            assert block.text == block.text.lstrip("\n")
            assert block.text.strip()
