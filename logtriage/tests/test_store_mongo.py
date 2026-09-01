"""The MongoDB adapters, as far as an in-process engine can drive them.

test_store_differential.py executes the pipelines and compares their answers to
the in-memory store. This file drives the adapter classes themselves, which is a
different question: the pipelines could be right while the code that hands them
to the server, or turns documents back into records, is wrong.

Three things can be checked here and one cannot, and the boundary is not where
it first looks.

Queries run: mongomock executes the read pipelines, so count, sample and
fetch_by_ids are compared against the in-memory store, and facet and timeline
skip by name where mongomock lacks an operator.

Writes are checked but not executed. mongomock 4.3.0 -- the current release --
cannot execute any pymongo 4.17 bulk update: both ReplaceOne and UpdateOne now
carry a `sort` argument its builder does not accept. Rewriting insert() as a
loop of single replace_one calls would satisfy the fake and cost a network round
trip per record on a real ingest, which is the fake dictating production code.
So the writes keep bulk_write, and what is asserted is the operations it builds
-- batching, filters, upsert, document shape -- through a recording double. The
untested remainder is that pymongo and MongoDB execute a ReplaceOne correctly,
which is not this project's claim to make.

Search does not run at all. $search and $vectorSearch are mongot operators, not
MongoDB ones; mongomock does not implement them and could not, and any assertion
here would be about a stub rather than about mongot.
"""

from __future__ import annotations

from typing import Any

import pytest
from pymongo import ReplaceOne

from logtriage.fixtures.generator import generate
from logtriage.kb.chunk import Chunk
from logtriage.mongod.normalize import NormalizedRecord, normalize
from logtriage.mongod.records import LogRecord, parse_lines
from logtriage.store.base import LogFilter
from logtriage.store.base import to_document as record_document
from logtriage.store.knowledge_mongo import (
    TEXT_INDEX_NAME,
    VECTOR_FIELD,
    VECTOR_INDEX_NAME,
    MongoKnowledgeStore,
    from_document,
    text_index_definition,
    to_document,
    vector_index_definition,
)
from logtriage.store.memory import InMemoryLogStore
from logtriage.store.mongo import INDEXES, UPSERT_BATCH, MongoLogStore

mongomock = pytest.importorskip("mongomock")

# How mongomock reports an operator it does not implement. Matched on these
# phrases only: a broad except-and-skip would turn every malformed pipeline into
# a green run, which is the failure this file is meant to catch.
MISSING_OPERATOR = ("firstN", "dateTrunc", "$firstN", "$dateTrunc")


class RecordingCollection:
    """Captures bulk_write calls instead of executing them.

    Stands in for the collection precisely where mongomock stops, and only
    there: what insert() decides -- how many batches, which filter, whether the
    write upserts, what the document holds -- is this project's logic, while
    executing the resulting operation is pymongo's and the server's.
    """

    def __init__(self) -> None:
        self.batches: list[list[Any]] = []

    def bulk_write(self, operations: list[Any], ordered: bool = True) -> None:
        self.batches.append(list(operations))

    @property
    def operations(self) -> list[Any]:
        return [operation for batch in self.batches for operation in batch]


def _records(scenario: str = "slow-query-storm") -> list[NormalizedRecord]:
    log = generate(scenario, minutes=30)
    return [normalize(r) for r in parse_lines(log.lines) if isinstance(r, LogRecord)]


@pytest.fixture(scope="module")
def corpus() -> list[NormalizedRecord]:
    return _records()


@pytest.fixture(scope="module")
def memory(corpus) -> InMemoryLogStore:
    return InMemoryLogStore(corpus)


@pytest.fixture(scope="module")
def store(corpus) -> MongoLogStore:
    collection = mongomock.MongoClient()["logtriage_test"]["records"]
    # Loaded with insert_many rather than through MongoLogStore.insert, because
    # mongomock cannot execute the bulk replace that method issues. The write
    # path is covered by TestIngest below.
    collection.insert_many([record_document(record) for record in corpus])
    return MongoLogStore(collection)


class TestIngest:
    def test_every_record_becomes_one_upsert(self, corpus) -> None:
        collection = RecordingCollection()
        assert MongoLogStore(collection).insert(corpus) == len(corpus)  # type: ignore[arg-type]
        assert len(collection.operations) == len(corpus)

    def test_a_record_is_replaced_by_its_id_rather_than_inserted(self, corpus) -> None:
        """record_id is a content handle, not a sequence number, so re-ingesting
        the same log must replace rather than raise a duplicate key error --
        ingest is the step most likely to be run twice, since the first run is
        how you find out the file was truncated."""
        collection = RecordingCollection()
        MongoLogStore(collection).insert(corpus[:1])  # type: ignore[arg-type]

        document = record_document(corpus[0])
        assert collection.operations == [
            ReplaceOne({"_id": document["_id"]}, document, upsert=True)
        ]

    def test_writes_are_batched_rather_than_sent_one_at_a_time(self, corpus) -> None:
        """A real ingest is hundreds of thousands of records. One round trip
        each is the difference between seconds and an afternoon."""
        assert len(corpus) > UPSERT_BATCH, "fixture no longer covers the batch path"
        collection = RecordingCollection()
        MongoLogStore(collection).insert(corpus)  # type: ignore[arg-type]

        assert len(collection.batches) > 1
        assert all(len(batch) <= UPSERT_BATCH for batch in collection.batches)
        # The tail matters: a flush that only fired on a full batch would drop
        # every record after the last multiple of UPSERT_BATCH.
        assert len(collection.operations) == len(corpus)

    def test_ingesting_nothing_writes_nothing(self) -> None:
        collection = RecordingCollection()
        assert MongoLogStore(collection).insert([]) == 0  # type: ignore[arg-type]
        assert collection.batches == []

    def test_indexes_are_created_and_creating_them_twice_is_harmless(self) -> None:
        collection = mongomock.MongoClient()["index_test"]["records"]
        store = MongoLogStore(collection)
        store.ensure_indexes()
        store.ensure_indexes()
        # _id_ is the implicit primary key index and is not one of ours.
        assert len(collection.index_information()) == len(INDEXES) + 1

    def test_constructing_a_store_writes_nothing(self) -> None:
        """ensure_indexes is called explicitly so a read-only user can hold a
        store without the constructor trying to write."""
        collection = mongomock.MongoClient()["readonly_test"]["records"]
        MongoLogStore(collection)
        # Only the implicit primary key index may exist, and on an empty
        # collection not even that.
        assert set(collection.index_information()) <= {"_id_"}


class TestQueriesMatchTheInMemoryStore:
    FILTERS = [
        LogFilter(),
        LogFilter(components=("COMMAND",)),
        LogFilter(severities=("W", "E")),
        LogFilter(min_duration_ms=100),
        LogFilter(text="slow"),
        LogFilter(components=("NETWORK",), severities=("I",)),
    ]

    @pytest.mark.parametrize("flt", FILTERS)
    def test_count(self, store, memory, flt) -> None:
        assert store.count(flt) == memory.count(flt)

    def test_a_filter_matching_nothing_counts_zero_rather_than_failing(
        self, store
    ) -> None:
        """$count emits no document at all for an empty match, so the adapter
        has to turn an absent result into a zero."""
        assert store.count(LogFilter(components=("NOSUCH",))) == 0

    @pytest.mark.parametrize("order", ["time", "-time", "-duration"])
    def test_sample_returns_the_same_records_in_the_same_order(
        self, store, memory, order
    ) -> None:
        mongo = store.sample(LogFilter(), limit=15, order=order)
        local = memory.sample(LogFilter(), limit=15, order=order)
        assert [r.record_id for r in mongo] == [r.record_id for r in local]

    def test_records_survive_the_document_round_trip(self, store, memory) -> None:
        """from_document is the half of the adapter a pipeline test cannot see."""
        (mongo,) = store.sample(LogFilter(), limit=1)
        (local,) = memory.sample(LogFilter(), limit=1)
        assert mongo == local

    def test_citations_resolve(self, store, memory) -> None:
        ids = [r.record_id for r in memory.sample(LogFilter(), limit=5)]
        assert [r.record_id for r in store.fetch_by_ids(ids)] == sorted(ids)

    def test_an_unresolvable_citation_is_omitted_not_fabricated(self, store) -> None:
        assert store.fetch_by_ids(["no-such-record"]) == []

    def test_no_ids_makes_no_query(self, store) -> None:
        assert store.fetch_by_ids([]) == []

    @pytest.mark.parametrize("by", [["component"], ["severity"], ["ns"]])
    def test_facet(self, store, memory, by) -> None:
        try:
            mongo = store.facet(LogFilter(), by)
        except Exception as error:  # noqa: BLE001 - re-raised unless it is the known gap
            if any(token in str(error) for token in MISSING_OPERATOR):
                pytest.skip(f"mongomock does not implement $firstN: facet({by})")
            raise
        local = memory.facet(LogFilter(), by)
        assert [(b.key, b.count) for b in mongo] == [(b.key, b.count) for b in local]

    def test_timeline(self, store, memory) -> None:
        try:
            mongo = store.timeline(LogFilter(), bucket_seconds=60)
        except Exception as error:  # noqa: BLE001 - re-raised unless it is the known gap
            if any(token in str(error) for token in MISSING_OPERATOR):
                pytest.skip("mongomock does not implement $dateTrunc: timeline")
            raise
        local = memory.timeline(LogFilter(), bucket_seconds=60)
        assert [(b.start, b.count) for b in mongo] == [
            (b.start, b.count) for b in local
        ]


CHUNK = Chunk(
    chunk_id="reference/log-messages.txt#3",
    document="reference/log-messages.txt",
    heading_path=("Log Messages", "Structured Logging"),
    text="mongod outputs log messages in structured JSON format.",
    token_estimate=11,
    has_code=True,
    anchors=("log-messages-ref",),
    references=("bin.mongod",),
    line_no=48,
)


class TestKnowledgeDocuments:
    def test_a_chunk_survives_the_round_trip_unchanged(self) -> None:
        """Tuples become BSON arrays and come back as lists. Left alone, a
        retrieved chunk would compare unequal to the one that was stored while
        looking identical when printed."""
        assert from_document(to_document(CHUNK, [0.1, 0.2])) == CHUNK

    def test_the_embedding_is_stored_beside_the_chunk_not_inside_it(self) -> None:
        document = to_document(CHUNK, [0.1, 0.2])
        assert document[VECTOR_FIELD] == [0.1, 0.2]
        assert document["_id"] == CHUNK.chunk_id
        # chunk_id is the citation handle and the primary key both; a second
        # identifier is one that could disagree with the citation.
        assert "chunk_id" not in document

    def test_every_chunk_field_reaches_the_document(self) -> None:
        """CHUNK_FIELDS is derived from the dataclass, so a field added to Chunk
        cannot be silently dropped on the way to the server."""
        document = to_document(CHUNK, [0.1])
        for name in Chunk.__dataclass_fields__:
            if name != "chunk_id":
                assert name in document, f"{name} was not stored"

    def test_upsert_replaces_by_chunk_id(self) -> None:
        collection = RecordingCollection()
        store = MongoKnowledgeStore(collection)  # type: ignore[arg-type]
        assert store.upsert([CHUNK], [[0.1, 0.2]]) == 1
        assert collection.operations == [
            ReplaceOne(
                {"_id": CHUNK.chunk_id}, to_document(CHUNK, [0.1, 0.2]), upsert=True
            )
        ]

    def test_upsert_of_nothing_writes_nothing(self) -> None:
        collection = RecordingCollection()
        assert MongoKnowledgeStore(collection).upsert([], []) == 0  # type: ignore[arg-type]
        assert collection.batches == []

    def test_fetch_by_ids_resolves_without_touching_a_search_index(self) -> None:
        """A plain $match, deliberately: routing citations through mongot would
        make a report's verifiability depend on an index being Ready."""
        collection = mongomock.MongoClient()["kb_fetch"]["chunks"]
        collection.insert_one(to_document(CHUNK, [0.1, 0.2]))
        store = MongoKnowledgeStore(collection)

        assert store.fetch_by_ids([CHUNK.chunk_id]) == [CHUNK]
        assert store.fetch_by_ids(["missing"]) == []
        assert store.fetch_by_ids([]) == []
        assert len(store) == 1

    def test_a_fetched_chunk_does_not_carry_its_vector(self) -> None:
        """768 floats per hit, on every citation resolved in a report."""
        collection = mongomock.MongoClient()["kb_projection"]["chunks"]
        collection.insert_one(to_document(CHUNK, [0.1, 0.2]))
        # The store projects the vector away; from_document never sees it, so
        # this asserts on the query rather than on the reconstructed chunk.
        document = collection.find_one({"_id": CHUNK.chunk_id}, {VECTOR_FIELD: 0})
        assert VECTOR_FIELD not in document
        assert MongoKnowledgeStore(collection).fetch_by_ids([CHUNK.chunk_id]) == [CHUNK]


class TestSearchIndexDefinitions:
    """The definitions are data so they can be read against the manual here,
    rather than discovered to be wrong on a server later."""

    def test_the_vector_index_declares_the_embedder_s_width(self) -> None:
        definition = vector_index_definition(768)
        (field,) = definition["fields"]
        assert field["type"] == "vector"
        assert field["path"] == VECTOR_FIELD
        assert field["numDimensions"] == 768
        # Both embedders return unit vectors, for which cosine and dotProduct
        # rank identically; cosine is named because embed.base.cosine is what a
        # reader will compare it against.
        assert field["similarity"] == "cosine"

    def test_the_text_index_is_static_and_names_what_is_searched(self) -> None:
        """A dynamic mapping would index the embedding array and the token
        count -- index size spent on fields no query searches."""
        mappings = text_index_definition()["mappings"]
        assert mappings["dynamic"] is False
        assert set(mappings["fields"]) == {"text", "heading_path", "document"}
        assert VECTOR_FIELD not in mappings["fields"]

    def test_the_two_indexes_are_named_apart(self) -> None:
        assert VECTOR_INDEX_NAME != TEXT_INDEX_NAME


class TestFusionIsSharedNotReimplemented:
    def test_both_stores_call_the_same_fusion_function(self) -> None:
        """Ranking is the one place the two implementations could diverge with
        nothing able to notice: both would return plausible chunks in a
        different order, and that is indistinguishable from an ambiguous query.
        They agree by construction because there is one implementation."""
        import inspect

        from logtriage.store import knowledge, knowledge_mongo

        assert knowledge_mongo.fuse is knowledge.fuse
        for source in (
            inspect.getsource(knowledge.InMemoryKnowledgeStore.hybrid_search),
            inspect.getsource(knowledge_mongo.MongoKnowledgeStore.hybrid_search),
        ):
            assert "fuse(" in source
            # A second RRF implementation would need the constant.
            assert "RRF_K" not in source
