"""The in-memory store, and the pipelines asserted as data.

The pipeline assertions matter more than they look. They are how the MongoDB
adapter is reviewed on a machine with no MongoDB, and they pin the two
decisions -- $firstN over $push, and the sort that makes it deterministic --
that are invisible until a corpus is large enough for them to fail.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from logtriage.fixtures.generator import generate
from logtriage.mongod.normalize import NormalizedRecord, normalize
from logtriage.mongod.records import LogRecord, parse_lines
from logtriage.store.base import LogFilter, from_document, to_document
from logtriage.store.memory import InMemoryLogStore, matches
from logtriage.store.pipelines import (
    facet_pipeline,
    fetch_by_ids_pipeline,
    match_stage,
    sample_pipeline,
    timeline_pipeline,
)


def _records(scenario: str = "slow-query-storm", **kwargs) -> list[NormalizedRecord]:
    log = generate(scenario, **kwargs)
    return [normalize(r) for r in parse_lines(log.lines) if isinstance(r, LogRecord)]


@pytest.fixture
def store() -> InMemoryLogStore:
    return InMemoryLogStore(_records())


class TestDocumentRoundTrip:
    def test_every_field_survives(self) -> None:
        """Derived from the dataclass, so a new field cannot be dropped silently."""
        for record in _records()[:200]:
            assert from_document(to_document(record)) == record

    def test_record_id_is_the_primary_key(self) -> None:
        """One identifier, so a citation and a primary key cannot disagree."""
        record = _records()[0]
        assert to_document(record)["_id"] == record.record_id
        assert "record_id" not in to_document(record)


class TestFilter:
    def test_an_empty_filter_matches_everything(self, store) -> None:
        assert store.count(LogFilter()) == len(store)

    def test_an_empty_tuple_does_not_match_nothing(self, store) -> None:
        """Filters are built from optional values, so an empty tuple means
        "unconstrained". MongoDB's $in over an empty list matches no document at
        all, which would turn an unfiltered query into a silent empty result."""
        assert store.count(LogFilter(components=())) == len(store)
        assert "component" not in match_stage(LogFilter(components=()))

    def test_component_and_severity_narrow(self, store) -> None:
        assert 0 < store.count(LogFilter(components=("NETWORK",))) < len(store)
        assert store.count(LogFilter(severities=("F",))) == 0

    def test_time_bounds_are_inclusive(self) -> None:
        """A window given as whole minutes must contain a record on its bound,
        or "the 14:00-14:15 window" quietly means fourteen minutes and change."""
        records = _records("baseline")
        first, last = records[0].ts, records[-1].ts
        store = InMemoryLogStore(records)
        assert store.count(LogFilter(start=first, end=last)) == len(records)
        assert match_stage(LogFilter(end=last))["ts"] == {"$lte": last}

    def test_duration_bound_excludes_records_without_a_duration(self, store) -> None:
        """MongoDB's $gte does not match a missing field. Treating None as zero
        would put every connection message into a query about slow operations."""
        slow = store.sample(LogFilter(min_duration_ms=1), limit=1000)
        assert slow
        assert all(r.duration_ms is not None for r in slow)
        connection = next(r for r in _records() if r.duration_ms is None)
        assert not matches(connection, LogFilter(min_duration_ms=0))

    def test_text_search_is_case_insensitive_and_not_a_regex(self, store) -> None:
        assert store.count(LogFilter(text="slow query")) > 0
        assert store.count(LogFilter(text="SLOW QUERY")) > 0
        # A metacharacter is a literal, not a pattern: an unescaped user string
        # is an unbounded regex on the server.
        assert store.count(LogFilter(text="Slow.query")) == 0
        assert match_stage(LogFilter(text="a.b"))["msg"]["$regex"] == r"a\.b"

    def test_conditions_are_conjunctive(self, store) -> None:
        both = store.count(LogFilter(components=("COMMAND",), plan_summary="COLLSCAN"))
        assert 0 < both <= store.count(LogFilter(components=("COMMAND",)))


class TestFacet:
    def test_counts_sum_to_the_matched_total(self, store) -> None:
        flt = LogFilter(components=("COMMAND",))
        buckets = store.facet(flt, ["ns"], top=100)
        assert sum(b.count for b in buckets) == store.count(flt)

    def test_ordered_by_count_with_a_total_tiebreak(self, store) -> None:
        buckets = store.facet(LogFilter(), ["component"], top=100)
        assert buckets == sorted(buckets, key=lambda b: -b.count)
        assert store.facet(LogFilter(), ["component"]) == store.facet(
            LogFilter(), ["component"]
        )

    def test_carries_the_aggregates_diagnostics_need(self, store) -> None:
        bucket = next(
            b for b in store.facet(LogFilter(), ["ns"], top=10) if b.key["ns"]
        )
        assert bucket.max_duration_ms is not None
        assert bucket.total_docs_examined is not None
        assert 1 <= len(bucket.examples) <= 3

    def test_examples_resolve_to_real_records(self, store) -> None:
        """A facet's examples are citations, so they must be fetchable."""
        for bucket in store.facet(LogFilter(), ["component"], top=10):
            assert len(store.fetch_by_ids(bucket.examples)) == len(bucket.examples)

    def test_top_bounds_the_result(self, store) -> None:
        assert len(store.facet(LogFilter(), ["log_id"], top=2)) == 2

    def test_pipeline_bounds_examples_at_the_group_rather_than_after(self) -> None:
        """$push would accumulate every id in a group before anything trimmed
        it, exceeding the 16MB document limit on exactly the namespace most
        worth profiling. And $firstN is only deterministic behind a sort."""
        stages = facet_pipeline(LogFilter(), ["ns"])
        group = next(s for s in stages if "$group" in s)["$group"]
        assert group["examples"] == {"$firstN": {"input": "$_id", "n": 3}}
        assert "$push" not in str(stages)
        assert stages[1] == {"$sort": {"_id": 1}}


class TestTimeline:
    def test_counts_sum_to_the_matched_total(self, store) -> None:
        buckets = store.timeline(LogFilter(), 60)
        assert sum(b.count for b in buckets) == len(store)

    def test_buckets_are_ordered_and_epoch_aligned(self, store) -> None:
        """Aligning to the epoch rather than to the first record is what makes
        two timelines over overlapping windows comparable."""
        buckets = store.timeline(LogFilter(), 300)
        assert [b.start for b in buckets] == sorted(b.start for b in buckets)
        for bucket in buckets:
            elapsed = bucket.start - datetime(1970, 1, 1, tzinfo=UTC)
            assert elapsed % timedelta(seconds=300) == timedelta(0)

    def test_rejects_a_non_positive_width(self, store) -> None:
        with pytest.raises(ValueError, match="must be positive"):
            store.timeline(LogFilter(), 0)

    def test_pipeline_truncates_rather_than_computing_offsets(self) -> None:
        stage = next(s for s in timeline_pipeline(LogFilter(), 60) if "$group" in s)
        assert stage["$group"]["_id"]["$dateTrunc"]["binSize"] == 60


class TestSample:
    def test_time_order_is_ascending(self, store) -> None:
        sampled = store.sample(LogFilter(), limit=50)
        assert [r.ts for r in sampled] == sorted(r.ts for r in sampled)

    def test_reverse_time_order_is_descending(self, store) -> None:
        sampled = store.sample(LogFilter(), limit=50, order="-time")
        assert [r.ts for r in sampled] == sorted((r.ts for r in sampled), reverse=True)

    def test_duration_order_puts_the_worst_first(self, store) -> None:
        sampled = store.sample(
            LogFilter(min_duration_ms=1), limit=20, order="-duration"
        )
        durations = [r.duration_ms or 0 for r in sampled]
        assert durations == sorted(durations, reverse=True)
        assert durations[0] == max(
            r.duration_ms or 0 for r in _records() if r.duration_ms
        )

    def test_duration_order_tolerates_records_without_one(self, store) -> None:
        """Python cannot compare None to an int, so this would raise rather
        than merely order oddly if the None case were not handled."""
        assert store.sample(LogFilter(), limit=5, order="-duration")

    def test_ordering_is_total_so_two_runs_cite_the_same_records(self, store) -> None:
        """mongod logs at millisecond resolution and ties are constant."""
        for order in ("time", "-time", "-duration"):
            first = store.sample(LogFilter(), limit=30, order=order)  # type: ignore[arg-type]
            second = store.sample(LogFilter(), limit=30, order=order)  # type: ignore[arg-type]
            assert [r.record_id for r in first] == [r.record_id for r in second]

    def test_every_pipeline_sort_ends_with_the_primary_key(self) -> None:
        for order in ("time", "-time", "-duration"):
            stage = next(
                s for s in sample_pipeline(LogFilter(), order=order) if "$sort" in s
            )  # type: ignore[arg-type]
            assert list(stage["$sort"])[-1] == "_id"


class TestFetchByIds:
    def test_resolves_citations(self, store) -> None:
        wanted = ["L1", "L2", "L3"]
        assert [r.record_id for r in store.fetch_by_ids(wanted)] == wanted

    def test_unknown_ids_are_omitted_rather_than_raising(self, store) -> None:
        """The caller checking a report's citations needs to know *which* ones
        failed to resolve; an exception only says that something did."""
        found = store.fetch_by_ids(["L1", "L999999", "nonsense"])
        assert [r.record_id for r in found] == ["L1"]

    def test_empty_input_is_empty_output(self, store) -> None:
        assert store.fetch_by_ids([]) == []

    def test_pipeline_matches_on_the_primary_key(self) -> None:
        assert fetch_by_ids_pipeline(["L1"])[0] == {"$match": {"_id": {"$in": ["L1"]}}}


class TestInsert:
    def test_reingesting_a_corpus_does_not_double_the_counts(self) -> None:
        records = _records("baseline")
        store = InMemoryLogStore(records)
        before = len(store)
        assert store.insert(records) == 0
        assert len(store) == before

    def test_reingest_takes_the_newer_record(self) -> None:
        records = _records("baseline")[:10]
        store = InMemoryLogStore(records)
        from dataclasses import replace

        store.insert([replace(records[0], msg="rewritten")])
        assert store.fetch_by_ids([records[0].record_id])[0].msg == "rewritten"
        assert len(store) == 10
