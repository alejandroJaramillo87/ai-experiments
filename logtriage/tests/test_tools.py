"""The tool surface, tested the way a model will actually use it.

Most of these are malformed calls. A model emits wrong types, invented enum
members and missing fields as a matter of course, and every one of those has to
come back as something it can act on -- an exception ends the run, a structured
error costs one turn.
"""

from __future__ import annotations

import json

import pytest

from logtriage.embed.hashing import HashingEmbedder
from logtriage.kb.chunk import chunk_document
from logtriage.mongod.normalize import normalize
from logtriage.mongod.records import LogRecord, parse_lines
from logtriage.store.knowledge import InMemoryKnowledgeStore
from logtriage.store.memory import InMemoryLogStore
from logtriage.tools.budget import estimate_tokens
from logtriage.tools.schemas import GROUPABLE, MAX_ROWS, SCHEMAS, ToolBox
from logtriage.triage.profile import build_profile, render_profile
from tests.sample_rst import MANUAL


@pytest.fixture(scope="module")
def toolbox() -> ToolBox:
    from logtriage.fixtures.generator import generate

    log = generate("slow-query-storm", minutes=45)
    records = [normalize(r) for r in parse_lines(log.lines) if isinstance(r, LogRecord)]
    logs = InMemoryLogStore(records)

    embedder = HashingEmbedder()
    chunks = chunk_document(MANUAL, "reference/log-messages.txt")
    knowledge = InMemoryKnowledgeStore()
    knowledge.upsert(chunks, embedder.embed([c.render() for c in chunks]))

    return ToolBox(
        logs=logs,
        knowledge=knowledge,
        embedder=embedder,
        profile_text=render_profile(build_profile(records)),
    )


class TestSchemas:
    def test_schemas_are_valid_openai_function_definitions(self) -> None:
        for schema in SCHEMAS:
            assert schema["type"] == "function"
            function = schema["function"]
            assert function["name"] and function["description"]
            params = function["parameters"]
            assert params["type"] == "object"
            for name in params.get("required", []):
                assert name in params["properties"], f"{function['name']}: {name}"

    def test_schemas_are_json_serialisable(self) -> None:
        """They go over the wire to llama-server verbatim."""
        assert json.loads(json.dumps(SCHEMAS)) == SCHEMAS

    def test_groupable_fields_are_advertised_in_the_schema(self) -> None:
        """A closed enum in the prompt beats validation after the fact: the
        model sees what is groupable instead of guessing and being corrected."""
        query = next(s for s in SCHEMAS if s["function"]["name"] == "query_logs")
        enum = query["function"]["parameters"]["properties"]["group_by"]["items"][
            "enum"
        ]
        assert set(enum) == set(GROUPABLE)

    def test_a_tool_without_its_store_is_not_offered(self) -> None:
        """Offering a tool that cannot work is worse than offering fewer: the
        model spends turns on it and reads the failures as evidence."""
        bare = ToolBox(logs=InMemoryLogStore())
        names = {s["function"]["name"] for s in bare.schemas()}
        assert "search_docs" not in names
        assert "query_logs" in names


class TestDispatch:
    def test_an_unknown_tool_lists_the_real_ones(self, toolbox) -> None:
        result = toolbox.call("grep_logs", {})
        assert not result.ok
        assert "no such tool" in result.text
        assert "query_logs" in result.text

    def test_non_object_arguments_are_rejected(self, toolbox) -> None:
        assert not toolbox.call("query_logs", ["oops"]).ok  # type: ignore[arg-type]

    def test_no_call_raises(self, toolbox) -> None:
        """The loop must survive anything the model emits."""
        for name, arguments in [
            ("query_logs", {"group_by": 5}),
            ("query_logs", {"severity": {"nested": "object"}}),
            ("search_logs", {}),
            ("search_logs", {"text": ""}),
            ("search_docs", {"query": None}),
            ("fetch_records", {"ids": "L1"}),
            ("fetch_records", {"ids": [1, 2]}),
            ("nonexistent", {}),
        ]:
            result = toolbox.call(name, arguments)  # type: ignore[arg-type]
            assert isinstance(result.text, str) and result.text


class TestLogProfile:
    def test_returns_the_computed_profile(self, toolbox) -> None:
        result = toolbox.call("log_profile", {})
        assert result.ok
        assert "corpus profile" in result.text

    def test_says_so_when_no_profile_was_computed(self) -> None:
        result = ToolBox(logs=InMemoryLogStore()).call("log_profile", {})
        assert not result.ok


class TestQueryLogs:
    def test_a_bare_count(self, toolbox) -> None:
        result = toolbox.call("query_logs", {"component": "NETWORK"})
        assert result.ok
        assert "records match" in result.text

    def test_grouping_returns_counts_with_citations(self, toolbox) -> None:
        result = toolbox.call(
            "query_logs", {"group_by": ["ns"], "plan_summary": "COLLSCAN"}
        )
        assert result.ok
        assert "shop.orders" in result.text
        assert result.cited_ids
        assert all(i.startswith("L") for i in result.cited_ids)

    def test_a_single_group_by_string_is_accepted(self, toolbox) -> None:
        """Models send a bare string for an array field constantly, and the
        intent is unambiguous."""
        assert toolbox.call("query_logs", {"group_by": "component"}).ok

    def test_an_invented_group_field_is_named_with_the_alternatives(
        self, toolbox
    ) -> None:
        result = toolbox.call("query_logs", {"group_by": ["hostname"]})
        assert not result.ok
        assert "hostname" in result.text
        assert "component" in result.text

    def test_a_string_integer_is_accepted(self, toolbox) -> None:
        assert toolbox.call("query_logs", {"min_duration_ms": "1000"}).ok

    def test_a_naive_timestamp_is_rejected_with_an_example(self) -> None:
        """There is no correct timezone to assume on the model's behalf, and
        comparing naive against aware raises."""
        box = ToolBox(logs=InMemoryLogStore())
        result = box.call("query_logs", {"start": "2026-08-30T14:00:00"})
        assert not result.ok
        assert "ISO-8601" in result.text
        assert "Z" in result.text

    def test_an_offset_timestamp_is_accepted(self, toolbox) -> None:
        assert toolbox.call("query_logs", {"start": "2026-08-30T09:00:00Z"}).ok
        assert toolbox.call("query_logs", {"end": "2026-08-30T09:30:00+00:00"}).ok

    def test_a_filter_matching_nothing_says_so(self, toolbox) -> None:
        """A blank result reads as a broken tool; "no records match" is data."""
        result = toolbox.call(
            "query_logs", {"group_by": ["ns"], "component": "SHARDING"}
        )
        assert result.ok
        assert "no records match" in result.text

    def test_limit_is_capped(self, toolbox) -> None:
        result = toolbox.call("query_logs", {"group_by": ["log_id"], "limit": 10_000})
        assert result.ok
        assert result.text.count("\n") <= MAX_ROWS + 2


class TestSearchLogs:
    def test_finds_and_cites_records(self, toolbox) -> None:
        result = toolbox.call("search_logs", {"text": "Slow query", "limit": 5})
        assert result.ok
        assert len(result.cited_ids) == 5
        assert "COMMAND:51803" in result.text

    def test_an_unknown_order_lists_the_valid_ones(self, toolbox) -> None:
        result = toolbox.call("search_logs", {"text": "x", "order": "duration"})
        assert not result.ok
        assert "-duration" in result.text

    def test_a_miss_is_reported_rather_than_returned_empty(self, toolbox) -> None:
        result = toolbox.call("search_logs", {"text": "quorum reconfiguration"})
        assert result.ok
        assert "no records contain" in result.text

    def test_rendered_records_redact_command_values(self, toolbox) -> None:
        """Literal values are most of a record's token cost and none of its
        diagnostic content."""
        result = toolbox.call(
            "search_logs", {"text": "Slow query", "limit": 3, "order": "-duration"}
        )
        # The field name survives -- it is what says whether an index would help.
        assert "customerEmail" in result.text
        # The literal value does not.
        assert "<string>" in result.text
        assert "value-" not in result.text


class TestSearchDocs:
    def test_returns_passages_with_resolvable_chunk_ids(self, toolbox) -> None:
        result = toolbox.call("search_docs", {"query": "slow query threshold"})
        assert result.ok
        assert result.cited_ids
        assert all("#" in i for i in result.cited_ids)

    def test_says_so_when_no_index_is_loaded(self) -> None:
        result = ToolBox(logs=InMemoryLogStore()).call("search_docs", {"query": "x"})
        assert not result.ok
        assert "no documentation index" in result.text


class TestFetchRecords:
    def test_resolves_citations(self, toolbox) -> None:
        result = toolbox.call("fetch_records", {"ids": ["L1", "L2"]})
        assert result.ok
        assert "L1" in result.text and "L2" in result.text

    def test_missing_ids_are_named_individually(self, toolbox) -> None:
        """The model needs to know *which* citation does not exist. This is the
        mechanism by which it can catch itself inventing one."""
        result = toolbox.call("fetch_records", {"ids": ["L1", "L999999"]})
        assert "not found: L999999" in result.text
        assert "L999999" not in result.cited_ids


class TestBudget:
    @pytest.mark.parametrize(
        ("name", "arguments"),
        [
            ("log_profile", {}),
            ("query_logs", {"group_by": ["log_id"], "limit": 50}),
            ("search_logs", {"text": "Slow query", "limit": 50}),
            ("search_docs", {"query": "log message format", "limit": 50}),
            ("fetch_records", {"ids": [f"L{n}" for n in range(1, 50)]}),
        ],
    )
    def test_every_result_stays_within_budget(self, toolbox, name, arguments) -> None:
        result = toolbox.call(name, arguments)
        # log_profile carries the profile's own larger budget; the rest are
        # bounded by the per-result one.
        ceiling = 4_000 if name == "log_profile" else toolbox.budget_tokens
        assert estimate_tokens(result.text) <= ceiling
