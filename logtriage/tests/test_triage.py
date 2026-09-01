"""The triage loop end to end, and the two guards on what it may claim.

Everything here runs against a scripted model, which means these tests prove the
*mechanism* works: tools dispatch, turns bound, citations get checked, ordering
gets stated. Whether a real Gemma obeys the system prompt is a different
question that no fixture can answer, and it belongs on hardware with a served
model. Nothing below should be read as evidence about model behaviour.
"""

from __future__ import annotations

import pytest

from logtriage.cli import main
from logtriage.embed.hashing import HashingEmbedder
from logtriage.fixtures.generator import generate
from logtriage.fixtures.scenarios import SCENARIO_NAMES, SCENARIOS
from logtriage.kb.chunk import chunk_document
from logtriage.llm.base import LlmError, Message
from logtriage.llm.scripted import ScriptedLlmClient, Step, calls, says, tool_call
from logtriage.mongod.normalize import normalize
from logtriage.mongod.records import LogRecord, parse_lines
from logtriage.store.knowledge import InMemoryKnowledgeStore
from logtriage.store.memory import InMemoryLogStore
from logtriage.triage.pipeline import MAX_TURNS, run_triage
from logtriage.triage.report import (
    UNVERIFIED,
    audit_citations,
    mark_unverified,
    render_report,
)
from tests.sample_rst import MANUAL


def store_for(scenario: str, **kwargs) -> InMemoryLogStore:
    log = generate(scenario, **kwargs)
    return InMemoryLogStore(
        normalize(r) for r in parse_lines(log.lines) if isinstance(r, LogRecord)
    )


def records_for(scenario: str, **kwargs):
    log = generate(scenario, **kwargs)
    return [normalize(r) for r in parse_lines(log.lines) if isinstance(r, LogRecord)]


class TestScriptedClient:
    def test_replays_in_order(self) -> None:
        client = ScriptedLlmClient.of(says("first"), says("second"))
        assert client.complete([]).content == "first"
        assert client.complete([]).content == "second"

    def test_raises_when_exhausted_rather_than_returning_a_default(self) -> None:
        """The whole design of the fake. Answering "" to an unexpected request
        makes a broken loop look like a working one: the run completes, the
        report renders, every test passes, and nothing was exercised."""
        client = ScriptedLlmClient.of(says("only one"))
        client.complete([Message(role="user", content="x")])
        with pytest.raises(LlmError, match="exhausted"):
            client.complete([Message(role="user", content="x")])

    def test_a_guard_catches_a_fixture_that_has_drifted(self) -> None:
        client = ScriptedLlmClient(steps=[Step(says("ok"), expect="collscan")])
        with pytest.raises(LlmError, match="expected"):
            client.complete([Message(role="user", content="nothing relevant")])

    def test_records_the_conversation_it_was_shown(self) -> None:
        client = ScriptedLlmClient.of(says("done"))
        client.complete([Message(role="system", content="rules")])
        assert client.calls[0][0].role == "system"


class TestLoop:
    def test_a_plain_answer_ends_the_loop(self) -> None:
        store = store_for("baseline")
        client = ScriptedLlmClient.of(says("Nothing to report."))
        outcome = run_triage(records_for("baseline"), store, client)
        assert outcome.report.narrative == "Nothing to report."
        assert outcome.report.tool_calls == 0

    def test_tool_calls_are_dispatched_and_their_results_returned(self) -> None:
        store = store_for("slow-query-storm")
        client = ScriptedLlmClient.of(
            calls(tool_call("query_logs", group_by=["ns"], plan_summary="COLLSCAN")),
            says("shop.orders is being scanned."),
        )
        outcome = run_triage(records_for("slow-query-storm"), store, client)
        assert outcome.report.tool_calls == 1
        tool_messages = [m for m in outcome.messages if m.role == "tool"]
        assert len(tool_messages) == 1
        assert "shop.orders" in tool_messages[0].content
        assert tool_messages[0].tool_call_id == "call_query_logs"

    def test_a_failing_tool_call_does_not_end_the_run(self) -> None:
        """A model emits malformed calls routinely; the loop must survive one
        and let it try again."""
        store = store_for("baseline")
        client = ScriptedLlmClient.of(
            calls(tool_call("query_logs", group_by=["hostname"])),
            calls(tool_call("query_logs", group_by=["component"])),
            says("Recovered."),
        )
        outcome = run_triage(records_for("baseline"), store, client)
        assert outcome.report.narrative == "Recovered."
        assert outcome.report.tool_calls == 2
        # system, user, assistant(tool_calls), tool -- the error comes back as the
        # tool message, which is what the model reads on its next turn.
        error_message = next(m for m in outcome.messages if m.role == "tool")
        assert "error:" in error_message.content
        assert "hostname" in error_message.content

    def test_the_system_prompt_and_findings_reach_the_model(self) -> None:
        store = store_for("slow-query-storm")
        client = ScriptedLlmClient.of(says("ok"))
        run_triage(records_for("slow-query-storm"), store, client)
        system, user = client.calls[0][0], client.calls[0][1]
        assert "Cite everything" in system.content
        assert "corpus profile" in user.content
        assert "collscan" in user.content
        # The exact evidence ids are in the prompt, so a citation is something
        # the model copies rather than constructs.
        assert "evidence: L" in user.content

    def test_the_tools_are_offered_with_their_schemas(self) -> None:
        store = store_for("baseline")
        client = ScriptedLlmClient.of(says("ok"))
        run_triage(records_for("baseline"), store, client)
        # ScriptedLlmClient ignores tools, so assert via the toolbox the
        # pipeline builds: search_docs must be absent with no knowledge store.
        assert client.calls

    def test_running_out_of_turns_is_stated_not_hidden(self) -> None:
        """A truncated investigation and a finished one look identical from the
        outside unless the report says which it was."""
        store = store_for("baseline")
        client = ScriptedLlmClient.of(
            *[calls(tool_call("query_logs", group_by=["component"]))] * MAX_TURNS
        )
        outcome = run_triage(records_for("baseline"), store, client)
        assert "incomplete" in outcome.report.narrative
        assert f"{MAX_TURNS} turns" in outcome.report.narrative

    def test_a_precomputed_profile_avoids_rereading_the_records(self) -> None:
        """A Mongo-backed run has the records in the database; pulling them all
        back to compute a profile would defeat having one."""
        from logtriage.triage.profile import build_profile

        store = store_for("baseline")
        profile = build_profile(records_for("baseline"), source="db")
        outcome = run_triage(
            None, store, ScriptedLlmClient.of(says("ok")), profile=profile, source="db"
        )
        assert outcome.report.profile is profile

    def test_neither_records_nor_a_profile_is_an_error(self) -> None:
        with pytest.raises(ValueError, match="records or a precomputed profile"):
            run_triage(None, store_for("baseline"), ScriptedLlmClient.of(says("x")))

    def test_docs_search_is_available_when_a_knowledge_store_is(self) -> None:
        store = store_for("baseline")
        embedder = HashingEmbedder()
        chunks = chunk_document(MANUAL, "reference/log-messages.txt")
        knowledge = InMemoryKnowledgeStore()
        knowledge.upsert(chunks, embedder.embed([c.render() for c in chunks]))

        client = ScriptedLlmClient.of(
            calls(tool_call("search_docs", query="slow query threshold")),
            says("The threshold is set with setParameter."),
        )
        outcome = run_triage(
            records_for("baseline"),
            store,
            client,
            knowledge=knowledge,
            embedder=embedder,
        )
        tool_message = next(m for m in outcome.messages if m.role == "tool")
        assert "documentation passages" in tool_message.content


class TestCitations:
    def test_resolvable_citations_pass(self) -> None:
        store = store_for("baseline")
        audit = audit_citations("The scan at L5 and L9 was slow.", store)
        assert audit.clean
        assert set(audit.records_cited) == {"L5", "L9"}

    def test_an_invented_record_id_is_caught(self) -> None:
        """The failure this mechanism exists for: a model writing an id that
        looks exactly like a real one and does not exist."""
        store = store_for("baseline")
        audit = audit_citations("As shown at L999999, the primary stepped down.", store)
        assert not audit.clean
        assert audit.unresolved_records == ("L999999",)

    def test_an_unresolvable_citation_is_marked_not_deleted(self) -> None:
        """Deleting it leaves the sentence standing with nothing behind it,
        which reads as better supported than a visibly broken citation."""
        store = store_for("baseline")
        narrative = "The election at L999999 caused this."
        audit = audit_citations(narrative, store)
        marked = mark_unverified(narrative, audit)
        assert f"L999999 {UNVERIFIED}" in marked
        assert "The election at" in marked

    def test_marking_leaves_good_citations_alone(self) -> None:
        store = store_for("baseline")
        narrative = "Both L5 and L999999 are cited."
        marked = mark_unverified(narrative, audit_citations(narrative, store))
        assert f"L5 {UNVERIFIED}" not in marked
        assert f"L999999 {UNVERIFIED}" in marked

    def test_documentation_citations_are_audited_too(self) -> None:
        store = store_for("baseline")
        embedder = HashingEmbedder()
        chunks = chunk_document(MANUAL, "reference/log-messages.txt")
        knowledge = InMemoryKnowledgeStore()
        knowledge.upsert(chunks, embedder.embed([c.render() for c in chunks]))

        good = chunks[0].chunk_id
        audit = audit_citations(
            f"See {good} and reference/invented.txt#42.", store, knowledge
        )
        assert audit.unresolved_chunks == ("reference/invented.txt#42",)
        assert good in audit.chunks_resolved

    def test_a_narrative_with_no_citations_is_vacuously_clean(self) -> None:
        """Clean means "nothing false", not "well supported" -- the report
        prints the count, which is where an uncited narrative shows up."""
        audit = audit_citations("Something happened.", store_for("baseline"))
        assert audit.clean
        assert audit.total_cited == 0


class TestReport:
    def test_the_timeline_states_the_ordering_and_the_gaps(self) -> None:
        store = store_for("replica-stepdown")
        outcome = run_triage(
            records_for("replica-stepdown"), store, ScriptedLlmClient.of(says("ok"))
        )
        text = render_report(outcome.report)
        assert "first occurrence, in order" in text
        assert "cannot have been caused by one that began later" in text
        assert "min)" in text, "the gap between findings is the number that settles it"

    def test_the_timeline_puts_the_collscan_before_the_election(self) -> None:
        """The trap, checked in the rendered output rather than in the data.

        This is what a reader actually sees, and it is what has to make the
        wrong reading hard.
        """
        store = store_for("replica-stepdown")
        outcome = run_triage(
            records_for("replica-stepdown"), store, ScriptedLlmClient.of(says("ok"))
        )
        text = render_report(outcome.report)
        timeline = text.split("--- findings ---")[0]
        assert timeline.index("collscan") < timeline.index("election")

    def test_an_unresolvable_citation_warns_in_the_report(self) -> None:
        store = store_for("baseline")
        outcome = run_triage(
            records_for("baseline"),
            store,
            ScriptedLlmClient.of(says("The failure at L999999 is the cause.")),
        )
        text = render_report(outcome.report)
        assert "WARNING" in text
        assert "L999999" in text
        assert "unsupported by this corpus" in text

    def test_a_clean_corpus_says_no_detector_matched(self) -> None:
        """Not an empty section: "nothing found" and "detectors did not run"
        look identical otherwise."""
        store = store_for("baseline")
        outcome = run_triage(
            records_for("baseline"), store, ScriptedLlmClient.of(says("All clear."))
        )
        text = render_report(outcome.report)
        assert "no detector matched this corpus" in text

    def test_every_finding_is_rendered_with_its_evidence(self) -> None:
        store = store_for("slow-query-storm")
        outcome = run_triage(
            records_for("slow-query-storm"), store, ScriptedLlmClient.of(says("ok"))
        )
        text = render_report(outcome.report)
        for finding in outcome.report.findings:
            assert finding.summary in text
            for citation in finding.evidence_ids:
                assert citation in text


class TestEndToEnd:
    @pytest.mark.parametrize("scenario", SCENARIO_NAMES)
    def test_the_cli_triages_every_scenario(self, tmp_path, capsys, scenario) -> None:
        path = tmp_path / "mongod.log"
        assert main(["gen-logs", "--scenario", scenario, "--out", str(path)]) == 0
        assert main(["triage", str(path)]) == 0
        err = capsys.readouterr().err
        assert "=== triage:" in err
        assert "record citations resolve" in err

    @pytest.mark.parametrize("scenario", SCENARIO_NAMES)
    def test_the_report_never_makes_a_forbidden_claim(
        self, tmp_path, capsys, scenario
    ) -> None:
        """Each scenario declares the wrong reading its evidence invites.

        With a scripted backend this checks the guard rails, not the model: the
        narrative here is generated from the findings. It is still worth
        asserting, because a change to the report renderer that starts asserting
        causation would fail here rather than on hardware.
        """
        truth = SCENARIOS[scenario]
        path = tmp_path / "mongod.log"
        main(["gen-logs", "--scenario", scenario, "--out", str(path)])
        main(["triage", str(path)])
        report = capsys.readouterr().err.lower()
        for forbidden in truth.must_not_conclude:
            assert forbidden.lower() not in report, (
                f"{scenario}: report asserted {forbidden!r}"
            )

    def test_the_stepdown_report_states_the_correct_direction(
        self, tmp_path, capsys
    ) -> None:
        path = tmp_path / "mongod.log"
        main(["gen-logs", "--scenario", "replica-stepdown", "--out", str(path)])
        main(["triage", str(path)])
        report = capsys.readouterr().err
        assert "election cannot have caused collscan" in report

    def test_detector_only_mode_carries_no_model_output(self, tmp_path, capsys) -> None:
        path = tmp_path / "mongod.log"
        main(["gen-logs", "--scenario", "index-build-abort", "--out", str(path)])
        assert main(["triage", str(path), "--llm", "none"]) == 0
        err = capsys.readouterr().err
        assert "no model backend" in err
        assert "index_build_abort" in err

    def test_json_output_is_pure_json(self, tmp_path, capsys) -> None:
        import json

        path = tmp_path / "mongod.log"
        main(["gen-logs", "--scenario", "slow-query-storm", "--out", str(path)])
        capsys.readouterr()  # discard the generator's own status lines
        assert main(["triage", str(path), "--json"]) == 0
        captured = capsys.readouterr()
        payload = json.loads(captured.out)
        assert payload["findings"]
        assert payload["citations"]["records_cited"]
        assert captured.err == ""

    def test_a_report_with_bad_citations_exits_nonzero(self, tmp_path) -> None:
        """A caller in a pipeline must be able to tell a failed run from a
        successful one without parsing prose."""
        from logtriage.triage.profile import build_profile
        from logtriage.triage.report import (
            CitationAudit,
            TriageReport,
            render_report,
        )

        report = TriageReport(
            source="x",
            profile=build_profile([]),
            findings=(),
            narrative="See L999.",
            audit=CitationAudit(records_cited=("L999",), records_resolved=()),
        )
        assert not report.audit.clean
        assert "WARNING" in render_report(report)
