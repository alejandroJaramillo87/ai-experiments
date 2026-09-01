"""The whole-corpus profile: correct, bounded, and honest about absence."""

from __future__ import annotations

import pytest

from logtriage.fixtures.generator import HOT_NS, generate
from logtriage.mongod.normalize import NormalizedRecord, normalize
from logtriage.mongod.records import LogRecord, parse_lines
from logtriage.tools.budget import (
    DEFAULT_PROFILE_TOKENS,
    estimate_tokens,
    truncate_rows,
)
from logtriage.triage.profile import build_profile, render_profile


def _normalized(scenario: str = "baseline", **kwargs) -> list[NormalizedRecord]:
    log = generate(scenario, **kwargs)
    return [normalize(r) for r in parse_lines(log.lines) if isinstance(r, LogRecord)]


def test_profile_counts_every_record() -> None:
    records = _normalized("slow-query-storm")
    profile = build_profile(records)
    assert profile.records_parsed == len(records)
    assert sum(profile.severity_counts.values()) == len(records)
    assert sum(profile.component_counts.values()) == len(records)


def test_profile_reports_the_span() -> None:
    profile = build_profile(_normalized("baseline", minutes=60))
    assert profile.duration_seconds is not None
    assert 55 * 60 < profile.duration_seconds <= 60 * 60


def test_profile_of_an_empty_corpus_does_not_crash() -> None:
    """An empty file is a real input -- a wrong path, or a rotated-away log."""
    profile = build_profile([])
    assert profile.records_parsed == 0
    assert profile.span_start is None
    assert profile.duration_seconds is None
    assert "empty corpus" in render_profile(profile)


def test_parse_errors_are_carried_into_the_profile() -> None:
    """Unparseable lines are reported, not silently dropped.

    A corpus where a tenth of the lines failed to parse is a different object
    from one where none did, and a summary that hides the difference invites
    conclusions drawn from a fraction of the evidence.
    """
    profile = build_profile(_normalized("baseline"), parse_errors=17)
    assert profile.parse_errors == 17
    assert "17 unparseable" in render_profile(profile)


@pytest.mark.parametrize(
    "scenario",
    ["baseline", "slow-query-storm", "connection-pool-exhaustion", "replica-stepdown"],
)
def test_rendered_profile_stays_inside_its_budget(scenario: str) -> None:
    """The whole point: a corpus no model can read becomes one it can."""
    text = render_profile(build_profile(_normalized(scenario, minutes=60)))
    assert estimate_tokens(text) <= DEFAULT_PROFILE_TOKENS


def test_profile_is_deterministic() -> None:
    records = _normalized("replica-stepdown")
    assert render_profile(build_profile(records)) == render_profile(
        build_profile(records)
    )


def test_collscan_namespace_ranks_first_and_is_cited() -> None:
    profile = build_profile(_normalized("slow-query-storm"))
    top = profile.namespaces[0]
    assert top.ns == HOT_NS
    assert top.collscan_count > 100
    assert top.worst_docs_per_returned and top.worst_docs_per_returned > 1000
    assert top.examples


def test_namespace_citations_point_at_the_collection_scans() -> None:
    """Citing three healthy queries on a namespace full of scans would mislead."""
    records = _normalized("slow-query-storm")
    by_id = {r.record_id: r for r in records}
    profile = build_profile(records)
    top = profile.namespaces[0]
    assert all(by_id[e].plan_summary == "COLLSCAN" for e in top.examples)


def test_baseline_profile_states_the_absence_of_errors() -> None:
    """Absence of evidence is a finding, so it is stated rather than omitted.

    A reader who simply does not see an error section cannot tell whether the
    corpus was clean or the section was dropped for budget.
    """
    text = render_profile(build_profile(_normalized("baseline")))
    assert "warnings and errors" in text
    assert "none: no records above severity I" in text


def test_error_corpus_reports_its_errors_with_citations() -> None:
    profile = build_profile(_normalized("index-build-abort"))
    assert profile.problem_templates
    text = render_profile(profile)
    assert "INDEX:" in text
    assert "[E]" in text


def test_clustered_error_text_reaches_the_profile() -> None:
    text = render_profile(build_profile(_normalized("index-build-abort")))
    assert "clustered error text" in text
    assert "memory limit" in text


def test_distinct_template_count_is_reported() -> None:
    """The collapse ratio is itself diagnostic: a corpus with a thousand
    distinct call sites is behaving very differently from one with twelve."""
    profile = build_profile(_normalized("connection-pool-exhaustion"))
    assert profile.distinct_templates > 0
    assert f"distinct call sites: {profile.distinct_templates}" in render_profile(
        profile
    )


class TestTruncation:
    """truncate_rows is the floor under every rendered payload."""

    def test_untruncated_output_has_no_marker(self) -> None:
        out = truncate_rows(["a", "b", "c"], 1_000)
        assert out == "a\nb\nc"
        assert "omitted" not in out

    def test_truncation_drops_from_the_tail_and_says_so(self) -> None:
        rows = [f"row-{n} " + "x" * 100 for n in range(50)]
        out = truncate_rows(rows, 60)
        assert "row-0" in out
        assert "row-49" not in out
        assert "omitted" in out

    def test_truncated_output_respects_the_budget(self) -> None:
        rows = [f"row-{n} " + "x" * 200 for n in range(200)]
        for budget in (20, 50, 200, 1_000):
            assert estimate_tokens(truncate_rows(rows, budget)) <= budget

    def test_rows_are_never_cut_in_half(self) -> None:
        """A model shown half a JSON record either refuses it or, worse, reads
        the fragment as complete -- and a claim built on half a record cites a
        record that does not say what the claim says."""
        rows = [f'{{"record": "L{n}", "duration": {n}}}' for n in range(100)]
        out = truncate_rows(rows, 40)
        for line in out.splitlines():
            if line.startswith("..."):
                continue
            assert line.endswith("}"), f"row was cut mid-record: {line!r}"

    def test_a_single_oversized_row_still_reports_the_omission(self) -> None:
        """The degenerate case must not silently produce an empty result."""
        out = truncate_rows(["x" * 10_000], 10)
        assert "1 more rows omitted" in out
