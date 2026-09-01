"""Template collapse: exact for mongod's own ids, clustered for free text."""

from __future__ import annotations

from logtriage.fixtures.generator import generate
from logtriage.mongod.records import LogRecord, parse_lines
from logtriage.templates.extract import (
    TemplateIndex,
    cluster_free_text,
    free_text_values,
    template_key,
)


def _records(scenario: str = "baseline", **kwargs) -> list[LogRecord]:
    log = generate(scenario, **kwargs)
    return [r for r in parse_lines(log.lines) if isinstance(r, LogRecord)]


def test_template_identity_is_component_and_id() -> None:
    record = _records()[0]
    assert template_key(record) == f"{record.component}:{record.log_id}"


def test_collapse_is_large_and_exact() -> None:
    """A corpus is a few hundred distinct call sites repeated.

    The count is asserted exactly rather than approximately because the identity
    is exact -- there is no similarity threshold that could move it. An
    inferred-template approach could not make this assertion at all.
    """
    records = _records("slow-query-storm", minutes=60)
    index = TemplateIndex()
    index.extend(records)
    assert len(records) > 1000
    assert len(index) < 20
    assert sum(stat.count for stat in index.stats()) == len(records)


def test_collapse_is_deterministic() -> None:
    first = TemplateIndex()
    first.extend(_records("replica-stepdown"))
    second = TemplateIndex()
    second.extend(_records("replica-stepdown"))
    assert first.stats() == second.stats()


def test_stats_are_ordered_by_count_with_a_total_tiebreak() -> None:
    index = TemplateIndex()
    index.extend(_records("connection-pool-exhaustion"))
    stats = index.stats()
    assert stats == sorted(stats, key=lambda s: (-s.count, s.template))


def test_index_tracks_span_and_examples() -> None:
    records = _records("baseline")
    index = TemplateIndex()
    index.extend(records)
    stat = index.stats()[0]
    assert stat.count > 1
    assert stat.first_seen <= stat.last_seen
    assert 1 <= len(stat.examples) <= 3
    # Examples are citation handles, so they must resolve to real lines.
    by_id = {r.record_id: r for r in records}
    for example in stat.examples:
        assert by_id[example].component == stat.component


def test_index_reports_the_worst_severity_ever_seen() -> None:
    """A call site that is usually I and occasionally E is the interesting one."""
    records = _records("index-build-abort")
    index = TemplateIndex()
    index.extend(records)
    failures = [s for s in index.stats() if s.severity == "E"]
    assert failures
    assert all(s.component == "INDEX" for s in failures)


def test_free_text_is_extracted_from_both_attr_shapes() -> None:
    """attr.error is a string in some components and a sub-document in others."""
    records = _records("index-build-abort")
    messages = [m for r in records for m in free_text_values(r)]
    assert any("exceeded memory limit" in m for m in messages)


def test_free_text_clustering_collapses_varying_messages() -> None:
    """The one job drain3 has here: messages that differ only in variables."""
    messages = [
        f"Connection refused to host-{n}:27017 after {n} retries" for n in range(20)
    ]
    messages += ["Index build failed: exceeded memory limit for external sort"] * 3
    clusters = cluster_free_text(messages)
    assert len(clusters) == 2
    assert clusters[0].count == 20
    assert clusters[1].count == 3
    # The variable part is masked out, the fixed part survives.
    assert "Connection refused to" in clusters[0].template


def test_free_text_clustering_is_deterministic() -> None:
    """Two profiles of one corpus must agree, so no state may persist."""
    messages = [f"Cursor {n} timed out on shard-{n % 4}" for n in range(50)]
    assert cluster_free_text(messages) == cluster_free_text(messages)


def test_free_text_clustering_handles_no_messages() -> None:
    """The common case: a healthy corpus has no error text at all."""
    assert cluster_free_text([]) == []


def test_first_occurrence_is_not_stranded_in_its_own_cluster() -> None:
    """Drain generalizes a cluster as it grows, and reports the template it had
    at insertion time -- so the first member comes back with its literal text and
    the rest with wildcards. Grouping on that returned template splits every
    cluster in two. Counting keys on the cluster id instead; this pins it.
    """
    messages = [f"Cursor {n} timed out on shard-{n % 4}" for n in range(30)]
    clusters = cluster_free_text(messages)
    assert len(clusters) == 1
    assert clusters[0].count == 30
