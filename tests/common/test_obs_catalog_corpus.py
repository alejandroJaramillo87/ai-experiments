"""Catalog tests against observability output the rust harness really wrote.

test_obs_catalog.py builds its records inline, so they parse by construction and
say nothing about whether the Rust writer and the Python reader agree on real
bytes. These run the same readers over a captured sample (see
fixtures/README.md), which is the only place the cross-language contract is
checked against data neither language authored for a test.

Scope is deliberately narrow: parsing real bytes, and the rendering properties
that only appear once a table has more than a handful of rows. Filter semantics
stay in test_obs_catalog.py, where synthetic records express the cases directly.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from common.obs import catalog

FIXTURES = Path(__file__).parent / "fixtures"
RUNS_SAMPLE = FIXTURES / "obs_runs_sample.jsonl"
EVENTS_SAMPLE = FIXTURES / "obs_events_sample.jsonl"

# The captured sample's shape, asserted so that replacing the fixture with a
# differently-shaped capture fails here rather than silently weakening the
# tests below.
SAMPLE_ROWS = 50


def args(**overrides) -> argparse.Namespace:
    """The runs subcommand's defaults, matching its parser registration."""
    values = {"all": False, "failed": False, "limit": 20, "json": False}
    values.update(overrides)
    return argparse.Namespace(**values)


@pytest.fixture
def catalog_is_the_sample(monkeypatch):
    """Point the reader at the captured sample instead of the log root.

    run_command resolves its own path from the environment, so redirecting
    catalog_path is what lets the real handler run over a fixture file.
    """
    monkeypatch.setattr(catalog, "catalog_path", lambda cli: RUNS_SAMPLE)


def test_every_captured_row_parses():
    """A drifting writer presents as an empty table, not an error.

    read_runs skips a line it cannot parse rather than raising, so this is the
    assertion the inline-record tests structurally cannot make: they cannot
    produce a line that fails to parse without writing one on purpose.
    """
    records = catalog.read_runs(RUNS_SAMPLE)
    assert len(records) == SAMPLE_ROWS


def test_captured_rows_carry_exactly_the_contract_keys():
    """The six catalog keys named by schemas/obs_record.json, no more, no less."""
    contract = json.loads(
        (Path(__file__).parents[2] / "schemas" / "obs_record.json").read_text()
    )
    expected = set(contract["catalog"]["required"])
    for record in catalog.read_runs(RUNS_SAMPLE):
        assert set(record) == expected, record


def test_the_event_sample_joins_to_a_catalog_row():
    """The correlation join the contract exists to protect.

    An event file and its catalog row are written by separate code paths and
    tied together only by run_id; if either side drifts, the join is what
    breaks, and it breaks silently.
    """
    events = [json.loads(line) for line in EVENTS_SAMPLE.read_text().splitlines()]
    run_ids = {event["run_id"] for event in events}
    assert len(run_ids) == 1, "one event file describes one run"
    run_id = run_ids.pop()

    catalog_ids = {r["run_id"] for r in catalog.read_runs(RUNS_SAMPLE)}
    assert run_id in catalog_ids

    assert [e["event"] for e in events] == ["cmd.start", "fail", "cmd.end"]


def test_the_table_renders_every_captured_row(capsys, catalog_is_the_sample):
    """A header plus one line per row, with no field falling back to the
    unknown-value marker the reader uses for a foreign or hand-edited row."""
    assert catalog.run_command("harness", args(limit=0)) == 0
    lines = capsys.readouterr().err.strip().splitlines()
    assert len(lines) == SAMPLE_ROWS + 1, "header plus one line per row"
    assert lines[0].startswith("FINISHED")
    for line in lines[1:]:
        assert "?" not in line, line


# Deliberately no column-alignment test here. Alignment is a property of a
# table whose commands differ in width, and every command in this corpus is at
# most five characters -- under the floor of seven that command_width applies.
# A per-row width bug is therefore invisible against this data: mutating
# catalog.py to compute the width per row rather than per table leaves all nine
# tests in this module green. The assertion lives in test_obs_catalog.py, where
# a row wide enough to make it bite can be written on purpose.


@pytest.mark.parametrize("limit", [1, 7, 20, SAMPLE_ROWS])
def test_limit_selects_from_the_tail_at_scale(capsys, catalog_is_the_sample, limit):
    """The limit takes the most recent rows, which only distinguishes itself
    from taking the first rows once the table is longer than the limit."""
    assert catalog.run_command("harness", args(limit=limit, json=True)) == 0
    shown = json.loads(capsys.readouterr().out)
    everything = catalog.read_runs(RUNS_SAMPLE)
    assert shown == everything[-limit:]
