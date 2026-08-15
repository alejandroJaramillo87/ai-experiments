"""The shared runs command surface (roadmap U5).

common/obs/runs.py is the one definition of the four selection flags, reached
three ways: the training subcommand, the interp subcommand, and the standalone
entry point the rust harness is browsed through. These tests pin the properties
that a refactor collapsing three call sites onto one module can silently lose --
a default drifting apart from the callers that assume it, a viewer regaining the
power to prune, and the standalone path resolving the wrong catalog.

The end-to-end case reads bytes the Rust writer produced (see
fixtures/README.md), so it checks the entry point against the real writer rather
than against a Python-authored imitation of it.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pytest

from common.obs import runs

FIXTURES = Path(__file__).parent / "fixtures"
RUNS_SAMPLE = FIXTURES / "obs_runs_sample.jsonl"

SAMPLE_ROWS = 50


@pytest.fixture
def log_root(tmp_path, monkeypatch):
    """A log root holding the captured harness catalog at its real path."""
    (tmp_path / "harness").mkdir()
    shutil.copy(RUNS_SAMPLE, tmp_path / "harness" / "runs.jsonl")
    monkeypatch.setenv("OBS_LOG_DIR", str(tmp_path))
    return tmp_path


def test_add_arguments_defines_the_documented_defaults():
    """catalog.run_command reads four attributes and tolerates none missing.

    test_obs_catalog_corpus.py builds this same Namespace by hand, commented
    "matching its parser registration". This is the assertion that keeps the two
    from drifting: the hand-written defaults there are the ones defined here.
    """
    parser = argparse.ArgumentParser()
    runs.add_arguments(parser)
    args = parser.parse_args([])
    assert vars(args) == {"limit": 20, "failed": False, "all": False, "json": False}


def test_main_reads_the_harness_catalog(capsys, log_root):
    """The standalone path resolves ${OBS_LOG_DIR}/harness/runs.jsonl.

    Bytes written by rust-harness/harness/src/obs.rs, read by the Python
    reader -- the cross-language seam the whole stage exists to expose.
    """
    assert runs.main(["--cli", "harness", "--json", "-n", "0"]) == 0
    records = json.loads(capsys.readouterr().out)
    assert len(records) == SAMPLE_ROWS


def test_main_renders_a_table_on_stderr(capsys, log_root):
    """stdout stays reserved for the --json payload."""
    assert runs.main(["--cli", "harness", "-n", "5"]) == 0
    captured = capsys.readouterr()
    assert captured.out == ""
    lines = captured.err.strip().splitlines()
    assert lines[0].startswith("FINISHED")
    assert len(lines) == 6, "header plus five rows"


def test_main_separates_the_tiers(capsys, log_root):
    """--cli selects the catalog; an empty tier is empty, not the harness one."""
    assert runs.main(["--cli", "training"]) == 0
    assert "No recorded runs" in capsys.readouterr().err


def test_an_unknown_cli_is_a_usage_error(capsys):
    """Exit 2 for bad arguments, per the Python output style guide."""
    with pytest.raises(SystemExit) as excinfo:
        runs.main(["--cli", "bogus"])
    assert excinfo.value.code == 2


def test_cli_choices_cover_every_catalog_writer():
    """A fourth writer must reach the browser, not just the log root."""
    assert set(runs.CLI_CHOICES) == {"training", "interp", "harness"}


@pytest.mark.parametrize("cli", ["training", "interp"])
def test_the_subcommand_wrappers_stay_viewers(cli):
    """obs_retention=False is what stops a browser pruning what it displays.

    The flag is set in one place now, so losing it would silently arm every
    runs invocation to delete the event files it was asked to list.
    """
    module = __import__(f"{cli}.commands.runs", fromlist=["configure"])
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command")
    module.configure(sub)
    args = parser.parse_args(["runs"])
    assert args.obs_retention is False
    assert args.limit == 20


@pytest.mark.parametrize("cli", ["training", "interp", "harness"])
def test_the_description_names_the_catalog_it_reads(cli):
    """Sharing the module must not cost the per-tier help text.

    Each wrapper previously carried a docstring naming its own path; a runs
    command whose help does not say which catalog it reads is worse than the
    duplication this stage removed.
    """
    assert f"/{cli}/runs.jsonl" in runs.describe(cli)
