"""RH.13, the run-level observability records on the workstation, as a test.

The harness writes one event file per run and one catalog line per invocation
under ``${OBS_LOG_DIR}/harness/``, in the shape ``common/obs`` reads. The
sandbox half of that claim is ``rust-harness/harness/tests/obs_records.rs``,
which spawns the binary over the tiny fixture into a scratch root on every
``cargo test``. What it cannot show is the half RH.13 was written for: the
default root, ``/mnt/ai-data/logs``, resolved when ``OBS_LOG_DIR`` is unset,
a real checkpoint behind the successful run, and the Python reader running in
its own interpreter over rows the release binary appended to the workstation's
real catalog. Until this module that evidence was a pasted terminal block,
which nothing re-checked after the code under it changed; U7 changed the
terminal output and a cold-start phase changed the event file, and each
reopened the entry by hand.

The successful run is ``--chat`` with the prompt on stdin, for the reason
``test_interp_chain.py`` gives: a positional task puts the binary in
coding-agent mode, where the model may answer with a tool call and stop at a
permission prompt no test can answer. The catalog row therefore carries
``command: chat`` where the runbook's pasted task run carries ``task``; the
entry's expected signal names neither. The subject is ``gemma-4-E2B-it``
because chat mode wraps the prompt in the chat template, and a base
checkpoint continues that text instead of answering it (the 2026-09-06 base
runs recorded under RH.13). Every run gets its own empty directory so no
``session.kv`` from another model is resumed.

Writing outside a temporary directory is what makes this ``hw_full`` rather
than ``hw_smoke``: the two rows and two event files stay in the real catalog,
as any workstation run's do. The rows present before the runs are counted
and left alone; only the appended rows are asserted.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path

import pytest

from common.obs.catalog import read_runs
from tests.hardware.conftest import (
    AI_DATA_ROOT,
    HARNESS,
    LOG_ROOT,
    REPO_ROOT,
    default_root_env,
    events,
    require_binary,
)

pytestmark = [
    pytest.mark.hw_full,
    pytest.mark.cpu,
    pytest.mark.ram,
    pytest.mark.ssd,
    pytest.mark.harness,
]

CONTRACT = os.path.join(REPO_ROOT, "schemas", "obs_record.json")
CATALOG = Path(LOG_ROOT) / "runs.jsonl"  # read_runs takes a Path, not a str

RUN_ID = re.compile(r"R-\d{8}-\d{6}-[0-9a-f]{6}")
FAILURE_LINE = "[ERROR] Task failed: "
MISSING = "/nonexistent-checkpoint"

# The 2026-09-06 run loaded E2B-it in 3.69 s and prefilled 252 tokens in
# 11.40 s; the failing run ends before any weight is read. Generous, because
# a timeout here would read as a records failure.
RUN_TIMEOUT_S = 10 * 60


def checkpoint() -> str:
    root = os.environ.get("GEMMA4_CHECKPOINT_ROOT", f"{AI_DATA_ROOT}/models/hf")
    return os.path.join(root, "gemma-4-E2B-it")


def harness(
    directory: str, args: list[str], stdin: str | None
) -> subprocess.CompletedProcess:
    os.makedirs(directory, exist_ok=True)
    return subprocess.run(
        [HARNESS, *args],
        cwd=directory,
        input=stdin,
        capture_output=True,
        text=True,
        timeout=RUN_TIMEOUT_S,
        env=default_root_env(),
    )


class Records:
    """The two runs and the catalog rows they appended, in invocation order."""

    def __init__(self, ok, failed, rows):
        self.ok = ok
        self.failed = failed
        self.rows = rows


@pytest.fixture(scope="module")
def records(tmp_path_factory) -> Records:
    """One success and one forced failure against the default root, run once
    per module: the model load is the cost, and every test reads the same rows."""
    require_binary()
    model = checkpoint()
    if not os.path.isfile(os.path.join(model, "model.safetensors")):
        pytest.skip(f"E2B-it checkpoint not on disk at {model}")
    # Read with the same reader the workstation tooling uses. The rows already
    # present are counted and not read: the assertion is about what these two
    # runs add.
    before = len(read_runs(CATALOG)) if CATALOG.is_file() else 0

    base = tmp_path_factory.mktemp("rh13")
    ok = harness(
        str(base / "success"),
        ["--model", model, "--text-only", "--chat"],
        "Reply with the single word: ready\n",
    )
    assert ok.returncode == 0, f"the chat run failed:\n{ok.stderr}"
    failed = harness(str(base / "failure"), ["--model", MISSING, "task"], None)
    assert failed.returncode == 1, (
        f"expected exit 1, got {failed.returncode}:\n{failed.stderr}"
    )

    rows = read_runs(CATALOG)[before:]
    return Records(ok, failed, rows)


def test_a_success_and_a_failure_append_two_rows_to_the_default_catalog(records):
    rows = records.rows
    assert len(rows) == 2, f"expected two appended rows, read {rows}"
    assert [r["exit_code"] for r in rows] == [0, 1], rows
    assert [r["command"] for r in rows] == ["chat", "task"], rows
    for row in rows:
        assert row["run_dir"] == os.path.join(LOG_ROOT, row["run_id"]), (
            f"run_dir must sit under the default root: {row}"
        )
        assert isinstance(row["duration_s"], (int, float)), row
    with open(CONTRACT, encoding="utf-8") as handle:
        required = set(json.load(handle)["catalog"]["required"])
    for row in rows:
        assert set(row) == required, f"catalog keys drifted from the contract: {row}"


def test_run_ids_have_the_documented_shape(records):
    ids = [r["run_id"] for r in records.rows]
    for run_id in ids:
        assert RUN_ID.fullmatch(run_id), (
            f"run id {run_id!r} is not R-YYYYMMDD-HHMMSS-xxxxxx"
        )
    assert len(set(ids)) == len(ids), f"run ids are not distinct: {ids}"


def test_the_failing_run_states_its_fix_and_records_only_the_diagnosis(records):
    err = records.failed.stderr
    lines = [
        line
        for line in err.splitlines()
        if line.startswith(FAILURE_LINE)
        or line.startswith("  Solution: ")
        or line.startswith("  Command: ")
    ]
    assert len(lines) == 3, f"expected the failure line and two remedy lines:\n{err}"
    assert lines[0].startswith(FAILURE_LINE), err
    assert lines[1].startswith("  Solution: "), err
    assert lines[2] == f"  Command: ls {MISSING}", err
    printed = lines[0][len(FAILURE_LINE) :]
    assert records.failed.stdout == "", (
        f"a failure is not payload: {records.failed.stdout!r}"
    )

    row = records.rows[1]
    recorded = events(row["run_id"])
    names = [e["event"] for e in recorded]
    # The missing path is rejected inside `model.load`, the phase that brackets
    # the cold start, before any weight is read: the phase closes at once and
    # no run.header follows. The 2026-08-14 record predates the phase.
    assert names == ["cmd.start", "phase.start", "phase.end", "fail", "cmd.end"], (
        recorded
    )
    assert recorded[1]["phase"] == recorded[2]["phase"] == "model.load", recorded
    for event in recorded:
        assert event["run_id"] == row["run_id"], event

    fail = recorded[3]
    assert fail["level"] == "error", fail
    assert fail["exc_type"] == "anyhow", fail
    assert fail["exit_code"] == 1, fail
    # RH.13's coupling: the recorded message and the printed one are one
    # string, which is what lets a catalog row's exit_code 1 be turned into a
    # cause without re-running anything. The remedy reaches neither.
    assert fail["exc_msg"] == printed, (
        f"recorded {fail['exc_msg']!r}\nprinted {printed!r}"
    )
    assert "Solution:" not in fail["exc_msg"], fail
    assert "Command:" not in fail["exc_msg"], fail
    assert recorded[-1]["exit_code"] == 1, recorded[-1]


def test_the_successful_run_records_no_failure(records):
    row = records.rows[0]
    recorded = events(row["run_id"])
    names = [e["event"] for e in recorded]
    assert names[0] == "cmd.start", names
    assert names[-1] == "cmd.end", names
    assert "fail" not in names, names
    assert recorded[-1]["exit_code"] == 0, recorded[-1]
    for event in recorded:
        assert event["run_id"] == row["run_id"], event
    # The reply is the payload and stays on stdout; its wording is the model's
    # and is not asserted.
    assert records.ok.stdout.strip(), (
        f"the chat run produced no reply on stdout:\n{records.ok.stderr}"
    )
