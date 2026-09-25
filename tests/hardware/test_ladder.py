"""RH.1b, the correctness ladder at Tier E, as a test the hardware tiers run.

`make ladder` is the numerics oracle: seven rungs against `transformers` on
the real checkpoint. Until this test it was a make target and not a marker, so
`make quality-full HW=full` never reached it and "I ran the full gate" said
nothing about whether the backend computes the right numbers. This wraps the
target rather than reimplementing any rung: the ladder binary owns the
comparison and its thresholds, and what is asserted here is the contract of
its closing line.

The reference artifacts are gitignored (tens of MB at a 262144-wide
vocabulary) and `make ladder` generates any that are missing before it climbs,
so the only precondition left is the checkpoint, which skips when absent per
the conftest rule for real-model loads. Anything the binary reports once it
runs is a verdict, and a rung skipped inside the run is a failure here even
though the binary exits 0 for it: after the generators ran, a skip means one
wrote somewhere other than where the climb read.

The L5 detail line is read as well, for RH.17: the window this run's loader
read has to equal the window the reference generator recorded, because both
sides derive the scored positions from their own window and the rows are
paired by index. Agreement there is what makes L5's row comparison a
comparison at all, and the binary's verdict line does not carry it.

The L6 detail line is read for RH.18: the rows the rung compared have to be
the rows the artifact holds, over the layers, heads and positions it names
and at the window it recorded. L6 refuses a disagreement itself; reading the
line here is what ties the transcript to the artifact on disk.

RHX.3 (rust-harness.md) rides the same climb. Until U9b the ladder wrote
nothing to the obs root, so a red rung months later could not be attributed
to the configuration it ran under. Now one climb is one run: a `ladder.header`
event naming the checkpoint, the reference, the size and the fold, a
`model.load` phase and one `rung.L<n>` phase per rung, and a `metrics.jsonl`
in the run directory with one record per rung and the rung index as `step`.
The three tests at the bottom read that record off the run the catalog says
this climb appended, so the gate climbs once and not twice.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path

import pytest

from common.obs.catalog import read_runs
from common.obs.dashboard import read_metrics
from tests.hardware.conftest import LOG_ROOT, REPO_ROOT, e4b_checkpoint, events

pytestmark = [
    pytest.mark.hw_full,
    pytest.mark.hw_e4b,
    pytest.mark.ssd,
    pytest.mark.harness,
]

HARNESS = os.path.join(REPO_ROOT, "rust-harness")
SIZE = "quick"
CATALOG = Path(LOG_ROOT) / "runs.jsonl"  # read_runs takes a Path, not a str

# The verdict line ladder.rs prints for a green climb, with the skip count
# the runbook says to read instead of the exit code.
VERDICT = re.compile(
    r"^\[OK\] Ladder: every rung within tolerance \((\d+) skipped\)", re.M
)

# L5's detail line: the schedule size, and the window this run's loader read.
WINDOW_LINE = re.compile(
    r"^\s+(\d+) positions across a (\d+)-token window over (\d+) tokens", re.M
)
# The artifact `make ladder` generates and reads for SIZE=quick; its
# `sliding_window` is what transformers attended under when the rows were
# written (fixtures/reference_long_context.py).
LONG_CONTEXT = os.path.join(
    HARNESS, "fixtures", "reference", "e4b", SIZE, "long_context.json"
)
# ladder.rs fixes the schedule at six anchors (LONG_CONTEXT_POSITIONS).
LONG_CONTEXT_POSITIONS = 6

# The rung rows of the table on stdout, in the order the climb ran them. The
# check name is left unread; the label is what the phase and metrics names
# carry.
RUNG_ROW = re.compile(r"^(L\d)\s+\S.*\s(PASS|FAIL|skipped)$", re.M)
# The reference directory the make target passes for SIZE, relative to the
# rust-harness cwd it runs from, as the header records it.
REFERENCE = f"fixtures/reference/e4b/{SIZE}"

# L6's detail line: what the rung compared, and the window this run read.
ATTENTION_LINE = re.compile(
    r"^\s+(\d+) rows across (\d+) layers x (\d+) heads at (\d+) positions; "
    r"window (\d+);",
    re.M,
)
# The artifact `make ladder` generates and reads for SIZE=quick
# (fixtures/reference_attention.py).
ATTENTION = os.path.join(
    HARNESS, "fixtures", "reference", "e4b", SIZE, "attention.json"
)

# A cold run is six generator passes through transformers on the CPU (the
# perplexity, long-context and attention ones dominate) and then seven model
# loads and two 2052-token forward passes on the backend; the observed
# six-rung climb alone took 347 s of rung bodies. Generous, because a timeout
# here would read as a numerics failure.
TIMEOUT_S = 120 * 60


class Climb:
    """One `make ladder` run: its transcript, and the obs run it appended."""

    def __init__(
        self, model: str, result: subprocess.CompletedProcess, run_id: str
    ) -> None:
        self.model = model
        self.result = result
        self.transcript = result.stdout + result.stderr
        self.run_id = run_id
        self.events = events(run_id)
        self.run_dir = Path(LOG_ROOT) / run_id

    def printed_rungs(self) -> list[str]:
        return [m.group(1) for m in RUNG_ROW.finditer(self.result.stdout)]

    def phases(self) -> list[dict]:
        return [e for e in self.events if e["event"] == "phase.end"]


@pytest.fixture(scope="module")
def climb() -> Climb:
    """The quick climb, once per module. The run id is the one catalog row it
    appends; rows already present are counted and not read, so a concurrent
    reader of the catalog is not a concern the test has to share."""
    model = e4b_checkpoint()
    if not os.path.isfile(os.path.join(model, "model.safetensors")):
        pytest.skip(f"E4B checkpoint not on disk at {model}")

    before = len(read_runs(CATALOG)) if CATALOG.is_file() else 0
    result = subprocess.run(
        ["make", "ladder", f"SIZE={SIZE}", f"MODEL={model}"],
        cwd=HARNESS,
        capture_output=True,
        text=True,
        timeout=TIMEOUT_S,
    )
    transcript = result.stdout + result.stderr
    assert result.returncode == 0, (
        f"make ladder exited {result.returncode}\n{transcript}"
    )
    rows = read_runs(CATALOG)[before:]
    assert len(rows) == 1, f"expected one appended catalog row, read {rows}"
    return Climb(model, result, rows[0]["run_id"])


def test_the_ladder_passes_every_rung_at_tier_e(climb):
    transcript = climb.transcript
    verdict = VERDICT.search(transcript)
    assert verdict is not None, (
        f"no [OK] Ladder verdict line in the transcript\n{transcript}"
    )
    skipped = int(verdict.group(1))
    assert skipped == 0, (
        f"{skipped} rung(s) skipped after the generators ran; a skipped rung is"
        f" missing evidence, not a pass\n{transcript}"
    )

    # RH.17's half of L5. The run's window comes from the loader's config read
    # and the artifact's from transformers' config read of the same checkpoint;
    # a run reading a different window scores different positions and compares
    # rows computed at different places, which diverges rather than mislabels.
    detail = WINDOW_LINE.search(transcript)
    assert detail is not None, f"no L5 detail line in the transcript\n{transcript}"
    positions, reported = int(detail.group(1)), int(detail.group(2))
    with open(LONG_CONTEXT, encoding="utf-8") as handle:
        recorded = int(json.load(handle)["sliding_window"])
    assert reported == recorded, (
        f"the run reports a {reported}-token window; the reference at "
        f"{LONG_CONTEXT} was written under {recorded}"
    )
    assert positions == LONG_CONTEXT_POSITIONS, detail.group(0)

    # RH.18. The rung compared exactly the rows the artifact holds: as many
    # layers, heads and positions as it names, at the window it recorded.
    attention = ATTENTION_LINE.search(transcript)
    assert attention is not None, f"no L6 detail line in the transcript\n{transcript}"
    rows, layers, heads, positions, window = (int(g) for g in attention.groups())
    with open(ATTENTION, encoding="utf-8") as handle:
        artifact = json.load(handle)
    assert (layers, heads, positions, window) == (
        len(artifact["layers"]),
        int(artifact["heads"]),
        len(artifact["positions"]),
        int(artifact["sliding_window"]),
    ), f"{attention.group(0)} against {ATTENTION}"
    assert rows == len(artifact["rows"]) == layers * heads * positions, (
        f"{rows} rows compared, {len(artifact['rows'])} in the artifact"
    )


def test_one_ladder_header_names_the_climb(climb):
    headers = [e for e in climb.events if e["event"] == "ladder.header"]
    assert len(headers) == 1, f"expected one ladder.header, read {headers}"
    header = headers[0]
    assert header["command"] == "ladder", header
    assert header["model"] == climb.model, header
    assert header["reference"] == REFERENCE, header
    assert header["size"] == SIZE, header
    assert isinstance(header["corpus_tokens"], int) and header["corpus_tokens"] > 0, (
        header
    )
    # The fold is the reason the header exists: the three knobs that hid the
    # DotKind regression, so a red rung can be attributed to them.
    for key in ("threads", "dot", "batch"):
        assert header.get(key) not in (None, ""), f"header lacks {key}: {header}"


def test_one_model_load_then_one_phase_per_printed_rung(climb):
    printed = climb.printed_rungs()
    assert printed, f"no rung rows in the table:\n{climb.result.stdout}"
    phases = climb.phases()
    names = [p["phase"] for p in phases]
    assert names == ["model.load"] + [f"rung.{r}" for r in printed], (
        f"phases {names} against printed rungs {printed}"
    )
    for phase in phases:
        duration = phase["duration_s"]
        assert isinstance(duration, (int, float)) and duration > 0, phase


def test_the_run_directory_holds_one_metrics_record_per_rung(climb):
    path = climb.run_dir / "metrics.jsonl"
    assert path.is_file(), f"no metrics.jsonl at {path}"
    header, records, skipped = read_metrics(path)
    assert skipped == 0, f"{skipped} unreadable line(s) in {path}"
    assert header.get("report") == "run_metrics", header
    assert header.get("cli") == "harness" and header.get("command") == "ladder", header
    assert header.get("run_id") == climb.run_id, header

    printed = climb.printed_rungs()
    assert [r["step"] for r in records] == list(range(len(printed))), records
    assert [r["rung"] for r in records] == printed, records
    # The phase is stamped around the rung body by the run log and the
    # record's duration by `climb` around the same closure: two clocks on
    # one span, so they agree once both are rounded to the millisecond, give
    # or take the guard's own overhead.
    by_phase = {p["phase"]: p["duration_s"] for p in climb.phases()}
    for record in records:
        assert record["status"] == "PASS", record
        recorded = by_phase[f"rung.{record['rung']}"]
        assert abs(record["rung_duration_s"] - recorded) < 0.05, (
            f"{record['rung']}: metrics {record['rung_duration_s']} s, "
            f"phase {recorded} s"
        )
