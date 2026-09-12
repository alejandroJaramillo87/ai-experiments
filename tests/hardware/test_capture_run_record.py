"""RHX.5, a capture's run record on real weights, as a test.

The capture manifests always recorded provenance about the artifact: the
model hash, the dataset digest, the git sha. Until U9c nothing recorded
provenance about the run, the thread count, the fold and the load time, and
that absence is the shape the `DotKind` regression hid in. The sandbox half,
`rust-harness/runtime/tests/capture_obs.rs`, proves the record's shape over the
tiny fixture for all three capture binaries: one header, one `model.load`
phase, one catalog row, no run directory.

What the fixture cannot show is the real checkpoint. This runs
`activation-capture` on E4B and reads the record where RHX.5 read it: the
header's `model` is the path passed, its `examples` is the dataset's length,
the load is seconds rather than the fixture's milliseconds, and the row lands
in the shared catalog under the default obs root beside the ladder's and the
chat's.

The dataset is the first eight lines of the refusal set, written to the test's
own directory. RHX.5's hand run captured all 258 in 293 s; the record is the
same shape at any length, and eight keeps the test to one load plus seconds.

`hw_full` rather than `hw_smoke`: the run resident-loads about 14 GiB and its
event file and catalog row stay in the real obs root under `/mnt/ai-data`.
`hw_e4b` because the released E4B checkpoint is the subject.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import pytest

from common.obs.catalog import read_runs
from tests.hardware.conftest import (
    HARNESS_DIR,
    LOG_ROOT,
    REPO_ROOT,
    default_root_env,
    e4b_checkpoint,
    events,
)

pytestmark = [
    pytest.mark.hw_full,
    pytest.mark.hw_e4b,
    pytest.mark.cpu,
    pytest.mark.ram,
    pytest.mark.ssd,
]

CAPTURE_BIN = os.path.join(HARNESS_DIR, "target", "release", "activation-capture")
CATALOG = Path(LOG_ROOT) / "runs.jsonl"  # read_runs takes a Path, not a str
SOURCE = os.path.join(REPO_ROOT, "interp", "probes", "prompts", "refusal.jsonl")
EXAMPLES = 8
LAYERS = "8,16,24"  # the band the runbook passes

OK_LINE = re.compile(
    r"^\[OK\] Capture: (\d+) examples, (\d+) layers, (\d+) shards -> (.+)$"
)

# The fixture loads in milliseconds; E4B took 5.33, 5.49 and 5.70 s on
# 2026-09-12 across the ladder, the capture and the chat. One second separates
# the two without pinning a page-cache state.
MIN_LOAD_S = 1.0
# One load plus eight forward passes of a dozen tokens each.
RUN_TIMEOUT_S = 10 * 60


def require_binary() -> None:
    """A missing binary fails rather than skips: it is built on the same
    host, so its absence is a misconfigured run and not the unavailable
    hardware the conftest rule exempts."""
    assert os.path.isfile(CAPTURE_BIN), (
        f"built binary absent: {CAPTURE_BIN}; run "
        f"`cargo build --release -p runtime --bin activation-capture` in {HARNESS_DIR}"
    )


class Run:
    """One capture: what was passed, what was printed, and what was recorded."""

    def __init__(
        self,
        model: str,
        dataset: str,
        out: str,
        result: subprocess.CompletedProcess,
        row: dict,
    ) -> None:
        self.model = model
        self.dataset = dataset
        self.out = out
        self.result = result
        self.row = row
        self.run_id = row["run_id"]
        self.events = events(self.run_id)

    def named(self, event: str) -> list[dict]:
        return [e for e in self.events if e["event"] == event]

    def one(self, event: str) -> dict:
        found = self.named(event)
        assert len(found) == 1, f"expected one {event}, read {found}"
        return found[0]


@pytest.fixture(scope="module")
def run(tmp_path_factory) -> Run:
    """One capture of eight examples on E4B, run once per module: the model
    load is the cost and every test reads the same record."""
    require_binary()
    model = e4b_checkpoint()
    if not os.path.isfile(os.path.join(model, "model.safetensors")):
        pytest.skip(f"E4B checkpoint not on disk at {model}")

    work = tmp_path_factory.mktemp("rhx5")
    dataset = work / "dataset.jsonl"
    with open(SOURCE, encoding="utf-8") as source:
        lines = [next(source) for _ in range(EXAMPLES)]
    dataset.write_text("".join(lines), encoding="utf-8")
    out = work / "out"

    # Rows already present are counted and not read: the run id comes from
    # the one row this run appends.
    before = len(read_runs(CATALOG)) if CATALOG.is_file() else 0
    result = subprocess.run(
        [
            CAPTURE_BIN,
            "--model",
            model,
            "--text-only",
            "--dataset",
            str(dataset),
            "--layers",
            LAYERS,
            "--out",
            str(out),
        ],
        cwd=work,
        capture_output=True,
        text=True,
        timeout=RUN_TIMEOUT_S,
        env=default_root_env(),
    )
    assert result.returncode == 0, f"the capture failed:\n{result.stderr}"
    rows = read_runs(CATALOG)[before:]
    assert len(rows) == 1, f"expected one appended catalog row, read {rows}"
    return Run(model, str(dataset), str(out), result, rows[0])


def test_one_header_names_the_invocation(run):
    header = run.one("activation-capture.header")
    assert header["model"] == run.model, header
    assert header["dataset"] == run.dataset, header
    assert header["out"] == run.out, header
    assert header["layers"] == LAYERS, header
    assert header["examples"] == EXAMPLES, header
    assert header["text_only"] is True, header
    # The fold and the build are the reason the header exists: the knobs
    # that hid the DotKind regression, and the binary that ran under them.
    for key in ("threads", "dot", "batch", "binary_version", "git_sha"):
        assert header.get(key) not in (None, ""), f"header lacks {key}: {header}"

    printed = [OK_LINE.match(line) for line in run.result.stderr.splitlines()]
    printed = [m for m in printed if m]
    assert len(printed) == 1, f"expected one [OK] Capture line:\n{run.result.stderr}"
    assert int(printed[0].group(1)) == header["examples"], printed[0].group(0)


def test_one_model_load_phase_in_seconds(run):
    phases = run.named("phase.end")
    assert len(phases) == 1, f"expected one phase.end, read {phases}"
    assert phases[0]["phase"] == "model.load", phases[0]
    assert phases[0]["duration_s"] >= MIN_LOAD_S, (
        f"a {phases[0]['duration_s']} s load is the fixture's, not E4B's"
    )
    # The header carries the fold the load resolved, so it is written after
    # the phase ends; the 2026-09-12 hand run showed that order.
    header = run.one("activation-capture.header")
    assert run.events.index(phases[0]) < run.events.index(header), [
        e["event"] for e in run.events
    ]


def test_no_run_directory_and_the_row_says_so(run):
    # The scope decision U9c made: a capture writes an artifact, not a run
    # directory, so there is nothing for `report` to render.
    # The catalog records the path either way, so `run_dir` means one thing,
    # "where this run's metrics would be" (obs/src/lib.rs); the directory
    # itself is created only by a metrics header, which a capture never
    # writes.
    run_dir = Path(LOG_ROOT) / run.run_id
    assert run.row["run_dir"] == str(run_dir), run.row
    assert not run_dir.exists(), f"a capture opened a run directory at {run_dir}"


def test_the_catalog_row_names_this_run(run):
    assert run.row["command"] == "activation-capture", run.row
    assert run.row["exit_code"] == 0, run.row
    header = run.one("activation-capture.header")
    end = run.one("cmd.end")
    assert header["run_id"] == end["run_id"] == run.run_id, (header, end, run.row)
    assert end["exit_code"] == 0, end
