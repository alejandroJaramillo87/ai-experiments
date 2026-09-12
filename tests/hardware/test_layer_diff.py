"""RH.21, layer-diff against a transformers capture of the real E4B, as a test
the E4B tier runs.

`layer-diff` prefills the token ids a capture recorded, compares the residual
stream row by row against the capture, and names the first row over
tolerance. The ladder's L1 rung makes the same comparison (same capture, same
metrics, same Tier E floor and ceiling) and RH.1b records it passing, but L1
prints only the worst pair. This test reads every row the tool prints, which
is the table L1 never shows, and then does what a table of passes cannot:
closes the ceiling until two layer rows are over it and checks that the
verdict names the earlier of the two rather than the larger.

First, not worst, is the tool's whole policy: once layer k diverges every
later layer is downstream of it, so a verdict naming the largest figure sends
a reader to the end of the stack. A ceiling under one row cannot test that
policy, because the first row over it and the largest row are then the same
row. The bracket below sits between the second- and third-largest layer
figures, so exactly two rows exceed it, and the verdict has to choose.

The prelude rows come first in the table and decide first: a stage over the
ceiling displaces the layer verdict with a different message, by design
(`runtime/src/bin/layer-diff.rs`). The closed ceiling is therefore checked to
sit above every stage row before it is applied, and a stage row that would
not allow it is a finding about the stage's magnitude rather than a pass.

The capture is the ladder's own: `fixtures/reference/e4b/quick/layers.json`,
written by `fixtures/reference_capture.py` through the Makefile rule that
`make ladder` uses, so this table is directly comparable with RH.1b's L1
figures. Generated through that rule when absent, never by a second
convention here.
"""

from __future__ import annotations

import json
import os
import subprocess
from typing import NamedTuple

import pytest

from tests.hardware.conftest import REPO_ROOT, e4b_checkpoint

pytestmark = [pytest.mark.hw_full, pytest.mark.hw_e4b, pytest.mark.cpu, pytest.mark.ram]

HARNESS_DIR = os.path.join(REPO_ROOT, "rust-harness")
LAYER_DIFF = os.path.join(HARNESS_DIR, "target", "release", "layer-diff")
# The Makefile's `$(REFERENCE)/layers.json` at SIZE=quick.
LAYERS_JSON_REL = os.path.join("fixtures", "reference", "e4b", "quick", "layers.json")
LAYERS_JSON = os.path.join(HARNESS_DIR, LAYERS_JSON_REL)

# The tool's defaults, `COSINE_FLOOR` and `REL_L2_CEILING` in layer-diff.rs;
# the same two numbers as the ladder's Tier E.
COSINE_FLOOR = 0.9999
REL_L2_CEILING = 1e-3

# The pre-stack stages, in the order `runtime/src/model/cpu.rs` names them in
# `PRELUDE_STAGES`; the capture records whichever of them the checkpoint
# reaches, and the tool prints those before the layer rows.
PRELUDE_STAGES = ("embed", "ple_identity", "ple_context", "ple")

# One float32 transformers pass for the capture when it is missing, then a
# release model load and a short prefill per run. Generous, because a timeout
# here would read as a divergence.
CAPTURE_TIMEOUT_S = 30 * 60
RUN_TIMEOUT_S = 15 * 60


class Row(NamedTuple):
    label: str
    cosine: float
    max_abs: float
    rel_l2: float
    marked: bool
    line: str


def rows(stdout: str) -> list[Row]:
    """The stage and layer rows as the binary printed them, in order."""
    out = []
    for line in stdout.splitlines():
        fields = line.split()
        # Blank lines, and the column header whose second field is the word
        # "cosine".
        if len(fields) < 4 or fields[1] == "cosine":
            continue
        out.append(
            Row(
                label=fields[0],
                cosine=float(fields[1]),
                max_abs=float(fields[2]),
                rel_l2=float(fields[3]),
                marked="<--" in line,
                line=line,
            )
        )
    return out


def require_binary() -> None:
    """A missing binary fails rather than skips: it is built on the same
    host, so its absence is a misconfigured run and not the unavailable
    hardware the conftest rule exempts."""
    assert os.path.isfile(LAYER_DIFF), (
        f"built binary absent: {LAYER_DIFF}; run "
        f"`cargo build --release -p runtime --bin layer-diff` in {HARNESS_DIR}"
    )


def require_capture(model: str) -> None:
    """The ladder's capture, generated through the ladder's own rule when it
    is not already on disk."""
    if os.path.isfile(LAYERS_JSON):
        return
    result = subprocess.run(
        ["make", "-C", HARNESS_DIR, LAYERS_JSON_REL, "SIZE=quick", f"MODEL={model}"],
        capture_output=True,
        text=True,
        timeout=CAPTURE_TIMEOUT_S,
    )
    assert result.returncode == 0 and os.path.isfile(LAYERS_JSON), (
        f"make exited {result.returncode} without writing {LAYERS_JSON}\n"
        f"{result.stdout}{result.stderr}"
    )


class Run:
    """One layer-diff invocation: its exit code, streams and parsed rows."""

    def __init__(self, result: subprocess.CompletedProcess[str]) -> None:
        self.code = result.returncode
        self.stdout = result.stdout
        self.stderr = result.stderr
        self.rows = rows(result.stdout)

    def transcript(self) -> str:
        return f"stdout:\n{self.stdout}\nstderr:\n{self.stderr}"

    def stages(self) -> list[Row]:
        return [r for r in self.rows if r.label in PRELUDE_STAGES]

    def layers(self) -> list[Row]:
        return [r for r in self.rows if r.label not in PRELUDE_STAGES]


class Session:
    """The checkpoint, the capture, and a way to run the tool over them."""

    def __init__(self, model: str, directory: str) -> None:
        self.model = model
        self.directory = directory
        with open(LAYERS_JSON, encoding="utf-8") as f:
            capture = json.load(f)
        self.n_layers = int(capture["n_layers"])
        self.stages = [s for s in PRELUDE_STAGES if s in capture]

    def run(self, *extra: str) -> Run:
        # Its own working directory and log root, so the obs record every
        # layer-diff run opens lands beside the test and not in the tree.
        result = subprocess.run(
            [
                LAYER_DIFF,
                "--text-only",
                "--model",
                self.model,
                "--reference",
                LAYERS_JSON,
                *extra,
            ],
            cwd=self.directory,
            env={**os.environ, "OBS_LOG_DIR": os.path.join(self.directory, "logs")},
            capture_output=True,
            text=True,
            timeout=RUN_TIMEOUT_S,
        )
        return Run(result)


@pytest.fixture(scope="module")
def session(tmp_path_factory) -> Session:
    model = e4b_checkpoint()
    if not os.path.isfile(os.path.join(model, "model.safetensors")):
        pytest.skip(f"E4B checkpoint not on disk at {model}")
    require_binary()
    require_capture(model)
    return Session(model, str(tmp_path_factory.mktemp("rh21")))


@pytest.fixture(scope="module")
def defaults(session: Session) -> Run:
    """The comparison at the tool's defaults, shared by the tests below."""
    return session.run()


def test_the_defaults_pass_every_row(session, defaults):
    assert defaults.code == 0, defaults.transcript()
    assert f"all {session.n_layers} layers within tolerance" in defaults.stderr, (
        defaults.transcript()
    )
    marked = [r.line for r in defaults.rows if r.marked]
    assert not marked, "\n".join(marked)
    # The stages the capture recorded, in the order the backend computes
    # them, then every layer of the checkpoint.
    labels = [r.label for r in defaults.rows]
    expected = session.stages + [str(i) for i in range(session.n_layers)]
    assert labels == expected, defaults.stdout


def test_every_row_is_within_the_tier_e_tolerances(defaults):
    for r in defaults.rows:
        assert r.cosine >= COSINE_FLOOR, r.line
        assert r.rel_l2 <= REL_L2_CEILING, r.line


def test_a_closed_ceiling_names_the_first_row_over_it_not_the_worst(session, defaults):
    layers = defaults.layers()
    assert len(layers) >= 3, defaults.stdout
    by_size = sorted(layers, key=lambda r: r.rel_l2, reverse=True)
    largest, second, third = by_size[0], by_size[1], by_size[2]
    # A tie at the printed precision leaves no room between the second and
    # third figures, and the bracket would put a third row over the ceiling.
    assert second.rel_l2 > third.rel_l2, (
        f"no bracket between the second- and third-largest layer figures:\n"
        f"{second.line}\n{third.line}"
    )
    ceiling = (second.rel_l2 + third.rel_l2) / 2
    # A stage over the ceiling displaces the layer verdict by design, so the
    # bracket has to clear every stage row. Failed rather than skipped: a
    # stage row that large is a finding about the stage's magnitude.
    over = [s.line for s in defaults.stages() if s.rel_l2 > ceiling]
    assert not over, (
        f"a stage row sits above the ceiling {ceiling!r} the bracket needs, and "
        f"the tool would report the stack's input instead of a layer:\n"
        + "\n".join(over)
    )

    first = min(int(largest.label), int(second.label))
    closed = session.run("--rel-l2-ceiling", repr(ceiling))
    assert closed.code == 1, closed.transcript()
    assert f"first divergence at layer {first}" in closed.stderr, (
        f"the verdict did not name layer {first}, the earlier of "
        f"{largest.label} and {second.label}\n{closed.transcript()}"
    )
    marked = [r for r in closed.rows if r.marked]
    assert sorted(r.label for r in marked) == sorted([largest.label, second.label]), (
        "\n".join(r.line for r in marked) or closed.stdout
    )
    for r in marked:
        assert "(magnitude)" in r.line, r.line
    # The table agrees with the verdict: the first marked row is the earlier.
    assert marked[0].label == str(first), "\n".join(r.line for r in marked)
