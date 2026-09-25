"""RHX.6, a failure's `Command:` remedy runs as printed, as a test.

U7 made three failures print `Solution:` and `Command:` beneath the diagnosis.
The sandbox half, `rust-harness/harness/src/actionable.rs`'s unit tests and
`rust-harness/harness/tests/actionable_surface.rs`, pins the shape: three
lines in order, no `<placeholder>`, the command beginning with the binary as
it was invoked. What the sandbox never does is run the printed command. RHX.6
did, on 2026-09-12, and got `harness: command not found` (KB-72): the line was
well-formed by every assertion and unusable when typed, because `harness` is
not on any PATH and the operator runs `target/release/harness`.

So this test does what the operator did. It provokes the two failures the
runbook provokes on the released E4B, takes each `Command:` line off stderr,
hands it to a shell verbatim, and holds the shell's output to what the remedy
claimed: the `ls` shows the checkpoint missing, and the `--status --deep`
preflight reports the same identity fault the run hit, digest for digest.

The foreign probe is built here by the real exporter at the checkpoint's own
width, with a `model_hash` that is not the weights' digest, rather than copied
from the registry as the hand run did: a registered probe is a host artifact
the test would otherwise skip on. Width is not the fault reached either way.
`main.rs` compares the artifact's `model_hash` to the served weights before it
compares widths, and does so before the load, so the run prints no header and
costs no load; the 2026-09-12 run showed both.

`hw_full` rather than `hw_smoke`: the two runs write fail records into the
real obs root under `/mnt/ai-data`, and the replayed `--deep` hashes the 15 GB
weights once, about 35 s on 2026-09-11 (`test_preflight.py`). `hw_e4b` because
the released E4B checkpoint is the subject.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path

import pytest

from tests.hardware.conftest import (
    HARNESS,
    altered,
    default_root_env,
    e4b_checkpoint,
    require_binary,
    write_probe,
)

pytestmark = [
    pytest.mark.hw_full,
    pytest.mark.hw_e4b,
    pytest.mark.cpu,
    pytest.mark.ssd,
    pytest.mark.harness,
]

ERROR_LINE = re.compile(r"^\[ERROR\] Task failed: (.+)$")
SOLUTION_LINE = re.compile(r"^  Solution: (.+)$")
COMMAND_LINE = re.compile(r"^  Command: (.+)$")

# The run path's diagnosis and the preflight's verdict for the same fault.
# Two spellings, one pair of digests: the test holds the pair equal.
RUN_IDENTITY = re.compile(
    r"probe was trained on a different model: probe model_hash ([0-9a-f]{64}), "
    r"served model sha256 ([0-9a-f]{64})$"
)
PREFLIGHT_IDENTITY = re.compile(
    r"^\[ERROR\] probe identity: trained on a different model: "
    r"artifact model_hash ([0-9a-f]{64}), served model sha256 ([0-9a-f]{64})$",
    re.M,
)

# The layer RH.16's test probes; any layer the checkpoint has would do, the
# identity check never reads it.
PROBE_LAYER = 8
# A digest that is not the weights': the served one is read off the run.
FOREIGN_HASH = altered("0" * 64)

# The two failing runs answer before any load; the replayed preflight hashes
# the weights once.
RUN_TIMEOUT_S = 60
REPLAY_TIMEOUT_S = 5 * 60


class Failure:
    """One provoked failure and the shell's replay of the remedy it printed."""

    def __init__(self, result: subprocess.CompletedProcess[str], work: Path) -> None:
        self.result = result
        self.stderr_lines = result.stderr.splitlines()
        commands = [m.group(1) for m in map(COMMAND_LINE.match, self.stderr_lines) if m]
        assert len(commands) == 1, f"expected one Command: line:\n{result.stderr}"
        self.command = commands[0]
        # Verbatim, through a shell, from a directory that is not the repo:
        # what an operator pasting the line does.
        self.replay = subprocess.run(
            self.command,
            shell=True,
            cwd=work,
            capture_output=True,
            text=True,
            timeout=REPLAY_TIMEOUT_S,
            env=default_root_env(),
        )

    def transcript(self) -> str:
        return (
            f"run stdout:\n{self.result.stdout}\nrun stderr:\n{self.result.stderr}\n"
            f"replay of `{self.command}`:\nstdout:\n{self.replay.stdout}\n"
            f"stderr:\n{self.replay.stderr}"
        )

    def diagnosis(self) -> str:
        found = [m.group(1) for m in map(ERROR_LINE.match, self.stderr_lines) if m]
        assert len(found) == 1, f"expected one [ERROR] line:\n{self.result.stderr}"
        return found[0]


class Failures:
    def __init__(
        self, model: str, missing: str, image: Path, absent: Failure, foreign: Failure
    ):
        self.model = model
        self.missing = missing
        self.image = image
        self.absent = absent
        self.foreign = foreign


def harness(args: list[str], work: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [HARNESS, *args],
        cwd=work,
        capture_output=True,
        text=True,
        timeout=RUN_TIMEOUT_S,
        env=default_root_env(),
    )


@pytest.fixture(scope="module")
def failures(tmp_path_factory) -> Failures:
    """The runbook's two failures, provoked once, each remedy replayed once."""
    require_binary()
    model = e4b_checkpoint()
    if not os.path.isfile(os.path.join(model, "model.safetensors")):
        pytest.skip(f"E4B checkpoint not on disk at {model}")
    # A sibling of the real checkpoint, as the runbook's `absent` is, so the
    # `ls` lists a path under the real models root and not under tmp.
    missing = os.path.join(os.path.dirname(model), "absent-rhx6")
    assert not os.path.exists(missing), f"{missing} exists; the test needs it absent"

    work = Path(tmp_path_factory.mktemp("rhx6"))
    with open(os.path.join(model, "config.json"), encoding="utf-8") as handle:
        width = json.load(handle)["text_config"]["hidden_size"]
    image = write_probe(work, width, PROBE_LAYER, FOREIGN_HASH)

    absent = Failure(
        harness(["--model", missing, "--text-only", "say ready"], work), work
    )
    foreign = Failure(
        harness(
            ["--model", model, "--text-only", "--probe", str(image), "say ready"], work
        ),
        work,
    )
    return Failures(model, missing, image, absent, foreign)


def three_lines_and_nothing_else(failure: Failure) -> None:
    assert failure.result.returncode == 1, failure.transcript()
    assert failure.result.stdout == "", failure.transcript()
    assert len(failure.stderr_lines) == 3, failure.transcript()
    assert ERROR_LINE.match(failure.stderr_lines[0]), failure.transcript()
    assert SOLUTION_LINE.match(failure.stderr_lines[1]), failure.transcript()
    assert COMMAND_LINE.match(failure.stderr_lines[2]), failure.transcript()
    assert "<" not in failure.command and ">" not in failure.command, failure.command


def test_a_missing_checkpoint_is_three_lines_and_nothing_else(failures):
    three_lines_and_nothing_else(failures.absent)
    assert failures.absent.command == f"ls {failures.missing}", failures.absent.command


def test_the_ls_remedy_runs_as_printed(failures):
    replay = failures.absent.replay
    assert replay.returncode == 2, failures.absent.transcript()
    assert failures.missing in replay.stderr, failures.absent.transcript()
    assert "No such file or directory" in replay.stderr, failures.absent.transcript()


def test_a_foreign_probe_is_refused_before_the_load_and_names_the_binary_invoked(
    failures,
):
    # Three lines and no header: the identity check runs before the load, so
    # the wrong probe costs nothing to discover.
    three_lines_and_nothing_else(failures.foreign)
    identity = RUN_IDENTITY.search(failures.foreign.diagnosis())
    assert identity, failures.foreign.transcript()
    assert identity.group(1) == FOREIGN_HASH, failures.foreign.transcript()
    # KB-72: the binary as this test invoked it, not a bare `harness`.
    assert failures.foreign.command == (
        f"{HARNESS} --status --deep --model {failures.model} --probe {failures.image}"
    ), failures.foreign.command


def test_the_preflight_remedy_runs_as_printed_and_reports_the_same_fault(failures):
    replay = failures.foreign.replay
    assert replay.returncode == 1, failures.foreign.transcript()
    assert replay.stdout == "", failures.foreign.transcript()
    verdict = PREFLIGHT_IDENTITY.search(replay.stderr)
    assert verdict, failures.foreign.transcript()
    diagnosis = RUN_IDENTITY.search(failures.foreign.diagnosis())
    assert (verdict.group(1), verdict.group(2)) == (
        diagnosis.group(1),
        diagnosis.group(2),
    ), failures.foreign.transcript()
