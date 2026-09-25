"""RHX.12, a chat at a terminal shows where the next turn goes, as a test.

KB-75: a chat at a terminal printed nothing between the end of a reply and
the next read. On 2026-09-13 the operator's typed `hello` and the base
model's sixteen-token reply, whose first line was also `hello`, were
indistinguishable on the screen, and nothing said the harness was waiting.
`chat::run_chat` now prints `> ` to stderr before every read when stdin is a
terminal, closes a dangling marker with a newline at end of input, and
prints nothing for a pipe. The unit tests in `chat.rs` pin the decision's two
arms and that the marker never reaches the reply stream. Neither they nor
any Rust integration test can open a pseudo-terminal without a new crate, so
the terminal arm had only the runbook's hand run behind it.

Python's `pty` module can. This test runs a chat on E4B with stdin a
pseudo-terminal and both output streams pipes: `is_terminal()` on stdin is
true, the captured streams hold the harness's bytes alone, and the tty's echo
of the typed line goes to the master, never to the child's stdout or stderr.
Input ends with Ctrl-D at the start of a line, which the tty's canonical mode
turns into a clean end of file, as a person's terminal does; closing the
master instead reads as EIO on Linux, which is not the path a person takes.
A second run on a pipe holds the other arm, the one every piped chat in the
tree depends on.

What a person still judges is whether the screen tells them the harness is
waiting. RHX.12's re-run is the record of that; this test holds the marker's
presence, placement and closing newline, and the reply stream's purity.

`hw_full` rather than `hw_smoke`: two resident loads of about 14 GiB, and the
event files and catalog rows stay in the real obs root under `/mnt/ai-data`.
`hw_e4b` because the released E4B checkpoint is the subject, as in the entry.
"""

from __future__ import annotations

import os
import pty
import subprocess
from pathlib import Path

import pytest

from tests.hardware.conftest import (
    HARNESS,
    default_root_env,
    e4b_checkpoint,
    require_binary,
)

pytestmark = [
    pytest.mark.hw_full,
    pytest.mark.hw_e4b,
    pytest.mark.cpu,
    pytest.mark.ram,
    pytest.mark.ssd,
    pytest.mark.harness,
]

TURN = "hello"
MAX_TOKENS = 16
MARKER = "> "
LOAD_PREFIX = "[INFO] Model load:"
PREFILL_PREFIX = "[INFO] Prefill:"
ENDED_LINE = "[INFO] Chat ended: 1 model turns"
EOF = b"\x04"  # Ctrl-D: end of file at the start of a line in canonical mode

# 2026-09-13: a 5.75 s load and a sub-second prefill of ten tokens.
# Generous, because a timeout here would read as a missing line.
RUN_TIMEOUT_S = 15 * 60


class Run:
    """One chat run: its exit code, both streams, and its directory."""

    def __init__(self, cwd: Path, returncode: int, stdout: str, stderr: str):
        self.cwd = cwd
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr
        self.err_lines = stderr.split("\n")

    def index_of(self, prefix: str) -> int:
        hits = [i for i, line in enumerate(self.err_lines) if line.startswith(prefix)]
        assert len(hits) == 1, f"expected one line starting {prefix!r}:\n{self.stderr}"
        return hits[0]

    def transcript_entries(self) -> int:
        with open(self.cwd / "session.jsonl", encoding="utf-8") as handle:
            return sum(1 for line in handle if line.strip())


class Runs:
    def __init__(self, terminal: Run, pipe: Run):
        self.terminal = terminal
        self.pipe = pipe


def argv(model: str) -> list[str]:
    return [
        HARNESS,
        "--model",
        model,
        "--text-only",
        "--chat",
        "--max-tokens",
        str(MAX_TOKENS),
    ]


def at_a_terminal(cwd: Path, model: str) -> Run:
    """Stdin a pseudo-terminal, stdout and stderr pipes."""
    cwd.mkdir()
    master, slave = pty.openpty()
    try:
        with subprocess.Popen(
            argv(model),
            stdin=slave,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=cwd,
            env=default_root_env(),
            text=True,
        ) as child:
            os.close(slave)
            # The tty buffers both writes until the harness reads after its
            # load: one line, then end of file where the entry typed `/quit`.
            os.write(master, f"{TURN}\n".encode())
            os.write(master, EOF)
            stdout, stderr = child.communicate(timeout=RUN_TIMEOUT_S)
            return Run(cwd, child.returncode, stdout, stderr)
    finally:
        os.close(master)


def on_a_pipe(cwd: Path, model: str) -> Run:
    cwd.mkdir()
    result = subprocess.run(
        argv(model),
        cwd=cwd,
        input=f"{TURN}\n",
        capture_output=True,
        text=True,
        timeout=RUN_TIMEOUT_S,
        env=default_root_env(),
    )
    return Run(cwd, result.returncode, result.stdout, result.stderr)


@pytest.fixture(scope="module")
def runs(tmp_path_factory) -> Runs:
    require_binary()
    model = e4b_checkpoint()
    if not os.path.isfile(os.path.join(model, "model.safetensors")):
        pytest.skip(f"E4B checkpoint not on disk at {model}")
    work = Path(tmp_path_factory.mktemp("rhx12"))
    return Runs(
        terminal=at_a_terminal(work / "terminal", model),
        pipe=on_a_pipe(work / "pipe", model),
    )


def test_the_marker_precedes_the_first_read_and_follows_the_reply_at_a_terminal(
    runs,
):
    run = runs.terminal
    assert run.returncode == 0, run.stderr
    # The typed line's echo went to the tty, so on stderr the marker printed
    # before the first read is followed directly by the prefill line.
    load = run.index_of(LOAD_PREFIX)
    assert run.err_lines[load + 1].startswith(f"{MARKER}{PREFILL_PREFIX}"), run.stderr
    # After the reply the marker is printed again, alone on its line once
    # end of input closes it.
    prefill = load + 1
    assert run.err_lines[prefill + 1] == MARKER, run.stderr
    assert run.stderr.count(MARKER) == 2, run.stderr


def test_the_reply_stream_carries_the_reply_alone_at_a_terminal(runs):
    run = runs.terminal
    assert run.stdout.strip(), run.stderr
    assert MARKER not in run.stdout, run.stdout
    assert "[INFO]" not in run.stdout, run.stdout
    # The typed turn and the model's reply.
    assert run.transcript_entries() == 2


def test_end_of_input_closes_the_marker_before_the_summary(runs):
    run = runs.terminal
    # KB-74's shape, a status line landing on an unterminated prompt, does
    # not recur: the summary starts its own line.
    ended = run.index_of(ENDED_LINE)
    assert run.err_lines[ended] == ENDED_LINE, run.stderr
    assert run.err_lines[ended - 1] == MARKER, run.stderr


def test_a_pipe_gets_no_marker(runs):
    run = runs.pipe
    assert run.returncode == 0, run.stderr
    assert MARKER not in run.stderr, run.stderr
    assert run.stdout.strip(), run.stderr
    assert MARKER not in run.stdout and "[INFO]" not in run.stdout, run.stdout
    assert run.index_of(ENDED_LINE) > run.index_of(PREFILL_PREFIX), run.stderr
