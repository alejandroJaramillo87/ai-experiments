"""RHX.10, the permissions layer on a model that makes the calls, as a test.

The sandbox half runs over a scripted generator and the tiny fixture.
`harness/tests/agent_e2e.rs` (`ask_decision_is_resolved_by_the_handler`)
proves an unmatched call reaches the Ask handler and its answer decides the
call; `harness/tests/session_status.rs` proves the two `Permissions:` header
lines over the fixture. Neither has a model that decides to call a tool, a
real stdin, or the prompt `main.rs::ask_stdin` prints, which is the whole
surface an operator meets.

This runs E2B-it on the runbook's task with the runbook's rule file (`read`
allowed, `bash` denied, `write` unmatched) four times: the prompt answered
`n`, answered `y`, stdin closed, and no rule file at all. It holds the harness
to its three lines and its one file. The model's calls are asserted, not
tolerated: if E2B-it stops calling `read` and `write` on this prompt, the
entry has nothing left to test and the test says so instead of passing on an
empty transcript. On 2026-09-13 it made both calls in every hand run.

The answers go through a pipe where the operator typed at a terminal. That is
the KB-74 fix in use: `ask_stdin` prints its prompt only to a terminal and
reads the line either way, so a piped `y` or `n` is honoured and a closed
stdin denies at EOF with nothing printed. Every run here therefore asserts an
empty prompt count; the terminal prompt itself is the one arm only a person
sees, and RHX.10's hand run is its record.

`hw_full`: four resident loads of 8.6 GiB, and records in the real obs root.
No `hw_e4b`: the subject is E2B-it, as in RH.9c and RH.37, because the base
E4B cannot follow the tool contract task mode installs.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

import pytest

from tests.hardware.conftest import (
    HARNESS,
    default_root_env,
    e2b_it_checkpoint,
    require_binary,
)

pytestmark = [
    pytest.mark.hw_full,
    pytest.mark.cpu,
    pytest.mark.ram,
    pytest.mark.ssd,
    pytest.mark.harness,
]

RULES = "rules:\n  - tool: read\n    action: allow\n  - tool: bash\n    action: deny\n"
TASK = "Read hello.txt, write its contents to copy.txt, then say done"
PROMPT = "allow "  # every Ask prompt begins so; none may appear off a terminal
RULES_LINE = (
    "[INFO] Permissions: 2 rules from permissions.yaml (an unmatched call asks)"
)
NO_RULES_LINE = "[INFO] Permissions: no permissions.yaml, every tool call asks"
DENIED = "permission denied: write copy.txt"
WROTE = "wrote 5 bytes to copy.txt"

# Hand runs on 2026-09-13: a 4 to 6 s load, a 12 s prefill of the task
# contract, then a few seconds of calls. Five minutes is a hung prompt.
RUN_TIMEOUT_S = 5 * 60
# The closed-stdin run may not take longer than the answered one by more than
# this: an Ask that waited on a stdin nobody writes would take the timeout.
NO_WAIT_MARGIN_S = 10.0


class Run:
    """One task run: its directory, streams, wall time, and transcript."""

    def __init__(
        self, cwd: Path, result: subprocess.CompletedProcess[str], elapsed_s: float
    ):
        self.cwd = cwd
        self.result = result
        self.elapsed_s = elapsed_s
        self.stderr_lines = result.stderr.splitlines()

    def transcript(self) -> list[dict]:
        with open(self.cwd / "session.jsonl", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]

    def calls(self, tool: str) -> list[tuple[int, dict]]:
        """Model entries that parse as a call to `tool`, with their index, so
        the tool entry that answered each is the one right after it."""
        found = []
        for i, entry in enumerate(self.transcript()):
            if entry["role"] != "model":
                continue
            try:
                call = json.loads(entry["content"])
            except ValueError:
                continue
            if isinstance(call, dict) and call.get("tool") == tool:
                found.append((i, call))
        return found

    def answer_to(self, tool: str) -> str:
        """The tool entry following the one call to `tool`. Asserting the call
        exists is deliberate: see the module docstring."""
        calls = self.calls(tool)
        assert len(calls) == 1, (
            f"expected one {tool} call, read {calls}\n{self.result.stderr}"
        )
        entries = self.transcript()
        index = calls[0][0] + 1
        assert entries[index]["role"] == "tool", entries[index]
        return entries[index]["content"]

    def prompts(self) -> list[str]:
        return [line for line in self.stderr_lines if line.startswith(PROMPT)]

    def index_of(self, prefix: str) -> int:
        hits = [
            i for i, line in enumerate(self.stderr_lines) if line.startswith(prefix)
        ]
        assert hits, f"no line starts with {prefix!r}:\n{self.result.stderr}"
        return hits[0]


class Runs:
    def __init__(self, denied: Run, allowed: Run, closed: Run, no_rules: Run):
        self.denied = denied
        self.allowed = allowed
        self.closed = closed
        self.no_rules = no_rules


def run_task(cwd: Path, model: str, task: str, rules: bool, stdin: str | None) -> Run:
    cwd.mkdir()
    (cwd / "hello.txt").write_text("hello\n", encoding="utf-8")
    if rules:
        (cwd / "permissions.yaml").write_text(RULES, encoding="utf-8")
    kwargs: dict = (
        {"input": stdin} if stdin is not None else {"stdin": subprocess.DEVNULL}
    )
    started = time.perf_counter()
    result = subprocess.run(
        [HARNESS, "--model", model, "--text-only", task],
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=RUN_TIMEOUT_S,
        env=default_root_env(),
        **kwargs,
    )
    return Run(cwd, result, time.perf_counter() - started)


@pytest.fixture(scope="module")
def runs(tmp_path_factory) -> Runs:
    require_binary()
    model = e2b_it_checkpoint()
    if not os.path.isfile(os.path.join(model, "model.safetensors")):
        pytest.skip(f"E2B-it checkpoint not on disk at {model}")
    work = Path(tmp_path_factory.mktemp("rhx10"))
    return Runs(
        denied=run_task(work / "denied", model, TASK, rules=True, stdin="n\n"),
        allowed=run_task(work / "allowed", model, TASK, rules=True, stdin="y\n"),
        closed=run_task(work / "closed", model, TASK, rules=True, stdin=None),
        no_rules=run_task(
            work / "no_rules", model, "say done", rules=False, stdin=None
        ),
    )


def test_the_header_states_the_rules_between_load_and_first_prefill(runs):
    for run in (runs.denied, runs.allowed, runs.closed):
        assert run.result.returncode == 0, run.result.stderr
        assert run.stderr_lines.count(RULES_LINE) == 1, run.result.stderr
        load, rules, prefill = (
            run.index_of("[INFO] Model load:"),
            run.index_of(RULES_LINE),
            run.index_of("[INFO] Prefill:"),
        )
        assert load < rules < prefill, run.result.stderr
    no_rules = runs.no_rules
    assert no_rules.result.returncode == 0, no_rules.result.stderr
    assert RULES_LINE not in no_rules.stderr_lines, no_rules.result.stderr
    assert (
        no_rules.index_of("[INFO] Model load:")
        < no_rules.index_of(NO_RULES_LINE)
        < no_rules.index_of("[INFO] Prefill:")
    ), no_rules.result.stderr


def test_an_allowed_read_asks_nothing_and_returns_the_file(runs):
    run = runs.denied
    assert run.answer_to("read") == "hello\n", run.transcript()
    # The read matched an allow rule: whatever the write did, nothing about
    # the read reached stderr, and off a terminal nothing about the write did.
    assert run.prompts() == [], run.result.stderr


def test_a_denied_write_leaves_a_denial_entry_and_no_file(runs):
    run = runs.denied
    assert run.result.returncode == 0, run.result.stderr
    assert run.answer_to("write") == DENIED, run.transcript()
    assert not (run.cwd / "copy.txt").exists(), os.listdir(run.cwd)


def test_an_allowed_write_writes_the_file(runs):
    run = runs.allowed
    assert run.result.returncode == 0, run.result.stderr
    assert run.answer_to("write") == WROTE, run.transcript()
    assert (run.cwd / "copy.txt").read_text(encoding="utf-8") == "hello", os.listdir(
        run.cwd
    )


def test_closed_stdin_resolves_to_denial_without_waiting(runs):
    run = runs.closed
    assert run.result.returncode == 0, run.result.stderr
    assert run.answer_to("write") == DENIED, run.transcript()
    assert not (run.cwd / "copy.txt").exists(), os.listdir(run.cwd)
    # KB-74: no prompt to a stdin nobody reads, so no status line lands on
    # the tail of one either.
    assert run.prompts() == [], run.result.stderr
    assert run.elapsed_s <= runs.allowed.elapsed_s + NO_WAIT_MARGIN_S, (
        f"closed stdin took {run.elapsed_s:.1f} s against {runs.allowed.elapsed_s:.1f} s "
        "answered: the Ask waited on nobody"
    )
