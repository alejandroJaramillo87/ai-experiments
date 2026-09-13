"""RHX.11, a resumed session says what it resumed, as a test.

The harness resumes `session.jsonl` and `session.kv` from the directory it
starts in. The sandbox half, `rust-harness/harness/tests/session_status.rs`
(`a_second_run_in_the_same_directory_says_what_it_resumed`), runs the fixture
twice in one directory and pins the `Session: resumed N entries` line with its
count. It does not pin where the line falls, that the `Session restored` count
is the count the first run snapshotted, or that the second prefill shrinks to
the new turn. Every other module in this directory starts in an empty
directory on purpose, so no test on real weights had ever resumed a session.

The delta prefill is the property that matters. RH.37 read `Prefill: 2504
tokens` on a second run where 252 were expected: the first run had exited
non-zero, saved no snapshot, and its successor re-prefilled the whole prior
transcript with only the token count to hint at why. The resume line exists
so that a person sees the resume, and the restored cache is what makes the
second prefill the new turn alone. Only a real KV cache exercises that.

This runs the runbook's two turns in one directory on E4B, piped as the hand
run piped them, and reads the second run where a person reads it. The second
turn is shorter than the first so that "fewer tokens than the first prefill"
is decisive: a re-sent transcript prefills more than the first turn, never
less (`test_status_costs.py` records why equal-length turns cannot show
this). The replies are the model's and are not asserted.

`hw_full` rather than `hw_smoke`: two resident loads of about 14 GiB, and the
event files and catalog rows stay in the real obs root under `/mnt/ai-data`.
`hw_e4b` because the released E4B checkpoint is the subject.
"""

from __future__ import annotations

import os
import re
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
]

# The runbook's turns: a fact, then a question about it, shorter than the fact.
FIRST_TURN = "The workstation has an RTX 5090 and a Ryzen 9950X."
SECOND_TURN = "Which GPU does it have?"
MAX_TOKENS = 16

LOAD_LINE = re.compile(r"^\[INFO\] Model load: (\d+\.\d{2}) s$")
# A `; rebuilt from 0` suffix fails this match, and should: nothing in two
# short turns far below the window may rebuild.
PREFILL_LINE = re.compile(
    r"^\[INFO\] Prefill: (\d+) tokens in (\d+\.\d{2}) s \((\d+) tok/s\)$"
)
SNAPSHOT_LINE = re.compile(r"^\[INFO\] Session snapshot: (\d+) cached tokens at (.+)$")
RESUME_LINE = re.compile(r"^\[INFO\] Session: resumed (\d+) entries from (.+)$")
RESTORED_LINE = re.compile(
    r"^\[INFO\] Session restored: (\d+) cached tokens from (.+)$"
)

# 2026-09-13: loads of 9.57 s cold and 5.97 s warm, prefills under 3 s.
# Generous, because a timeout here would read as a missing line.
RUN_TIMEOUT_S = 15 * 60


class Run:
    """One chat run: its streams and the status lines parsed off stderr."""

    def __init__(self, result: subprocess.CompletedProcess[str]) -> None:
        self.result = result
        self.lines = result.stderr.splitlines()

    def matches(self, pattern: re.Pattern[str]) -> list[tuple[int, re.Match[str]]]:
        found = []
        for i, line in enumerate(self.lines):
            match = pattern.match(line)
            if match:
                found.append((i, match))
        return found

    def one(self, pattern: re.Pattern[str]) -> tuple[int, re.Match[str]]:
        found = self.matches(pattern)
        assert len(found) == 1, (
            f"expected one line matching {pattern.pattern!r}, "
            f"read {len(found)}:\n{self.result.stderr}"
        )
        return found[0]

    def prefill_tokens(self) -> int:
        return int(self.one(PREFILL_LINE)[1].group(1))

    def cached_tokens(self) -> int:
        return int(self.one(SNAPSHOT_LINE)[1].group(1))


class Runs:
    def __init__(self, directory: Path, first: Run, entries: int, second: Run):
        self.directory = directory
        self.first = first
        self.entries = entries  # transcript entries on disk between the runs
        self.second = second


def chat(directory: Path, model: str, turn: str) -> Run:
    result = subprocess.run(
        [
            HARNESS,
            "--model",
            model,
            "--text-only",
            "--chat",
            "--max-tokens",
            str(MAX_TOKENS),
        ],
        cwd=directory,
        input=f"{turn}\n",
        capture_output=True,
        text=True,
        timeout=RUN_TIMEOUT_S,
        env=default_root_env(),
    )
    assert result.returncode == 0, f"the chat run failed:\n{result.stderr}"
    return Run(result)


def transcript_entries(directory: Path) -> int:
    with open(directory / "session.jsonl", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


@pytest.fixture(scope="module")
def runs(tmp_path_factory) -> Runs:
    """The runbook's two runs in one directory, the second resuming the first."""
    require_binary()
    model = e4b_checkpoint()
    if not os.path.isfile(os.path.join(model, "model.safetensors")):
        pytest.skip(f"E4B checkpoint not on disk at {model}")
    directory = Path(tmp_path_factory.mktemp("rhx11"))
    first = chat(directory, model, FIRST_TURN)
    entries = transcript_entries(directory)
    second = chat(directory, model, SECOND_TURN)
    return Runs(directory, first, entries, second)


def test_a_fresh_directory_reports_no_resume_and_leaves_a_snapshot(runs):
    first = runs.first
    assert first.matches(RESUME_LINE) == [], first.result.stderr
    assert first.matches(RESTORED_LINE) == [], first.result.stderr
    _, snapshot = first.one(SNAPSHOT_LINE)
    assert snapshot.group(2) == str(runs.directory / "session.kv"), first.result.stderr
    assert (runs.directory / "session.kv").is_file(), os.listdir(runs.directory)
    # The user's turn and the model's reply.
    assert runs.entries == 2, runs.entries


def test_the_second_run_says_what_it_resumed_before_the_load(runs):
    second = runs.second
    index, resumed = second.one(RESUME_LINE)
    assert int(resumed.group(1)) == runs.entries, second.result.stderr
    assert resumed.group(2) == str(runs.directory / "session.jsonl"), (
        second.result.stderr
    )
    load, _ = second.one(LOAD_LINE)
    assert index < load, second.result.stderr


def test_the_restored_cache_is_the_first_runs_snapshot(runs):
    second = runs.second
    index, restored = second.one(RESTORED_LINE)
    assert int(restored.group(1)) == runs.first.cached_tokens(), second.result.stderr
    assert restored.group(2) == str(runs.directory / "session.kv"), second.result.stderr
    load, _ = second.one(LOAD_LINE)
    assert index < load, second.result.stderr


def test_the_second_prefill_is_the_new_turn_alone(runs):
    first_prefill = runs.first.prefill_tokens()
    second_prefill = runs.second.prefill_tokens()
    # A re-sent transcript would prefill the first turn, its reply and the
    # new turn: more than the first prefill, never fewer.
    assert second_prefill < first_prefill, (
        f"second prefill {second_prefill} against first {first_prefill}:\n"
        f"{runs.second.result.stderr}"
    )
    # What the second run added to the cache is its prefill plus its reply,
    # and the reply is between one token and the cap.
    added = runs.second.cached_tokens() - runs.first.cached_tokens()
    assert second_prefill + 1 <= added <= second_prefill + MAX_TOKENS, (
        f"cache grew by {added} over a {second_prefill}-token prefill:\n"
        f"{runs.second.result.stderr}"
    )
