"""RHX.2, the load and prefill lines on real weights, as a test.

Before U8 a cold start was silent: the load read about 14 GiB and printed
nothing, and prefill printed nothing at all. Now a run states what its two
long operations cost, one `[INFO] Model load:` line and one `[INFO] Prefill:`
line per turn. The sandbox half, `rust-harness/harness/tests/status_streams.rs`,
pins one of each over the tiny fixture, the load before the prefill, the
printed load equal to the recorded `model.load` phase, and the absence of a
carriage return in either stream.

What the fixture cannot show is the second turn. A chat re-sends the whole
transcript, and the KV cache should make the second prefill the new turn
alone; a delta prefill that quietly stopped working would look exactly like
one that works, because nothing else displays it. This test runs two turns on
E4B and reads the delta where a person reads it, on the second `Prefill:` line.

The runbook's hand run on 2026-09-12 showed how not to assert it. Its two
turns were 17 tokens each, so "the second prefill is a small fraction of the
first" could never hold, and the delta was proved from the snapshot instead:
98 cached tokens was 17 + 32 + 17 + 32, where a re-sent transcript would have
prefilled 66. This test does both. Its first turn is long and its second short,
so the second prefill being smaller than the first is decisive, and the
snapshot arithmetic bounds everything the cache holds beyond the two prefills
by the two replies.

The run is `--chat` with the turns on stdin, for the reason RH.37 recorded: a
positional task puts the base model in coding-agent mode. The replies are the
model's and are not asserted.

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
    HARNESS,
    LOG_ROOT,
    default_root_env,
    e4b_checkpoint,
    events,
    require_binary,
)

pytestmark = [
    pytest.mark.hw_full,
    pytest.mark.hw_e4b,
    pytest.mark.cpu,
    pytest.mark.ram,
    pytest.mark.ssd,
]

CATALOG = Path(LOG_ROOT) / "runs.jsonl"  # read_runs takes a Path, not a str

# Two turns of unequal length. The first is long so its prefill is well over
# a hundred tokens; the second is the runbook's own short follow-up. The
# 2026-09-12 hand run used two 17-token turns, and equal lengths make the
# delta invisible on the prefill lines.
FIRST_TURN = (
    "Summarize the Unix philosophy in three sentences, covering why programs "
    "should do one thing well, why they should work together through text "
    "streams, why plain text is the universal interface, why building a tool "
    "beats doing a job by hand, why prototypes come before optimization, and "
    "why a design that is simple to explain is usually the one that survives "
    "contact with its users and its maintainers over the following decades."
)
SECOND_TURN = "Now say that again in one sentence"
SEED = 7
MAX_TOKENS = 32

INFO = "[INFO] "
LOAD_LINE = re.compile(r"^\[INFO\] Model load: (\d+\.\d{2}) s$")
# A `; rebuilt from 0` suffix fails this match, and should: a rebuild is the
# one case where a turn's prefill is the whole prompt again, and it cannot
# happen in a two-turn session far below the window.
PREFILL_LINE = re.compile(
    r"^\[INFO\] Prefill: (\d+) tokens in (\d+\.\d{2}) s \((\d+) tok/s\)$"
)
SNAPSHOT_LINE = re.compile(r"^\[INFO\] Session snapshot: (\d+) cached tokens at (.+)$")
CHAT_ENDED = f"{INFO}Chat ended: 2 model turns"

# 2026-09-12: the E4B load took 5.70 s on a warm page cache and 9.42 s cold;
# the 17-token prefills ran at 13 tok/s, so a first turn of a few hundred
# tokens costs under half a minute. Generous, because a timeout here would
# read as a missing line.
RUN_TIMEOUT_S = 15 * 60


class Prefill:
    """One printed `Prefill:` line, parsed, with its index in stderr."""

    def __init__(self, index: int, match: re.Match) -> None:
        self.index = index
        self.tokens = int(match.group(1))
        self.seconds = float(match.group(2))
        self.rate = int(match.group(3))


class Run:
    """One two-turn chat, its two streams, and the phases it recorded."""

    def __init__(
        self, result: subprocess.CompletedProcess, recorded: list[dict]
    ) -> None:
        self.result = result
        self.lines = result.stderr.splitlines()
        self.phases = [e for e in recorded if e["event"] == "phase.end"]

    def load_lines(self) -> list[tuple[int, float]]:
        out = []
        for i, line in enumerate(self.lines):
            match = LOAD_LINE.match(line)
            if match:
                out.append((i, float(match.group(1))))
        return out

    def prefills(self) -> list[Prefill]:
        out = []
        for i, line in enumerate(self.lines):
            match = PREFILL_LINE.match(line)
            if match:
                out.append(Prefill(i, match))
        return out

    def cached_tokens(self) -> int:
        found = [SNAPSHOT_LINE.match(line) for line in self.lines]
        found = [m for m in found if m]
        assert len(found) == 1, f"expected one snapshot line:\n{self.result.stderr}"
        return int(found[0].group(1))


@pytest.fixture(scope="module")
def run(tmp_path_factory) -> Run:
    """Two chat turns against E4B from an empty directory, so no `session.kv`
    is resumed and the first prefill is the whole first turn, run once per
    module: the model load is the cost and every test reads the same lines."""
    require_binary()
    model = e4b_checkpoint()
    if not os.path.isfile(os.path.join(model, "model.safetensors")):
        pytest.skip(f"E4B checkpoint not on disk at {model}")

    # Rows already present are counted and not read: the run id comes from
    # the one row this run appends.
    before = len(read_runs(CATALOG)) if CATALOG.is_file() else 0
    result = subprocess.run(
        [
            HARNESS,
            "--model",
            model,
            "--text-only",
            "--chat",
            "--seed",
            str(SEED),
            "--max-tokens",
            str(MAX_TOKENS),
        ],
        cwd=tmp_path_factory.mktemp("rhx2"),
        input=f"{FIRST_TURN}\n{SECOND_TURN}\n",
        capture_output=True,
        text=True,
        timeout=RUN_TIMEOUT_S,
        env=default_root_env(),
    )
    assert result.returncode == 0, f"the chat run failed:\n{result.stderr}"
    rows = read_runs(CATALOG)[before:]
    assert len(rows) == 1, f"expected one appended catalog row, read {rows}"
    return Run(result, events(rows[0]["run_id"]))


def test_one_load_line_then_one_prefill_line_per_turn(run):
    loads = run.load_lines()
    prefills = run.prefills()
    assert len(loads) == 1, f"expected one Model load line:\n{run.result.stderr}"
    assert len(prefills) == 2, f"expected two Prefill lines:\n{run.result.stderr}"
    assert CHAT_ENDED in run.lines, run.result.stderr
    assert loads[0][0] < prefills[0].index < prefills[1].index, run.result.stderr


def test_the_printed_load_is_the_recorded_phase_to_two_decimals(run):
    [(_, printed)] = run.load_lines()
    loads = [p for p in run.phases if p["phase"] == "model.load"]
    assert len(loads) == 1, f"expected one model.load phase, read {run.phases}"
    # One clock: `main.rs` prints the span's elapsed time and the guard
    # records the same span, so the two figures differ only by rounding.
    assert round(loads[0]["duration_s"], 2) == printed, (
        f"printed {printed} s, recorded {loads[0]['duration_s']} s"
    )


def test_each_prefill_line_states_its_own_rate(run):
    for prefill in run.prefills():
        line = run.lines[prefill.index]
        assert prefill.tokens > 0 and prefill.seconds > 0, line
        assert prefill.rate == round(prefill.tokens / prefill.seconds), line


def test_the_second_prefill_is_the_delta_not_the_transcript(run):
    first, second = run.prefills()
    cached = run.cached_tokens()
    figures = (
        f"first prefill {first.tokens}, second prefill {second.tokens}, "
        f"{cached} cached tokens, max_tokens {MAX_TOKENS}"
    )
    # A re-sent transcript prefills the first turn, its reply and the second
    # turn, which is more than the first prefill alone.
    assert second.tokens < first.tokens, (
        f"the second prefill is not the delta: {figures}"
    )
    # The replies are in the cache and were never prefilled.
    assert first.tokens + second.tokens < cached, figures
    # Everything the cache holds beyond the two prefills is generated tokens,
    # at most two replies, so no token was prefilled twice.
    assert cached - first.tokens - second.tokens <= 2 * MAX_TOKENS, figures


def test_neither_stream_carries_a_carriage_return(run):
    # The animation was deliberately not built: a pasted Observed block gets
    # the same bytes a terminal does.
    assert "\r" not in run.result.stderr, repr(run.result.stderr)
    assert "\r" not in run.result.stdout, repr(run.result.stdout)
    assert run.result.stdout.strip(), (
        f"the chat run produced no reply on stdout:\n{run.result.stderr}"
    )
