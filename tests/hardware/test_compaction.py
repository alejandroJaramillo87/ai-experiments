"""RH.20, compaction past the rewind horizon, as a test the E4B tier runs.

A sliding-window cache evicts rather than masks, so `Session::sync` cannot
rewind to a position an evicting layer has dropped. Below that floor it
clears and re-prefills. Compaction rewinds much further than the floor by
construction: it rewrites the middle of the transcript, so the new prompt
shares only its head with the cache, and on any real checkpoint the head sits
well below the floor once the window has filled. The rebuild is therefore the
routine path after a compaction, and the property under test is that it is
taken as a rebuild, exactly, rather than refused.

The fixture-backed sandbox tests reach the floor on a four-position window in
a few hundred tokens and never leave the library. This test reaches it on the
real E4B window of 512, through the chat loop, with the threshold placed past
the point at which eviction begins, and reads the decision where the binary
reports it: the `Prefill:` line names the rewind the window refused and the
floor it fell below. The library records that fact and prints nothing
(`runtime/src/session.rs`, `Prefill`); the line is the binary's.

Arithmetic the settings rest on, all read from the run header rather than
assumed: window 512 and a batch group of 64 give a window capacity of 719
positions, so eviction begins past 719 and the horizon is 208 behind the tip.
`--context 2048 --compact-at 0.4` fires compaction near 819 rendered tokens,
past 719, so the floor is several hundred at every sync a compaction causes.

A compaction causes two such syncs, not one, and the first run of this test
by hand (2026-09-11, entry RH.20) is what showed it. The compactor renders its
summarization prompt as BOS plus the middle entries, which begin at the first
model reply; the chat prompt is BOS plus the head user entry. The two diverge
at their third token, so the summarization sync shares two positions with the
cache and rebuilds the whole transcript. The post-compaction prompt then
shares those same two positions with the summarization prompt the cache holds
and rebuilds again, this time the short compacted prompt. Large then small,
once per compaction, is the shape asserted below.
"""

from __future__ import annotations

import json
import os
import re
import subprocess

import pytest

from tests.hardware.conftest import REPO_ROOT, e4b_checkpoint

pytestmark = [
    pytest.mark.hw_full,
    pytest.mark.hw_e4b,
    pytest.mark.cpu,
    pytest.mark.ram,
    pytest.mark.harness,
]

HARNESS_DIR = os.path.join(REPO_ROOT, "rust-harness")
HARNESS = os.path.join(HARNESS_DIR, "target", "release", "harness")

# The session: sixty turns of thirty words each. The content does not matter,
# only that each turn is long enough for the threshold to be crossed more than
# once in the session. Varied rather than one repeated word, so the summary the
# compactor asks for has something to summarize.
TURNS = 60
WORDS_PER_TURN = 30
CONTEXT = "2048"
COMPACT_AT = "0.4"
MAX_TOKENS = "8"

HORIZON_LINE = re.compile(
    r"^\[INFO\] Rewind horizon: (\d+) tokens behind the tip", re.M
)
PREFILL_LINE = re.compile(r"^\[INFO\] Prefill: (\d+) tokens in", re.M)
REBUILT_LINE = re.compile(
    r"^\[INFO\] Prefill: (\d+) tokens in .*; rebuilt from 0: a rewind to (\d+) "
    r"fell below the window floor at (\d+)$",
    re.M,
)

# Sixty turns of a thirty-token delta plus eight decodes, and one full
# re-prefill of about 830 tokens per compaction at the 11 tok/s RH.19's run
# measured. Generous, because a timeout here would read as a refused rewind.
TIMEOUT_S = 40 * 60


def turns() -> str:
    """The chat input: one user turn per line, ending with a newline."""
    words = "matter energy motion field charge wave particle heat light sound".split()
    lines = []
    for i in range(TURNS):
        body = " ".join(words[(i + j) % len(words)] for j in range(WORDS_PER_TURN))
        lines.append(f"turn {i + 1}: {body}")
    return "\n".join(lines) + "\n"


def require_binary() -> None:
    """A missing binary fails rather than skips: it is built on the same
    host, so its absence is a misconfigured run and not the unavailable
    hardware the conftest rule exempts."""
    assert os.path.isfile(HARNESS), (
        f"built binary absent: {HARNESS}; run "
        f"`cargo build --release -p harness` in {HARNESS_DIR}"
    )


class Run:
    """One chat session's transcript records and its stderr."""

    def __init__(self, directory: str, transcript: str) -> None:
        self.transcript = transcript
        with open(os.path.join(directory, "session.jsonl"), encoding="utf-8") as f:
            self.records = [json.loads(line) for line in f if line.strip()]

    def compactions(self) -> list[dict]:
        return [r for r in self.records if r.get("role") == "compaction"]

    def rebuilt(self) -> list[tuple[int, int, int]]:
        """(prefilled, common, floor) per rebuilt prefill line, in order."""
        return [
            tuple(int(g) for g in m.groups())
            for m in REBUILT_LINE.finditer(self.transcript)
        ]


@pytest.fixture(scope="module")
def run(tmp_path_factory) -> Run:
    """One session for the module, in its own directory: the harness writes
    `session.jsonl` and `session.kv` where it starts and resumes from them."""
    model = e4b_checkpoint()
    if not os.path.isfile(os.path.join(model, "model.safetensors")):
        pytest.skip(f"E4B checkpoint not on disk at {model}")
    require_binary()

    directory = str(tmp_path_factory.mktemp("rh20"))
    result = subprocess.run(
        [
            HARNESS,
            "--model",
            model,
            "--text-only",
            "--chat",
            "--context",
            CONTEXT,
            "--compact-at",
            COMPACT_AT,
            "--max-tokens",
            MAX_TOKENS,
        ],
        cwd=directory,
        input=turns(),
        capture_output=True,
        text=True,
        timeout=TIMEOUT_S,
    )
    transcript = result.stdout + result.stderr
    # A rewind past the floor is a slower path, not an error: a nonzero exit
    # here means the floor was enforced as a refusal rather than a rebuild.
    assert result.returncode == 0, f"harness exited {result.returncode}\n{transcript}"
    return Run(directory, transcript)


def test_the_header_reports_the_horizon(run):
    horizon = HORIZON_LINE.search(run.transcript)
    assert horizon is not None, f"no Rewind horizon line\n{run.transcript}"
    assert int(horizon.group(1)) > 0, horizon.group(0)


def test_the_session_compacts_more_than_once(run):
    compactions = run.compactions()
    assert len(compactions) >= 2, (
        f"{len(compactions)} compaction record(s) in {len(run.records)} entries; "
        f"the threshold was not crossed twice\n{run.transcript}"
    )
    # Every turn reported its prefill, so the line the next test reads was
    # printed at all.
    assert len(PREFILL_LINE.findall(run.transcript)) >= TURNS, run.transcript


def test_every_compaction_rebuilds_the_cache_past_the_floor(run):
    compactions = run.compactions()
    rebuilt = run.rebuilt()
    # Two rebuilds per compaction, the summarization sync and the sync after
    # it (module doc). Fewer means a compaction fired inside a window the
    # cheap rewind still covered, which is not the path this test exists for;
    # more means a rewind past the floor with no compaction to cause it.
    assert len(rebuilt) == 2 * len(compactions), (
        f"{len(rebuilt)} rebuilt prefill(s) against {len(compactions)} "
        f"compaction(s), two expected each\n{run.transcript}"
    )
    for prefilled, common, floor in rebuilt:
        assert 1 <= common < floor, (
            f"a rebuild reported a prefix of {common} against a floor of {floor}"
        )
        # Rebuilt from zero: the prefill is the whole prompt, so it is longer
        # than the prefix that was refused.
        assert prefilled > common, (prefilled, common, floor)
    # In order: the summarization prompt is the whole transcript at the
    # threshold, the compacted prompt that follows it is a fraction of that.
    for summarize, compacted in zip(rebuilt[0::2], rebuilt[1::2], strict=True):
        assert summarize[0] > compacted[0], (
            f"a compaction's rebuilds are not large then small: {summarize} "
            f"then {compacted}"
        )
