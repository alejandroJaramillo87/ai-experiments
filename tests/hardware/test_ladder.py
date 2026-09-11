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
"""

from __future__ import annotations

import json
import os
import re
import subprocess

import pytest

from tests.hardware.conftest import AI_DATA_ROOT, REPO_ROOT

pytestmark = [pytest.mark.hw_full, pytest.mark.hw_e4b, pytest.mark.ssd]

HARNESS = os.path.join(REPO_ROOT, "rust-harness")
SIZE = "quick"

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


def checkpoint() -> str:
    """The E4B tier names the checkpoint through HARNESS_E4B_MODEL; outside
    that tier fall back to the same root the sibling weight test reads."""
    named = os.environ.get("HARNESS_E4B_MODEL")
    if named:
        return named
    root = os.environ.get("GEMMA4_CHECKPOINT_ROOT", f"{AI_DATA_ROOT}/models/hf")
    return os.path.join(root, "gemma-4-E4B")


def test_the_ladder_passes_every_rung_at_tier_e():
    model = checkpoint()
    if not os.path.isfile(os.path.join(model, "model.safetensors")):
        pytest.skip(f"E4B checkpoint not on disk at {model}")

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
