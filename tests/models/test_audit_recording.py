"""The recording audit's parsing and measures, on synthetic recordings.

The script's forward pass needs the workstation; everything before and after
it is numpy over a JSONL file, and that half is what can go wrong quietly:
a sequence stitched from the wrong offsets compares transformers against a
prefix it never saw and reports a defect in the backend. These pin the
reconstruction against the recorder's own line shapes (record.rs `Record`,
`SelectionLine`) and the measures against cases with known answers.
"""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "rust-harness" / "fixtures" / "audit_recording.py"


def load():
    spec = importlib.util.spec_from_file_location("audit_recording", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


audit = load()


def header(vocab=8, sampling=None):
    line = {"vocab_size": vocab}
    if sampling:
        line["sampling"] = sampling
    return json.dumps(line)


def forward(start, tokens, rows):
    return json.dumps(
        {
            "start": start,
            "tokens": tokens,
            "rows": [
                {"position": p, "logits": list(map(float, row))} for p, row in rows
            ],
        }
    )


def selected(position, token, draws=1):
    return json.dumps(
        {"selected": {"position": position, "token": token, "draws": draws}}
    )


def test_a_prefill_and_two_steps_reconstruct_the_sequence_and_the_rows():
    lines = [
        header(sampling={"temperature": 0.8, "top_p": 1.0, "seed": 0}),
        forward(0, [2, 5, 7], [(2, [0, 1, 2, 3, 4, 5, 6, 7])]),
        selected(3, 6),
        forward(3, [6], [(3, [7, 6, 5, 4, 3, 2, 1, 0])]),
        selected(4, 1),
        forward(4, [1], [(4, [1] * 8)]),
    ]
    hdr, rows, sels, seq = audit.parse_recording(lines)
    assert hdr["sampling"]["temperature"] == 0.8
    assert seq == [2, 5, 7, 6, 1]
    assert sorted(rows) == [2, 3, 4]
    assert rows[3][0] == 7.0
    assert sels == {3: 6, 4: 1}


def test_a_second_branch_at_a_written_position_is_refused():
    lines = [
        header(),
        forward(0, [2, 5], [(1, [0] * 8)]),
        forward(1, [9], [(1, [0] * 8)]),
    ]
    with pytest.raises(ValueError, match="two branches"):
        audit.parse_recording(lines)


def test_a_forward_that_skips_positions_is_refused():
    lines = [header(), forward(0, [2], [(0, [0] * 8)]), forward(5, [1], [(5, [0] * 8)])]
    with pytest.raises(ValueError, match="gap"):
        audit.parse_recording(lines)


def test_a_recording_must_open_with_its_header():
    with pytest.raises(ValueError, match="header"):
        audit.parse_recording([forward(0, [1], [(0, [0] * 8)])])


def test_identical_rows_have_zero_kl_full_overlap_and_the_same_argmax():
    rng = np.random.default_rng(0)
    row = rng.normal(size=1000)
    kl, argmax, overlap = audit.compare_row(row, row.copy())
    assert kl == pytest.approx(0.0, abs=1e-12)
    assert argmax is True
    assert overlap == 1.0


def test_a_shifted_argmax_is_seen_and_the_kl_is_positive():
    rng = np.random.default_rng(1)
    row = rng.normal(size=1000)
    other = row.copy()
    other[int(np.argmax(row))] -= 10.0
    kl, argmax, overlap = audit.compare_row(row, other)
    assert kl > 0.01
    assert argmax is False
    assert overlap < 1.0


def test_kl_is_scale_free_in_the_direction_the_ladder_uses():
    # KL(reference || ours): a row that spreads mass the reference gives none
    # to costs little, the reverse costs much. The same asymmetry L2 has.
    ref = np.array([10.0, 0.0, 0.0, 0.0])
    ours = np.array([10.0, 5.0, 5.0, 5.0])
    kl_ref_ours, _, _ = audit.compare_row(ref, ours, top_k=2)
    kl_ours_ref, _, _ = audit.compare_row(ours, ref, top_k=2)
    assert kl_ref_ours < kl_ours_ref


def test_rank_is_one_based_and_the_probability_is_a_softmax():
    logits = np.array([1.0, 3.0, 2.0])
    assert audit.rank_of(logits, 1) == (
        1,
        pytest.approx(math.exp(3) / (math.e + math.exp(3) + math.exp(2))),
    )
    assert audit.rank_of(logits, 2)[0] == 2
    assert audit.rank_of(logits, 0)[0] == 3


def test_the_verdict_applies_every_l2_threshold():
    good = [
        {"position": i, "kl": 0.0001, "argmax": True, "overlap": 1.0} for i in range(10)
    ]
    ok, m = audit.verdict(good)
    assert ok and m["positions"] == 10
    one_bad = [dict(p) for p in good]
    one_bad[4]["kl"] = 0.01
    ok, m = audit.verdict(one_bad)
    assert not ok and m["worst_kl_position"] == 4
    argmax_off = [dict(p) for p in good]
    argmax_off[0]["argmax"] = False
    ok, _ = audit.verdict(argmax_off)
    assert not ok, "one wrong argmax in ten is below the 0.98 floor"
    assert audit.verdict([]) == (False, {})
