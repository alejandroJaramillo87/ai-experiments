"""F7 judge logic: position-swapped pairing, verdict parsing, aggregation.
Pure logic; the judge endpoint is a stub. The live judging path is a host
acceptance run."""

import pytest

from training.eval.judge import (
    build_pairs,
    judge_messages,
    parse_verdict,
    resolve_pair,
    score_judgments,
)


def test_build_pairs_swaps_positions_per_prompt():
    pairs = build_pairs(["p1", "p2"])
    assert pairs == [
        {"prompt_id": "p1", "first": "baseline", "second": "tuned"},
        {"prompt_id": "p1", "first": "tuned", "second": "baseline"},
        {"prompt_id": "p2", "first": "baseline", "second": "tuned"},
        {"prompt_id": "p2", "first": "tuned", "second": "baseline"},
    ]


def test_judge_messages_carry_prompt_and_both_responses():
    messages = judge_messages("the prompt", "resp A", "resp B")
    assert messages[0]["role"] == "system"
    assert "VERDICT" in messages[0]["content"]
    user = messages[1]["content"]
    assert "the prompt" in user
    assert user.index("resp A") < user.index("resp B")


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Rationale here.\nVERDICT: FIRST", "FIRST"),
        ("verdict: second", "SECOND"),
        ("Some reasoning\nVERDICT: TIE\n", "TIE"),
        # The last verdict line wins over one quoted in the rationale
        ("The format is VERDICT: FIRST or similar.\nVERDICT: SECOND", "SECOND"),
        ("I cannot compare these responses.", None),
        ("", None),
        (None, None),
        ("VERDICT: BOTH", None),
    ],
)
def test_parse_verdict(text, expected):
    assert parse_verdict(text) == expected


def test_resolve_pair_maps_positions_back_to_sides():
    tuned_first = {"prompt_id": "p", "first": "tuned", "second": "baseline"}
    assert resolve_pair(tuned_first, "FIRST") == "tuned"
    assert resolve_pair(tuned_first, "SECOND") == "baseline"
    assert resolve_pair(tuned_first, "TIE") == "tie"
    assert resolve_pair(tuned_first, None) == "malformed"


def _judgment(pid, first, outcome):
    second = "tuned" if first == "baseline" else "baseline"
    return {"prompt_id": pid, "first": first, "second": second, "outcome": outcome}


def test_score_judgments_aggregates_and_reports_exceptions():
    judgments = [
        _judgment("p1", "baseline", "tuned"),
        _judgment("p1", "tuned", "tuned"),  # consistent tuned win
        _judgment("p2", "baseline", "baseline"),
        _judgment("p2", "tuned", "tie"),  # inconsistent
        _judgment("p3", "baseline", "malformed"),
        _judgment("p3", "tuned", "error"),
    ]
    section = score_judgments(judgments)
    assert section["tuned_wins"] == 2
    assert section["baseline_wins"] == 1
    assert section["ties"] == 1
    assert section["malformed"] == 1
    assert section["errors"] == 1
    # malformed/error excluded from the decided denominator (2+1+1)
    assert section["win_rate"] == 0.5
    assert section["position_consistent"] == 1
    assert section["prompts"] == 3


def test_score_judgments_empty_decided_has_no_rate():
    section = score_judgments([_judgment("p", "baseline", "malformed")])
    assert section["win_rate"] is None
    assert section["position_consistency"] is None


def test_null_test_shape_win_rate_near_half():
    """A model judged against itself: symmetric outcomes -> 50 percent."""
    judgments = []
    for i in range(10):
        judgments.append(_judgment(f"p{i}", "baseline", "tuned"))
        judgments.append(_judgment(f"p{i}", "tuned", "baseline"))
    section = score_judgments(judgments)
    assert section["win_rate"] == 0.5
