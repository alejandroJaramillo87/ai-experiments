"""Pairwise model-as-judge scoring (Phase 1 F7).

Pure pairing and verdict-parsing logic; no serving I/O. The runner feeds each
pair through the judge endpoint and hands the raw judge text back to
parse_verdict. Every prompt is judged twice with the response order swapped
so position bias cancels in the aggregate; a judge refusal or malformed
verdict is counted and reported, never silently dropped.

The two compared sides are named "baseline" and "tuned" throughout:
"baseline" is the --base / --baseline model, "tuned" the fine-tuned one.
"""

from __future__ import annotations

import re

# The verdict must land on its own final line so parsing never depends on the
# rationale's wording. FIRST/SECOND refer to presentation order, which the
# pairing layer maps back to baseline/tuned.
JUDGE_SYSTEM = (
    "You are an impartial judge comparing two AI assistant responses to the "
    "same user prompt. Judge which response is better on correctness, "
    "helpfulness, completeness, and clarity. Do not let response length, "
    "style, or presentation order influence the judgment. First give a "
    "brief rationale (at most three sentences), then output the verdict on "
    "the last line, exactly in the form:\n"
    "VERDICT: FIRST\n"
    "or VERDICT: SECOND\n"
    "or VERDICT: TIE"
)

_VERDICT_RE = re.compile(r"VERDICT:\s*(FIRST|SECOND|TIE)\b", re.IGNORECASE)


class JudgeError(RuntimeError):
    """Non-recoverable judging failure; message says how to proceed."""


def judge_messages(prompt: str, first: str, second: str) -> list[dict]:
    """Chat messages presenting one ordered pair to the judge."""
    user = (
        f"[User prompt]\n{prompt}\n\n"
        f"[Response FIRST]\n{first}\n\n"
        f"[Response SECOND]\n{second}"
    )
    return [{"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": user}]


def build_pairs(prompt_ids: list[str]) -> list[dict]:
    """Two ordered judgments per prompt, orders swapped.

    Deterministic: order within the list is (baseline-first, tuned-first) per
    prompt, prompts in input order. Each pair record carries which side was
    presented FIRST so verdicts map back to sides.
    """
    pairs = []
    for pid in prompt_ids:
        pairs.append({"prompt_id": pid, "first": "baseline", "second": "tuned"})
        pairs.append({"prompt_id": pid, "first": "tuned", "second": "baseline"})
    return pairs


def parse_verdict(text: str | None) -> str | None:
    """FIRST/SECOND/TIE from the judge's final verdict line, else None.

    The last VERDICT: occurrence wins, so a rationale that quotes the format
    does not confuse parsing. None means malformed (including empty or
    refused output); the caller counts and persists it.
    """
    if not text:
        return None
    matches = _VERDICT_RE.findall(text)
    if not matches:
        return None
    return matches[-1].upper()


def resolve_pair(pair: dict, verdict: str | None) -> str:
    """Map an ordered verdict back to sides: tuned/baseline/tie/malformed."""
    if verdict == "TIE":
        return "tie"
    if verdict == "FIRST":
        return pair["first"]
    if verdict == "SECOND":
        return pair["second"]
    return "malformed"


def score_judgments(judgments: list[dict]) -> dict:
    """Aggregate resolved judgments into the scorecard's judge section.

    Each judgment record: {"prompt_id", "first", "second", "outcome"} where
    outcome is tuned/baseline/tie/malformed/error. Win rate is tuned wins
    over decided judgments (wins plus losses plus ties); malformed and
    errored judgments are excluded from the rate but reported alongside it.
    Position consistency counts prompts whose two swapped judgments agree.
    """
    counts = {"tuned": 0, "baseline": 0, "tie": 0, "malformed": 0, "error": 0}
    by_prompt: dict[str, list[str]] = {}
    for record in judgments:
        outcome = record["outcome"]
        counts[outcome] = counts.get(outcome, 0) + 1
        by_prompt.setdefault(record["prompt_id"], []).append(outcome)

    decided = counts["tuned"] + counts["baseline"] + counts["tie"]
    complete = [o for o in by_prompt.values() if len(o) == 2]
    consistent = sum(
        1 for pair in complete
        if pair[0] == pair[1] and pair[0] in ("tuned", "baseline", "tie"))
    return {
        "judgments": len(judgments),
        "prompts": len(by_prompt),
        "tuned_wins": counts["tuned"],
        "baseline_wins": counts["baseline"],
        "ties": counts["tie"],
        "malformed": counts["malformed"],
        "errors": counts["error"],
        "win_rate": round(counts["tuned"] / decided, 4) if decided else None,
        "position_consistent": consistent,
        "position_consistency": round(consistent / len(complete), 4)
        if complete else None,
    }
