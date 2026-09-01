"""Keeping what reaches the model inside a context budget.

The operating constraint this tool is designed around is not the model's maximum
context but the one it is fast at. Prefill on Apple silicon is roughly an order
of magnitude slower than on a discrete GPU, so a 128K-token prompt that is
technically accepted costs minutes before the first token appears. Everything
here exists to keep a triage run in the 16-32K range where it stays interactive.

Estimation is characters divided by four. That is an approximation, and it is a
deliberate one: importing a tokenizer would add a heavyweight dependency, tie
the budget to one model's vocabulary, and still be wrong whenever the served
model changed. Four characters per token is close for English prose and JSON
alike, and it errs high on the dense punctuation that dominates a log record --
which is the direction an error must go, since underestimating is what blows the
budget.
"""

from __future__ import annotations

from collections.abc import Sequence

# Mean characters per token. See the module docstring for why this is a constant
# rather than a tokenizer call.
CHARS_PER_TOKEN = 4

# Default ceiling for one rendered tool result. Sized so several results plus the
# system prompt and the deterministic findings fit the interactive range above.
DEFAULT_RESULT_TOKENS = 2_000

# Default ceiling for the whole-corpus profile, which is the largest single
# payload a run sends and the one that most repays spending tokens on.
DEFAULT_PROFILE_TOKENS = 4_000


def estimate_tokens(text: str) -> int:
    """Approximate the token cost of a string. See the module docstring."""
    return (len(text) + CHARS_PER_TOKEN - 1) // CHARS_PER_TOKEN


def fits(text: str, budget_tokens: int) -> bool:
    return estimate_tokens(text) <= budget_tokens


def truncate_rows(
    rows: Sequence[str],
    budget_tokens: int,
    *,
    header: str = "",
    footer_hint: str = "more rows omitted",
) -> str:
    """Join rows under a budget, dropping from the tail and saying so.

    Truncation is structural rather than a character slice. Cutting a rendered
    result mid-row hands the model a fragment of JSON, and a model shown
    malformed evidence will either refuse it or, worse, read the fragment as
    complete -- and a claim built on half a record cites a record that does not
    say what the claim says.

    Rows are dropped from the tail because every caller sorts by relevance
    first: the interesting rows are at the front, so the tail is what can be
    lost cheaply. The omission marker is always emitted when anything was
    dropped, so a truncated result is never mistaken for a complete one.
    """
    parts: list[str] = [header] if header else []
    used = estimate_tokens("\n".join(parts)) if parts else 0

    # Reserve room for the marker up front. Discovering it does not fit after
    # filling the budget would force dropping another row anyway, and doing it
    # here keeps the result honest in the worst case rather than the average.
    marker_reserve = estimate_tokens(f"... {len(rows)} {footer_hint}")

    kept: list[str] = []
    for row in rows:
        cost = estimate_tokens(row) + 1
        if used + cost + marker_reserve > budget_tokens:
            break
        kept.append(row)
        used += cost

    parts.extend(kept)
    dropped = len(rows) - len(kept)
    if dropped:
        parts.append(f"... {dropped} {footer_hint}")
    return "\n".join(parts)
