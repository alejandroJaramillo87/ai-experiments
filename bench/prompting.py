"""Building prompts of an exact token length, without a tokenizer dependency.

The suite is stdlib-only and does not import transformers, so it cannot tokenize
locally. It does not need to: the engine reports `usage.prompt_tokens` for every
request, so the prompt length that lands in the report is always the server's own count
rather than an estimate. What this module provides is a way to hit a target length
closely enough that a sweep has distinct, well-separated points, plus the per-repeat
prefix that keeps a prompt cache from answering the question.
"""

from __future__ import annotations

import os

CORPUS = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "prompts", "natural.txt"
)

# English prose runs about four characters to the token across common BPE vocabularies.
# Only used to slice a candidate; the reported length is always the server's usage count.
CHARS_PER_TOKEN = 4


def _corpus() -> str:
    with open(CORPUS, encoding="utf-8") as handle:
        return handle.read()


def slices(count: int) -> list[str]:
    """`count` disjoint regions of the corpus.

    An MoE's decode rate moves with which experts a prompt routes to, so the decode arm
    draws three disjoint prompts and reports the spread between them as the routing
    sensitivity -- a measured quantity rather than an unmeasured confound. A dense model
    does identical work per token and takes one.
    """
    text = _corpus()
    if count < 1:
        raise ValueError("need at least one slice")
    width = len(text) // count
    return [text[i * width : (i + 1) * width] for i in range(count)]


def build(target_tokens: int, region: str, nonce: str) -> str:
    """A prompt of roughly `target_tokens`, unique to `nonce`.

    The nonce leads the prompt rather than trailing it, and that placement is the point:
    a prefix cache keys on the leading tokens, so a unique prefix defeats it while a
    unique suffix would not. vLLM's prefix caching is a launch flag with no per-request
    override, so this is the only way to be sure a measured prefill actually happened.
    """
    budget = max(1, target_tokens * CHARS_PER_TOKEN)
    head = f"[{nonce}] "
    body = region[: max(0, budget - len(head))]
    if len(body) < budget - len(head):
        # Short region: repeat it rather than silently returning a shorter prompt.
        body = (region * (budget // max(1, len(region)) + 1))[: budget - len(head)]
    return head + body
