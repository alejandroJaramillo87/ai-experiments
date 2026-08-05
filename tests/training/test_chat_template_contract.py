"""The Python half of the chat-template contract.

The Rust half lives at rust-harness/harness/tests/template_contract.rs. Each
side asserts its own constants against schemas/chat_template.json, so a
one-sided edit fails on the side that was edited rather than producing two
tiers that render the same conversation to different prompts.

This is the two-step failure schemas/README.md describes: changing a spec
field fails the tier that changed, and updating the contract to match then
fails the other tier until it follows.
"""

import json
from pathlib import Path

import pytest

from training.data.sft import render_segments
from training.models.families import CHAT_TEMPLATE_VERSION, family_spec

CONTRACT = Path(__file__).resolve().parents[2] / "schemas" / "chat_template.json"


@pytest.fixture(scope="module")
def contract() -> dict:
    if not CONTRACT.is_file():
        pytest.skip(f"contract absent: {CONTRACT}")
    return json.loads(CONTRACT.read_text())


def test_spec_fields_match_the_shared_contract(contract: dict) -> None:
    spec = family_spec("gemma4").chat_template
    declared = contract["spec"]
    assert spec.bos == declared["bos"]
    assert spec.turn_prefix == declared["turn_prefix"]
    assert spec.eot == declared["eot"]
    assert spec.turn_suffix == declared["turn_suffix"]
    assert spec.assistant_role == declared["assistant_role"]
    assert spec.trim_content == declared["trim_content"]


def test_template_version_matches_the_shared_contract(contract: dict) -> None:
    # The version reaches prepared SFT dataset manifests, which sft.py
    # refuses when stale, so a drift here silently invalidates artifacts.
    assert CHAT_TEMPLATE_VERSION == contract["chat_template_version"]


def test_rendered_markers_are_a_subset_of_the_neutralized_set(contract: dict) -> None:
    # The renderer emits only turn markers, but content is neutralized of
    # every marker the tokenizer defines. Rendering a marker that is not
    # neutralized would let untrusted content forge that structure.
    rendered = set(contract["markers"]["rendered"])
    neutralized = set(contract["markers"]["neutralized_in_content"])
    assert rendered <= neutralized


def test_thinking_markers_are_neutralized_by_the_harness(contract: dict) -> None:
    # The training tier renders curated conversations and does not sanitize,
    # so it has no marker set of its own to compare. What it does define is
    # the thinking markers sft.py refuses in a source dataset; those are
    # exactly the strings an inference-time prompt must not let content
    # forge, so the harness's set has to cover them.
    neutralized = set(contract["markers"]["neutralized_in_content"])
    for marker in family_spec("gemma4").thinking_markers:
        # <|channel>thought is a marker plus a literal word; the marker is
        # the part the tokenizer resolves.
        assert any(marker.startswith(m) for m in neutralized), marker


def test_the_bos_is_neutralized_in_content(contract: dict) -> None:
    # The harness writes bos as prompt text rather than prepending an id, so
    # content carrying the literal would open a second sequence.
    assert contract["spec"]["bos"] in contract["markers"]["neutralized_in_content"]


def test_a_rendered_turn_uses_only_contract_markers(contract: dict) -> None:
    spec = family_spec("gemma4").chat_template
    messages = [{"role": "user", "content": "hello"}]
    text = "".join(t for t, _ in render_segments(messages, spec))
    assert text == f"{spec.bos}{spec.turn_prefix.format(role='user')}hello{spec.eot}\n"
    for marker in contract["markers"]["rendered"]:
        assert marker in text or marker == spec.eot
