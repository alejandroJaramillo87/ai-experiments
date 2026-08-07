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

# Google's own template, extracted verbatim from the GGUF checkpoint metadata by
# tests/models/test_gemma4_facts_contract.py. Until this file existed, the marker
# list below was hand-transcribed from release material and compared against
# nothing -- the same posture that let the Gemma 3 markers survive into the
# harness's Gemma 4 path (see schemas/README.md).
CANONICAL_TEMPLATE = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "models"
    / "gemma4-chat-template.jinja"
)


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


# --------------------------------------------------------------------------
# Agreement with Google's canonical template
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def canonical() -> str:
    if not CANONICAL_TEMPLATE.is_file():
        pytest.skip(f"canonical template absent: {CANONICAL_TEMPLATE}")
    return CANONICAL_TEMPLATE.read_text()


def test_rendered_markers_occur_in_the_canonical_template(
    contract: dict, canonical: str
) -> None:
    # A marker this tier renders that Google's template never emits is a marker
    # the tokenizer has no trained id for, which is precisely the Gemma 3
    # failure: fluent output, wrong tokenization, nothing raised.
    for marker in contract["markers"]["rendered"]:
        assert marker in canonical, marker


def test_spec_framing_occurs_in_the_canonical_template(
    contract: dict, canonical: str
) -> None:
    spec = contract["spec"]
    assert spec["eot"] in canonical
    # turn_prefix is a format string; the literal part is what the template emits.
    assert spec["turn_prefix"].split("{")[0] in canonical


def test_canonical_template_emits_bos_through_the_jinja_variable(
    canonical: str,
) -> None:
    # The literal "<bos>" never appears in Google's template: it emits the
    # bos_token variable, which the tokenizer resolves. Both tiers here write
    # the resolved literal instead, which is why the contract's spec.bos is a
    # string and not a variable name. Asserting the literal against the
    # template would fail for the wrong reason, so assert the emission point.
    assert "{{- bos_token -}}" in canonical
    assert "<bos>" not in canonical


def test_assistant_role_is_the_one_the_canonical_template_opens(
    contract: dict, canonical: str
) -> None:
    # Gemma 4 names the assistant turn "model", not "assistant". Getting this
    # wrong shifts every generation prompt by one turn label.
    role = contract["spec"]["assistant_role"]
    prefix = contract["spec"]["turn_prefix"].format(role=role)
    assert repr(prefix)[1:-1] in canonical or prefix in canonical


def test_thinking_markers_occur_in_the_canonical_template(canonical: str) -> None:
    # sft.py refuses these in source datasets; they must be real markers.
    for marker in family_spec("gemma4").thinking_markers:
        assert marker in canonical, marker
