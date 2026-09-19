"""F7 scorecard assembly: delta math, confound refusal, rendering, and the
section builders that fold two per-side records into one comparison."""

import pytest

from training.eval.runner import holdout_section, lmeval_section
from training.eval.scorecard import (
    ScorecardError,
    assemble_scorecard,
    delta_section,
    render_markdown,
)


def metadata(base_thinking="off", tuned_thinking="off"):
    return {
        "git_sha": "abc123",
        "baseline": {
            "model": "/app/models/hf/gemma-4-E2B",
            "model_hash": "a" * 64,
            "thinking_mode": base_thinking,
        },
        "tuned": {
            "model": "/app/models/trained/run/hf",
            "model_hash": "b" * 64,
            "thinking_mode": tuned_thinking,
        },
    }


def test_delta_section_direction_aware():
    lower_better = delta_section(2.0, 1.5, higher_is_better=False)
    assert lower_better["delta"] == -0.5
    assert lower_better["improved"] is True
    higher_better = delta_section(0.6, 0.5, higher_is_better=True)
    assert higher_better["improved"] is False
    partial = delta_section(None, 0.5, higher_is_better=True)
    assert "delta" not in partial


def test_assemble_marks_absent_evaluators_skipped():
    card = assemble_scorecard(
        metadata(),
        {"holdout": {"loss": delta_section(2.0, 1.8, higher_is_better=False)}},
    )
    assert card["schema_version"] == 1
    assert card["evaluators"]["lmeval"] == {"skipped": True}
    assert not card["evaluators"]["holdout"].get("skipped")


def test_assemble_refuses_empty_sections():
    with pytest.raises(ScorecardError, match="No evaluator sections"):
        assemble_scorecard(metadata(), {})


def test_assemble_refuses_thinking_mismatch():
    with pytest.raises(ScorecardError, match="thinking_mode differs"):
        assemble_scorecard(metadata(tuned_thinking="on"), {"holdout": {"loss": {}}})


def test_assemble_allows_unhashed_sides():
    # holdout and lmeval accept a hub id or an adapter checkpoint, which
    # the harness cannot serve and so carries no model_hash
    meta = metadata()
    meta["baseline"]["model_hash"] = None
    meta["tuned"]["model_hash"] = None
    card = assemble_scorecard(meta, {"holdout": {"loss": {}}})
    assert card["evaluators"]["lmeval"] == {"skipped": True}


def test_differing_hashes_are_the_point_not_a_confound():
    card = assemble_scorecard(metadata(), {"holdout": {"loss": {}}})
    assert (
        card["metadata"]["baseline"]["model_hash"]
        != (card["metadata"]["tuned"]["model_hash"])
    )


def holdout_record(loss, precision="bf16"):
    return {
        "loss": loss,
        "perplexity": 2.718**loss,
        "examples": 10,
        "supervised_tokens": 1000,
        "precision": precision,
        "dataset": "/data/sft/x",
    }


def test_holdout_section_deltas_and_precision_guard():
    section = holdout_section(holdout_record(2.0), holdout_record(1.5))
    assert section["loss"]["improved"] is True
    assert section["precision"] == "bf16"
    with pytest.raises(ScorecardError, match="precision differs"):
        holdout_section(holdout_record(2.0), holdout_record(1.5, "nf4-bf16"))


def test_holdout_section_refuses_mismatched_run_parameters():
    # KB-23: a side record resumed from a run with a different dataset or
    # example cap must be refused, not merged into the scorecard
    other_dataset = holdout_record(1.5)
    other_dataset["dataset"] = "/data/sft/y"
    with pytest.raises(ScorecardError, match="dataset differs"):
        holdout_section(holdout_record(2.0), other_dataset)
    fewer_examples = holdout_record(1.5)
    fewer_examples["examples"] = 5
    with pytest.raises(ScorecardError, match="example count differs"):
        holdout_section(holdout_record(2.0), fewer_examples)


def lmeval_record(value, version="0.4.12"):
    return {
        "lm_eval_version": version,
        "limit": None,
        "results": {"tinyMMLU": {"metric": "acc_norm", "value": value, "stderr": 0.01}},
    }


def test_lmeval_section_deltas_and_version_guard():
    section = lmeval_section(lmeval_record(0.5), lmeval_record(0.55))
    task = section["tasks"]["tinyMMLU"]
    assert task["improved"] is True
    assert task["metric"] == "acc_norm"
    with pytest.raises(ScorecardError, match="version differs"):
        lmeval_section(lmeval_record(0.5), lmeval_record(0.55, "0.4.11"))
    with pytest.raises(ScorecardError, match="missing from the baseline"):
        lmeval_section(
            {"lm_eval_version": "0.4.12", "results": {}}, lmeval_record(0.55)
        )


def test_lmeval_section_refuses_mismatched_limit():
    # KB-23: a side resumed from a run with a different --limit is not
    # comparable with the other side
    limited = lmeval_record(0.55)
    limited["limit"] = 100
    with pytest.raises(ScorecardError, match="limit differs"):
        lmeval_section(lmeval_record(0.5), limited)


def test_render_markdown_covers_all_sections():
    card = assemble_scorecard(
        metadata(),
        {
            "holdout": holdout_section(holdout_record(2.0), holdout_record(1.5)),
            "lmeval": lmeval_section(lmeval_record(0.5), lmeval_record(0.55)),
        },
    )
    text = render_markdown(card)
    assert "| holdout | loss |" in text
    assert "tinyMMLU" in text
    assert "judge" not in text


def test_render_markdown_partial_run_says_skipped():
    card = assemble_scorecard(
        metadata(),
        {"holdout": holdout_section(holdout_record(2.0), holdout_record(1.5))},
    )
    text = render_markdown(card)
    assert "| lmeval | - | skipped |" in text
