"""Unified interp CLI: parser wiring only (no heavy deps executed).

build_parser() registers every subcommand; each minimal argv parses into the
right handler function. Handlers are not invoked, so torch/transformers and
the interp pipeline modules stay unimported.
"""

from __future__ import annotations

import pytest

from interp import cli
from interp.commands import (
    harvest,
    interp_gc,
    organism,
    probe_capture,
    probe_eval,
    probe_monitor,
    probe_train,
    sae_eval,
    sae_train,
    smoke_test,
    steer_extract,
    steer_scorecard,
)

CASES = [
    (["harvest", "--recipe", "r.yaml"], harvest.cmd),
    (["gc"], interp_gc.cmd),
    (["organism", "generate", "--recipe", "r.yaml"], organism.cmd),
    (["organism", "verify", "--recipe", "r.yaml",
      "--base-metrics", "b.json", "--subject-metrics", "s.json"], organism.cmd),
    (["organism", "seal", "--recipe", "r.yaml"], organism.cmd),
    (["organism", "check-seal", "--dir", "d"], organism.cmd),
    (["probe-capture", "--model", "m", "--dataset", "d.jsonl",
      "--layers", "3,5", "--out", "o"], probe_capture.cmd),
    (["probe-train", "--capture", "c", "--name", "n"], probe_train.cmd),
    (["probe-eval", "--probe", "p", "--label-name", "l"], probe_eval.cmd),
    (["probe-monitor", "--probes", "a,b", "--model", "m",
      "--model-hash", "h"], probe_monitor.cmd),
    (["sae-train", "--recipe", "r.yaml"], sae_train.cmd),
    (["sae-eval", "--sae-dir", "s", "--capture-dir", "c"], sae_eval.cmd),
    (["steer-extract", "--model", "m", "--out", "o"], steer_extract.cmd),
    (["steer-scorecard", "--vector", "v", "--attribute", "a"],
     steer_scorecard.cmd),
    (["smoke"], smoke_test.cmd),
]


@pytest.mark.parametrize("argv,func", CASES, ids=lambda c: c[0] if isinstance(c, list) else "")
def test_subcommand_parses_to_handler(argv, func):
    args = cli.build_parser().parse_args(argv)
    assert args.func is func


def test_missing_subcommand_is_usage_error():
    with pytest.raises(SystemExit) as exc:
        cli.build_parser().parse_args([])
    assert exc.value.code == 2


def test_organism_requires_verb():
    with pytest.raises(SystemExit) as exc:
        cli.build_parser().parse_args(["organism"])
    assert exc.value.code == 2


def test_defaults_preserved():
    args = cli.build_parser().parse_args(["gc"])
    assert args.root == "/mnt/ai-data/interp/activations"
    assert args.dry_run is False
    args = cli.build_parser().parse_args(
        ["probe-eval", "--probe", "p", "--label-name", "l"])
    assert args.subject_url == "http://127.0.0.1:8002"
    assert args.judge_url == "http://127.0.0.1:8003"
    assert args.max_tokens == 256


def test_steer_scorecard_malformed_scales_is_styled_usage_error(capsys):
    # Handler-level contract: bad --scales is a styled exit 2, not a raw
    # ValueError traceback.
    args = cli.build_parser().parse_args(
        ["steer-scorecard", "--vector", "v", "--attribute", "honesty",
         "--scales", "0.5,fast"])
    assert args.func(args) == 2
    captured = capsys.readouterr().out
    assert "[ERROR]" in captured
    assert "--scales" in captured or "scales" in captured
