"""Unified training CLI parser tests: each migrated training/commands/
subcommand parses its minimal argument set and binds the module's cmd
entry point. Parse-only; no command bodies execute."""

from __future__ import annotations

import pytest

from training import cli
from training.commands import (
    build_contamination,
    generate_teacher,
    make_replay_synthetic,
    prepare_corpus,
    prepare_data,
    prepare_sft,
    smoke_test,
    train_tokenizer,
    validate_recipe,
)

CASES = [
    (["prepare-data", "--source", "d.jsonl", "--out", "/tmp/o",
      "--tokenizer", "tok"], prepare_data.cmd),
    (["prepare-corpus", "--source", "d.jsonl", "--out", "/tmp/o",
      "--work-dir", "/tmp/w", "--tokenizer", "tok"], prepare_corpus.cmd),
    (["build-contamination", "--out", "/tmp/o"], build_contamination.cmd),
    (["prepare-sft", "--source", "d.jsonl", "--out", "/tmp/o",
      "--tokenizer", "tok", "--model-type", "gemma4"], prepare_sft.cmd),
    (["train-tokenizer", "--corpus", "c.jsonl", "--out", "/tmp/o"],
     train_tokenizer.cmd),
    (["validate", "recipe.yaml"], validate_recipe.cmd),
    (["validate", "recipe.yaml", "optimizer.lr=1e-4", "--print"],
     validate_recipe.cmd),
    (["smoke"], smoke_test.cmd),
    (["smoke", "--build"], smoke_test.cmd),
    (["generate-teacher", "--pipeline", "respond", "--prompts", "p.jsonl",
      "--teacher", "/app/models/t", "--out", "/tmp/o"], generate_teacher.cmd),
    (["make-replay-synthetic", "--out", "/tmp/o", "--gb", "1"],
     make_replay_synthetic.cmd),
]


@pytest.mark.parametrize("argv,expected_func",
                         CASES, ids=lambda v: v if callable(v) else v[0])
def test_subcommand_binds_cmd(argv, expected_func):
    args = cli.build_parser().parse_args(argv)
    assert args.func is expected_func


def test_validate_keeps_positional_shape():
    args = cli.build_parser().parse_args(
        ["validate", "recipe.yaml", "data.micro_batch=8"])
    assert args.config == "recipe.yaml"
    assert args.overrides == ["data.micro_batch=8"]


def test_make_replay_size_group_is_exclusive():
    parser = cli.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["make-replay-synthetic", "--out", "/tmp/o",
                           "--gb", "1", "--tokens", "10"])


class TestPositiveInt:
    """First coverage for the shared argparse helper (previously four
    untested copies)."""

    def test_accepts_underscored_ints(self):
        from training.commands.args import positive_int

        assert positive_int("2_000_000") == 2_000_000
        assert positive_int("7") == 7

    @pytest.mark.parametrize("bad", ["0", "-3", "1.5", "many"])
    def test_rejects_non_positive_and_non_int(self, bad):
        import argparse

        from training.commands.args import positive_int

        with pytest.raises((argparse.ArgumentTypeError, ValueError)):
            positive_int(bad)
