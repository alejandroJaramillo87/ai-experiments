"""CLI handler contracts for the data-tooling subcommands: validate,
build-contamination, and make-replay-synthetic.

Each test parses real argv through build_parser() and invokes the bound
handler, asserting the exit-code contract (0 success, 1 failure, 2 invalid
usage), the styled [OK]/[INFO]/[ERROR] lines, and the artifacts written to
disk. Every path runs for real: validate against the committed reference
recipes, build-contamination against a local JSONL benchmark fixture, and
make-replay-synthetic with a tiny token count. No seams are stubbed.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest
from support import TINY_RECIPE, write_benchmarks_yaml

from training import cli
from training.data import contamination
from training.data.shards import load_manifest, verify_shards
from training.recipes import load_recipe, resolved_yaml


def run_cli(argv: list[str]) -> int:
    args = cli.build_parser().parse_args(argv)
    return args.func(args)


# -- validate ---------------------------------------------------------------


class TestValidate:
    def test_success_reports_schema_and_invariants(self, capsys):
        assert run_cli(["validate", TINY_RECIPE]) == 0
        out = capsys.readouterr().err
        assert f"[OK] Schema: {TINY_RECIPE} loads as a typed recipe" in out
        assert (
            "[OK] Validation: run tiny-e2e passes all cross-field invariants"
        ) in out
        # without overrides or --print, neither the override banner nor the
        # resolved yaml may appear
        assert "[INFO] Overrides applied" not in out
        assert "seed: 1234" not in out

    def test_print_emits_resolved_yaml_with_override(self, capsys):
        overrides = ["optimizer.lr=1e-4"]
        assert run_cli(["validate", TINY_RECIPE, *overrides, "--print"]) == 0
        captured = capsys.readouterr()
        assert f"[INFO] Overrides applied: {overrides[0]}" in captured.err
        # the resolved yaml is machine output: stdout holds exactly the
        # canonical snapshot serialization of the overridden recipe
        tail = captured.out
        assert tail == resolved_yaml(load_recipe(TINY_RECIPE, overrides))
        import yaml

        parsed = yaml.safe_load(tail)
        assert parsed["optimizer"]["lr"] == pytest.approx(1e-4)
        assert parsed["run"]["name"] == "tiny-e2e"

    def test_override_failing_invariant_lists_problems(self, capsys):
        assert run_cli(["validate", TINY_RECIPE, "optimizer.lr=-1"]) == 1
        out = capsys.readouterr().err
        assert "[INFO] Overrides applied: optimizer.lr=-1" in out
        assert "[ERROR] Recipe validation failed:" in out
        assert "  - optimizer.lr: must be positive, got -1.0" in out

    def test_unknown_override_key_is_load_error(self, capsys):
        assert run_cli(["validate", TINY_RECIPE, "optimizer.bogus=1"]) == 1
        out = capsys.readouterr().err
        assert "[ERROR] Recipe load failed:" in out
        assert "'bogus'" in out
        assert (
            "  Solution: Fix the reported key or type to match "
            "training/recipes/schema.py"
        ) in out
        # load fails before the schema line is reached
        assert "[OK] Schema" not in out

    def test_missing_file_is_load_error(self, tmp_path, capsys):
        missing = str(tmp_path / "nope.yaml")
        assert run_cli(["validate", missing]) == 1
        out = capsys.readouterr().err
        assert "[ERROR] Recipe load failed:" in out
        assert f"Recipe file not found: {missing}" in out


# -- build-contamination ---------------------------------------------------------------

# One 20-word document yields exactly 20 - 13 + 1 = 8 unique 13-grams, so the
# per-benchmark and total lines are asserted with exact counts.
BENCH_WORDS = 20
BENCH_NGRAMS = BENCH_WORDS - 13 + 1


class TestBuildContamination:
    def test_success_builds_index_and_reports_hash(self, tmp_path, capsys):
        rows = [{"question": " ".join(f"word{i}" for i in range(BENCH_WORDS))}]
        bench_yaml = write_benchmarks_yaml(tmp_path, rows)
        index_dir = str(tmp_path / "index")
        assert (
            run_cli(
                ["build-contamination", "--benchmarks", bench_yaml, "--out", index_dir]
            )
            == 0
        )

        assert os.path.isfile(os.path.join(index_dir, "ngrams.bin"))
        with open(os.path.join(index_dir, "index_manifest.json")) as f:
            manifest = json.load(f)
        assert manifest["total_ngrams"] == BENCH_NGRAMS

        out = capsys.readouterr().err
        assert "=== Contamination index build (P2-D1) ===" in out
        assert f"[INFO] Benchmarks: {bench_yaml}" in out
        assert f"[INFO] Output: {index_dir}" in out
        assert "[RUN] Extracting benchmark n-grams" in out
        assert f"  [OK] toy: 1 documents, {BENCH_NGRAMS} new n-grams" in out
        assert f"[OK] Total: {BENCH_NGRAMS} unique 13-grams" in out
        # the reported decontaminated_against must be the recomputable
        # manifest hash that shard manifests reference downstream
        assert (
            f"[OK] decontaminated_against: {contamination.index_hash(index_dir)}"
        ) in out

    @pytest.mark.parametrize(
        "body,detail",
        [
            (None, "No such file or directory"),
            ("version: 1\nbenchmarks: []\n", "missing required key 'ngram_size'"),
        ],
        ids=["missing-yaml", "incomplete-yaml"],
    )
    def test_build_error_exits_1(self, body, detail, tmp_path, capsys):
        bench_yaml = tmp_path / "benchmarks.yaml"
        if body is not None:
            bench_yaml.write_text(body)
        assert (
            run_cli(
                [
                    "build-contamination",
                    "--benchmarks",
                    str(bench_yaml),
                    "--out",
                    str(tmp_path / "index"),
                ]
            )
            == 1
        )
        out = capsys.readouterr().err
        assert "[ERROR] Index build failed:" in out
        assert detail in out
        assert "[OK] Total" not in out


# -- make-replay-synthetic ---------------------------------------------------------------


class TestMakeReplaySynthetic:
    def test_tokens_run_writes_shards_and_manifest(self, tmp_path, capsys):
        out_dir = str(tmp_path / "replay")
        # 3000 tokens at 1000/shard forces the multi-shard roll
        assert (
            run_cli(
                [
                    "make-replay-synthetic",
                    "--out",
                    out_dir,
                    "--tokens",
                    "3000",
                    "--vocab",
                    "512",
                    "--shard-tokens",
                    "1000",
                    "--seed",
                    "7",
                ]
            )
            == 0
        )

        manifest = load_manifest(out_dir)
        assert manifest["dataset"] == "replay-synthetic"
        assert manifest["dtype"] == "uint16"
        assert manifest["tokenizer_hash"] == "synthetic"
        assert manifest["seed"] == 7
        assert manifest["total_tokens"] == 3000
        assert len(manifest["shards"]) == 3
        # recorded shard hashes must match the bytes on disk
        verify_shards(out_dir, manifest)
        first = np.fromfile(
            os.path.join(out_dir, manifest["shards"][0]["file"]), dtype=np.uint16
        )
        assert int(first.max()) < 512

        out = capsys.readouterr().err
        assert "=== Synthetic replay shard set ===" in out
        assert f"[INFO] Output: {out_dir}" in out
        assert "[INFO] Tokens: 3000 (0.00GB at uint16)" in out
        assert "[RUN] Writing shards: seed 7" in out
        assert "[OK] Shards written: 3" in out
        assert f"[OK] Manifest: {os.path.join(out_dir, 'manifest.json')}" in out

    def test_gb_run_derives_token_count(self, tmp_path, capsys):
        out_dir = str(tmp_path / "replay-gb")
        gb = 0.000004
        expected = int(gb * (1 << 30) / 2)  # uint16 itemsize
        assert (
            run_cli(
                [
                    "make-replay-synthetic",
                    "--out",
                    out_dir,
                    "--gb",
                    str(gb),
                    "--vocab",
                    "512",
                    "--tokenizer-hash",
                    "abc123",
                    "--eos-id",
                    "2",
                ]
            )
            == 0
        )

        manifest = load_manifest(out_dir)
        assert manifest["total_tokens"] == expected
        assert manifest["tokenizer_hash"] == "abc123"
        assert manifest["eos_id"] == 2

        out = capsys.readouterr().err
        assert f"[INFO] Tokens: {expected} (0.00GB at uint16)" in out
        assert "[OK] Shards written: 1" in out

    def test_zero_tokens_is_usage_error(self, tmp_path, capsys):
        out_dir = str(tmp_path / "replay-zero")
        assert (
            run_cli(["make-replay-synthetic", "--out", out_dir, "--tokens", "0"]) == 2
        )
        out = capsys.readouterr().err
        assert "[ERROR] Token count: must be positive, got 0" in out
        # rejected before the writer is constructed, so nothing is created
        assert not os.path.exists(out_dir)
