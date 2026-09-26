"""CLI handler contracts for the data-preparation subcommands: prepare-data,
prepare-corpus, and prepare-sft.

Each test parses real argv through build_parser() and invokes the bound
handler, asserting the exit-code contract (0 success, 1 pipeline failure,
2 invalid usage), the styled [OK]/[WARN]/[ERROR] lines, and the manifests
written to disk. Tokenize-and-shard paths run for real against the tiny BPE
tokenizer and synthetic JSONL corpus from conftest; only the datatrove
quality stages (engine not installed in the sandbox) are stubbed at their
home module, training.data.pipeline.
"""

from __future__ import annotations

import json
import os

import pytest

from training import cli
from training.data import pipeline as pipeline_mod


def run_cli(argv: list[str]) -> int:
    args = cli.build_parser().parse_args(argv)
    return args.func(args)


def read_manifest(out_dir: str) -> dict:
    with open(os.path.join(out_dir, "manifest.json")) as f:
        return json.load(f)


# -- prepare-data ---------------------------------------------------------------


class TestPrepareData:
    def test_success_writes_shards_and_reports_manifest(
        self, tokenizer_dir, jsonl_path, tmp_path, documents, capsys
    ):
        out_dir = str(tmp_path / "shards")
        # 64 tokens/shard forces multiple shards out of the 32-document corpus
        assert (
            run_cli(
                [
                    "prepare-data",
                    "--source",
                    jsonl_path,
                    "--out",
                    out_dir,
                    "--tokenizer",
                    tokenizer_dir,
                    "--shard-tokens",
                    "64",
                ]
            )
            == 0
        )

        manifest = read_manifest(out_dir)
        assert manifest["total_documents"] == len(documents)
        assert len(manifest["shards"]) >= 2
        for rec in manifest["shards"]:
            assert os.path.isfile(os.path.join(out_dir, rec["file"]))

        out = capsys.readouterr().err
        assert "=== Data preparation (Phase 0 D2) ===" in out
        assert f"[INFO] Source: {jsonl_path}" in out
        assert f"[INFO] Tokenizer: {tokenizer_dir}" in out
        assert "[RUN] Tokenizing and writing shards" in out
        # 32 documents fit one encode batch, so the single progress call
        # already carries the final totals
        assert (
            f"  [INFO] Progress: {manifest['total_tokens']:,} tokens, "
            f"{manifest['total_documents']:,} documents"
        ) in out
        assert (
            f"[OK] Shards written: {len(manifest['shards'])} "
            f"({manifest['dtype']}, 64 tokens/shard)"
        ) in out
        assert (
            f"[OK] Total: {manifest['total_tokens']:,} tokens, "
            f"{manifest['total_documents']:,} documents"
        ) in out
        assert f"[OK] Manifest: {os.path.join(out_dir, 'manifest.json')}" in out

    def test_existing_manifest_exits_1_with_solution(
        self, tokenizer_dir, jsonl_path, tmp_path, capsys
    ):
        out_dir = tmp_path / "shards"
        out_dir.mkdir()
        (out_dir / "manifest.json").write_text("{}\n")
        assert (
            run_cli(
                [
                    "prepare-data",
                    "--source",
                    jsonl_path,
                    "--out",
                    str(out_dir),
                    "--tokenizer",
                    tokenizer_dir,
                ]
            )
            == 1
        )
        out = capsys.readouterr().err
        assert "[ERROR]" in out
        assert "refusing to overwrite" in out
        assert "Solution: choose a new output directory" in out

    @pytest.mark.parametrize(
        "extra,detail",
        [
            (["--text-field", "body"], "no field"),
            (["--eos-token", "<|missing|>"], "Cannot determine EOS id"),
        ],
        ids=["missing-text-field", "unknown-eos-token"],
    )
    def test_pipeline_error_exits_1(
        self, extra, detail, tokenizer_dir, jsonl_path, tmp_path, capsys
    ):
        assert (
            run_cli(
                [
                    "prepare-data",
                    "--source",
                    jsonl_path,
                    "--out",
                    str(tmp_path / "shards"),
                    "--tokenizer",
                    tokenizer_dir,
                    *extra,
                ]
            )
            == 1
        )
        out = capsys.readouterr().err
        assert "[ERROR] Pipeline failed:" in out
        assert detail in out


# -- prepare-corpus ---------------------------------------------------------------

FUNNEL = [{"stage": "filter", "docs_in": 32, "docs_out": 32, "drops": {}}]


def stub_quality_stages(monkeypatch) -> list:
    """Replace the datatrove-backed stage runner with a pass-through; the
    engine is not installed in the sandbox. Returns the recorded calls."""
    calls: list = []

    def fake(input_dir, work_dir, config, contamination_dir):
        calls.append((input_dir, contamination_dir))
        return input_dir, [dict(stage) for stage in FUNNEL]

    monkeypatch.setattr(pipeline_mod, "run_quality_stages", fake)
    return calls


class TestPrepareCorpus:
    def test_success_without_contamination(
        self, monkeypatch, tokenizer_dir, jsonl_path, tmp_path, capsys
    ):
        calls = stub_quality_stages(monkeypatch)
        out_dir = str(tmp_path / "shards")
        work_dir = str(tmp_path / "work")
        assert (
            run_cli(
                [
                    "prepare-corpus",
                    "--source",
                    jsonl_path,
                    "--out",
                    out_dir,
                    "--work-dir",
                    work_dir,
                    "--tokenizer",
                    tokenizer_dir,
                    "--num-proc",
                    "4",
                    "--shard-tokens",
                    "64",
                ]
            )
            == 0
        )

        # ingest normalizes the local JSONL into the work dir before staging
        ingest = os.path.join(work_dir, pipeline_mod.STAGE_INGEST)
        assert os.path.isfile(os.path.join(ingest, "00000.jsonl"))
        assert calls == [(ingest, None)]

        manifest = read_manifest(out_dir)
        assert manifest["pipeline"]["stage_funnel"] == FUNNEL
        assert manifest["decontaminated_against"] is None
        assert manifest["dataset"] == jsonl_path
        assert manifest["total_documents"] == 32

        out = capsys.readouterr().err
        assert "=== Corpus preparation (P2-D1) ===" in out
        assert f"[INFO] Work dir: {work_dir}" in out
        assert "[WARN] Decontamination: skipped (no --contamination index)" in out
        assert "[RUN] Running pipeline stages (4 workers)" in out
        assert (
            f"  [INFO] Progress: {manifest['total_tokens']:,} tokens, 32 documents"
        ) in out
        assert "  [OK] filter: 32 -> 32 (0 dropped)" in out
        assert (
            f"[OK] Shards written: {len(manifest['shards'])} "
            f"({manifest['dtype']}, 64 tokens/shard)"
        ) in out
        assert (f"[OK] Total: {manifest['total_tokens']:,} tokens, 32 documents") in out
        assert "[OK] decontaminated_against: None" in out
        assert f"[OK] Manifest: {os.path.join(out_dir, 'manifest.json')}" in out

    def test_existing_manifest_exits_1_with_solution(
        self, monkeypatch, tokenizer_dir, jsonl_path, tmp_path, capsys
    ):
        stub_quality_stages(monkeypatch)
        out_dir = tmp_path / "shards"
        out_dir.mkdir()
        (out_dir / "manifest.json").write_text("{}\n")
        assert (
            run_cli(
                [
                    "prepare-corpus",
                    "--source",
                    jsonl_path,
                    "--out",
                    str(out_dir),
                    "--work-dir",
                    str(tmp_path / "work"),
                    "--tokenizer",
                    tokenizer_dir,
                ]
            )
            == 1
        )
        out = capsys.readouterr().err
        assert "[ERROR]" in out
        assert "refusing to overwrite" in out
        assert "Solution: choose a new output directory" in out

    def test_bad_pipeline_config_exits_1(
        self, tokenizer_dir, jsonl_path, tmp_path, capsys
    ):
        # Real config-load failure: the handler must fail before any stage
        # runs. --contamination exercises the index banner line.
        config = tmp_path / "pipeline.yaml"
        config.write_text("bogus: 1\n")
        contamination = str(tmp_path / "contamination")
        assert (
            run_cli(
                [
                    "prepare-corpus",
                    "--source",
                    jsonl_path,
                    "--out",
                    str(tmp_path / "shards"),
                    "--work-dir",
                    str(tmp_path / "work"),
                    "--tokenizer",
                    tokenizer_dir,
                    "--pipeline-config",
                    str(config),
                    "--contamination",
                    contamination,
                ]
            )
            == 1
        )
        out = capsys.readouterr().err
        assert f"[INFO] Contamination index: {contamination}" in out
        assert "[ERROR] Pipeline failed:" in out
        assert "unknown pipeline config key 'bogus'" in out


# -- prepare-sft ---------------------------------------------------------------

# The handler's progress callback fires every 1000 records, so the success
# corpus carries exactly enough conversations to see one progress line.
CONVERSATION_COUNT = 1000


@pytest.fixture()
def chat_jsonl(tmp_path) -> str:
    path = tmp_path / "conversations.jsonl"
    with open(path, "w") as f:
        for i in range(CONVERSATION_COUNT):
            f.write(
                json.dumps(
                    {
                        "messages": [
                            {"role": "user", "content": f"question number {i}"},
                            {
                                "role": "assistant",
                                "content": f"the answer to question {i}",
                            },
                        ]
                    }
                )
                + "\n"
            )
    return str(path)


class TestPrepareSft:
    def test_success_writes_split_files_and_reports_manifest(
        self, tokenizer_dir, chat_jsonl, tmp_path, capsys
    ):
        out_dir = str(tmp_path / "sft")
        assert (
            run_cli(
                [
                    "prepare-sft",
                    "--source",
                    chat_jsonl,
                    "--out",
                    out_dir,
                    "--tokenizer",
                    tokenizer_dir,
                    "--model-type",
                    "gemma4",
                    "--holdout",
                    "0.1",
                ]
            )
            == 0
        )

        manifest = read_manifest(out_dir)
        assert manifest["holdout_fraction"] == 0.1
        assert manifest["dropped_examples"] == 0
        assert (
            manifest["train_examples"] + manifest["holdout_examples"]
            == CONVERSATION_COUNT
        )
        assert manifest["holdout_examples"] > 0
        with open(os.path.join(out_dir, "train.jsonl")) as f:
            assert sum(1 for _ in f) == manifest["train_examples"]

        out = capsys.readouterr().err
        assert "=== SFT data preparation (Phase 1 F1) ===" in out
        assert "[INFO] Model family: gemma4" in out
        assert "[RUN] Rendering, tokenizing, and masking conversations" in out
        assert "  [INFO] Progress: 1,000 examples," in out
        assert f"[OK] Train examples: {manifest['train_examples']:,}" in out
        assert (
            f"[OK] Holdout examples: {manifest['holdout_examples']:,} "
            "(fraction 0.1, seed 42)"
        ) in out
        assert "[WARN] Dropped examples" not in out
        assert (
            f"[OK] Tokens: {manifest['total_tokens']:,} total, "
            f"{manifest['supervised_tokens']:,} supervised"
        ) in out
        assert f"[OK] Manifest: {os.path.join(out_dir, 'manifest.json')}" in out

    def test_replay_flags_reach_the_pipeline_and_print(
        self, tokenizer_dir, chat_jsonl, tmp_path, capsys
    ):
        out_dir = str(tmp_path / "sft")
        assert (
            run_cli(
                [
                    "prepare-sft",
                    "--source",
                    chat_jsonl,
                    "--out",
                    out_dir,
                    "--tokenizer",
                    tokenizer_dir,
                    "--model-type",
                    "gemma4",
                    "--holdout",
                    "0",
                    "--repeat",
                    "2",
                    "--replay-source",
                    chat_jsonl,
                    "--replay-split",
                    "test_sft",
                    "--replay-examples",
                    "3",
                ]
            )
            == 0
        )
        manifest = read_manifest(out_dir)
        assert manifest["train_examples"] == 2 * CONVERSATION_COUNT + 3
        assert manifest["repeat"] == 2
        assert manifest["replay"]["examples"] == 3
        assert manifest["replay"]["split"] == "test_sft"
        out = capsys.readouterr().err
        assert f"[INFO] Replay: 3 examples from {chat_jsonl} (test_sft)" in out
        assert "[INFO] Corpus repeat: 2" in out

    def test_replay_source_without_a_count_is_usage_error(
        self, tokenizer_dir, chat_jsonl, tmp_path, capsys
    ):
        assert (
            run_cli(
                [
                    "prepare-sft",
                    "--source",
                    chat_jsonl,
                    "--out",
                    str(tmp_path / "sft"),
                    "--tokenizer",
                    tokenizer_dir,
                    "--model-type",
                    "gemma4",
                    "--replay-source",
                    chat_jsonl,
                ]
            )
            == 2
        )
        assert "--replay-source and --replay-examples go together" in (
            capsys.readouterr().err
        )

    def test_truncation_drops_warn(self, tokenizer_dir, tmp_path, capsys):
        # max-seq-len 2 leaves only masked header tokens, so every example
        # drops; the run still succeeds and warns with the dropped count.
        source = tmp_path / "tiny.jsonl"
        with open(source, "w") as f:
            for i in range(5):
                f.write(json.dumps({"prompt": f"q{i}", "completion": f"a{i}"}) + "\n")
        assert (
            run_cli(
                [
                    "prepare-sft",
                    "--source",
                    str(source),
                    "--out",
                    str(tmp_path / "sft"),
                    "--tokenizer",
                    tokenizer_dir,
                    "--model-type",
                    "gemma4",
                    "--max-seq-len",
                    "2",
                ]
            )
            == 0
        )
        out = capsys.readouterr().err
        assert (
            "[WARN] Dropped examples: 5 (no supervised tokens after truncation)"
        ) in out
        assert "[OK] Train examples: 0" in out

    @pytest.mark.parametrize("value", ["1.0", "-0.1"])
    def test_holdout_out_of_range_is_usage_error(self, value, capsys):
        with pytest.raises(SystemExit) as exc:
            cli.build_parser().parse_args(
                [
                    "prepare-sft",
                    "--source",
                    "d.jsonl",
                    "--out",
                    "/tmp/o",
                    "--tokenizer",
                    "tok",
                    "--model-type",
                    "gemma4",
                    "--holdout",
                    value,
                ]
            )
        assert exc.value.code == 2
        assert "must be in [0, 1)" in capsys.readouterr().err

    def test_holdout_zero_parses(self):
        args = cli.build_parser().parse_args(
            [
                "prepare-sft",
                "--source",
                "d.jsonl",
                "--out",
                "/tmp/o",
                "--tokenizer",
                "tok",
                "--model-type",
                "gemma4",
                "--holdout",
                "0",
            ]
        )
        assert args.holdout == 0.0

    def test_unknown_model_type_exits_1(
        self, tokenizer_dir, chat_jsonl, tmp_path, capsys
    ):
        assert (
            run_cli(
                [
                    "prepare-sft",
                    "--source",
                    chat_jsonl,
                    "--out",
                    str(tmp_path / "sft"),
                    "--tokenizer",
                    tokenizer_dir,
                    "--model-type",
                    "llama9",
                ]
            )
            == 1
        )
        out = capsys.readouterr().err
        assert (
            "[ERROR] HF model_type 'llama9' has no entry in the family registry"
        ) in out
        # registry misses get the registry message verbatim, not the
        # generic pipeline-failure framing
        assert "Pipeline failed" not in out

    def test_existing_manifest_exits_1_with_solution(
        self, tokenizer_dir, chat_jsonl, tmp_path, capsys
    ):
        out_dir = tmp_path / "sft"
        out_dir.mkdir()
        (out_dir / "manifest.json").write_text("{}\n")
        assert (
            run_cli(
                [
                    "prepare-sft",
                    "--source",
                    chat_jsonl,
                    "--out",
                    str(out_dir),
                    "--tokenizer",
                    tokenizer_dir,
                    "--model-type",
                    "gemma4",
                ]
            )
            == 1
        )
        out = capsys.readouterr().err
        assert "[ERROR]" in out
        assert "refusing to overwrite" in out
        assert "Solution: choose a new output directory" in out

    def test_malformed_record_exits_1(self, tokenizer_dir, tmp_path, capsys):
        source = tmp_path / "bad.jsonl"
        source.write_text(json.dumps({"foo": "bar"}) + "\n")
        assert (
            run_cli(
                [
                    "prepare-sft",
                    "--source",
                    str(source),
                    "--out",
                    str(tmp_path / "sft"),
                    "--tokenizer",
                    tokenizer_dir,
                    "--model-type",
                    "gemma4",
                ]
            )
            == 1
        )
        out = capsys.readouterr().err
        assert "[ERROR] Pipeline failed:" in out
        assert "record 0" in out
