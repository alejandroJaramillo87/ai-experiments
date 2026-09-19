"""CLI handler contracts for the train-tokenizer subcommand.

Each test parses real argv through build_parser() and invokes the bound
handler, asserting the exit-code contract (0 success, 1 failure, 2 invalid
usage), the styled [OK]/[WARN]/[ERROR] lines, and concrete file side effects,
running the real tokenizers-library pipeline on a tiny corpus. The
generate-teacher handler's tests are test_cli_handlers_teacher.py.
"""

from __future__ import annotations

import json
import os

import pytest
from support import build_bpe_tokenizer
from tokenizers import Tokenizer

from common.output import Printer
from training import cli
from training.commands.train_tokenizer import eval_corpus_pair, metrics_table

# Smallest round vocab above MIN_VOCAB_SIZE (256-byte alphabet + 19 specials)
# so real BPE training stays in the milliseconds
VOCAB = 300

PANGRAMS = [
    "the quick brown fox jumps over the lazy dog",
    "pack my box with five dozen liquor jugs",
    "sphinx of black quartz judge my vow",
    "how vexingly quick daft zebras jump",
    "the five boxing wizards jump quickly",
    "jackdaws love my big sphinx of quartz",
    "waltz bad nymph for quick jigs vex",
    "quick zephyrs blow vexing daft jim",
]
TRAIN_DOCS = PANGRAMS * 4

EVAL_WEB = [
    "the sun rises over the quiet harbor",
    "ships depart at dawn with cargo",
    "a light rain falls on the pier",
]
EVAL_CODE = [
    "def add(a, b): return a + b",
    "for i in range(10): print(i)",
    "x = [n * n for n in values]",
]


def run_cli(argv: list[str]) -> int:
    args = cli.build_parser().parse_args(argv)
    return args.func(args)


def write_jsonl(path, texts: list[str], field: str = "text") -> str:
    with open(path, "w") as f:
        for text in texts:
            f.write(json.dumps({field: text}) + "\n")
    return str(path)


# -- eval_corpus_pair and metrics_table (module-level helpers) ------------------


def test_eval_corpus_pair_splits_on_first_equals():
    assert eval_corpus_pair("web=/data/web.jsonl") == ("web", "/data/web.jsonl")
    # Only the first = separates; the path may contain more
    assert eval_corpus_pair("a=b=c") == ("a", "b=c")


def test_eval_corpus_pair_without_equals_is_usage_error(tmp_path, capsys):
    with pytest.raises(SystemExit) as exc:
        cli.build_parser().parse_args(
            [
                "train-tokenizer",
                "--corpus",
                "c.jsonl",
                "--out",
                str(tmp_path),
                "--eval-corpus",
                "webonly",
            ]
        )
    assert exc.value.code == 2
    assert "expected DOMAIN=PATH" in capsys.readouterr().err


def test_metrics_table_skips_reference_missing_a_domain(capsys):
    def stats(fertility: float, compression: float) -> dict:
        return {"fertility": fertility, "compression": compression}

    metrics = {"web": stats(1.5, 4.0), "code": stats(2.25, 3.5)}
    references = {"gpt2": {"web": stats(1.75, 3.75)}}  # no code entry
    metrics_table(Printer(), metrics, references)
    lines = capsys.readouterr().err.splitlines()
    assert "fertility" in lines[0] and "compression" in lines[0]
    trained_rows = [row for row in lines if row.startswith("trained")]
    ref_rows = [row for row in lines if row.startswith("gpt2")]
    assert len(trained_rows) == 2
    assert len(ref_rows) == 1 and "web" in ref_rows[0]
    assert "1.5000" in trained_rows[0] and "4.0000" in trained_rows[0]


# -- train-tokenizer handler ----------------------------------------------------


class TestTrainTokenizerCmd:
    def test_success_with_two_eval_domains(self, tmp_path, capsys):
        corpus = write_jsonl(tmp_path / "corpus.jsonl", TRAIN_DOCS)
        web = write_jsonl(tmp_path / "web.jsonl", EVAL_WEB)
        code = write_jsonl(tmp_path / "code.jsonl", EVAL_CODE)
        out_dir = str(tmp_path / "artifact")
        # --eval-docs 2 with 3-doc files exercises the per-domain read cap
        assert (
            run_cli(
                [
                    "train-tokenizer",
                    "--corpus",
                    corpus,
                    "--out",
                    out_dir,
                    "--vocab-size",
                    str(VOCAB),
                    "--eval-corpus",
                    f"web={web}",
                    "--eval-corpus",
                    f"code={code}",
                    "--eval-docs",
                    "2",
                    "--no-references",
                ]
            )
            == 0
        )
        out = capsys.readouterr().err
        assert "=== Tokenizer training (Phase 0 D3) ===" in out
        assert "[INFO] Corpus files: 1" in out
        assert "[INFO] Vocab size: 300" in out
        assert "[RUN] Training byte-level BPE" in out
        assert "[OK] Trained: 32 documents" in out
        assert "  [OK] web: 2 documents round-trip lossless" in out
        assert "  [OK] code: 2 documents round-trip lossless" in out
        assert "[SKIP] Reference comparison disabled (--no-references)" in out
        assert "[WARN]" not in out

        tokenizer_path = os.path.join(out_dir, "tokenizer.json")
        assert os.path.isfile(tokenizer_path)
        with open(os.path.join(out_dir, "manifest.json")) as f:
            manifest = json.load(f)
        assert (
            manifest["vocab_size"]
            == Tokenizer.from_file(tokenizer_path).get_vocab_size()
        )
        assert manifest["requested_vocab_size"] == VOCAB
        assert manifest["corpus"] == [corpus]
        assert manifest["special_tokens"]["<|bos|>"] == 0
        assert set(manifest["metrics"]) == {"web", "code"}
        assert manifest["references"] == {}
        assert f"[OK] Artifact: {tokenizer_path}" in out
        assert f"[OK] Tokenizer hash: {manifest['tokenizer_hash']}" in out
        # Metrics table rows for the trained tokenizer, one per domain
        assert [
            row
            for row in out.splitlines()
            if row.startswith("trained") and "web" in row
        ]

    def test_corpus_tail_fallback_warns_and_succeeds(self, tmp_path, capsys):
        corpus = write_jsonl(tmp_path / "corpus.jsonl", TRAIN_DOCS)
        out_dir = str(tmp_path / "artifact")
        # A 200-byte budget stops sampling after ~5 pangrams, leaving a tail
        assert (
            run_cli(
                [
                    "train-tokenizer",
                    "--corpus",
                    corpus,
                    "--out",
                    out_dir,
                    "--vocab-size",
                    str(VOCAB),
                    "--sample-bytes",
                    "200",
                    "--eval-docs",
                    "4",
                    "--no-references",
                ]
            )
            == 0
        )
        out = capsys.readouterr().err
        assert "[WARN] No --eval-corpus given; evaluating on the corpus tail" in out
        assert "[WARN] Only 1 evaluation domain(s); D3 recommends at least two" in out
        assert "  [OK] corpus-tail: 4 documents round-trip lossless" in out
        assert os.path.isfile(os.path.join(out_dir, "manifest.json"))

    def test_exhausted_corpus_without_eval_corpus_exits_1(self, tmp_path, capsys):
        corpus = write_jsonl(tmp_path / "corpus.jsonl", TRAIN_DOCS)
        out_dir = str(tmp_path / "artifact")
        # Default 10 GB budget consumes the whole corpus: no held-out tail
        assert (
            run_cli(
                [
                    "train-tokenizer",
                    "--corpus",
                    corpus,
                    "--out",
                    out_dir,
                    "--vocab-size",
                    str(VOCAB),
                    "--no-references",
                ]
            )
            == 1
        )
        out = capsys.readouterr().err
        assert (
            "[ERROR] Corpus exhausted during training; no held-out documents remain"
        ) in out
        assert "Solution: pass --eval-corpus DOMAIN=PATH or lower --sample-bytes" in out
        assert not os.path.exists(os.path.join(out_dir, "tokenizer.json"))

    def test_missing_corpus_exits_2(self, tmp_path, capsys):
        pattern = str(tmp_path / "absent-*.jsonl")
        assert (
            run_cli(
                [
                    "train-tokenizer",
                    "--corpus",
                    pattern,
                    "--out",
                    str(tmp_path / "artifact"),
                ]
            )
            == 2
        )
        out = capsys.readouterr().err
        assert f"[ERROR] corpus pattern matches nothing: {pattern}" in out
        assert "Solution: check the corpus path or quote the glob pattern" in out

    def test_existing_artifact_refused_then_force_overwrites(self, tmp_path, capsys):
        corpus = write_jsonl(tmp_path / "corpus.jsonl", TRAIN_DOCS)
        web = write_jsonl(tmp_path / "web.jsonl", EVAL_WEB)
        out_dir = tmp_path / "artifact"
        out_dir.mkdir()
        (out_dir / "tokenizer.json").write_text("stale")
        argv = [
            "train-tokenizer",
            "--corpus",
            corpus,
            "--out",
            str(out_dir),
            "--vocab-size",
            str(VOCAB),
            "--eval-corpus",
            f"web={web}",
            "--no-references",
        ]
        assert run_cli(argv) == 1
        out = capsys.readouterr().err
        assert f"[ERROR] {out_dir} already contains tokenizer.json" in out
        assert "Solution: choose a new output directory or pass --force" in out
        assert (out_dir / "tokenizer.json").read_text() == "stale"

        assert run_cli(argv + ["--force"]) == 0
        assert (out_dir / "tokenizer.json").read_text() != "stale"
        assert os.path.isfile(out_dir / "manifest.json")

    def test_vocab_below_minimum_is_training_failure(self, tmp_path, capsys):
        corpus = write_jsonl(tmp_path / "corpus.jsonl", TRAIN_DOCS)
        assert (
            run_cli(
                [
                    "train-tokenizer",
                    "--corpus",
                    corpus,
                    "--out",
                    str(tmp_path / "artifact"),
                    "--vocab-size",
                    "100",
                ]
            )
            == 1
        )
        out = capsys.readouterr().err
        assert "[ERROR] Training failed: vocab_size 100 below minimum 275" in out

    def test_roundtrip_failure_exits_1_without_artifact(
        self, tmp_path, monkeypatch, capsys
    ):
        # Byte-level BPE is lossless by construction, so the failure arm is
        # only reachable through the evaluate seam
        corpus = write_jsonl(tmp_path / "corpus.jsonl", TRAIN_DOCS)
        web = write_jsonl(tmp_path / "web.jsonl", EVAL_WEB)
        out_dir = str(tmp_path / "artifact")

        def lossy_evaluate(tok, domains):
            return {
                name: {
                    "documents": len(texts),
                    "tokens": 10,
                    "fertility": 1.0,
                    "compression": 4.0,
                    "roundtrip_failures": 2,
                }
                for name, texts in domains.items()
            }

        monkeypatch.setattr("training.data.tokenizer.evaluate", lossy_evaluate)
        assert (
            run_cli(
                [
                    "train-tokenizer",
                    "--corpus",
                    corpus,
                    "--out",
                    out_dir,
                    "--vocab-size",
                    str(VOCAB),
                    "--eval-corpus",
                    f"web={web}",
                    "--no-references",
                ]
            )
            == 1
        )
        out = capsys.readouterr().err
        assert "[ERROR] Round-trip losslessness failed: {'web': 2}" in out
        assert not os.path.exists(os.path.join(out_dir, "tokenizer.json"))

    def test_local_reference_compared_and_bad_reference_degrades_to_warn(
        self, tmp_path, capsys
    ):
        corpus = write_jsonl(tmp_path / "corpus.jsonl", TRAIN_DOCS)
        web = write_jsonl(tmp_path / "web.jsonl", EVAL_WEB)
        # A local tokenizer.json loads without transformers or network
        ref_path = os.path.join(
            build_bpe_tokenizer(
                str(tmp_path / "ref"),
                PANGRAMS,
                specials=["<|bos|>", "<|eos|>", "<|pad|>"],
            ),
            "tokenizer.json",
        )
        bogus = str(tmp_path / "absent" / "tokenizer.json")
        out_dir = str(tmp_path / "artifact")
        assert (
            run_cli(
                [
                    "train-tokenizer",
                    "--corpus",
                    corpus,
                    "--out",
                    out_dir,
                    "--vocab-size",
                    str(VOCAB),
                    "--eval-corpus",
                    f"web={web}",
                    "--reference",
                    ref_path,
                    "--reference",
                    bogus,
                ]
            )
            == 0
        )
        out = capsys.readouterr().err
        assert f"[RUN] Reference metrics: {ref_path}" in out
        assert f"[WARN] Reference {bogus} unavailable:" in out
        assert "Solution: check network/credentials, or pass --no-references" in out
        with open(os.path.join(out_dir, "manifest.json")) as f:
            manifest = json.load(f)
        assert list(manifest["references"]) == [ref_path]
        assert "web" in manifest["references"][ref_path]
        assert [row for row in out.splitlines() if row.startswith(ref_path)]

    def test_write_artifact_race_exits_1(self, tmp_path, monkeypatch, capsys):
        # The upfront existence check passed, then the artifact appeared
        # before write time; only the write_artifact seam can model the race
        corpus = write_jsonl(tmp_path / "corpus.jsonl", TRAIN_DOCS)
        web = write_jsonl(tmp_path / "web.jsonl", EVAL_WEB)

        def racing_write(*args, **kwargs):
            raise FileExistsError(
                "artifact already contains tokenizer.json; pass --force to overwrite"
            )

        monkeypatch.setattr("training.data.tokenizer.write_artifact", racing_write)
        assert (
            run_cli(
                [
                    "train-tokenizer",
                    "--corpus",
                    corpus,
                    "--out",
                    str(tmp_path / "artifact"),
                    "--vocab-size",
                    str(VOCAB),
                    "--eval-corpus",
                    f"web={web}",
                    "--no-references",
                ]
            )
            == 1
        )
        out = capsys.readouterr().err
        assert "[ERROR] artifact already contains tokenizer.json" in out
        assert "Solution: choose a new output directory or pass --force" in out
