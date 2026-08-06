"""CLI handler contracts for the train-tokenizer and generate-teacher
subcommands.

Each test parses real argv through build_parser() and invokes the bound
handler, asserting the exit-code contract (0 success, 1 failure, 2 invalid
usage), the styled [OK]/[WARN]/[ERROR] lines, and concrete file side effects.
train-tokenizer runs the real tokenizers-library pipeline on a tiny corpus;
generate-teacher runs the real teacher harness with the serving transport
(swap_model, wait_healthy, chat_completion) faked at training.serving, since
the sandbox has no docker or inference services.
"""

from __future__ import annotations

import hashlib
import json
import os

import pytest
from support import build_bpe_tokenizer
from tokenizers import Tokenizer

from common.output import Printer
from training import cli, serving
from training.commands.train_tokenizer import eval_corpus_pair, metrics_table
from training.data import teacher

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


# -- generate-teacher handler ----------------------------------------------------


def write_prompts(path, n: int) -> str:
    with open(path, "w") as f:
        for i in range(n):
            f.write(json.dumps({"instruction": f"q{i}"}) + "\n")
    return str(path)


def patch_transport(monkeypatch, chat=None, healthy_after: float = 1.5):
    """Fake the serving transport at training.serving; returns recorded
    (budget, chat-call) lists."""
    budgets: list[int] = []
    calls: list[tuple] = []

    def fake_wait(service, budget):
        budgets.append(budget)
        return healthy_after

    def default_chat(service, messages, sampling, timeout=None):
        calls.append((messages, sampling, timeout))
        return {"content": "ans:" + messages[-1]["content"]}

    monkeypatch.setattr("training.serving.wait_healthy", fake_wait)
    monkeypatch.setattr("training.serving.chat_completion", chat or default_chat)
    return budgets, calls


def teacher_argv(prompts: str, out_dir: str, *extra: str) -> list[str]:
    return [
        "generate-teacher",
        "--pipeline",
        "respond",
        "--prompts",
        prompts,
        "--teacher",
        "/app/models/t-Q8_0.gguf",
        "--out",
        out_dir,
        *extra,
    ]


class TestGenerateTeacherCmd:
    def test_missing_prompts_exits_2(self, tmp_path, capsys):
        missing = str(tmp_path / "absent.jsonl")
        assert run_cli(teacher_argv(missing, str(tmp_path / "out"))) == 2
        out = capsys.readouterr().err
        assert f"[ERROR] Prompt set: Prompt path not found: {missing}" in out

    def test_bad_temperature_exits_1(self, tmp_path, capsys):
        prompts = write_prompts(tmp_path / "p.jsonl", 1)
        assert (
            run_cli(
                teacher_argv(prompts, str(tmp_path / "out"), "--temperature", "3.0")
            )
            == 1
        )
        out = capsys.readouterr().err
        assert "[ERROR] temperature out of range [0, 2]: 3.0" in out

    def test_teacher_outside_models_roots_exits_1(self, tmp_path, capsys):
        prompts = write_prompts(tmp_path / "p.jsonl", 1)
        assert (
            run_cli(
                [
                    "generate-teacher",
                    "--pipeline",
                    "respond",
                    "--prompts",
                    prompts,
                    "--teacher",
                    "relative/t.gguf",
                    "--out",
                    str(tmp_path / "out"),
                ]
            )
            == 1
        )
        out = capsys.readouterr().err
        assert "[ERROR] Model must live under /app/models/" in out

    def test_teacher_missing_on_host_exits_1(self, tmp_path, capsys):
        prompts = write_prompts(tmp_path / "p.jsonl", 1)
        assert run_cli(teacher_argv(prompts, str(tmp_path / "out"))) == 1
        out = capsys.readouterr().err
        assert (
            "[ERROR] Teacher model not found on host: /mnt/ai-data/models/t-Q8_0.gguf"
        ) in out

    def test_training_lock_present_exits_1(self, tmp_path, capsys):
        # The autouse train_lock_path fixture points the D9 lock here
        (tmp_path / ".train.lock").touch()
        prompts = write_prompts(tmp_path / "p.jsonl", 1)
        assert run_cli(teacher_argv(prompts, str(tmp_path / "out"), "--no-swap")) == 1
        out = capsys.readouterr().err
        assert "[ERROR] Training lock present:" in out

    def test_no_swap_respond_success_resumes_completed(
        self, tmp_path, monkeypatch, capsys
    ):
        budgets, calls = patch_transport(monkeypatch)
        prompts = write_prompts(tmp_path / "p.jsonl", 3)
        out_dir = str(tmp_path / "out")
        # One completed record from a previous (killed) run: q0 must be skipped
        os.makedirs(out_dir)
        done = {
            "key": teacher.prompt_key(0, [{"role": "user", "content": "q0"}]),
            "messages": [
                {"role": "user", "content": "q0"},
                {"role": "assistant", "content": "old"},
            ],
        }
        with open(os.path.join(out_dir, "data.jsonl"), "w") as f:
            f.write(json.dumps(done) + "\n")

        assert (
            run_cli(
                teacher_argv(
                    prompts,
                    out_dir,
                    "--no-swap",
                    "--concurrency",
                    "1",
                    "--temperature",
                    "0.5",
                    "--max-tokens",
                    "64",
                    "--timeout",
                    "7",
                )
            )
            == 0
        )
        out = capsys.readouterr().err
        assert "[OK] Preflight: training is idle" in out
        assert "[INFO] Pipeline: respond, 3 prompts, concurrency 1" in out
        assert f"[INFO] Resume: 1 completed records found in {out_dir}" in out
        assert "[RUN] Using already-served model on vllm-gpu" in out
        assert "[OK] vllm-gpu healthy after 1.5s" in out
        assert "[INFO] Progress: 3/3" in out
        assert "Progress: 1/3" not in out  # gated to every 25th or the last
        assert f"[OK] Dataset complete: 3 examples in {out_dir}" in out
        assert "Restoring previous serving state" not in out
        assert budgets == [600]  # vllm-gpu default start budget

        asked = [messages[-1]["content"] for messages, _, _ in calls]
        assert sorted(asked) == ["q1", "q2"]
        for _, sampling, timeout in calls:
            assert sampling == {"temperature": 0.5, "top_p": 0.9, "max_tokens": 64}
            assert timeout == 7
        with open(os.path.join(out_dir, "data.jsonl")) as f:
            records = [json.loads(line) for line in f]
        assert len(records) == 3
        assert records[1]["messages"][-1] == {"role": "assistant", "content": "ans:q1"}
        with open(os.path.join(out_dir, "manifest.json")) as f:
            manifest = json.load(f)
        assert manifest["examples"] == 3
        assert manifest["pipeline"] == "respond"
        assert manifest["teacher"] == {
            "model": "/app/models/t-Q8_0.gguf",
            "quant_type": "Q8_0",
        }
        with open(prompts, "rb") as f:
            assert manifest["prompt_set_sha256"] == hashlib.sha256(f.read()).hexdigest()

    def test_synthesize_pipeline_evolves_then_answers(
        self, tmp_path, monkeypatch, capsys
    ):
        def synth_chat(service, messages, sampling, timeout=None):
            if "Rewrite the following instruction" in messages[-1]["content"]:
                return {"content": "evolved q0"}
            return {"content": "final answer"}

        patch_transport(monkeypatch, chat=synth_chat)
        prompts = write_prompts(tmp_path / "p.jsonl", 1)
        out_dir = str(tmp_path / "out")
        assert (
            run_cli(
                [
                    "generate-teacher",
                    "--pipeline",
                    "synthesize",
                    "--prompts",
                    prompts,
                    "--teacher",
                    "/app/models/t-Q8_0.gguf",
                    "--out",
                    out_dir,
                    "--no-swap",
                    "--concurrency",
                    "1",
                ]
            )
            == 0
        )
        out = capsys.readouterr().err
        assert "[INFO] Pipeline: synthesize, 1 prompts, concurrency 1" in out
        assert f"[OK] Dataset complete: 1 examples in {out_dir}" in out
        with open(os.path.join(out_dir, "data.jsonl")) as f:
            record = json.loads(f.readline())
        assert record["messages"] == [
            {"role": "user", "content": "evolved q0"},
            {"role": "assistant", "content": "final answer"},
        ]

    def swap_argv(self, prompts: str, out_dir: str, *extra: str) -> list[str]:
        return [
            "generate-teacher",
            "--pipeline",
            "respond",
            "--prompts",
            prompts,
            "--teacher",
            "/mnt/ai-data/models/tiny-Q4_K_M.gguf",
            "--out",
            out_dir,
            *extra,
        ]

    def patch_swap(self, monkeypatch, restore_error: str | None = None):
        """Make the host-side teacher file appear present and record
        swap_model calls; restore_error fails the container_path=None call."""
        host = "/mnt/ai-data/models/tiny-Q4_K_M.gguf"
        real_exists = os.path.exists
        monkeypatch.setattr(
            os.path, "exists", lambda p: True if p == host else real_exists(p)
        )
        swaps: list[tuple[str, str | None]] = []

        def fake_swap(service, container_path, env=None):
            if container_path is None and restore_error:
                raise serving.ServingError(restore_error)
            swaps.append((service.name, container_path))

        monkeypatch.setattr("training.serving.swap_model", fake_swap)
        return swaps

    def test_swap_success_restores_previous_state(self, tmp_path, monkeypatch, capsys):
        swaps = self.patch_swap(monkeypatch)
        budgets, _ = patch_transport(monkeypatch)
        prompts = write_prompts(tmp_path / "p.jsonl", 1)
        out_dir = str(tmp_path / "out")
        assert run_cli(self.swap_argv(prompts, out_dir, "--start-budget", "5")) == 0
        out = capsys.readouterr().err
        assert "[INFO] Pipeline: respond, 1 prompts, concurrency 16" in out
        assert "[RUN] Swapping vllm-gpu to /app/models/tiny-Q4_K_M.gguf" in out
        assert "[RUN] Restoring previous serving state" in out
        assert "[OK] Previous serving state restored" in out
        assert swaps == [
            ("vllm-gpu", "/app/models/tiny-Q4_K_M.gguf"),
            ("vllm-gpu", None),
        ]
        assert budgets == [5]
        assert os.path.isfile(os.path.join(out_dir, "manifest.json"))

    def test_keep_serving_skips_restore(self, tmp_path, monkeypatch, capsys):
        swaps = self.patch_swap(monkeypatch)
        patch_transport(monkeypatch)
        prompts = write_prompts(tmp_path / "p.jsonl", 1)
        assert (
            run_cli(self.swap_argv(prompts, str(tmp_path / "out"), "--keep-serving"))
            == 0
        )
        out = capsys.readouterr().err
        assert "Restoring previous serving state" not in out
        assert swaps == [("vllm-gpu", "/app/models/tiny-Q4_K_M.gguf")]

    def test_restore_failure_exits_1_with_compose_hint(
        self, tmp_path, monkeypatch, capsys
    ):
        self.patch_swap(monkeypatch, restore_error="compose exited 1")
        patch_transport(monkeypatch)
        prompts = write_prompts(tmp_path / "p.jsonl", 1)
        out_dir = str(tmp_path / "out")
        assert run_cli(self.swap_argv(prompts, out_dir)) == 1
        out = capsys.readouterr().err
        # Generation itself completed; only the restore failed
        assert f"[OK] Dataset complete: 1 examples in {out_dir}" in out
        assert "[ERROR] Restore failed: compose exited 1" in out
        # The hint is copy-pasteable, so it carries -f: the inference compose file
        # has a domain suffix and a bare `docker compose` would not find it.
        assert (
            "Solution: docker compose -f docker-compose.inference.yaml "
            "up -d --force-recreate vllm-gpu"
        ) in out

    def test_failed_swap_still_restores(self, tmp_path, monkeypatch, capsys):
        # KB-10: a force-recreate that fails mid-swap can already have
        # disturbed serving state; restore is owed from the attempt, not only
        # after a completed swap
        host = "/mnt/ai-data/models/tiny-Q4_K_M.gguf"
        real_exists = os.path.exists
        monkeypatch.setattr(
            os.path, "exists", lambda p: True if p == host else real_exists(p)
        )
        swaps = []

        def failing_swap(service, container_path, env=None):
            swaps.append((service.name, container_path))
            if container_path is not None:
                raise serving.ServingError("force-recreate exited 1")

        monkeypatch.setattr("training.serving.swap_model", failing_swap)
        patch_transport(monkeypatch)
        prompts = write_prompts(tmp_path / "p.jsonl", 1)
        assert run_cli(self.swap_argv(prompts, str(tmp_path / "out"))) == 1
        out = capsys.readouterr().err
        assert "[ERROR] Generation aborted: force-recreate exited 1" in out
        assert "[RUN] Restoring previous serving state" in out
        assert swaps == [
            ("vllm-gpu", "/app/models/tiny-Q4_K_M.gguf"),
            ("vllm-gpu", None),
        ]

    def test_changed_prompt_refusal_styled_and_restores_despite_keep_serving(
        self, tmp_path, monkeypatch, capsys
    ):
        # KB-11: the harness's ValueError refusal must land in the abort
        # handler (styled error, exit 1, restore even under --keep-serving),
        # not escape as a raw traceback
        swaps = self.patch_swap(monkeypatch)
        patch_transport(monkeypatch)

        def refusing_generate(*args, **kwargs):
            raise ValueError("prompt set changed: 2 completed records are orphans")

        monkeypatch.setattr("training.data.teacher.generate", refusing_generate)
        prompts = write_prompts(tmp_path / "p.jsonl", 1)
        assert (
            run_cli(self.swap_argv(prompts, str(tmp_path / "out"), "--keep-serving"))
            == 1
        )
        out = capsys.readouterr().err
        assert "[ERROR] Generation aborted: prompt set changed" in out
        # --keep-serving keeps the swap only on success; a failed run restores
        assert swaps[-1] == ("vllm-gpu", None)

    def test_keyboard_interrupt_reports_class_name(self, tmp_path, monkeypatch, capsys):
        # KB-12: a bare KeyboardInterrupt has an empty str(); the abort line
        # must fall back to the class name instead of printing nothing
        swaps = self.patch_swap(monkeypatch)
        patch_transport(monkeypatch)

        def interrupted_generate(*args, **kwargs):
            raise KeyboardInterrupt

        monkeypatch.setattr("training.data.teacher.generate", interrupted_generate)
        prompts = write_prompts(tmp_path / "p.jsonl", 1)
        assert run_cli(self.swap_argv(prompts, str(tmp_path / "out"))) == 1
        out = capsys.readouterr().err
        assert "[ERROR] Generation aborted: KeyboardInterrupt" in out
        assert swaps[-1] == ("vllm-gpu", None)

    def test_generation_failure_aborts_resumable_and_exits_1(
        self, tmp_path, monkeypatch, capsys
    ):
        def broken_chat(service, messages, sampling, timeout=None):
            # Non-transient, so with_retry raises without sleeping
            raise serving.ServingError("HTTP 400 from vllm-gpu: bad request")

        patch_transport(monkeypatch, chat=broken_chat)
        prompts = write_prompts(tmp_path / "p.jsonl", 2)
        out_dir = str(tmp_path / "out")
        assert (
            run_cli(teacher_argv(prompts, out_dir, "--no-swap", "--concurrency", "1"))
            == 1
        )
        out = capsys.readouterr().err
        assert "[ERROR] Generation aborted: 2 of 2 prompts failed" in out
        # No manifest: the partial run stays resumable
        assert not os.path.exists(os.path.join(out_dir, "manifest.json"))
