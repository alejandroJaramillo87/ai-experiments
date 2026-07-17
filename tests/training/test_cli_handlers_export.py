"""CLI handler contracts for cmd_export, cmd_eval_holdout, and cmd_eval_lmeval.

Each test parses real argv through build_parser() and invokes the bound
handler, asserting the exit-code contract (0 success, 1 failure), the styled
[OK]/[WARN]/[ERROR]/[INFO] lines, --json stdout purity, and the concrete file
side effects (GGUF, manifest, record JSONs). For export, only the torch-bound
seams are faked (stage_model, convert_to_gguf, quantize); name derivation,
converter lookup, checkpoint/adapter resolution, the F6 base-identity
preflight, architecture checking, and manifest build/write all run for real
against tmp-dir checkpoints. The eval pipelines (GPU-only) are faked at their
home modules training.eval.holdout / training.eval.lmeval.
"""

from __future__ import annotations

import hashlib
import json
import os

import pytest
from support import REPO_ROOT

from training import cli
from training.eval.lmeval import LmEvalError
from training.export import read_pin
from training.export.merge import StagingInfo

PIN = read_pin()

ARCH = "LlamaForCausalLM"

GGUF_PAYLOAD = b"g" * 4096  # 4,096 bytes: proves the comma-grouped size line


def run_cli(argv: list[str]) -> int:
    args = cli.build_parser().parse_args(argv)
    return args.func(args)


# -- cmd_export -----------------------------------------------------------------


def make_llama_cpp(tmp_path) -> str:
    """A --llama-cpp dir whose converter registers exactly one architecture."""
    d = tmp_path / "llama.cpp"
    d.mkdir()
    (d / "convert_hf_to_gguf.py").write_text(
        f'@ModelBase.register("{ARCH}")\nclass Llama(TextModel):\n    pass\n'
    )
    return str(d)


def write_json(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f)


def full_model_checkpoint(tmp_path) -> str:
    """D6-layout checkpoint dir holding a plain HF model (no adapter)."""
    ckpt = tmp_path / "myrun" / "checkpoints" / "step_20"
    write_json(str(ckpt / "config.json"), {"architectures": [ARCH]})
    return str(ckpt)


def adapter_checkpoint(tmp_path, meta: dict | None = None) -> tuple[str, str]:
    """(checkpoint, base_dir): adapter-only checkpoint over a local base."""
    base = tmp_path / "base-model"
    write_json(str(base / "config.json"), {"architectures": [ARCH]})
    ckpt = tmp_path / "adapter-ckpt"
    write_json(
        str(ckpt / "adapter_config.json"), {"base_model_name_or_path": str(base)}
    )
    if meta is not None:
        write_json(str(ckpt / "meta.json"), meta)
    return str(ckpt), str(base)


def patch_pipeline(
    monkeypatch, adapter_path=None, base_model=None, tokenizer_source="tok-src"
) -> dict:
    """Fake the three torch/toolchain seams; returns the recorded calls.

    The staging fake writes the config.json check_architecture reads, and the
    quantize fake writes the GGUF bytes build_manifest hashes, so everything
    downstream of the seams executes for real.
    """
    calls: dict = {}

    def fake_stage(
        checkpoint, staging_dir, tokenizer_dir=None, meta=None, base_revision=None
    ):
        calls["stage"] = {
            "checkpoint": checkpoint,
            "staging_dir": staging_dir,
            "tokenizer_dir": tokenizer_dir,
            "meta": meta,
            "base_revision": base_revision,
        }
        write_json(os.path.join(staging_dir, "config.json"), {"architectures": [ARCH]})
        return StagingInfo(
            model_dir=checkpoint,
            staging_dir=staging_dir,
            adapter_path=adapter_path,
            base_model=base_model,
            merge_method="peft.merge_and_unload" if adapter_path else None,
            tokenizer_source=tokenizer_source,
        )

    def fake_convert(staging_dir, out_path, outtype="bf16", llama_cpp_dir=None):
        calls["convert"] = {
            "staging_dir": staging_dir,
            "out_path": out_path,
            "outtype": outtype,
        }

    def fake_quantize(in_path, out_path, quant_type, llama_cpp_dir=None):
        calls["quantize"] = {
            "in_path": in_path,
            "out_path": out_path,
            "quant_type": quant_type,
        }
        with open(out_path, "wb") as f:
            f.write(GGUF_PAYLOAD)

    monkeypatch.setattr("training.export.stage_model", fake_stage)
    monkeypatch.setattr("training.export.convert.convert_to_gguf", fake_convert)
    monkeypatch.setattr("training.export.convert.quantize", fake_quantize)
    return calls


class TestCmdExport:
    def test_missing_converter_exits_1(self, tmp_path, capsys):
        empty = tmp_path / "no-llama-cpp"
        empty.mkdir()
        assert (
            run_cli(
                [
                    "export",
                    "--checkpoint",
                    str(tmp_path / "ckpt"),
                    "--llama-cpp",
                    str(empty),
                ]
            )
            == 1
        )
        out = capsys.readouterr().err
        assert (
            f"[ERROR] Pinned converter not found: {empty / 'convert_hf_to_gguf.py'}"
        ) in out
        assert "Solution: run inside the export image" in out

    def test_full_model_happy_path(self, tmp_path, capsys, monkeypatch):
        ckpt = full_model_checkpoint(tmp_path)
        calls = patch_pipeline(monkeypatch)
        out_root = str(tmp_path / "out")
        staging_root = str(tmp_path / "staging")
        # No --name: the D6 layout must derive "<run>-step_N" for real.
        assert (
            run_cli(
                [
                    "export",
                    "--checkpoint",
                    ckpt,
                    "--quant",
                    "q8_0",
                    "--llama-cpp",
                    make_llama_cpp(tmp_path),
                    "--out-root",
                    out_root,
                    "--staging-root",
                    staging_root,
                    "--tokenizer",
                    str(tmp_path / "tok"),
                ]
            )
            == 0
        )
        out = capsys.readouterr().err

        name = "myrun-step_20"
        gguf = os.path.join(out_root, name, f"{name}-Q8_0.gguf")
        assert f"[RUN] Staging model: {ckpt}" in out
        assert f"[OK] Architecture: {ARCH} (supported by llama.cpp {PIN[:12]})" in out
        assert "[RUN] Converting to GGUF (bf16)" in out
        assert "[RUN] Quantizing: Q8_0" in out
        assert f"[OK] GGUF written: {gguf} (4,096 bytes)" in out

        sha = hashlib.sha256(GGUF_PAYLOAD).hexdigest()
        assert f"SHA-256: {sha}" in out
        with open(gguf, "rb") as f:
            assert f.read() == GGUF_PAYLOAD

        manifest_file = os.path.join(out_root, name, f"{name}-Q8_0.manifest.json")
        assert f"[OK] Manifest written: {manifest_file}" in out
        with open(manifest_file) as f:
            manifest = json.load(f)
        assert manifest["architecture"] == ARCH
        assert manifest["quant_type"] == "Q8_0"
        assert manifest["llama_cpp_commit"] == PIN
        assert manifest["adapter"] is None
        assert manifest["gguf"] == {"path": gguf, "sha256": sha, "size_bytes": 4096}

        # Intermediate GGUF lives in the staging dir, which is removed by
        # default; --tokenizer reaches staging.
        staging = calls["stage"]["staging_dir"]
        assert calls["convert"]["out_path"] == os.path.join(
            staging, f"{name}-bf16.gguf"
        )
        assert calls["quantize"]["in_path"] == calls["convert"]["out_path"]
        assert calls["quantize"]["quant_type"] == "Q8_0"
        assert calls["stage"]["tokenizer_dir"] == str(tmp_path / "tok")
        assert calls["stage"]["base_revision"] is None
        assert not os.path.exists(staging)
        # No adapter, tokenizer found: none of the optional arms fire.
        assert "Adapter merged" not in out
        assert "[WARN]" not in out

    def test_adapter_base_verified(self, tmp_path, capsys, monkeypatch):
        ckpt, base = adapter_checkpoint(tmp_path)
        patch_pipeline(monkeypatch, adapter_path=ckpt, base_model=base)
        assert (
            run_cli(
                [
                    "export",
                    "--checkpoint",
                    ckpt,
                    "--name",
                    "tuned",
                    "--llama-cpp",
                    make_llama_cpp(tmp_path),
                    "--out-root",
                    str(tmp_path / "out"),
                    "--staging-root",
                    str(tmp_path / "staging"),
                ]
            )
            == 0
        )
        out = capsys.readouterr().err
        # Local-path base: no hub revision, so the identity reads "local".
        assert f"[OK] Base model verified: {base} (revision local)" in out
        assert f"[OK] Adapter merged: {ckpt} into {base}" in out
        assert "[WARN]" not in out

        manifest_file = os.path.join(
            str(tmp_path / "out"), "tuned", "tuned-Q8_0.manifest.json"
        )
        with open(manifest_file) as f:
            manifest = json.load(f)
        assert manifest["adapter"]["base_model"] == base
        assert manifest["adapter"]["merge_method"] == "peft.merge_and_unload"
        # Fields the checkpoint never recorded are noted, never fatal.
        assert manifest["base_drift"] == [
            "base_revision: not recorded at train time",
            "base_config_sha256: not recorded at train time",
        ]

    def test_allow_base_drift_warns_and_keeps_staging(
        self, tmp_path, capsys, monkeypatch
    ):
        meta = {
            "adapter_only": True,
            "base_revision": "f" * 40,
            "base_config_sha256": "0" * 64,
        }
        ckpt, base = adapter_checkpoint(tmp_path, meta=meta)
        with open(os.path.join(base, "config.json"), "rb") as f:
            real_sha = hashlib.sha256(f.read()).hexdigest()
        calls = patch_pipeline(
            monkeypatch, adapter_path=ckpt, base_model=base, tokenizer_source=None
        )
        assert (
            run_cli(
                [
                    "export",
                    "--checkpoint",
                    ckpt,
                    "--name",
                    "drifted",
                    "--allow-base-drift",
                    "--keep-staging",
                    "--llama-cpp",
                    make_llama_cpp(tmp_path),
                    "--out-root",
                    str(tmp_path / "out"),
                    "--staging-root",
                    str(tmp_path / "staging"),
                ]
            )
            == 0
        )
        out = capsys.readouterr().err

        note = (
            f"base_config_sha256: trained against {'0' * 64}, "
            f"export resolved {real_sha}"
        )
        assert "[WARN] Base drift allowed (--allow-base-drift):" in out
        assert f"  {note}" in out
        assert "Base model verified" not in out
        assert "[WARN] No tokenizer files found in the checkpoint" in out
        assert "Solution: pass --tokenizer" in out

        staging = calls["stage"]["staging_dir"]
        assert f"[INFO] Staging kept: {staging}" in out
        assert os.path.isdir(staging)
        # The recorded revision pins the merge and lands in the manifest.
        assert calls["stage"]["base_revision"] == "f" * 40
        manifest_file = os.path.join(
            str(tmp_path / "out"), "drifted", "drifted-Q8_0.manifest.json"
        )
        with open(manifest_file) as f:
            manifest = json.load(f)
        assert note in manifest["base_drift"]
        assert manifest["adapter"]["base_revision"] == "f" * 40
        assert manifest["tokenizer_source"] is None

    def test_base_drift_refused_exits_1(self, tmp_path, capsys, monkeypatch):
        meta = {"adapter_only": True, "base_config_sha256": "0" * 64}
        ckpt, _base = adapter_checkpoint(tmp_path, meta=meta)

        def must_not_stage(*a, **kw):
            raise AssertionError("stage_model reached despite drift refusal")

        monkeypatch.setattr("training.export.stage_model", must_not_stage)
        assert (
            run_cli(
                [
                    "export",
                    "--checkpoint",
                    ckpt,
                    "--name",
                    "drifted",
                    "--llama-cpp",
                    make_llama_cpp(tmp_path),
                    "--out-root",
                    str(tmp_path / "out"),
                    "--staging-root",
                    str(tmp_path / "staging"),
                ]
            )
            == 1
        )
        out = capsys.readouterr().err
        assert "[ERROR] Export failed: Base model drifted since training:" in out
        assert "--allow-base-drift" in out

    def test_json_stdout_is_pure_manifest(self, tmp_path, capsys, monkeypatch):
        ckpt = full_model_checkpoint(tmp_path)
        patch_pipeline(monkeypatch)
        out_root = str(tmp_path / "out")
        assert (
            run_cli(
                [
                    "export",
                    "--checkpoint",
                    ckpt,
                    "--name",
                    "quiet",
                    "--json",
                    "--llama-cpp",
                    make_llama_cpp(tmp_path),
                    "--out-root",
                    out_root,
                    "--staging-root",
                    str(tmp_path / "staging"),
                ]
            )
            == 0
        )
        data = json.loads(capsys.readouterr().out)
        manifest_file = os.path.join(out_root, "quiet", "quiet-Q8_0.manifest.json")
        with open(manifest_file) as f:
            assert data == json.load(f)
        assert data["gguf"]["sha256"] == hashlib.sha256(GGUF_PAYLOAD).hexdigest()


# -- cmd_eval-holdout -------------------------------------------------------------

HOLDOUT_RECORD = {
    "model": "/ckpt/sft",
    "dataset": "/data/sft",
    "loss": 1.25,
    "perplexity": 3.4903,
    "examples": 12,
    "supervised_tokens": 3456,
    "precision": "bfloat16",
}


class TestCmdEvalHoldout:
    def test_success_reports_loss_and_passes_flags(self, monkeypatch, capsys):
        seen = {}

        def fake_eval(model, dataset, load_4bit=False, max_examples=None):
            seen.update(
                model=model,
                dataset=dataset,
                load_4bit=load_4bit,
                max_examples=max_examples,
            )
            return dict(HOLDOUT_RECORD)

        monkeypatch.setattr("training.eval.holdout.evaluate_holdout", fake_eval)
        assert (
            run_cli(
                [
                    "eval-holdout",
                    "--model",
                    "/ckpt/sft",
                    "--dataset",
                    "/data/sft",
                    "--load-4bit",
                    "--max-examples",
                    "5",
                ]
            )
            == 0
        )
        out = capsys.readouterr().err
        assert "[RUN] Held-out loss: /ckpt/sft over /data/sft" in out
        assert (
            "[OK] Loss 1.250000, perplexity 3.4903 (12 examples, "
            "3456 supervised tokens, bfloat16)"
        ) in out
        assert "Record written" not in out
        assert seen == {
            "model": "/ckpt/sft",
            "dataset": "/data/sft",
            "load_4bit": True,
            "max_examples": 5,
        }

    def test_out_writes_record_json(self, monkeypatch, tmp_path, capsys):
        monkeypatch.setattr(
            "training.eval.holdout.evaluate_holdout",
            lambda *a, **kw: dict(HOLDOUT_RECORD),
        )
        # Nested path: the handler must create the parent directory.
        target = str(tmp_path / "records" / "holdout.json")
        assert (
            run_cli(["eval-holdout", "--model", "m", "--dataset", "d", "--out", target])
            == 0
        )
        assert f"[OK] Record written: {target}" in capsys.readouterr().err
        with open(target) as f:
            text = f.read()
        assert text.endswith("}\n")
        assert json.loads(text) == HOLDOUT_RECORD

    @pytest.mark.parametrize(
        "exc",
        [
            RuntimeError("CUDA out of memory"),
            FileNotFoundError("no prepared dataset"),
            ValueError("Dataset has no held-out split: /data/sft"),
        ],
        ids=["runtime", "missing-file", "value"],
    )
    def test_failure_exits_1(self, exc, monkeypatch, capsys):
        def failing(*a, **kw):
            raise exc

        monkeypatch.setattr("training.eval.holdout.evaluate_holdout", failing)
        assert run_cli(["eval-holdout", "--model", "m", "--dataset", "d"]) == 1
        assert f"[ERROR] Held-out evaluation failed: {exc}" in (capsys.readouterr().err)

    def test_json_stdout_is_pure_record(self, monkeypatch, capsys):
        monkeypatch.setattr(
            "training.eval.holdout.evaluate_holdout",
            lambda *a, **kw: dict(HOLDOUT_RECORD),
        )
        assert (
            run_cli(["eval-holdout", "--model", "m", "--dataset", "d", "--json"]) == 0
        )
        assert json.loads(capsys.readouterr().out) == HOLDOUT_RECORD


# -- cmd_eval-lmeval --------------------------------------------------------------

LMEVAL_RECORD = {
    "model": "/ckpt/sft",
    "lm_eval_version": "0.4.9",
    "tasks_file": "/app/training/eval/tasks.yaml",
    "limit": 50,
    "results": {
        "hellaswag": {
            "metric": "acc,none",
            "value": 0.55,
            "stderr": None,
            "version": 1,
        },
        "arc_challenge": {
            "metric": "acc_norm,none",
            "value": 0.4321,
            "stderr": 0.0145,
            "version": 1,
        },
    },
}


class TestCmdEvalLmeval:
    def test_success_reports_sorted_tasks_with_stderr_detail(self, monkeypatch, capsys):
        seen = {}

        def fake_run(model, tasks_file=None, tasks=None, limit=None):
            seen.update(model=model, tasks_file=tasks_file, tasks=tasks, limit=limit)
            return dict(LMEVAL_RECORD)

        monkeypatch.setattr("training.eval.lmeval.run_lmeval", fake_run)
        assert (
            run_cli(
                [
                    "eval-lmeval",
                    "--model",
                    "/ckpt/sft",
                    "--tasks",
                    "arc_challenge,hellaswag",
                    "--limit",
                    "50",
                ]
            )
            == 0
        )
        out = capsys.readouterr().err
        assert "[RUN] lm-evaluation-harness: /ckpt/sft" in out
        # Sorted task order; stderr detail only when the harness reports one.
        arc = out.index("[OK] arc_challenge: acc_norm,none = 0.4321 (stderr 0.0145)")
        hella = out.index("[OK] hellaswag: acc,none = 0.5500")
        assert arc < hella
        assert "hellaswag: acc,none = 0.5500 (stderr" not in out
        assert seen["tasks"] == ["arc_challenge", "hellaswag"]
        assert seen["limit"] == 50
        # Default --tasks-file is the pinned set beside the eval package.
        assert seen["tasks_file"] == os.path.join(
            REPO_ROOT, "training", "eval", "tasks.yaml"
        )

    def test_no_tasks_flag_passes_none(self, monkeypatch, capsys):
        seen = {}

        def fake_run(model, tasks_file=None, tasks=None, limit=None):
            seen.update(tasks=tasks, limit=limit)
            return dict(LMEVAL_RECORD)

        monkeypatch.setattr("training.eval.lmeval.run_lmeval", fake_run)
        assert run_cli(["eval-lmeval", "--model", "m"]) == 0
        assert seen == {"tasks": None, "limit": None}
        assert "[OK] arc_challenge" in capsys.readouterr().err

    def test_lmeval_error_exits_1_verbatim(self, monkeypatch, capsys):
        def failing(*a, **kw):
            raise LmEvalError(
                "lm-eval version mismatch: installed 0.4.8, tasks.yaml pins 0.4.9"
            )

        monkeypatch.setattr("training.eval.lmeval.run_lmeval", failing)
        assert run_cli(["eval-lmeval", "--model", "m"]) == 1
        out = capsys.readouterr().err
        # LmEvalError messages carry their own fix; no extra prefix.
        assert "[ERROR] lm-eval version mismatch: installed 0.4.8" in out
        assert "Harness run failed" not in out

    def test_runtime_failure_exits_1(self, monkeypatch, capsys):
        def failing(*a, **kw):
            raise RuntimeError("CUDA out of memory")

        monkeypatch.setattr("training.eval.lmeval.run_lmeval", failing)
        assert run_cli(["eval-lmeval", "--model", "m"]) == 1
        assert "[ERROR] Harness run failed: CUDA out of memory" in (
            capsys.readouterr().err
        )

    def test_out_writes_record_json(self, monkeypatch, tmp_path, capsys):
        monkeypatch.setattr(
            "training.eval.lmeval.run_lmeval", lambda *a, **kw: dict(LMEVAL_RECORD)
        )
        target = str(tmp_path / "records" / "lmeval.json")
        assert run_cli(["eval-lmeval", "--model", "m", "--out", target]) == 0
        assert f"[OK] Record written: {target}" in capsys.readouterr().err
        with open(target) as f:
            text = f.read()
        assert text.endswith("}\n")
        assert json.loads(text) == LMEVAL_RECORD

    def test_json_stdout_is_pure_record(self, monkeypatch, capsys):
        monkeypatch.setattr(
            "training.eval.lmeval.run_lmeval", lambda *a, **kw: dict(LMEVAL_RECORD)
        )
        assert run_cli(["eval-lmeval", "--model", "m", "--json"]) == 0
        assert json.loads(capsys.readouterr().out) == LMEVAL_RECORD
