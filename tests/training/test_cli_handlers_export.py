"""CLI handler contracts for cmd_export, cmd_eval_holdout, and cmd_eval_lmeval.

Each test parses real argv through build_parser() and invokes the bound
handler, asserting the exit-code contract (0 success, 1 failure), the styled
[OK]/[WARN]/[ERROR]/[INFO] lines, --json stdout purity, and the concrete file
side effects (the hf/ directory, manifest, record JSONs). For export, the
copy path runs with nothing faked; the merge path fakes stage_model, the one
torch-bound seam, and everything around it (name derivation, checkpoint and
adapter resolution, the F6 base-identity preflight, promotion, hashing,
manifest build and write) runs for real against tmp-dir checkpoints. The eval pipelines (GPU-only) are faked at their
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
from training.export.errors import ExportError
from training.export.merge import StagingInfo

ARCH = "LlamaForCausalLM"


def run_cli(argv: list[str]) -> int:
    args = cli.build_parser().parse_args(argv)
    return args.func(args)


# -- cmd_export -----------------------------------------------------------------

np = pytest.importorskip("numpy")
safetensors_numpy = pytest.importorskip("safetensors.numpy")


def write_json(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f)


def write_weights(model_dir: str) -> bytes:
    """A real one-tensor safetensors file; returns its bytes for hashing."""
    path = os.path.join(model_dir, "model.safetensors")
    safetensors_numpy.save_file(
        {"w": np.arange(4, dtype=np.float32)}, path, metadata={"format": "pt"}
    )
    with open(path, "rb") as f:
        return f.read()


def full_model_checkpoint(tmp_path) -> tuple[str, bytes]:
    """TRL-layout checkpoint dir holding a plain HF model (no adapter)."""
    ckpt = tmp_path / "myrun" / "checkpoints" / "checkpoint-20"
    write_json(str(ckpt / "config.json"), {"architectures": [ARCH]})
    write_json(str(ckpt / "generation_config.json"), {"do_sample": False})
    (ckpt / "tokenizer.json").write_text("{}")
    return str(ckpt), write_weights(str(ckpt))


def adapter_checkpoint(tmp_path, meta: dict | None = None) -> tuple[str, str]:
    """(checkpoint, base_dir): adapter-only checkpoint over a local base."""
    base = tmp_path / "base-model"
    write_json(str(base / "config.json"), {"architectures": [ARCH]})
    ckpt = tmp_path / "adapter-ckpt"
    write_json(
        str(ckpt / "adapter_config.json"), {"base_model_name_or_path": str(base)}
    )
    (ckpt / "adapter_model.safetensors").write_bytes(b"lora")
    if meta is not None:
        write_json(str(ckpt / "meta.json"), meta)
    return str(ckpt), str(base)


def patch_stage(
    monkeypatch, adapter_path=None, base_model=None, tokenizer_source="tok-src"
) -> dict:
    """Fake the torch-bound seam only; returns the recorded call.

    The fake writes a real config.json and model.safetensors into the
    staging dir, so promotion, the single-file assertion, hashing and the
    manifest all execute for real.
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
        os.makedirs(staging_dir, exist_ok=True)
        write_json(os.path.join(staging_dir, "config.json"), {"architectures": [ARCH]})
        calls["weights"] = write_weights(staging_dir)
        return StagingInfo(
            model_dir=checkpoint,
            staging_dir=staging_dir,
            adapter_path=adapter_path,
            base_model=base_model,
            merge_method="peft.merge_and_unload" if adapter_path else None,
            tokenizer_source=tokenizer_source,
        )

    monkeypatch.setattr("training.export.stage_model", fake_stage)
    return calls


def read_manifest(out_root: str, name: str) -> dict:
    with open(os.path.join(out_root, name, "manifest.json")) as f:
        return json.load(f)


class TestCmdExport:
    def test_full_model_copy_happy_path(self, tmp_path, capsys):
        """No seam faked: the copy path, promotion, hash and manifest are real."""
        ckpt, weights = full_model_checkpoint(tmp_path)
        out_root = str(tmp_path / "out")
        # No --name: the TRL layout must derive "<run>-checkpoint-N" for real.
        assert run_cli(["export", "--checkpoint", ckpt, "--out-root", out_root]) == 0
        out = capsys.readouterr().err

        name = "myrun-checkpoint-20"
        hf = os.path.join(out_root, name, "hf")
        assert f"[RUN] Copying model: {ckpt}" in out
        assert f"[OK] HF export written: {hf} ({len(weights):,} bytes)" in out
        manifest_file = os.path.join(out_root, name, "manifest.json")
        assert f"[OK] Manifest written: {manifest_file}" in out
        sha = hashlib.sha256(weights).hexdigest()
        assert f"model_hash: {sha}" in out
        assert "[WARN]" not in out and "Adapter merged" not in out

        assert sorted(os.listdir(hf)) == [
            "config.json",
            "generation_config.json",
            "model.safetensors",
            "tokenizer.json",
        ]
        assert not os.path.exists(hf + ".partial")
        manifest = read_manifest(out_root, name)
        assert manifest["artifact"] == "hf_export"
        assert manifest["mode"] == "copy"
        assert manifest["model_hash"] == sha
        assert manifest["hf_dir"] == hf
        assert manifest["adapter"] is None
        assert manifest["tokenizer_source"] == ckpt
        assert manifest["base"]["model"] == ckpt
        assert manifest["base"]["revision"] is None

    def test_adapter_base_verified(self, tmp_path, capsys, monkeypatch):
        ckpt, base = adapter_checkpoint(tmp_path)
        calls = patch_stage(monkeypatch, adapter_path=ckpt, base_model=base)
        out_root = str(tmp_path / "out")
        assert (
            run_cli(
                [
                    "export",
                    "--checkpoint",
                    ckpt,
                    "--name",
                    "tuned",
                    "--out-root",
                    out_root,
                ]
            )
            == 0
        )
        out = capsys.readouterr().err
        # Local-path base: no hub revision, so the identity reads "local".
        assert f"[OK] Base model verified: {base} (revision local)" in out
        assert f"[RUN] Merging model: {ckpt}" in out
        assert f"[OK] Adapter merged: {ckpt} into {base}" in out
        assert "[WARN]" not in out

        hf = os.path.join(out_root, "tuned", "hf")
        assert calls["stage"]["staging_dir"] == hf + ".partial"
        assert os.path.isdir(hf) and not os.path.exists(hf + ".partial")
        manifest = read_manifest(out_root, "tuned")
        assert manifest["mode"] == "merge"
        assert manifest["base"]["model"] == base
        assert manifest["adapter"]["path"] == ckpt
        assert manifest["adapter"]["merge_method"] == "peft.merge_and_unload"
        assert manifest["adapter"]["sha256"] == hashlib.sha256(b"lora").hexdigest()
        assert manifest["model_hash"] == hashlib.sha256(calls["weights"]).hexdigest()
        # Fields the checkpoint never recorded are noted, never fatal.
        assert manifest["base_drift"] == [
            "base_revision: not recorded at train time",
            "base_config_sha256: not recorded at train time",
        ]

    def test_allow_base_drift_warns_and_records_drift(
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
        calls = patch_stage(
            monkeypatch, adapter_path=ckpt, base_model=base, tokenizer_source=None
        )
        out_root = str(tmp_path / "out")
        assert (
            run_cli(
                [
                    "export",
                    "--checkpoint",
                    ckpt,
                    "--name",
                    "drifted",
                    "--allow-base-drift",
                    "--out-root",
                    out_root,
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
        # The recorded revision pins the merge and lands in the manifest.
        assert calls["stage"]["base_revision"] == "f" * 40
        manifest = read_manifest(out_root, "drifted")
        assert note in manifest["base_drift"]
        assert manifest["base"]["revision"] == "f" * 40
        assert manifest["tokenizer_source"] is None

    def test_base_drift_refused_exits_1(self, tmp_path, capsys, monkeypatch):
        meta = {"adapter_only": True, "base_config_sha256": "0" * 64}
        ckpt, _base = adapter_checkpoint(tmp_path, meta=meta)

        def must_not_stage(*a, **kw):
            raise AssertionError("stage_model reached despite drift refusal")

        monkeypatch.setattr("training.export.stage_model", must_not_stage)
        out_root = str(tmp_path / "out")
        assert (
            run_cli(
                [
                    "export",
                    "--checkpoint",
                    ckpt,
                    "--name",
                    "drifted",
                    "--out-root",
                    out_root,
                ]
            )
            == 1
        )
        out = capsys.readouterr().err
        assert "[ERROR] Export failed: Base model drifted since training:" in out
        assert "--allow-base-drift" in out
        assert not os.path.exists(os.path.join(out_root, "drifted", "hf"))

    def test_existing_export_refused_without_force(self, tmp_path, capsys):
        ckpt, weights = full_model_checkpoint(tmp_path)
        out_root = str(tmp_path / "out")
        argv = [
            "export",
            "--checkpoint",
            ckpt,
            "--name",
            "twice",
            "--out-root",
            out_root,
        ]
        assert run_cli(argv) == 0
        first = read_manifest(out_root, "twice")
        capsys.readouterr()
        assert run_cli(argv) == 1
        out = capsys.readouterr().err
        hf = os.path.join(out_root, "twice", "hf")
        assert f"[ERROR] Export exists: {hf}" in out
        assert "Solution: pass --force" in out
        assert read_manifest(out_root, "twice") == first
        assert not os.path.exists(hf + ".partial")

    def test_force_replaces_the_export(self, tmp_path, capsys):
        ckpt, _weights = full_model_checkpoint(tmp_path)
        out_root = str(tmp_path / "out")
        argv = [
            "export",
            "--checkpoint",
            ckpt,
            "--name",
            "again",
            "--out-root",
            out_root,
        ]
        assert run_cli(argv) == 0
        first = read_manifest(out_root, "again")
        # Change the source weights: the new export must carry the new hash.
        safetensors_numpy.save_file(
            {"w": np.zeros(4, dtype=np.float32)},
            os.path.join(ckpt, "model.safetensors"),
        )
        assert run_cli([*argv, "--force"]) == 0
        second = read_manifest(out_root, "again")
        assert second["model_hash"] != first["model_hash"]
        hf = os.path.join(out_root, "again", "hf")
        assert sorted(os.listdir(hf)) == [
            "config.json",
            "generation_config.json",
            "model.safetensors",
            "tokenizer.json",
        ]

    def test_failed_stage_leaves_no_partial_and_no_export(
        self, tmp_path, capsys, monkeypatch
    ):
        ckpt, _weights = full_model_checkpoint(tmp_path)

        def fail_stage(checkpoint, staging_dir, **kw):
            os.makedirs(staging_dir, exist_ok=True)
            write_json(os.path.join(staging_dir, "config.json"), {})
            raise ExportError("shard ended early")

        monkeypatch.setattr("training.export.stage_model", fail_stage)
        out_root = str(tmp_path / "out")
        assert (
            run_cli(
                [
                    "export",
                    "--checkpoint",
                    ckpt,
                    "--name",
                    "broken",
                    "--out-root",
                    out_root,
                ]
            )
            == 1
        )
        out = capsys.readouterr().err
        assert "[ERROR] Export failed: shard ended early" in out
        assert not os.path.exists(os.path.join(out_root, "broken", "hf"))
        assert not os.path.exists(os.path.join(out_root, "broken", "hf.partial"))
        assert not os.path.exists(os.path.join(out_root, "broken", "manifest.json"))

    def test_sharded_source_is_consolidated(self, tmp_path, capsys):
        model = tmp_path / "model"
        model.mkdir()
        write_json(str(model / "config.json"), {"architectures": [ARCH]})
        safetensors_numpy.save_file(
            {"a": np.ones(2, dtype=np.float32)},
            str(model / "model-00001-of-00002.safetensors"),
        )
        safetensors_numpy.save_file(
            {"b": np.ones(3, dtype=np.float32)},
            str(model / "model-00002-of-00002.safetensors"),
        )
        write_json(
            str(model / "model.safetensors.index.json"),
            {
                "weight_map": {
                    "a": "model-00001-of-00002.safetensors",
                    "b": "model-00002-of-00002.safetensors",
                }
            },
        )
        out_root = str(tmp_path / "out")
        assert (
            run_cli(
                [
                    "export",
                    "--checkpoint",
                    str(model),
                    "--name",
                    "joined",
                    "--out-root",
                    out_root,
                ]
            )
            == 0
        )
        hf = os.path.join(out_root, "joined", "hf")
        assert sorted(os.listdir(hf)) == ["config.json", "model.safetensors"]
        loaded = safetensors_numpy.load_file(os.path.join(hf, "model.safetensors"))
        assert sorted(loaded) == ["a", "b"]

    @pytest.mark.parametrize(
        "flag", ["--quant", "--outtype", "--llama-cpp", "--format"]
    )
    def test_gguf_era_flags_are_usage_errors(self, tmp_path, flag):
        """The GGUF flags left with the converter; the two staging flags are
        covered by the retired-names gate, which refuses their spelling
        anywhere in the tree, this file included."""
        ckpt, _weights = full_model_checkpoint(tmp_path)
        with pytest.raises(SystemExit) as exc:
            run_cli(["export", "--checkpoint", ckpt, flag, "x"])
        assert exc.value.code == 2

    def test_json_stdout_is_pure_manifest(self, tmp_path, capsys):
        ckpt, weights = full_model_checkpoint(tmp_path)
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
                    "--out-root",
                    out_root,
                ]
            )
            == 0
        )
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert captured.err == ""
        assert data == read_manifest(out_root, "quiet")
        assert data["model_hash"] == hashlib.sha256(weights).hexdigest()


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
