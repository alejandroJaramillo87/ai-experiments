"""interp probe CLI handlers (interp/commands/probe_{capture,train,eval,
export,monitor}.py) driven through build_parser() with real tmp-dir pipeline
execution. The only fakes sit at the torch/service seams: the bf16 nnsight
capture (capture_activations), the subject/judge chat services
(interp.steering.scorecard.chat), and the healthcheck HTTP client
(requests.get). Contracts checked: exit codes 0/1, styled
[OK]/[ERROR]/[WARN]/[INFO] lines, and concrete file side effects (shards,
manifests, evaluation and monitor reports)."""

from __future__ import annotations

import datetime
import json
import os

from interp import cli
from tests.interp.conftest import fake_capture


def run_cli(argv: list[str]) -> int:
    args = cli.build_parser().parse_args(argv)
    return args.func(args)


def fake_capture_activations(model_id, texts, layers, revision="main", batch_size=16):
    """capture_activations stand-in with the real nnsight-path signature,
    delegating to the deterministic conftest fake."""
    return fake_capture(texts, layers)


class TestProbeCapture:
    def test_capture_writes_shards_and_manifest(
        self, tmp_path, labeled_dataset, monkeypatch, capsys
    ):
        monkeypatch.setattr(
            "interp.probes.capture.capture_activations", fake_capture_activations
        )
        out_dir = str(tmp_path / "cap")
        assert (
            run_cli(
                [
                    "probe-capture",
                    "--model",
                    "test-model",
                    "--revision",
                    "rev-2",
                    "--model-hash",
                    "mh-1",
                    "--dataset",
                    labeled_dataset,
                    "--layers",
                    "3,5",
                    "--out",
                    out_dir,
                ]
            )
            == 0
        )
        printed = capsys.readouterr().err
        assert "[RUN] Capturing activations: test-model" in printed
        assert (
            "[OK] Capture complete: 120 examples, 1 shards, layers [3, 5]"
        ) in printed
        # --revision/--model-hash plumb through to the durable manifest
        with open(os.path.join(out_dir, "manifest.json")) as f:
            manifest = json.load(f)
        assert manifest["model_revision"] == "rev-2"
        assert manifest["model_hash"] == "mh-1"
        assert manifest["layers"] == [3, 5]
        assert manifest["count"] == 120
        assert os.path.isfile(os.path.join(out_dir, "shard_00000.npz"))
        assert os.path.isfile(os.path.join(out_dir, "examples.jsonl"))

    def test_malformed_dataset_is_styled_failure(self, tmp_path, capsys):
        dataset = tmp_path / "bad.jsonl"
        dataset.write_text('{"text": "x"}\n')  # label key missing
        out_dir = tmp_path / "cap"
        assert (
            run_cli(
                [
                    "probe-capture",
                    "--model",
                    "m",
                    "--dataset",
                    str(dataset),
                    "--layers",
                    "3",
                    "--out",
                    str(out_dir),
                ]
            )
            == 1
        )
        printed = capsys.readouterr().err
        assert "[ERROR] Capture failed:" in printed
        assert "needs 'text' and 'label' keys" in printed
        assert not out_dir.exists()  # refusal leaves no partial capture

    def test_malformed_layers_is_styled_usage_error(self, tmp_path, capsys):
        # Handler-level contract: bad --layers is a styled exit 2, not a raw
        # ValueError traceback (mirrors the steer-scorecard --scales arm).
        assert (
            run_cli(
                [
                    "probe-capture",
                    "--model",
                    "m",
                    "--dataset",
                    str(tmp_path / "d.jsonl"),
                    "--layers",
                    "3,x",
                    "--out",
                    str(tmp_path / "cap"),
                ]
            )
            == 2
        )
        printed = capsys.readouterr().err
        assert "[ERROR] Invalid --layers value: '3,x'" in printed
        assert "comma-separated integers" in printed

    def test_missing_dataset_is_styled_failure(self, tmp_path, capsys):
        # OSError arm: a bad path is a styled exit 1, not a traceback
        assert (
            run_cli(
                [
                    "probe-capture",
                    "--model",
                    "m",
                    "--dataset",
                    str(tmp_path / "none.jsonl"),
                    "--layers",
                    "3",
                    "--out",
                    str(tmp_path / "cap"),
                ]
            )
            == 1
        )
        assert "[ERROR] Capture failed:" in capsys.readouterr().err


class TestProbeTrain:
    def test_train_registers_probe(self, tmp_path, capture_dir, capsys):
        out_root = str(tmp_path / "cli-probes")
        assert (
            run_cli(
                [
                    "probe-train",
                    "--capture",
                    capture_dir,
                    "--name",
                    "cli-probe",
                    "--out",
                    out_root,
                ]
            )
            == 0
        )
        probe_dir = os.path.join(out_root, "cli-probe")
        with open(os.path.join(probe_dir, "manifest.json")) as f:
            manifest = json.load(f)
        printed = capsys.readouterr().err
        assert f"[RUN] Training probes from {capture_dir}" in printed
        # the [OK] line reports the registered layer and both AUROCs verbatim
        assert (
            f"[OK] Probe registered: cli-probe (layer {manifest['layer']},"
            f" val AUROC {manifest['metrics']['val_auroc']}, "
            f"test AUROC {manifest['metrics']['test_auroc']})"
        ) in printed
        assert "  [INFO] layer 3: val" in printed
        assert "  [INFO] layer 5: val" in printed
        assert manifest["name"] == "cli-probe"
        assert manifest["layer"] in (3, 5)
        assert os.path.isfile(os.path.join(probe_dir, "probe.npz"))

    def test_missing_capture_is_styled_failure(self, tmp_path, capsys):
        assert (
            run_cli(
                [
                    "probe-train",
                    "--capture",
                    str(tmp_path / "missing"),
                    "--name",
                    "n",
                    "--out",
                    str(tmp_path / "cli-probes"),
                ]
            )
            == 1
        )
        printed = capsys.readouterr().err
        assert "[ERROR] Training failed:" in printed
        assert "manifest not found" in printed


def scripted_eval_chat(subject_reply, judge_reply):
    """interp.steering.scorecard.chat stand-in routing subject prompts
    (port 8002) and judge prompts (port 8003) to the given callables."""

    def fake_chat(url, prompt, system=None, max_tokens=256, temperature=0.0, **kwargs):
        if url.endswith("8003"):
            return {"text": judge_reply(prompt)}
        return {"text": subject_reply(prompt)}

    return fake_chat


class TestProbeEval:
    def test_deployable_probe_beats_baselines(self, probe_dir, monkeypatch, capsys):
        # constant subject output starves the BoW baseline (AUROC 0.5) and an
        # unparseable judge verdict counts as wrong, so the probe wins
        monkeypatch.setattr(
            "interp.steering.scorecard.chat",
            scripted_eval_chat(lambda p: "a plain reply", lambda p: "not json"),
        )
        assert (
            run_cli(["probe-eval", "--probe", probe_dir, "--label-name", "negativity"])
            == 0
        )
        printed = capsys.readouterr().err
        assert (
            "[RUN] Evaluating probe test-probe against black-box baselines"
        ) in printed
        assert "[OK] Probe AUROC:" in printed
        assert "[OK] Baselines: judge accuracy 0.0, BoW AUROC 0.5" in printed
        assert (
            "[OK] Verdict: probe beats both black-box baselines; "
            "eligible for monitor duty"
        ) in printed
        report_path = os.path.join(probe_dir, "evaluation.json")
        assert f"[OK] Report written: {report_path}" in printed
        with open(report_path) as f:
            report = json.load(f)
        assert report["deployable"] is True
        assert report["label_name"] == "negativity"
        assert report["probe_metrics"]["auroc"] > 0.5

    def test_echoing_subject_fails_baseline_and_warns(
        self, probe_dir, monkeypatch, capsys
    ):
        # a subject that echoes its input hands the BoW baseline the label
        # tokens (AUROC 1.0), which no probe can strictly beat: exit stays 0
        # but the verdict is a [WARN] and the report says not deployable
        def judge_reply(prompt):
            return json.dumps({"label": 0 if "neg" in prompt else 1})

        monkeypatch.setattr(
            "interp.steering.scorecard.chat",
            scripted_eval_chat(lambda p: p, judge_reply),
        )
        assert (
            run_cli(["probe-eval", "--probe", probe_dir, "--label-name", "negativity"])
            == 0
        )
        printed = capsys.readouterr().err
        assert "[WARN] Verdict:" in printed
        assert "NOT deployed" in printed
        with open(os.path.join(probe_dir, "evaluation.json")) as f:
            assert json.load(f)["deployable"] is False

    def test_missing_probe_dir_is_styled_failure(self, tmp_path, capsys):
        assert (
            run_cli(
                ["probe-eval", "--probe", str(tmp_path / "nope"), "--label-name", "l"]
            )
            == 1
        )
        printed = capsys.readouterr().err
        assert "[ERROR] Evaluation failed:" in printed
        assert "manifest not found" in printed

    def test_missing_capture_manifest_is_styled_failure(self, probe_dir, capsys):
        # Captures are a regenerable cache, so the capture manifest can be
        # gone by eval time; the handler must style the ManifestError like
        # probe-train and probe-monitor do, not escape with a traceback
        with open(os.path.join(probe_dir, "manifest.json")) as f:
            capture_dir = json.load(f)["capture_dir"]
        os.remove(os.path.join(capture_dir, "manifest.json"))
        assert (
            run_cli(["probe-eval", "--probe", probe_dir, "--label-name", "negativity"])
            == 1
        )
        printed = capsys.readouterr().err
        assert "[ERROR] Evaluation failed:" in printed
        assert "manifest not found" in printed


class TestProbeExport:
    """The step between probe-train's probe.npz and the image harness-serve
    loads. Without it a registered probe cannot reach the serving process at
    all, which is why the handler exists rather than only the library call."""

    def test_export_writes_the_pair_beside_the_probe(self, probe_dir, capsys):
        assert run_cli(["probe-export", "--probe", probe_dir]) == 0
        image = os.path.join(probe_dir, "probe.safetensors")
        assert os.path.isfile(image)
        assert os.path.isfile(os.path.join(probe_dir, "probe.json"))
        printed = capsys.readouterr().err
        assert f"[OK] Image written: {image}" in printed

    def test_out_overrides_the_default_path(self, probe_dir, tmp_path, capsys):
        out = tmp_path / "elsewhere" / "p.safetensors"
        out.parent.mkdir()
        assert run_cli(["probe-export", "--probe", probe_dir, "--out", str(out)]) == 0
        assert out.is_file()
        assert out.with_suffix(".json").is_file()

    def test_an_image_the_harness_cannot_see_is_a_warning(self, probe_dir, capsys):
        # harness-serve mounts only /mnt/ai-data/interp/probes, so an image
        # written anywhere else is unreachable from the container. Printing a
        # HARNESS_PROBE command for it would hand over one that fails at
        # startup with "Probe not found".
        assert run_cli(["probe-export", "--probe", probe_dir]) == 0
        printed = capsys.readouterr().err
        assert "HARNESS_PROBE" not in printed
        assert "harness-serve mounts only that tree" in printed

    def test_a_tampered_probe_is_a_styled_failure(self, probe_dir, capsys):
        # The registry validates the hash on load, and the handler must style
        # that refusal rather than let RegistryError escape as a traceback.
        with open(os.path.join(probe_dir, "probe.npz"), "ab") as f:
            f.write(b"tamper")
        assert run_cli(["probe-export", "--probe", probe_dir]) == 1
        printed = capsys.readouterr().err
        assert "[ERROR] Export failed:" in printed
        assert "hash" in printed

    def test_missing_probe_dir_is_a_styled_failure(self, tmp_path, capsys):
        assert run_cli(["probe-export", "--probe", str(tmp_path / "nope")]) == 1
        printed = capsys.readouterr().err
        assert "[ERROR] Export failed:" in printed
        assert "manifest not found" in printed


def write_log_window(tmp_path) -> str:
    """A served-request log of 12 scored records inside the default 7d window
    (the handler passes now=None, so timestamps must be near real time)."""
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    now = datetime.datetime.now(datetime.UTC)
    lines = []
    for i in range(12):
        stamp = (now - datetime.timedelta(hours=1, seconds=i)).strftime(
            "%Y-%m-%dT%H:%M:%S.%fZ"
        )
        lines.append(
            json.dumps(
                {
                    "timestamp": stamp,
                    "target_service": "harness",
                    "messages": [{"role": "user", "content": f"request {i}"}],
                    "response_text": "a neg response" if i % 2 else "a pos response",
                    "probe": {
                        "name": "test-probe",
                        "layer": 4,
                        "site": "layer_output",
                        "threshold": 0.5,
                        "scored": 8,
                        "position_first": 0,
                        "max_score": round(0.08 * i, 4),
                        "flags": [],
                    },
                }
            )
        )
    (log_dir / f"{now.strftime('%Y-%m-%d')}.jsonl").write_text("\n".join(lines) + "\n")
    return str(log_dir)


class TestProbeMonitor:
    def test_aggregates_scores_and_writes_report(self, tmp_path, capsys):
        log_dir = write_log_window(tmp_path)
        monitor_root = str(tmp_path / "monitor")
        assert (
            run_cli(
                [
                    "probe-monitor",
                    "--log-dir",
                    log_dir,
                    "--monitor-root",
                    monitor_root,
                ]
            )
            == 0
        )
        printed = capsys.readouterr().err
        assert "[RUN] Monitor: window 7d" in printed
        assert "[OK] Reported 12/12 records" in printed
        assert "[OK] test-probe: mean " in printed
        assert "over-threshold" in printed
        run_dirs = os.listdir(monitor_root)
        assert len(run_dirs) == 1
        run_dir = os.path.join(monitor_root, run_dirs[0])
        assert f"[OK] Report written: {run_dir}/report.json" in printed
        with open(os.path.join(run_dir, "report.json")) as f:
            report = json.load(f)
        assert report["counts"]["reported"] == 12
        assert report["scored"] == "in_band"
        assert "test-probe" in report["probes"]
        assert os.path.isfile(os.path.join(run_dir, "flagged.jsonl"))

    def test_runs_while_inference_is_up(self, tmp_path, monkeypatch, capsys):
        # The refusal this replaces existed because scoring meant replaying a
        # model on the GPU that serving owned. Nothing here loads a model, so
        # a healthy service is no longer a reason to refuse -- and monitoring
        # traffic you cannot monitor while it is being served is not
        # monitoring.
        class UpResponse:
            status_code = 200

        monkeypatch.setattr(
            "requests.get", lambda url, timeout=None, **kwargs: UpResponse()
        )
        log_dir = write_log_window(tmp_path)
        assert (
            run_cli(
                [
                    "probe-monitor",
                    "--log-dir",
                    log_dir,
                    "--monitor-root",
                    str(tmp_path / "monitor"),
                ]
            )
            == 0
        )

    def test_an_empty_window_is_a_failure_naming_the_directory(self, tmp_path, capsys):
        empty = tmp_path / "logs"
        empty.mkdir()
        assert (
            run_cli(
                [
                    "probe-monitor",
                    "--log-dir",
                    str(empty),
                    "--monitor-root",
                    str(tmp_path / "monitor"),
                ]
            )
            == 1
        )
        assert "[ERROR] Monitor failed: no scored records" in capsys.readouterr().err
