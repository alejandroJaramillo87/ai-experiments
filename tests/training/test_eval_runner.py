"""F7 runner pieces: GGUF identity, quant-match preflight, resumable
generation and judging. Serving I/O is stubbed; the live phases are host
acceptance runs."""

import json
import os
from unittest import mock

import pytest

from training import serving
from training.eval import generate
from training.eval.lmeval import LmEvalError, load_task_config
from training.eval.runner import (
    EvalRunError,
    check_quant_match,
    gguf_identity,
    load_judgments,
    quant_from_name,
    run_judgments,
)

EVAL_DIR = os.path.join(os.path.dirname(__file__), "..", "..",
                        "training", "eval")


def write_gguf(tmp_path, name, quant=None, manifest_quant=None):
    path = tmp_path / name
    path.write_bytes(b"gguf")
    if manifest_quant is not None:
        manifest = path.with_suffix("").as_posix() + ".manifest.json"
        with open(manifest, "w") as f:
            json.dump({"quant_type": manifest_quant,
                       "gguf": {"sha256": "deadbeef"}}, f)
    return str(path)


def test_quant_from_name():
    assert quant_from_name("model-Q8_0.gguf") == "Q8_0"
    assert quant_from_name("model.gguf") is None
    assert quant_from_name("model.bin") is None


def test_gguf_identity_prefers_manifest(tmp_path, monkeypatch):
    monkeypatch.setattr(serving, "HOST_MODELS", str(tmp_path) + "/")
    path = write_gguf(tmp_path, "m-Q4_0.gguf", manifest_quant="Q8_0")
    identity = gguf_identity(path)
    assert identity["quant_type"] == "Q8_0"
    assert identity["sha256"] == "deadbeef"
    assert identity["thinking_mode"] == "off"
    assert identity["model"].startswith("/app/models/")


def test_gguf_identity_falls_back_to_name(tmp_path, monkeypatch):
    monkeypatch.setattr(serving, "HOST_MODELS", str(tmp_path) + "/")
    identity = gguf_identity(write_gguf(tmp_path, "m-Q4_0.gguf"))
    assert identity["quant_type"] == "Q4_0"
    assert identity["sha256"] is None


def test_gguf_identity_missing_file(tmp_path, monkeypatch):
    monkeypatch.setattr(serving, "HOST_MODELS", str(tmp_path) + "/")
    with pytest.raises(EvalRunError, match="not found"):
        gguf_identity(str(tmp_path / "absent.gguf"))


def test_check_quant_match():
    check_quant_match({"quant_type": "Q8_0"}, {"quant_type": "Q8_0"})
    with pytest.raises(EvalRunError, match="mismatch"):
        check_quant_match({"quant_type": "Q8_0"}, {"quant_type": "Q4_0"})
    with pytest.raises(EvalRunError, match="unknown"):
        check_quant_match({"quant_type": None}, {"quant_type": "Q8_0"})


def test_default_judge_prompts_load_unique():
    prompts = generate.load_prompts(
        os.path.join(EVAL_DIR, "judge_prompts.jsonl"))
    assert len(prompts) >= 20
    assert all("prompt" in p and "id" in p for p in prompts)


def test_load_prompts_rejects_duplicates(tmp_path):
    path = tmp_path / "p.jsonl"
    path.write_text('{"id": "a", "prompt": "x"}\n{"id": "a", "prompt": "y"}\n')
    with pytest.raises(ValueError, match="Duplicate"):
        generate.load_prompts(str(path))


def test_generate_responses_resumes_and_records_errors(tmp_path):
    out_path = str(tmp_path / "responses.jsonl")
    prompts = [{"id": "a", "prompt": "pa"}, {"id": "b", "prompt": "pb"}]
    service = serving.SERVICES["llama-gpu"]

    calls = []

    def fake_completion(_service, messages, _sampling, timeout):
        calls.append(messages[0]["content"])
        if messages[0]["content"] == "pb":
            raise serving.ServingError("bad request")
        return {"content": "ra", "usage": {}, "latency_ms": 1.0}

    with mock.patch.object(serving, "chat_completion", fake_completion):
        responses = generate.generate_responses(service, prompts, out_path)
    assert responses["a"]["response"] == "ra"
    assert "error" in responses["b"]

    # Resume: only the errored prompt regenerates, and its record heals
    def fixed_completion(_service, messages, _sampling, timeout):
        calls.append(messages[0]["content"])
        return {"content": "rb", "usage": {}, "latency_ms": 1.0}

    calls.clear()
    with mock.patch.object(serving, "chat_completion", fixed_completion):
        responses = generate.generate_responses(service, prompts, out_path)
    assert calls == ["pb"]
    assert responses["b"]["response"] == "rb"
    assert generate.load_responses(out_path)["b"]["response"] == "rb"


def test_run_judgments_resumable_and_position_swapped(tmp_path):
    out_path = str(tmp_path / "judgments.jsonl")
    prompts = {"p1": {"id": "p1", "prompt": "question"}}
    responses = {"baseline": {"p1": {"response": "base answer"}},
                 "tuned": {"p1": {"response": "tuned answer"}}}

    seen_orders = []

    def complete(messages):
        user = messages[1]["content"]
        seen_orders.append(user.index("base answer") < user.index("tuned answer"))
        return "Rationale.\nVERDICT: FIRST"

    results = run_judgments(complete, prompts, responses, out_path)
    assert seen_orders == [True, False]  # both presentation orders judged
    outcomes = sorted(r["outcome"] for r in results)
    assert outcomes == ["baseline", "tuned"]  # FIRST maps back per order

    # Resume: nothing re-judged
    def explode(_messages):
        raise AssertionError("should not be called on resume")

    resumed = run_judgments(explode, prompts, responses, out_path)
    assert len(resumed) == 2
    assert load_judgments(out_path)


def test_run_judgments_missing_generation_is_error_outcome(tmp_path):
    prompts = {"p1": {"id": "p1", "prompt": "q"}}
    responses = {"baseline": {"p1": {"response": "a"}},
                 "tuned": {"p1": {"error": "generation failed"}}}
    results = run_judgments(lambda m: "VERDICT: TIE", prompts, responses,
                            str(tmp_path / "j.jsonl"))
    assert all(r["outcome"] == "error" for r in results)


def test_run_judgments_malformed_verdict_counted(tmp_path):
    prompts = {"p1": {"id": "p1", "prompt": "q"}}
    responses = {"baseline": {"p1": {"response": "a"}},
                 "tuned": {"p1": {"response": "b"}}}
    results = run_judgments(lambda m: "I refuse to answer.", prompts,
                            responses, str(tmp_path / "j.jsonl"))
    assert [r["outcome"] for r in results] == ["malformed", "malformed"]
    assert all(r["rationale"] == "I refuse to answer." for r in results)


def test_task_config_pins_and_filters():
    config = load_task_config()
    assert config["lm_eval_version"] == "0.4.12"
    assert set(config["tasks"]) == {"ifeval", "gsm8k", "tinyMMLU", "hellaswag"}
    subset = load_task_config(tasks=["tinyMMLU"])
    assert list(subset["tasks"]) == ["tinyMMLU"]
    with pytest.raises(LmEvalError, match="not in the pinned set"):
        load_task_config(tasks=["mmlu_pro"])
