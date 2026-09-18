"""F7 runner pieces: GGUF identity, quant-match preflight, resumable
generation and judging. Serving I/O is stubbed; the live phases are host
acceptance runs."""

import hashlib
import json
import os
from unittest import mock

import pytest

from common import serving as common_serving
from training import serving
from training.eval import generate
from training.eval.lmeval import LmEvalError, load_task_config
from training.eval.runner import (
    EvalRunError,
    hf_identity,
    judge_identity,
    load_judgments,
    quant_from_name,
    run_judgments,
)

EVAL_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "training", "eval")


def write_gguf(tmp_path, name, quant=None, manifest_quant=None):
    path = tmp_path / name
    path.write_bytes(b"gguf")
    if manifest_quant is not None:
        manifest = path.with_suffix("").as_posix() + ".manifest.json"
        with open(manifest, "w") as f:
            json.dump({"quant_type": manifest_quant, "gguf": {"sha256": "deadbeef"}}, f)
    return str(path)


def test_quant_from_name():
    assert quant_from_name("model-Q8_0.gguf") == "Q8_0"
    # Some published GGUF names use lowercase suffixes; the confound check
    # compares case-sensitively, so parsing normalizes to uppercase.
    assert quant_from_name("gemma-4-E2B-q8_0.gguf") == "Q8_0"
    assert quant_from_name("model.gguf") is None
    assert quant_from_name("model.bin") is None
    # A published GGUF may ship unquantized; the storage types name
    # themselves so the identity is recorded rather than read as unknown.
    assert quant_from_name("tuned-BF16.gguf") == "BF16"
    assert quant_from_name("tuned-bf16.gguf") == "BF16"
    assert quant_from_name("model-F16.gguf") == "F16"
    assert quant_from_name("model-F32.gguf") == "F32"


def write_checkpoint(root, name, weights=b"weights", manifest_hash=None):
    """A checkpoint directory under a fake models root, optionally exported
    (an hf/ directory with the manifest beside it)."""
    if manifest_hash is None:
        directory = root / name
    else:
        directory = root / name / "hf"
    directory.mkdir(parents=True)
    (directory / "config.json").write_text(json.dumps({"model_type": "gemma4"}))
    (directory / "model.safetensors").write_bytes(weights)
    if manifest_hash is not None:
        (root / name / "manifest.json").write_text(
            json.dumps(
                {
                    "artifact": "hf_export",
                    "hf_dir": str(directory),
                    "model_hash": manifest_hash,
                }
            )
        )
    return str(directory)


def test_thinking_mode_is_family_aware():
    from training.eval.runner import thinking_mode_for

    # Gemma 4 thinking is opt-in; nothing in the eval pipeline opts in. The
    # composite config says gemma4, the text-only one gemma4_text.
    assert thinking_mode_for("gemma4") == "off"
    assert thinking_mode_for("gemma4_text") == "off"
    # An unregistered family is recorded as "not recorded", never guessed.
    assert thinking_mode_for("mystery") is None
    assert thinking_mode_for(None) is None


def test_hf_identity_reads_the_export_manifest(tmp_path, monkeypatch):
    monkeypatch.setattr(common_serving, "HOST_MODELS", str(tmp_path) + "/")
    path = write_checkpoint(tmp_path, "trained/tuned", manifest_hash="a" * 64)
    identity = hf_identity(path)
    assert identity["model_hash"] == "a" * 64
    assert identity["manifest"] == str(tmp_path / "trained" / "tuned" / "manifest.json")
    assert identity["thinking_mode"] == "off"
    assert identity["model"] == "/app/models/trained/tuned/hf"
    assert identity["host_path"] == path


def test_hf_identity_hashes_a_base_without_a_manifest(tmp_path, monkeypatch):
    monkeypatch.setattr(common_serving, "HOST_MODELS", str(tmp_path) + "/")
    path = write_checkpoint(tmp_path, "hf/base", weights=b"base weights")
    identity = hf_identity(path)
    assert identity["model_hash"] == hashlib.sha256(b"base weights").hexdigest()
    assert identity["manifest"] is None
    # The container spelling resolves to the same directory.
    assert hf_identity("/app/models/hf/base")["model_hash"] == identity["model_hash"]


def test_hf_identity_refuses_an_incomplete_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(common_serving, "HOST_MODELS", str(tmp_path) + "/")
    (tmp_path / "hf" / "partial").mkdir(parents=True)
    (tmp_path / "hf" / "partial" / "config.json").write_text("{}")
    with pytest.raises(EvalRunError, match="missing model.safetensors"):
        hf_identity(str(tmp_path / "hf" / "partial"))


def test_hf_identity_refuses_a_path_outside_the_models_tree(tmp_path, monkeypatch):
    monkeypatch.setattr(common_serving, "HOST_MODELS", str(tmp_path) + "/")
    with pytest.raises(EvalRunError, match="pass the exported hf directory"):
        hf_identity("google/gemma-4-E2B")
    with pytest.raises(EvalRunError, match="pass the exported hf directory"):
        hf_identity("/somewhere/else")


def test_judge_identity_names_the_published_gguf(tmp_path, monkeypatch):
    monkeypatch.setattr(common_serving, "HOST_MODELS", str(tmp_path) + "/")
    path = write_gguf(tmp_path, "judge-Q4_0.gguf")
    identity = judge_identity(path)
    assert identity["quant_type"] == "Q4_0"
    assert identity["model"] == "/app/models/judge-Q4_0.gguf"
    with pytest.raises(EvalRunError, match="not found"):
        judge_identity(str(tmp_path / "absent.gguf"))


def test_default_judge_prompts_load_unique():
    prompts = generate.load_prompts(os.path.join(EVAL_DIR, "judge_prompts.jsonl"))
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


def test_generate_responses_transient_retry_heals(tmp_path):
    # A transient failure retries within the same prompt (retries+1 attempts);
    # a later success clears the error field and records the response.
    out_path = str(tmp_path / "responses.jsonl")
    prompts = [{"id": "a", "prompt": "pa"}]
    service = serving.SERVICES["llama-gpu"]

    attempts = {"n": 0}

    def flaky_completion(_service, _messages, _sampling, timeout):
        attempts["n"] += 1
        if attempts["n"] <= 2:
            raise serving.ServingTransientError("upstream 503")
        return {"content": "healed", "usage": {}, "latency_ms": 1.0}

    with mock.patch.object(serving, "chat_completion", flaky_completion):
        responses = generate.generate_responses(service, prompts, out_path)
    assert attempts["n"] == 3  # two transient failures, then success
    assert responses["a"]["response"] == "healed"
    assert "error" not in responses["a"]  # the transient error field was cleared
    # Persisted record reflects the healed response, not the failures.
    assert generate.load_responses(out_path)["a"]["response"] == "healed"


def test_load_responses_repairs_torn_tail(tmp_path):
    # A run killed mid-append leaves a partial final line; the loader drops
    # it and truncates the file so the record regenerates and the next
    # append cannot fuse with the fragment.
    out_path = tmp_path / "responses.jsonl"
    good = json.dumps({"id": "a", "response": "ra"})
    out_path.write_text(good + "\n" + '{"id": "b", "resp')
    responses = generate.load_responses(str(out_path))
    assert list(responses) == ["a"]
    assert out_path.read_text() == good + "\n"


def test_load_responses_repairs_newline_less_complete_tail(tmp_path):
    # A record fully written but killed before its trailing newline still
    # parses as valid JSON; it must still be dropped and truncated, or the
    # next append fuses onto it and every later load is poisoned.
    out_path = tmp_path / "responses.jsonl"
    a = json.dumps({"id": "a", "response": "ra"})
    b = json.dumps({"id": "b", "response": "rb"})  # complete, no trailing newline
    out_path.write_text(a + "\n" + b)
    responses = generate.load_responses(str(out_path))
    assert list(responses) == ["a"]  # b dropped, regenerates next run
    assert out_path.read_text() == a + "\n"  # file left cleanly terminated
    # A subsequent append lands on a clean line, not fused onto b.
    with open(out_path, "a") as f:
        f.write(b + "\n")
    assert list(generate.load_responses(str(out_path))) == ["a", "b"]


def test_load_resume_jsonl_mid_file_corruption_raises(tmp_path):
    # Only the tail is repairable; corruption before it means the file is
    # not a killed append and must fail loudly.
    out_path = tmp_path / "responses.jsonl"
    out_path.write_text('{"id": "a"\n{"id": "b"}\n')
    with pytest.raises(json.JSONDecodeError):
        generate.load_resume_jsonl(str(out_path))


def test_run_judgments_error_then_resume_rejudges_only_errored(tmp_path):
    # First pass: the judge is down for one prompt's pairs (recorded as
    # error) while the other prompt's pairs succeed. A resumed run re-judges
    # only the errored pairs and preserves the successful ones untouched.
    out_path = str(tmp_path / "judgments.jsonl")
    prompts = {
        "p1": {"id": "p1", "prompt": "alpha"},
        "p2": {"id": "p2", "prompt": "beta"},
    }
    responses = {
        "baseline": {"p1": {"response": "a1"}, "p2": {"response": "a2"}},
        "tuned": {"p1": {"response": "b1"}, "p2": {"response": "b2"}},
    }

    def first_pass(messages):
        if "beta" in messages[1]["content"]:
            raise serving.ServingError("judge endpoint down")
        return "Rationale.\nVERDICT: FIRST"

    first = run_judgments(first_pass, prompts, responses, out_path)
    outcomes = {(r["prompt_id"], r["first"]): r["outcome"] for r in first}
    assert outcomes[("p1", "baseline")] == "baseline"
    assert outcomes[("p1", "tuned")] == "tuned"
    assert outcomes[("p2", "baseline")] == "error"
    assert outcomes[("p2", "tuned")] == "error"

    seen = []

    def resume_pass(messages):
        seen.append(messages[1]["content"])
        return "Rationale.\nVERDICT: TIE"

    resumed = run_judgments(resume_pass, prompts, responses, out_path)
    # Only the two errored (beta) pairs were re-judged; p1 was never re-called.
    assert all("beta" in text for text in seen)
    assert len(seen) == 2
    resumed_outcomes = {(r["prompt_id"], r["first"]): r["outcome"] for r in resumed}
    assert resumed_outcomes[("p1", "baseline")] == "baseline"  # preserved
    assert resumed_outcomes[("p1", "tuned")] == "tuned"  # preserved
    assert resumed_outcomes[("p2", "baseline")] == "tie"  # healed
    assert resumed_outcomes[("p2", "tuned")] == "tie"  # healed
    # Persisted state agrees: the errored pairs now read as decided verdicts.
    persisted = load_judgments(out_path)
    assert persisted[("p2", "baseline")]["outcome"] == "tie"
    assert persisted[("p1", "baseline")]["outcome"] == "baseline"


def test_run_judgments_resumable_and_position_swapped(tmp_path):
    out_path = str(tmp_path / "judgments.jsonl")
    prompts = {"p1": {"id": "p1", "prompt": "question"}}
    responses = {
        "baseline": {"p1": {"response": "base answer"}},
        "tuned": {"p1": {"response": "tuned answer"}},
    }

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
    responses = {
        "baseline": {"p1": {"response": "a"}},
        "tuned": {"p1": {"error": "generation failed"}},
    }
    results = run_judgments(
        lambda m: "VERDICT: TIE", prompts, responses, str(tmp_path / "j.jsonl")
    )
    assert all(r["outcome"] == "error" for r in results)


def test_run_judgments_malformed_verdict_counted(tmp_path):
    prompts = {"p1": {"id": "p1", "prompt": "q"}}
    responses = {
        "baseline": {"p1": {"response": "a"}},
        "tuned": {"p1": {"response": "b"}},
    }
    results = run_judgments(
        lambda m: "I refuse to answer.", prompts, responses, str(tmp_path / "j.jsonl")
    )
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
