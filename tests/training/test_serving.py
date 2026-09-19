"""Shared serving module: service registry, model path mapping, training-idle
preflight, and swap env construction. Compose and HTTP are mocked; the live
swap path is a host acceptance run."""

import json
import os
import urllib.error
from unittest import mock

import pytest

from common import serving as common_serving
from training import serving
from training.serving import SERVICES, ServingError, ServingTransientError, model_paths


def test_quant_from_name():
    assert common_serving.quant_from_name("model-Q8_0.gguf") == "Q8_0"
    # Some published GGUF names use lowercase suffixes; the confound check
    # compares case-sensitively, so parsing normalizes to uppercase.
    assert common_serving.quant_from_name("gemma-4-E2B-q8_0.gguf") == "Q8_0"
    assert common_serving.quant_from_name("model.gguf") is None
    assert common_serving.quant_from_name("model.bin") is None
    # A published GGUF may ship unquantized; the storage types name
    # themselves so the identity is recorded rather than read as unknown.
    assert common_serving.quant_from_name("tuned-BF16.gguf") == "BF16"
    assert common_serving.quant_from_name("tuned-bf16.gguf") == "BF16"
    assert common_serving.quant_from_name("model-F16.gguf") == "F16"
    assert common_serving.quant_from_name("model-F32.gguf") == "F32"


class _FakeHTTPResponse:
    """urlopen() stand-in: a context manager exposing .status and a read()
    that json.load can consume. Serves both the chat client (reads a body)
    and the health poll (checks status)."""

    def __init__(self, body=b"", status=200):
        self._body = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.status = status

    def read(self, *args):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _StepClock:
    """Deterministic monotonic clock: each call advances by a fixed step so
    wait_healthy's deadline arithmetic runs without real time passing."""

    def __init__(self, step: float = 1.0):
        self.t = 0.0
        self.step = step
        self.calls = 0

    def __call__(self) -> float:
        self.calls += 1
        self.t += self.step
        return self.t


def test_registry_matches_compose_definitions():
    # Parse the compose files rather than re-declaring the constants by hand:
    # the drift this test exists to catch must fail it. Every entry is checked,
    # including the harness, which lives in its own compose file under a
    # different service name -- the two spellings the spec carries.
    import yaml

    loaded: dict[str, dict] = {}
    for name, spec in SERVICES.items():
        if spec.compose_file not in loaded:
            with open(spec.compose_file) as f:
                loaded[spec.compose_file] = yaml.safe_load(f)
        compose = loaded[spec.compose_file]
        service = compose["services"][spec.service]
        published = [p.split(":")[1] for p in service["ports"]]
        assert str(spec.port) in published, name
        env = "\n".join(service.get("environment", []))
        assert f"${{{spec.model_env}" in env, name
        # The port spelling differs by probe: llama and vLLM curl a URL with
        # `:8001`, the harness opens /dev/tcp/127.0.0.1/8005. Both name the
        # port and the path, which is what must not drift from the registry.
        health_cmd = " ".join(service["healthcheck"]["test"])
        assert str(spec.port) in health_cmd, name
        assert spec.health_path in health_cmd, name
    assert SERVICES["vllm-gpu"].chat_url == "http://127.0.0.1:8003/v1/chat/completions"


def test_no_two_services_publish_the_same_port():
    # The registry spans two compose files now, so a collision is no longer
    # something one file's own validation would catch.
    ports = [spec.port for spec in SERVICES.values()]
    assert len(ports) == len(set(ports)), sorted(ports)


def test_swap_model_drives_the_compose_file_that_defines_the_service():
    # The harness is a compose service, but not in the inference file and not
    # under its registry name. A recreate aimed at the wrong file reports
    # "no such service", which reads as a broken compose file.
    seen = {}

    def fake_compose(env, *args, compose_file):
        seen["args"] = args
        seen["compose_file"] = compose_file
        seen["model"] = env.get("HARNESS_MODEL")

    original = common_serving.compose
    common_serving.compose = fake_compose
    try:
        serving.swap_model(SERVICES["harness"], "/app/models/hf/m", env={})
    finally:
        common_serving.compose = original

    assert seen["compose_file"] == serving.HARNESS_COMPOSE
    assert seen["args"][-1] == "harness-serve"
    assert seen["model"] == "/app/models/hf/m"


def test_model_paths_maps_both_forms():
    assert model_paths("/app/models/x/y.gguf") == (
        "/app/models/x/y.gguf",
        "/mnt/ai-data/models/x/y.gguf",
    )
    assert model_paths("/mnt/ai-data/models/x/y.gguf") == (
        "/app/models/x/y.gguf",
        "/mnt/ai-data/models/x/y.gguf",
    )
    with pytest.raises(ServingError, match="must live under"):
        model_paths("/tmp/y.gguf")


def test_ensure_training_idle_refuses_on_lock(tmp_path, monkeypatch):
    lock = tmp_path / ".train.lock"
    lock.write_text("{}")
    monkeypatch.setenv("TRAIN_LOCK_PATH", str(lock))
    with pytest.raises(ServingError, match="one GPU consumer"):
        serving.ensure_training_idle()


def test_swap_model_sets_and_clears_env():
    calls = []

    def fake_run(cmd, cwd=None, env=None):
        calls.append((cmd, env))
        return mock.Mock(returncode=0)

    base_env = {"PATH": "/usr/bin"}
    with mock.patch.object(common_serving.subprocess, "run", side_effect=fake_run):
        serving.swap_model(SERVICES["vllm-gpu"], "/app/models/t.gguf", env=base_env)
        serving.swap_model(
            SERVICES["vllm-gpu"],
            None,
            env={**base_env, "VLLM_GPU_MODEL": "/app/models/t.gguf"},
        )
    swap_cmd, swap_env = calls[0]
    # -f is not optional: the inference compose file carries a domain suffix, so
    # a bare `docker compose` would not find it and every swap would fail.
    assert swap_cmd == [
        "docker",
        "compose",
        "-f",
        serving.INFER_COMPOSE,
        "up",
        "-d",
        "--force-recreate",
        "vllm-gpu",
    ]
    assert swap_env["VLLM_GPU_MODEL"] == "/app/models/t.gguf"
    restore_cmd, restore_env = calls[1]
    assert "VLLM_GPU_MODEL" not in restore_env
    assert restore_cmd[-1] == "vllm-gpu"
    assert base_env == {"PATH": "/usr/bin"}  # caller env never mutated


def test_sha256_file_matches_known_digest(tmp_path):
    path = tmp_path / "f"
    path.write_bytes(b"abc")
    assert serving.sha256_file(str(path)) == (
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )


def test_default_lock_path_used_without_override(monkeypatch):
    monkeypatch.delenv("TRAIN_LOCK_PATH", raising=False)
    with mock.patch.object(serving.os.path, "exists", return_value=True) as ex:
        with pytest.raises(ServingError, match="Training lock present"):
            serving.ensure_training_idle()
    assert ex.call_args[0][0] == "/mnt/ai-data/training/.train.lock"


def test_scripts_import_the_shared_module():
    # The host entry points must resolve the shared implementation, not carry
    # their own copies of swap/health/client logic.
    for script in (
        "training/commands/eval_ft.py",
        "training/commands/generate_teacher.py",
    ):
        path = os.path.join(serving.REPO_ROOT, script)
        with open(path) as f:
            source = f.read()
        assert "from training import serving" in source, script


def test_chat_completion_success(monkeypatch):
    body = {
        "choices": [{"message": {"content": "hi"}}],
        "usage": {"prompt_tokens": 3, "completion_tokens": 1},
    }
    monkeypatch.setattr(
        common_serving.urllib.request,
        "urlopen",
        lambda *a, **k: _FakeHTTPResponse(body),
    )
    out = serving.chat_completion(
        SERVICES["llama-gpu"], [{"role": "user", "content": "q"}]
    )
    assert out["content"] == "hi"
    assert out["usage"] == body["usage"]
    assert isinstance(out["latency_ms"], float)


def test_chat_completion_5xx_is_transient(monkeypatch):
    err = urllib.error.HTTPError(
        SERVICES["llama-gpu"].chat_url, 503, "Service Unavailable", {}, None
    )

    def boom(*a, **k):
        raise err

    monkeypatch.setattr(common_serving.urllib.request, "urlopen", boom)
    with pytest.raises(ServingTransientError, match="HTTP 503"):
        serving.chat_completion(
            SERVICES["llama-gpu"], [{"role": "user", "content": "q"}]
        )


def test_chat_completion_4xx_is_non_transient(monkeypatch):
    err = urllib.error.HTTPError(
        SERVICES["llama-gpu"].chat_url, 400, "Bad Request", {}, None
    )

    def boom(*a, **k):
        raise err

    monkeypatch.setattr(common_serving.urllib.request, "urlopen", boom)
    with pytest.raises(ServingError) as exc:
        serving.chat_completion(
            SERVICES["llama-gpu"], [{"role": "user", "content": "q"}]
        )
    # A 4xx is a client error a retry cannot fix; taxonomy must not mark it transient.
    assert not isinstance(exc.value, ServingTransientError)
    assert "HTTP 400" in str(exc.value)


def test_chat_completion_urlerror_is_transient(monkeypatch):
    def boom(*a, **k):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(common_serving.urllib.request, "urlopen", boom)
    with pytest.raises(ServingTransientError, match="failed"):
        serving.chat_completion(
            SERVICES["llama-gpu"], [{"role": "user", "content": "q"}]
        )


def test_chat_completion_empty_body_is_non_transient(monkeypatch):
    # An empty or malformed body decodes to a ValueError: a request error a
    # retry cannot fix, so ServingError (not the transient subclass).
    monkeypatch.setattr(
        common_serving.urllib.request, "urlopen", lambda *a, **k: _FakeHTTPResponse(b"")
    )
    with pytest.raises(ServingError) as exc:
        serving.chat_completion(
            SERVICES["llama-gpu"], [{"role": "user", "content": "q"}]
        )
    assert not isinstance(exc.value, ServingTransientError)
    assert "Non-JSON" in str(exc.value)


def test_chat_completion_missing_choices_is_malformed(monkeypatch):
    # Valid JSON but no choices: extraction fails with a non-transient error.
    monkeypatch.setattr(
        common_serving.urllib.request,
        "urlopen",
        lambda *a, **k: _FakeHTTPResponse({"choices": []}),
    )
    with pytest.raises(ServingError) as exc:
        serving.chat_completion(
            SERVICES["llama-gpu"], [{"role": "user", "content": "q"}]
        )
    assert not isinstance(exc.value, ServingTransientError)
    assert "Malformed" in str(exc.value)


def test_wait_healthy_times_out(monkeypatch):
    clock = _StepClock()
    monkeypatch.setattr(common_serving.time, "monotonic", clock)
    monkeypatch.setattr(common_serving.time, "sleep", lambda _s: None)

    def never(*a, **k):
        raise urllib.error.URLError("still loading")

    monkeypatch.setattr(common_serving.urllib.request, "urlopen", never)
    with pytest.raises(ServingError, match="did not become healthy"):
        serving.wait_healthy(SERVICES["llama-gpu"], budget_s=5)
    # Polled repeatedly before giving up, not a single probe-and-fail.
    assert clock.calls > 1


def test_wait_healthy_returns_when_probe_succeeds(monkeypatch):
    clock = _StepClock()
    monkeypatch.setattr(common_serving.time, "monotonic", clock)
    monkeypatch.setattr(common_serving.time, "sleep", lambda _s: None)
    attempts = {"n": 0}

    def probe(*a, **k):
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise urllib.error.URLError("warming up")
        return _FakeHTTPResponse(status=200)

    monkeypatch.setattr(common_serving.urllib.request, "urlopen", probe)
    elapsed = serving.wait_healthy(SERVICES["vllm-gpu"], budget_s=100)
    assert attempts["n"] == 3  # returned on the first 200, not before
    assert elapsed >= 0.0
