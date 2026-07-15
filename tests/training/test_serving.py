"""Shared serving module: service registry, model path mapping, training-idle
preflight, and swap env construction. Compose and HTTP are mocked; the live
swap path is a host acceptance run."""

import os
from unittest import mock

import pytest

from training import serving
from training.serving import SERVICES, ServingError, model_paths


def test_registry_matches_compose_definitions():
    llama = SERVICES["llama-gpu"]
    assert (llama.port, llama.model_env, llama.health_path) == (
        8002, "LLAMA_GPU_MODEL", "/v1/health")
    vllm = SERVICES["vllm-gpu"]
    assert (vllm.port, vllm.model_env, vllm.health_path) == (
        8003, "VLLM_GPU_MODEL", "/health")
    assert vllm.chat_url == "http://127.0.0.1:8003/v1/chat/completions"


def test_model_paths_maps_both_forms():
    assert model_paths("/app/models/x/y.gguf") == (
        "/app/models/x/y.gguf", "/mnt/ai-data/models/x/y.gguf")
    assert model_paths("/mnt/ai-data/models/x/y.gguf") == (
        "/app/models/x/y.gguf", "/mnt/ai-data/models/x/y.gguf")
    with pytest.raises(ServingError, match="must live under"):
        model_paths("/tmp/y.gguf")


def test_ensure_training_idle_refuses_on_lock(tmp_path, monkeypatch):
    lock = tmp_path / ".train.lock"
    lock.write_text("{}")
    monkeypatch.setenv("TRAIN_LOCK_PATH", str(lock))
    with pytest.raises(ServingError, match="one GPU consumer"):
        serving.ensure_training_idle()


def test_ensure_training_idle_refuses_on_running_container(monkeypatch):
    completed = mock.Mock(returncode=0, stdout="abc123\n")
    with mock.patch.object(serving.subprocess, "run", return_value=completed):
        with pytest.raises(ServingError, match="Training container"):
            serving.ensure_training_idle()


def test_swap_model_sets_and_clears_env():
    calls = []

    def fake_run(cmd, cwd=None, env=None):
        calls.append((cmd, env))
        return mock.Mock(returncode=0)

    base_env = {"PATH": "/usr/bin"}
    with mock.patch.object(serving.subprocess, "run", side_effect=fake_run):
        serving.swap_model(SERVICES["vllm-gpu"], "/app/models/t.gguf",
                           env=base_env)
        serving.swap_model(SERVICES["vllm-gpu"], None,
                           env={**base_env, "VLLM_GPU_MODEL": "/app/models/t.gguf"})
    swap_cmd, swap_env = calls[0]
    assert swap_cmd == ["docker", "compose", "up", "-d", "--force-recreate",
                        "vllm-gpu"]
    assert swap_env["VLLM_GPU_MODEL"] == "/app/models/t.gguf"
    restore_cmd, restore_env = calls[1]
    assert "VLLM_GPU_MODEL" not in restore_env
    assert restore_cmd[-1] == "vllm-gpu"
    assert base_env == {"PATH": "/usr/bin"}  # caller env never mutated


def test_sha256_file_matches_known_digest(tmp_path):
    path = tmp_path / "f"
    path.write_bytes(b"abc")
    assert serving.sha256_file(str(path)) == (
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")


def test_default_lock_path_used_without_override(monkeypatch):
    monkeypatch.delenv("TRAIN_LOCK_PATH", raising=False)
    with mock.patch.object(serving.os.path, "exists", return_value=True) as ex:
        with pytest.raises(ServingError, match="Training lock present"):
            serving.ensure_training_idle()
    assert ex.call_args[0][0] == "/mnt/ai-data/training/.train.lock"


def test_scripts_import_the_shared_module():
    # The host entry points must resolve the shared implementation, not carry
    # their own copies of swap/health/client logic.
    for script in ("scripts/eval/serve_eval.py",
                   "training/commands/generate_teacher.py"):
        path = os.path.join(serving.REPO_ROOT, script)
        with open(path) as f:
            source = f.read()
        assert "from training import serving" in source, script
