"""Branch coverage for export seams and common probes, all torch-free.

Covers the paths the D7 export tests leave to the host acceptance run, executed
here against stub modules and fake tools instead: the PEFT adapter merge
(torch/peft/transformers injected at the sys.modules seam), GGUF conversion and
quantization (real subprocesses against fake converter/quantizer executables),
the calculator's HF-recipe and teacher-memory branches, and the common
version-probe and git-provenance helpers.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import types
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from common import provenance, smoke
from training.calculator.components import (
    ArchitectureSpec,
    UnknownArchitectureError,
    optimizer_bytes,
    spec_from_recipe,
    teacher_forward_bytes,
    teacher_weights_bytes,
)
from training.export import convert as convert_mod
from training.export import merge as merge_mod


def write_json(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f)


# -- D7 staging: PEFT adapter merge -------------------------------------------
#
# torch/peft/transformers are not installed in the sandbox; _merge_adapter
# imports them lazily, so stub modules injected into sys.modules let the real
# function body run. The stubs write real files so the staging contract
# (converter-ready safetensors dir) is asserted on disk, not on mock calls.

def install_merge_stubs(monkeypatch, *, fail_load=False):
    """Stub torch/peft/transformers for _merge_adapter; returns a call log."""
    calls = {}

    torch_stub = types.ModuleType("torch")
    torch_stub.bfloat16 = object()   # sentinel: identity-checked below

    class AutoModelForCausalLM:
        @staticmethod
        def from_pretrained(base, revision=None, dtype=None,
                            local_files_only=None):
            calls["load"] = {"base": base, "revision": revision,
                             "dtype": dtype,
                             "local_files_only": local_files_only}
            if fail_load:
                raise RuntimeError("snapshot not in cache")
            return SimpleNamespace(base=base)

    transformers_stub = types.ModuleType("transformers")
    transformers_stub.AutoModelForCausalLM = AutoModelForCausalLM

    class FakeMergedModel:
        def save_pretrained(self, staging_dir, safe_serialization=False):
            calls["safe_serialization"] = safe_serialization
            write_json(os.path.join(staging_dir, "config.json"),
                       {"architectures": ["LlamaForCausalLM"]})
            with open(os.path.join(staging_dir, "model.safetensors"), "wb") as f:
                f.write(b"merged-weights")

    class FakePeftModel:
        @classmethod
        def from_pretrained(cls, model, adapter_dir):
            calls["adapter_dir"] = adapter_dir
            return cls()

        def merge_and_unload(self):
            return FakeMergedModel()

    peft_stub = types.ModuleType("peft")
    peft_stub.PeftModel = FakePeftModel

    monkeypatch.setitem(sys.modules, "torch", torch_stub)
    monkeypatch.setitem(sys.modules, "transformers", transformers_stub)
    monkeypatch.setitem(sys.modules, "peft", peft_stub)
    return calls, torch_stub


def make_adapter_dir(tmp_path, base="org/base"):
    adapter = tmp_path / "adapter"
    write_json(str(adapter / "adapter_config.json"),
               {"base_model_name_or_path": base})
    (adapter / "adapter_model.safetensors").write_bytes(b"lora")
    return adapter


def test_merge_adapter_loads_base_offline_and_saves_merged(tmp_path, monkeypatch):
    calls, torch_stub = install_merge_stubs(monkeypatch)
    adapter = make_adapter_dir(tmp_path)
    staging = tmp_path / "staging"
    staging.mkdir()

    base = merge_mod._merge_adapter(str(adapter), str(staging),
                                    revision="b" * 40)

    assert base == "org/base"
    # F6 decision 2: a non-local base id must resolve cache-first in bf16.
    assert calls["load"] == {"base": "org/base", "revision": "b" * 40,
                             "dtype": torch_stub.bfloat16,
                             "local_files_only": True}
    assert calls["adapter_dir"] == str(adapter)
    assert calls["safe_serialization"] is True
    assert (staging / "model.safetensors").read_bytes() == b"merged-weights"
    assert (staging / "config.json").is_file()


def test_merge_adapter_wraps_base_load_failure(tmp_path, monkeypatch):
    install_merge_stubs(monkeypatch, fail_load=True)
    adapter = make_adapter_dir(tmp_path)
    staging = tmp_path / "staging"
    staging.mkdir()

    with pytest.raises(convert_mod.ExportError, match="local HF cache") as exc:
        merge_mod._merge_adapter(str(adapter), str(staging))
    assert "org/base" in str(exc.value)
    assert "snapshot not in cache" in str(exc.value)


def test_stage_model_merges_adapter_checkpoint(tmp_path, monkeypatch):
    install_merge_stubs(monkeypatch)
    ckpt = tmp_path / "step_000000020"
    model = ckpt / "model"
    write_json(str(model / "adapter_config.json"),
               {"base_model_name_or_path": "org/base"})
    (model / "adapter_model.safetensors").write_bytes(b"lora")
    (model / "tokenizer.json").write_text("{}")
    staging = tmp_path / "staging"

    info = merge_mod.stage_model(str(ckpt), str(staging))

    assert info.adapter_path == str(model)
    assert info.base_model == "org/base"
    assert info.merge_method == "peft.merge_and_unload"
    assert info.tokenizer_source == str(model)
    assert (staging / "model.safetensors").is_file()
    assert (staging / "config.json").is_file()
    assert (staging / "tokenizer.json").is_file()


def test_stage_model_rejects_configless_full_model(tmp_path):
    # Weights without config.json cannot feed the converter; staging must
    # refuse rather than hand convert_hf_to_gguf.py a broken directory.
    ckpt = tmp_path / "step_000000030"
    model = ckpt / "model"
    os.makedirs(model)
    (model / "model.safetensors").write_bytes(b"weights")

    with pytest.raises(FileNotFoundError, match="No config.json"):
        merge_mod.stage_model(str(ckpt), str(tmp_path / "staging"))


# -- D7 conversion and quantization subprocess wrappers ------------------------
#
# The pinned llama.cpp toolchain is host-only; fake tools at the exact paths
# converter_path/quantize_path resolve prove the wrappers build the right
# command lines and honor the _run log/failure contract with real subprocesses.

FAKE_CONVERT_SCRIPT = """\
import sys

out = sys.argv[sys.argv.index("--outfile") + 1]
outtype = sys.argv[sys.argv.index("--outtype") + 1]
with open(out, "w") as f:
    f.write("GGUF:" + outtype)
print("converted", sys.argv[1])
"""


@pytest.fixture
def fake_llama_cpp(tmp_path):
    """Fake llama.cpp checkout: working converter script and quantize binary."""
    root = tmp_path / "llama.cpp"
    bin_dir = root / "build" / "bin"
    os.makedirs(bin_dir)
    (root / "convert_hf_to_gguf.py").write_text(FAKE_CONVERT_SCRIPT)
    quant = bin_dir / "llama-quantize"
    quant.write_text('#!/bin/sh\ncp "$1" "$2"\necho "type=$3"\n')
    quant.chmod(0o755)
    return root


def test_convert_to_gguf_invokes_pinned_converter(tmp_path, fake_llama_cpp):
    staging = tmp_path / "staged"
    staging.mkdir()
    out = tmp_path / "model-bf16.gguf"

    convert_mod.convert_to_gguf(str(staging), str(out), outtype="bf16",
                                llama_cpp_dir=str(fake_llama_cpp))

    assert out.read_text() == "GGUF:bf16"
    log = tmp_path / "model-bf16.gguf.convert.log"
    assert f"converted {staging}" in log.read_text()


def test_quantize_invokes_llama_quantize(tmp_path, fake_llama_cpp):
    src = tmp_path / "model-bf16.gguf"
    src.write_bytes(b"raw gguf")
    out = tmp_path / "model-Q8_0.gguf"

    convert_mod.quantize(str(src), str(out), "Q8_0",
                         llama_cpp_dir=str(fake_llama_cpp))

    assert out.read_bytes() == b"raw gguf"
    assert "type=Q8_0" in (tmp_path / "model-Q8_0.gguf.quantize.log").read_text()


def test_quantize_failure_surfaces_log_tail(tmp_path):
    root = tmp_path / "llama.cpp"
    bin_dir = root / "build" / "bin"
    os.makedirs(bin_dir)
    quant = bin_dir / "llama-quantize"
    quant.write_text('#!/bin/sh\necho "tensor block corrupt"\nexit 3\n')
    quant.chmod(0o755)
    src = tmp_path / "in.gguf"
    src.write_bytes(b"raw gguf")
    out = tmp_path / "out.gguf"

    with pytest.raises(convert_mod.ExportError) as exc:
        convert_mod.quantize(str(src), str(out), "Q8_0",
                             llama_cpp_dir=str(root))
    message = str(exc.value)
    assert "Command failed (3)" in message
    assert "tensor block corrupt" in message       # log tail is user-facing
    assert str(out) + ".quantize.log" in message
    assert "tensor block corrupt" in (tmp_path / "out.gguf.quantize.log").read_text()


def test_check_architecture_rejects_missing_architectures(tmp_path):
    write_json(str(tmp_path / "staged" / "config.json"), {"architectures": []})
    with pytest.raises(convert_mod.ExportError, match="no architectures"):
        convert_mod.check_architecture(str(tmp_path / "staged"), "unused.py")


# -- D4 calculator: HF-recipe branch and teacher memory ------------------------

# Same hand-computed tiny shape test_calculator.py pins: layers=2, hidden=64,
# heads=4, kv_heads=2, ffn=128, vocab=256, tied.
#   param_count  = 256*64 + 2*(12,288 + 24,576 + 128) + 64 = 90,432
#   linear_count = 2*(12,288 + 24,576)                     = 73,728
TINY_PARAMS = 90_432
TINY_LINEAR = 73_728


def tiny_spec() -> ArchitectureSpec:
    return ArchitectureSpec(layers=2, hidden=64, heads=4, kv_heads=2,
                            ffn=128, vocab=256, tie_embeddings=True)


def test_spec_from_recipe_hf_maps_through_autoconfig(monkeypatch):
    hf_config = SimpleNamespace(
        model_type="llama", num_hidden_layers=2, hidden_size=64,
        num_attention_heads=4, num_key_value_heads=2, intermediate_size=128,
        vocab_size=256, tie_word_embeddings=True)
    seen = {}

    class AutoConfig:
        @staticmethod
        def from_pretrained(model_id):
            seen["id"] = model_id
            return hf_config

    transformers_stub = types.ModuleType("transformers")
    transformers_stub.AutoConfig = AutoConfig
    monkeypatch.setitem(sys.modules, "transformers", transformers_stub)

    recipe = OmegaConf.create(
        {"model": {"scratch": None, "hf": {"id": "org/tiny"}}})
    spec = spec_from_recipe(recipe)

    assert seen["id"] == "org/tiny"
    assert spec == tiny_spec()
    assert spec.param_count() == TINY_PARAMS


def test_spec_from_recipe_requires_scratch_or_hf():
    recipe = OmegaConf.create({"model": {"scratch": None, "hf": None}})
    with pytest.raises(UnknownArchitectureError,
                       match="neither model.scratch nor model.hf"):
        spec_from_recipe(recipe)


def test_optimizer_bytes_unknown_optimizer_fails_loudly():
    recipe = OmegaConf.create({
        "peft": {"enabled": False},
        "model": {"trainable_filter": "all"},
        "memory": {"offload": "none"},
        "optimizer": {"name": "sgd"},
    })
    with pytest.raises(ValueError, match="No optimizer memory model.*sgd"):
        optimizer_bytes(tiny_spec(), recipe)


def test_teacher_weights_bf16():
    assert teacher_weights_bytes(tiny_spec(), "bf16") == 2 * TINY_PARAMS


def test_teacher_weights_nf4_double_quant():
    # NF4+dq linears at 0.5 + 1/64 + 4/(64*256) B/param, bf16 for the rest:
    #   linear = int(73,728 * 0.515869140625) = 38,034
    #   rest   = (90,432 - 73,728) * 2        = 33,408
    assert teacher_weights_bytes(tiny_spec(), "nf4") == 38_034 + 33_408
    assert teacher_weights_bytes(tiny_spec(), "nf4") < 2 * TINY_PARAMS


def test_teacher_forward_bytes_hand_computed():
    # b=2, s=16 (tokens=32), hidden=64, kv_dim=32, ffn=128, vocab=256:
    #   hidden_stream = 32*2*64                       =  4,096
    #   layer_working = 32*2*(3*64 + 2*32 + 3*128)    = 40,960
    #   logits        = 32*256*(2+4)                  = 49,152
    assert teacher_forward_bytes(tiny_spec(), micro_batch=2, seq_len=16) == \
        4_096 + 40_960 + 49_152


# -- common.smoke version probes ------------------------------------------------

def test_pkg_version_reports_installed_distribution():
    from importlib.metadata import version

    assert smoke.pkg_version("pytest") == version("pytest")


def test_pkg_version_unknown_for_missing_distribution():
    assert smoke.pkg_version("no-such-distribution-xyzzy") == "unknown"


def test_driver_version_queries_nvml(monkeypatch):
    # nvmlSystemGetDriverVersion raises unless nvmlInit ran first, so the
    # probe's init-before-query ordering is asserted, not just the value.
    stub = types.ModuleType("pynvml")
    state = {"init": False}
    stub.nvmlInit = lambda: state.update(init=True)

    def get_driver_version():
        if not state["init"]:
            raise RuntimeError("NVML not initialized")
        return "580.65.06"

    stub.nvmlSystemGetDriverVersion = get_driver_version
    monkeypatch.setitem(sys.modules, "pynvml", stub)
    assert smoke.driver_version() == "580.65.06"


def test_driver_version_unknown_when_nvml_unavailable(monkeypatch):
    stub = types.ModuleType("pynvml")

    def failing_init():
        raise RuntimeError("driver not loaded")

    stub.nvmlInit = failing_init
    monkeypatch.setitem(sys.modules, "pynvml", stub)
    assert smoke.driver_version() == "unknown"


# -- common.provenance git_sha ---------------------------------------------------

def make_git_repo(tmp_path):
    """A tmp git repo with one commit; returns (path, HEAD sha)."""
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(
        ["git", "-C", str(repo), "-c", "user.email=t@example.com",
         "-c", "user.name=t", "commit", "--allow-empty", "-q", "-m", "seed"],
        check=True)
    head = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True).stdout.strip()
    return repo, head


def test_git_sha_repo_root_override(tmp_path):
    repo, head = make_git_repo(tmp_path)
    assert provenance.git_sha(repo_root=str(repo)) == head
    # The override must actually move the git cwd off this checkout.
    assert provenance.git_sha() != head


def test_git_sha_returns_unknown_outside_a_repo(tmp_path):
    assert provenance.git_sha(repo_root=str(tmp_path)) == "unknown"


def test_git_sha_strict_reraises(tmp_path):
    with pytest.raises(subprocess.CalledProcessError):
        provenance.git_sha(repo_root=str(tmp_path), strict=True)
