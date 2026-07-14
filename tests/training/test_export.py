"""Tests for the D7 GGUF export pipeline's pure-logic paths.

Conversion and quantization need the pinned llama.cpp toolchain (export image)
and are covered by the host acceptance run; staging's adapter merge needs
torch/peft and is exercised there too. Everything else — pin format, converter
architecture parsing, adapter detection, staging of full models, output naming,
and manifest reproducibility — runs here in the CPU sandbox.
"""

from __future__ import annotations

import json
import os

import pytest

from training.export import convert as convert_mod
from training.export import manifest as manifest_mod
from training.export import merge as merge_mod

FAKE_CONVERTER = '''
class LlamaModel(TextModel):
    pass

@ModelBase.register("LlamaForCausalLM", "Llama4ForCausalLM")
class Llama(TextModel):
    pass

@ModelBase.register("Qwen2ForCausalLM")
class Qwen2(TextModel):
    pass
'''


def write_json(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f)


@pytest.fixture
def fake_converter(tmp_path):
    path = tmp_path / "convert_hf_to_gguf.py"
    path.write_text(FAKE_CONVERTER)
    return str(path)


def test_pin_is_a_full_sha():
    sha = convert_mod.read_pin()
    assert len(sha) == 40
    assert convert_mod.read_pin(convert_mod.PIN_FILE) == sha


def test_read_pin_rejects_garbage(tmp_path):
    bad = tmp_path / "pin.txt"
    bad.write_text("main\n")
    with pytest.raises(convert_mod.ExportError):
        convert_mod.read_pin(str(bad))


def test_supported_architectures_parses_registrations(fake_converter):
    archs = convert_mod.supported_architectures(fake_converter)
    assert archs == {"LlamaForCausalLM", "Llama4ForCausalLM", "Qwen2ForCausalLM"}


def test_supported_architectures_rejects_empty(tmp_path):
    empty = tmp_path / "convert_hf_to_gguf.py"
    empty.write_text("print('no registrations here')\n")
    with pytest.raises(convert_mod.ExportError):
        convert_mod.supported_architectures(str(empty))


def test_check_architecture_accepts_supported(tmp_path, fake_converter):
    write_json(str(tmp_path / "staged" / "config.json"),
               {"architectures": ["LlamaForCausalLM"]})
    arch = convert_mod.check_architecture(str(tmp_path / "staged"), fake_converter)
    assert arch == "LlamaForCausalLM"


def test_check_architecture_rejects_unsupported(tmp_path, fake_converter):
    write_json(str(tmp_path / "staged" / "config.json"),
               {"architectures": ["MambaForCausalLM"]})
    with pytest.raises(convert_mod.ExportError, match="MambaForCausalLM"):
        convert_mod.check_architecture(str(tmp_path / "staged"), fake_converter)


def test_find_adapter_in_model_dir(tmp_path):
    model = tmp_path / "model"
    write_json(str(model / "adapter_config.json"), {"base_model_name_or_path": "b"})
    assert merge_mod.find_adapter(str(model)) == str(model)


def test_find_adapter_sibling_dir(tmp_path):
    ckpt = tmp_path / "step_000000010"
    write_json(str(ckpt / "model" / "config.json"), {})
    write_json(str(ckpt / "adapter" / "adapter_config.json"), {})
    assert merge_mod.find_adapter(str(ckpt / "model")) == str(ckpt / "adapter")


def test_find_adapter_none_for_full_model(tmp_path):
    model = tmp_path / "model"
    write_json(str(model / "config.json"), {})
    assert merge_mod.find_adapter(str(model)) is None


def test_resolve_model_dir_checkpoint_layout(tmp_path):
    ckpt = tmp_path / "step_000000010"
    write_json(str(ckpt / "model" / "config.json"), {})
    assert merge_mod.resolve_model_dir(str(ckpt)) == str(ckpt / "model")


def test_resolve_model_dir_rejects_junk(tmp_path):
    with pytest.raises(FileNotFoundError):
        merge_mod.resolve_model_dir(str(tmp_path))


def test_derive_output_name_from_d6_layout(tmp_path):
    ckpt = tmp_path / "runs" / "tiny-e2e" / "checkpoints" / "step_000000010"
    os.makedirs(ckpt)
    assert merge_mod.derive_output_name(str(ckpt)) == "tiny-e2e-step_000000010"
    assert merge_mod.derive_output_name(str(ckpt / "model")) == \
        "tiny-e2e-step_000000010"


def test_derive_output_name_fallback(tmp_path):
    plain = tmp_path / "SmolLM2-1.7B"
    os.makedirs(plain)
    assert merge_mod.derive_output_name(str(plain)) == "SmolLM2-1.7B"


def make_full_checkpoint(tmp_path, with_tokenizer=False):
    ckpt = tmp_path / "step_000000010"
    model = ckpt / "model"
    write_json(str(model / "config.json"), {"architectures": ["LlamaForCausalLM"]})
    (model / "model.safetensors").write_bytes(b"weights")
    write_json(str(ckpt / "meta.json"), {"step": 10, "snapshot_hash": "abc"})
    if with_tokenizer:
        (model / "tokenizer.json").write_text("{}")
    return ckpt


def test_stage_model_copies_full_model(tmp_path):
    ckpt = make_full_checkpoint(tmp_path, with_tokenizer=True)
    staging = tmp_path / "staging"
    info = merge_mod.stage_model(str(ckpt), str(staging))
    assert info.adapter_path is None and info.merge_method is None
    assert (staging / "config.json").is_file()
    assert (staging / "model.safetensors").is_file()
    assert info.tokenizer_source == str(ckpt / "model")
    assert (staging / "tokenizer.json").is_file()


def test_stage_model_takes_external_tokenizer(tmp_path):
    ckpt = make_full_checkpoint(tmp_path)
    tok = tmp_path / "tok"
    os.makedirs(tok)
    (tok / "tokenizer.json").write_text("{}")
    staging = tmp_path / "staging"
    info = merge_mod.stage_model(str(ckpt), str(staging), tokenizer_dir=str(tok))
    assert info.tokenizer_source == str(tok)
    assert (staging / "tokenizer.json").is_file()


def test_stage_model_reports_missing_tokenizer(tmp_path):
    ckpt = make_full_checkpoint(tmp_path)
    info = merge_mod.stage_model(str(ckpt), str(tmp_path / "staging"))
    assert info.tokenizer_source is None


def test_manifest_is_reproducible(tmp_path):
    ckpt = make_full_checkpoint(tmp_path)
    gguf = tmp_path / "out" / "tiny-Q8_0.gguf"
    os.makedirs(gguf.parent)
    gguf.write_bytes(b"gguf bytes")
    info = merge_mod.StagingInfo(
        model_dir=str(ckpt / "model"), staging_dir="s", adapter_path=None,
        base_model=None, merge_method=None, tokenizer_source=None)
    kwargs = {"checkpoint": str(ckpt), "architecture": "LlamaForCausalLM",
              "quant_type": "Q8_0", "outtype": "bf16",
              "llama_cpp_commit": "a" * 40, "gguf_path": str(gguf),
              "staging_info": info, "created": "T0"}
    first = manifest_mod.build_manifest(**kwargs)
    second = manifest_mod.build_manifest(**kwargs)
    assert first == second
    assert first["gguf"]["sha256"] == manifest_mod.sha256_file(str(gguf))
    assert first["checkpoint_meta"] == {"step": 10, "snapshot_hash": "abc"}
    assert first["quant_type"] == "Q8_0"


def test_manifest_records_adapter_provenance(tmp_path):
    ckpt = make_full_checkpoint(tmp_path)
    adapter = tmp_path / "adapter"
    os.makedirs(adapter)
    (adapter / "adapter_model.safetensors").write_bytes(b"lora")
    gguf = tmp_path / "tiny-Q8_0.gguf"
    gguf.write_bytes(b"gguf bytes")
    info = merge_mod.StagingInfo(
        model_dir=str(ckpt / "model"), staging_dir="s",
        adapter_path=str(adapter), base_model="org/base",
        merge_method="peft.merge_and_unload", tokenizer_source=None)
    manifest = manifest_mod.build_manifest(
        checkpoint=str(ckpt), architecture="LlamaForCausalLM", quant_type="Q8_0",
        outtype="bf16", llama_cpp_commit="a" * 40, gguf_path=str(gguf),
        staging_info=info, created="T0")
    assert manifest["adapter"]["base_model"] == "org/base"
    assert manifest["adapter"]["sha256"] == manifest_mod.sha256_file(
        str(adapter / "adapter_model.safetensors"))


def test_write_manifest_lands_beside_gguf(tmp_path):
    gguf = tmp_path / "tiny-Q8_0.gguf"
    gguf.write_bytes(b"x")
    path = manifest_mod.write_manifest({"schema_version": 1}, str(gguf))
    assert path == str(tmp_path / "tiny-Q8_0.manifest.json")
    with open(path) as f:
        assert json.load(f) == {"schema_version": 1}
