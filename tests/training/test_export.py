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


def test_supported_architectures_scans_conversion_modules(tmp_path):
    # Current llama.cpp pins register models in conversion/*.py next to a
    # thin wrapper script (the pinned SHA registers Gemma4ForCausalLM there).
    wrapper = tmp_path / "convert_hf_to_gguf.py"
    wrapper.write_text("from conversion import ModelBase\n")
    conversion = tmp_path / "conversion"
    conversion.mkdir()
    (conversion / "gemma.py").write_text(
        '@ModelBase.register("Gemma4ForConditionalGeneration", "Gemma4ForCausalLM")\n'
        "class Gemma4Model(TextModel):\n    pass\n")
    (conversion / "qwen.py").write_text(
        '@ModelBase.register("Qwen3ForCausalLM", "Qwen3Model")\n'
        "class Qwen3Model(TextModel):\n    pass\n")
    archs = convert_mod.supported_architectures(str(wrapper))
    assert archs == {"Gemma4ForConditionalGeneration", "Gemma4ForCausalLM",
                     "Qwen3ForCausalLM", "Qwen3Model"}


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


# -- F6 base-identity preflight ----------------------------------------------

from training.export import preflight as preflight_mod  # noqa: E402


def make_local_base(tmp_path, config=None):
    base = tmp_path / "base"
    write_json(str(base / "config.json"),
               config or {"architectures": ["LlamaForCausalLM"]})
    return base


def test_commit_from_cache_path():
    commit = "b" * 40
    path = f"/root/.cache/huggingface/models--org--m/snapshots/{commit}/config.json"
    assert preflight_mod.commit_from_cache_path(path) == commit
    assert preflight_mod.commit_from_cache_path("/some/local/config.json") is None


def test_resolve_base_identity_local_dir(tmp_path):
    base = make_local_base(tmp_path)
    identity = preflight_mod.resolve_base_identity(str(base))
    assert identity.revision is None
    assert identity.config_sha256 == manifest_mod.sha256_file(
        str(base / "config.json"))


def test_resolve_base_identity_rejects_configless_dir(tmp_path):
    empty = tmp_path / "empty"
    os.makedirs(empty)
    with pytest.raises(convert_mod.ExportError, match="config.json"):
        preflight_mod.resolve_base_identity(str(empty))


def identity_for(tmp_path, revision="b" * 40):
    base = make_local_base(tmp_path)
    identity = preflight_mod.resolve_base_identity(str(base))
    identity.revision = revision
    return identity


def test_check_base_identity_match(tmp_path):
    identity = identity_for(tmp_path)
    meta = {"base_revision": identity.revision,
            "base_config_sha256": identity.config_sha256}
    assert preflight_mod.check_base_identity(meta, identity) == []


def test_check_base_identity_mismatch_refuses(tmp_path):
    identity = identity_for(tmp_path)
    meta = {"base_revision": "c" * 40,
            "base_config_sha256": identity.config_sha256}
    with pytest.raises(convert_mod.ExportError, match="allow-base-drift"):
        preflight_mod.check_base_identity(meta, identity)


def test_check_base_identity_mismatch_allowed_returns_notes(tmp_path):
    identity = identity_for(tmp_path)
    meta = {"base_revision": "c" * 40, "base_config_sha256": "0" * 64}
    notes = preflight_mod.check_base_identity(meta, identity, allow_drift=True)
    assert len(notes) == 2
    assert any("base_revision" in n for n in notes)
    assert any("base_config_sha256" in n for n in notes)


def test_check_base_identity_unrecorded_fields_never_fatal(tmp_path):
    identity = identity_for(tmp_path)
    notes = preflight_mod.check_base_identity({"step": 10}, identity)
    assert all("not recorded" in n for n in notes)
    assert preflight_mod.check_base_identity(None, identity)  # no meta at all


# -- F6 meta.json cross-check in staging --------------------------------------

def make_adapter(tmp_path, base="org/base"):
    adapter = tmp_path / "ckpt"
    write_json(str(adapter / "adapter_config.json"),
               {"base_model_name_or_path": base})
    (adapter / "adapter_model.safetensors").write_bytes(b"lora")
    return adapter


def test_meta_consistency_accepts_agreement(tmp_path):
    adapter = make_adapter(tmp_path)
    meta = {"adapter_only": True, "base_model": "org/base"}
    merge_mod.check_meta_consistency(meta, str(adapter))
    merge_mod.check_meta_consistency({"adapter_only": False}, None)
    merge_mod.check_meta_consistency(None, str(adapter))  # plain HF dir


def test_meta_consistency_rejects_type_mismatch(tmp_path):
    adapter = make_adapter(tmp_path)
    with pytest.raises(convert_mod.ExportError, match="adapter_only"):
        merge_mod.check_meta_consistency({"adapter_only": False}, str(adapter))
    with pytest.raises(convert_mod.ExportError, match="adapter_only"):
        merge_mod.check_meta_consistency({"adapter_only": True}, None)


def test_meta_consistency_rejects_base_disagreement(tmp_path):
    adapter = make_adapter(tmp_path, base="org/other")
    meta = {"adapter_only": True, "base_model": "org/base"}
    with pytest.raises(convert_mod.ExportError, match="disagreement"):
        merge_mod.check_meta_consistency(meta, str(adapter))


def test_stage_model_runs_meta_cross_check(tmp_path):
    ckpt = make_full_checkpoint(tmp_path)
    with pytest.raises(convert_mod.ExportError, match="adapter_only"):
        merge_mod.stage_model(str(ckpt), str(tmp_path / "staging"),
                              meta={"adapter_only": True})


# -- F6 manifest fields --------------------------------------------------------

def test_manifest_records_base_identity_and_drift(tmp_path):
    ckpt = make_full_checkpoint(tmp_path)
    adapter = make_adapter(tmp_path / "a")
    gguf = tmp_path / "tiny-Q8_0.gguf"
    gguf.write_bytes(b"gguf bytes")
    info = merge_mod.StagingInfo(
        model_dir=str(ckpt / "model"), staging_dir="s",
        adapter_path=str(adapter), base_model="org/base",
        merge_method="peft.merge_and_unload", tokenizer_source=None)
    identity = preflight_mod.BaseIdentity(
        model="org/base", revision="b" * 40, config_sha256="f" * 64,
        config_path="/cache/config.json")
    manifest = manifest_mod.build_manifest(
        checkpoint=str(ckpt), architecture="LlamaForCausalLM", quant_type="Q8_0",
        outtype="bf16", llama_cpp_commit="a" * 40, gguf_path=str(gguf),
        staging_info=info, base_identity=identity,
        base_drift=["base_revision: trained against x, export resolved y"],
        created="T0")
    assert manifest["adapter"]["base_revision"] == "b" * 40
    assert manifest["adapter"]["base_config_sha256"] == "f" * 64
    assert manifest["base_drift"] == [
        "base_revision: trained against x, export resolved y"]


def test_manifest_base_fields_default_none(tmp_path):
    ckpt = make_full_checkpoint(tmp_path)
    gguf = tmp_path / "tiny-Q8_0.gguf"
    gguf.write_bytes(b"gguf bytes")
    info = merge_mod.StagingInfo(
        model_dir=str(ckpt / "model"), staging_dir="s", adapter_path=None,
        base_model=None, merge_method=None, tokenizer_source=None)
    manifest = manifest_mod.build_manifest(
        checkpoint=str(ckpt), architecture="LlamaForCausalLM", quant_type="Q8_0",
        outtype="bf16", llama_cpp_commit="a" * 40, gguf_path=str(gguf),
        staging_info=info, created="T0")
    assert manifest["adapter"] is None
    assert manifest["base_drift"] == []
