"""Tests for the exporter's pure-logic paths.

Staging's adapter merge needs torch/peft and is exercised on the workstation
(and against stub modules in test_export_common_branches.py). Everything
else runs here in the CPU sandbox: adapter detection, staging of full models
including shard consolidation, output naming, the base-identity preflight,
the hf export manifest and its contract.
"""

from __future__ import annotations

import json
import os
import struct

import pytest

from training.export import consolidate as consolidate_mod
from training.export import manifest as manifest_mod
from training.export import merge as merge_mod
from training.export.errors import ExportError


def write_json(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f)


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
    assert (
        merge_mod.derive_output_name(str(ckpt / "model")) == "tiny-e2e-step_000000010"
    )


def test_derive_output_name_from_trl_layout(tmp_path):
    ckpt = tmp_path / "runs" / "ft3-straight" / "checkpoints" / "checkpoint-60"
    os.makedirs(ckpt)
    assert merge_mod.derive_output_name(str(ckpt)) == "ft3-straight-checkpoint-60"


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


def test_stage_model_consolidates_sharded_copy(tmp_path):
    model, _expected = make_sharded_model(tmp_path)
    write_json(str(model / "config.json"), {"architectures": ["LlamaForCausalLM"]})
    write_json(str(model / "generation_config.json"), {"do_sample": False})
    staging = tmp_path / "staging"
    info = merge_mod.stage_model(str(model), str(staging))
    assert info.adapter_path is None
    assert sorted(os.listdir(staging)) == [
        "config.json",
        "generation_config.json",
        "model.safetensors",
    ]
    header, _ = consolidate_mod.read_header(str(staging / "model.safetensors"))
    assert sorted(k for k in header if k != "__metadata__") == [
        "embed.w",
        "layers.0.w",
        "layers.1.w",
    ]


def test_stage_model_renames_a_single_file_under_another_name(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    write_json(str(model / "config.json"), {})
    t = np.zeros((2,), dtype=np.float32)
    safetensors_numpy.save_file({"w": t}, str(model / "weights.safetensors"))
    staging = tmp_path / "staging"
    merge_mod.stage_model(str(model), str(staging))
    assert sorted(os.listdir(staging)) == ["config.json", "model.safetensors"]


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


# -- F6 base-identity preflight ----------------------------------------------

from training.export import preflight as preflight_mod  # noqa: E402


def make_local_base(tmp_path, config=None):
    base = tmp_path / "base"
    write_json(
        str(base / "config.json"), config or {"architectures": ["LlamaForCausalLM"]}
    )
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
    assert identity.config_sha256 == manifest_mod.sha256_file(str(base / "config.json"))


def test_resolve_base_identity_rejects_configless_dir(tmp_path):
    empty = tmp_path / "empty"
    os.makedirs(empty)
    with pytest.raises(ExportError, match="config.json"):
        preflight_mod.resolve_base_identity(str(empty))


def identity_for(tmp_path, revision="b" * 40):
    base = make_local_base(tmp_path)
    identity = preflight_mod.resolve_base_identity(str(base))
    identity.revision = revision
    return identity


def test_check_base_identity_match(tmp_path):
    identity = identity_for(tmp_path)
    meta = {
        "base_revision": identity.revision,
        "base_config_sha256": identity.config_sha256,
    }
    assert preflight_mod.check_base_identity(meta, identity) == []


def test_check_base_identity_mismatch_refuses(tmp_path):
    identity = identity_for(tmp_path)
    meta = {"base_revision": "c" * 40, "base_config_sha256": identity.config_sha256}
    with pytest.raises(ExportError, match="allow-base-drift"):
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
    write_json(str(adapter / "adapter_config.json"), {"base_model_name_or_path": base})
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
    with pytest.raises(ExportError, match="adapter_only"):
        merge_mod.check_meta_consistency({"adapter_only": False}, str(adapter))
    with pytest.raises(ExportError, match="adapter_only"):
        merge_mod.check_meta_consistency({"adapter_only": True}, None)


def test_meta_consistency_rejects_base_disagreement(tmp_path):
    adapter = make_adapter(tmp_path, base="org/other")
    meta = {"adapter_only": True, "base_model": "org/base"}
    with pytest.raises(ExportError, match="disagreement"):
        merge_mod.check_meta_consistency(meta, str(adapter))


def test_meta_consistency_accepts_a_staged_id_for_its_directory(tmp_path, monkeypatch):
    """The trainer loads the base from the staged directory (KB-79), so the
    adapter records that path while meta.json keeps the pinned id; the two
    spellings name one base. The workstation full gate failed here at 39b35e9
    after ten real QLoRA steps."""
    from training.models import staged

    root = tmp_path / "models" / "hf"
    base_dir = root / "base"
    base_dir.mkdir(parents=True)
    write_json(str(base_dir / "config.json"), {"model_type": "x"})
    monkeypatch.setattr(staged, "STAGED_ROOTS", (str(root),))
    adapter = make_adapter(tmp_path, base=str(base_dir))
    meta = {"adapter_only": True, "base_model": "org/base"}
    merge_mod.check_meta_consistency(meta, str(adapter))
    with pytest.raises(ExportError, match="disagreement"):
        merge_mod.check_meta_consistency(
            {"adapter_only": True, "base_model": "org/other"}, str(adapter)
        )


def test_stage_model_runs_meta_cross_check(tmp_path):
    ckpt = make_full_checkpoint(tmp_path)
    with pytest.raises(ExportError, match="adapter_only"):
        merge_mod.stage_model(
            str(ckpt), str(tmp_path / "staging"), meta={"adapter_only": True}
        )


# -- F6 manifest fields --------------------------------------------------------


# -- consolidate ----------------------------------------------------------------

safetensors_numpy = pytest.importorskip("safetensors.numpy")
np = pytest.importorskip("numpy")


def write_bf16_shard(path, name, values: bytes, shape):
    """A shard written by hand at the container level: bf16 is not a numpy
    dtype, so this is the only way to produce one without torch, and it is
    the case the consolidator exists for."""
    header = {
        "__metadata__": {"format": "pt"},
        name: {"dtype": "BF16", "shape": shape, "data_offsets": [0, len(values)]},
    }
    raw = json.dumps(header).encode()
    raw += b" " * ((-len(raw)) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(raw)) + raw + values)


def load_f32(path, names):
    """The reference reader, one tensor at a time: numpy has no bf16, so a
    whole-file load of a mixed checkpoint raises before reaching the f32
    tensors this asserts on."""
    from safetensors import safe_open

    with safe_open(path, framework="np") as f:
        return {name: f.get_tensor(name) for name in names}


def make_sharded_model(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    a = np.arange(6, dtype=np.float32).reshape(2, 3)
    b = np.full((4,), 2.5, dtype=np.float32)
    safetensors_numpy.save_file(
        {"layers.1.w": a},
        str(model / "model-00001-of-00003.safetensors"),
        metadata={"format": "pt"},
    )
    safetensors_numpy.save_file(
        {"embed.w": b},
        str(model / "model-00002-of-00003.safetensors"),
        metadata={"format": "pt"},
    )
    bf16_bytes = bytes(range(8))  # four bf16 values, content irrelevant
    write_bf16_shard(
        str(model / "model-00003-of-00003.safetensors"),
        "layers.0.w",
        bf16_bytes,
        [2, 2],
    )
    write_json(
        str(model / "model.safetensors.index.json"),
        {
            "metadata": {"total_size": 0},
            "weight_map": {
                "layers.1.w": "model-00001-of-00003.safetensors",
                "embed.w": "model-00002-of-00003.safetensors",
                "layers.0.w": "model-00003-of-00003.safetensors",
            },
        },
    )
    return model, {"layers.1.w": a, "embed.w": b, "layers.0.w": bf16_bytes}


class TestConsolidate:
    def test_joins_shards_into_one_file_in_sorted_order(self, tmp_path):
        model, expected = make_sharded_model(tmp_path)
        out = tmp_path / "model.safetensors"
        info = consolidate_mod.consolidate_shards(str(model), str(out))
        assert info["tensors"] == 3 and info["shards"] == 3
        assert info["size_bytes"] == out.stat().st_size
        header, data_start = consolidate_mod.read_header(str(out))
        assert header["__metadata__"] == {"format": "pt"}
        names = [k for k in header if k != "__metadata__"]
        assert names == sorted(names) == ["embed.w", "layers.0.w", "layers.1.w"]
        # offsets are contiguous from zero in that order
        cursor = 0
        for name in names:
            start, end = header[name]["data_offsets"]
            assert start == cursor
            cursor = end
        # the f32 tensors read back through the reference reader unchanged
        loaded = load_f32(str(out), ["embed.w", "layers.1.w"])
        assert np.array_equal(loaded["embed.w"], expected["embed.w"])
        assert np.array_equal(loaded["layers.1.w"], expected["layers.1.w"])
        # the bf16 tensor's bytes are carried verbatim
        start, end = header["layers.0.w"]["data_offsets"]
        assert header["layers.0.w"]["dtype"] == "BF16"
        with open(out, "rb") as f:
            f.seek(data_start + start)
            assert f.read(end - start) == expected["layers.0.w"]
        assert not (tmp_path / "model.safetensors.tmp").exists()

    def test_hash_is_independent_of_the_shard_split(self, tmp_path):
        model, _ = make_sharded_model(tmp_path)
        one = tmp_path / "one.safetensors"
        consolidate_mod.consolidate_shards(str(model), str(one))
        # re-shard differently: one shard holding everything, no index
        other = tmp_path / "other"
        other.mkdir()
        loaded = load_f32(str(one), ["embed.w", "layers.1.w"])
        safetensors_numpy.save_file(
            {"embed.w": loaded["embed.w"], "layers.1.w": loaded["layers.1.w"]},
            str(other / "a.safetensors"),
            metadata={"format": "pt"},
        )
        header, data_start = consolidate_mod.read_header(str(one))
        start, end = header["layers.0.w"]["data_offsets"]
        with open(one, "rb") as f:
            f.seek(data_start + start)
            write_bf16_shard(
                str(other / "b.safetensors"), "layers.0.w", f.read(end - start), [2, 2]
            )
        two = tmp_path / "two.safetensors"
        consolidate_mod.consolidate_shards(str(other), str(two))
        assert manifest_mod.sha256_file(str(one)) == manifest_mod.sha256_file(str(two))

    def test_index_selects_the_shards(self, tmp_path):
        model, _ = make_sharded_model(tmp_path)
        stray = np.zeros((2,), dtype=np.float32)
        safetensors_numpy.save_file({"stray": stray}, str(model / "stray.safetensors"))
        out = tmp_path / "model.safetensors"
        consolidate_mod.consolidate_shards(str(model), str(out))
        header, _ = consolidate_mod.read_header(str(out))
        assert "stray" not in header

    def test_duplicate_tensor_refused(self, tmp_path):
        model = tmp_path / "model"
        model.mkdir()
        t = np.zeros((2,), dtype=np.float32)
        safetensors_numpy.save_file({"w": t}, str(model / "a.safetensors"))
        safetensors_numpy.save_file({"w": t}, str(model / "b.safetensors"))
        with pytest.raises(ExportError, match="appears in two shards"):
            consolidate_mod.consolidate_shards(str(model), str(tmp_path / "out"))

    def test_index_naming_a_missing_shard_refused(self, tmp_path):
        model, _ = make_sharded_model(tmp_path)
        os.remove(model / "model-00002-of-00003.safetensors")
        with pytest.raises(ExportError, match="absent"):
            consolidate_mod.consolidate_shards(str(model), str(tmp_path / "out"))

    def test_truncated_shard_refused_and_leaves_no_output(self, tmp_path):
        model, _ = make_sharded_model(tmp_path)
        shard = model / "model-00002-of-00003.safetensors"
        shard.write_bytes(shard.read_bytes()[:-4])
        out = tmp_path / "model.safetensors"
        with pytest.raises(ExportError, match="ended before"):
            consolidate_mod.consolidate_shards(str(model), str(out))
        assert not out.exists()

    def test_not_a_safetensors_file_refused(self, tmp_path):
        model = tmp_path / "model"
        model.mkdir()
        (model / "a.safetensors").write_bytes(b"weights")
        with pytest.raises(ExportError, match="Not a safetensors file"):
            consolidate_mod.consolidate_shards(str(model), str(tmp_path / "out"))

    def test_no_weights_refused(self, tmp_path):
        with pytest.raises(ExportError, match="No \\*.safetensors"):
            consolidate_mod.shard_paths(str(tmp_path))


class TestSingleWeightsFile:
    def test_accepts_the_one_file(self, tmp_path):
        (tmp_path / "model.safetensors").write_bytes(b"x")
        (tmp_path / "config.json").write_text("{}")
        assert consolidate_mod.check_single_weights_file(str(tmp_path)) == str(
            tmp_path / "model.safetensors"
        )

    @pytest.mark.parametrize(
        "names",
        [
            [],
            ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"],
            ["model.safetensors", "model.safetensors.index.json"],
            ["weights.safetensors"],
        ],
    )
    def test_refuses_anything_else(self, tmp_path, names):
        for name in names:
            (tmp_path / name).write_bytes(b"x")
        with pytest.raises(ExportError, match="exactly one model.safetensors"):
            consolidate_mod.check_single_weights_file(str(tmp_path))


# -- the hf export manifest ------------------------------------------------------

from common.reports import validate_report  # noqa: E402

HF_CONTRACT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "schemas",
    "hf_export.manifest.json",
)


@pytest.fixture(scope="module")
def hf_contract() -> dict:
    with open(HF_CONTRACT, encoding="utf-8") as f:
        return json.load(f)


def make_hf_export(tmp_path, name="tiny", weights=b"weights"):
    """An exported directory as the copy path writes it, plus the manifest."""
    ckpt = make_full_checkpoint(tmp_path)
    out_dir = tmp_path / "trained" / name
    hf = out_dir / "hf"
    os.makedirs(hf)
    write_json(str(hf / "config.json"), {"architectures": ["Gemma4ForCausalLM"]})
    (hf / "model.safetensors").write_bytes(weights)
    info = merge_mod.StagingInfo(
        model_dir=str(ckpt / "model"),
        staging_dir=str(hf),
        adapter_path=None,
        base_model=None,
        merge_method=None,
        tokenizer_source=None,
    )
    identity = preflight_mod.resolve_base_identity(str(ckpt / "model"))
    manifest = manifest_mod.build_hf_manifest(
        checkpoint=str(ckpt),
        hf_dir=str(hf),
        staging_info=info,
        base_identity=identity,
        mode="copy",
        created="T0",
    )
    manifest_mod.write_hf_manifest(manifest, str(out_dir))
    return hf, manifest


class TestHfManifestContract:
    def test_writer_emits_exactly_the_required_keys(self, tmp_path, hf_contract):
        _hf, manifest = make_hf_export(tmp_path)
        assert sorted(manifest) == sorted(hf_contract["required"])
        assert hf_contract["optional"] == []

    def test_values_the_contract_allows(self, tmp_path, hf_contract):
        _hf, manifest = make_hf_export(tmp_path)
        for key, allowed in hf_contract["values"].items():
            assert manifest[key] in allowed, key

    def test_envelope_conforms(self, tmp_path):
        _hf, manifest = make_hf_export(tmp_path)
        validate_report(manifest)

    def test_model_hash_is_the_sha256_of_the_written_weights(self, tmp_path):
        hf, manifest = make_hf_export(tmp_path)
        assert manifest["model_hash"] == manifest_mod.sha256_file(
            str(hf / "model.safetensors")
        )
        assert manifest["size_bytes"] == len(b"weights")
        assert manifest["weights_file"] == "model.safetensors"
        assert manifest["hf_dir"] == str(hf)
        assert manifest["checkpoint_meta"] == {"step": 10, "snapshot_hash": "abc"}
        assert manifest["adapter"] is None and manifest["base_drift"] == []
        assert manifest["base"]["revision"] is None
        assert manifest["base"]["config_sha256"] == manifest_mod.sha256_file(
            str(manifest["source_checkpoint"] + "/model/config.json")
        )
        assert set(manifest["tool_versions"]) == {"torch", "transformers", "peft"}

    def test_reproducible_for_a_fixed_stamp(self, tmp_path):
        hf, first = make_hf_export(tmp_path)
        second = manifest_mod.load_hf_manifest(str(hf))
        rebuilt = manifest_mod.build_hf_manifest(
            checkpoint=first["source_checkpoint"],
            hf_dir=str(hf),
            staging_info=merge_mod.StagingInfo(
                model_dir=first["source_checkpoint"] + "/model",
                staging_dir=str(hf),
                adapter_path=None,
                base_model=None,
                merge_method=None,
                tokenizer_source=None,
            ),
            base_identity=preflight_mod.resolve_base_identity(
                first["source_checkpoint"] + "/model"
            ),
            mode="copy",
            created="T0",
        )
        assert first == second == rebuilt

    def test_merge_mode_records_adapter_and_drift(self, tmp_path):
        ckpt = make_full_checkpoint(tmp_path)
        adapter = make_adapter(tmp_path / "a")
        hf = tmp_path / "trained" / "tuned" / "hf"
        os.makedirs(hf)
        (hf / "model.safetensors").write_bytes(b"merged")
        info = merge_mod.StagingInfo(
            model_dir=str(ckpt / "model"),
            staging_dir=str(hf),
            adapter_path=str(adapter),
            base_model="org/base",
            merge_method="peft.merge_and_unload",
            tokenizer_source=None,
        )
        identity = preflight_mod.BaseIdentity(
            model="org/base",
            revision="b" * 40,
            config_sha256="f" * 64,
            config_path="/cache/config.json",
        )
        manifest = manifest_mod.build_hf_manifest(
            checkpoint=str(ckpt),
            hf_dir=str(hf),
            staging_info=info,
            base_identity=identity,
            mode="merge",
            base_drift=["base_revision: trained against x, export resolved y"],
            created="T0",
        )
        assert manifest["mode"] == "merge"
        assert manifest["base"] == {
            "model": "org/base",
            "revision": "b" * 40,
            "config_sha256": "f" * 64,
        }
        assert manifest["adapter"] == {
            "path": str(adapter),
            "sha256": manifest_mod.sha256_file(
                str(adapter / "adapter_model.safetensors")
            ),
            "merge_method": "peft.merge_and_unload",
        }
        assert manifest["base_drift"] == [
            "base_revision: trained against x, export resolved y"
        ]


class TestLoadHfManifest:
    def test_round_trip(self, tmp_path):
        hf, manifest = make_hf_export(tmp_path)
        assert manifest_mod.load_hf_manifest(str(hf)) == manifest
        assert manifest_mod.hf_manifest_path(str(hf.parent)) == str(
            hf.parent / "manifest.json"
        )

    def test_missing_manifest_refused(self, tmp_path):
        hf = tmp_path / "trained" / "x" / "hf"
        os.makedirs(hf)
        with pytest.raises(ExportError, match="No export manifest"):
            manifest_mod.load_hf_manifest(str(hf))

    def test_manifest_for_another_directory_refused(self, tmp_path):
        hf, _manifest = make_hf_export(tmp_path)
        other = tmp_path / "trained" / "other" / "hf"
        os.makedirs(other)
        os.rename(hf.parent / "manifest.json", other.parent / "manifest.json")
        with pytest.raises(ExportError, match="describes"):
            manifest_mod.load_hf_manifest(str(other))

    def test_foreign_artifact_refused(self, tmp_path):
        hf, _manifest = make_hf_export(tmp_path)
        write_json(str(hf.parent / "manifest.json"), {"artifact": "probe"})
        with pytest.raises(ExportError, match="Not an hf export manifest"):
            manifest_mod.load_hf_manifest(str(hf))
