"""Tests for scripts/models/gemma4_header_transcript.sh.

The transcripts this script produces are used to correct
docs/models/gemma-4.md, which rust-harness, training, and interp all build
against. A wrong value does not stay in the transcript, so these tests treat
"the script ran and printed something plausible" as insufficient: every
assertion compares the transcript against a value recomputed from the
ModelSpec that generated the fixture.

Test names carry a d1..d7 marker where they pin a specific confirmed defect,
so `-k defect` selects exactly the regressions.
"""

import importlib.util
import json
import re
import struct
from pathlib import Path

import pytest

from tests.models import specbuild
from tests.models.specbuild import ModelSpec, mutate

SCRIPT = "models/gemma4_header_transcript.sh"
CONTRACT_PATH = (
    Path(__file__).resolve().parent.parent.parent / "schemas" / "gemma4_facts.json"
)

ANSI_RE = re.compile(r"\x1b\[")
STATUS_RE = re.compile(r"^\s*\[([A-Z]+)\]", re.MULTILINE)
ALLOWED_TAGS = {"OK", "WARN", "ERROR", "INFO", "RUN", "SKIP"}

requires_safetensors = pytest.mark.skipif(
    importlib.util.find_spec("safetensors") is None,
    reason="safetensors package required for the differential check",
)


def flat_lines(sections, needle):
    return specbuild.section_matching(sections, needle)


def layer_indices_in_order(lines):
    """Layer indices in the order the MLP section lists them."""
    out = []
    for line in lines:
        m = re.search(r"\.layers\.(\d+)\.", line)
        if m:
            out.append(int(m.group(1)))
    return out


# --------------------------------------------------------------------------
# Round-trip property: transcript values equal spec-derived values
# --------------------------------------------------------------------------


def test_tensor_count_matches_spec(transcribe, model_dir, spec):
    result, sections, _ = transcribe(SCRIPT, str(model_dir))
    body = "\n".join(flat_lines(sections, "safetensors header"))
    expected = len(spec.tensors())
    assert f"tensor count: {expected}" in body, body


def test_every_spec_tensor_appears_with_its_shape(transcribe, model_dir, spec):
    _, sections, out = transcribe(SCRIPT, str(model_dir))
    text = out.read_text()
    for name, shape, _dtype in spec.tensors():
        collapsed = re.sub(r"\.\d+\.", ".{L}.", name)
        assert collapsed in text or name in text, f"{name} missing from transcript"
        assert str(list(shape)) in text, f"shape {list(shape)} for {name} missing"


def test_dtype_tally_matches_spec(transcribe, model_dir, spec):
    _, sections, _ = transcribe(SCRIPT, str(model_dir))
    counted = specbuild.parse_counted(flat_lines(sections, "dtype distribution"))
    assert counted == spec.expected_dtype_tally(), counted


def test_distinct_mlp_widths_match_spec(transcribe, model_dir, spec):
    _, sections, _ = transcribe(SCRIPT, str(model_dir))
    counted = specbuild.parse_counted(flat_lines(sections, "distinct MLP widths"))
    widths = spec.expected_mlp_widths()
    expected = {str(w): widths.count(w) for w in set(widths)}
    assert counted == expected, counted


def test_full_attention_indices_match_spec(transcribe, model_dir, spec):
    _, sections, _ = transcribe(SCRIPT, str(model_dir))
    body = "\n".join(flat_lines(sections, "documented fields"))
    assert f"full_attention indices: {spec.full_attention_indices()}" in body, body
    assert f"layer_types length: {spec.num_layers}" in body, body


def test_attention_shapes_split_by_head_dim(transcribe, model_dir, spec):
    """Sliding layers use head_dim, full layers global_head_dim."""
    _, sections, _ = transcribe(SCRIPT, str(model_dir))
    counted = specbuild.parse_counted(
        flat_lines(sections, "attention projection shapes")
    )
    n_full = len(spec.full_attention_indices())
    n_slide = spec.num_layers - n_full
    q_slide = [spec.num_attention_heads * spec.head_dim, spec.hidden_size]
    q_full = [spec.num_attention_heads * spec.global_head_dim, spec.hidden_size]
    got = {k: v for k, v in counted.items() if "q_proj" in k}
    assert sum(got.values()) == spec.num_layers, got
    assert any(str(q_slide) in k and v == n_slide for k, v in got.items()), got
    assert any(str(q_full) in k and v == n_full for k, v in got.items()), got


def test_zero_dimensional_tensors_are_reported(transcribe, model_dir, spec):
    _, sections, _ = transcribe(SCRIPT, str(model_dir))
    counted = specbuild.parse_counted(flat_lines(sections, "zero-dimensional tensors"))
    expected = sum(1 for _, shape, _ in spec.tensors() if shape == ())
    assert sum(counted.values()) == expected, counted


# --------------------------------------------------------------------------
# Released-variant structures
#
# The fixture defaults (7 layers, period 3, double_wide, kv_heads 1) produce
# no checkpoint that matches a released variant's structure: the final layer
# is sliding, the double-wide-off branch never runs, and kv_heads is never
# above 1. These build the two real shapes at fixture widths, so the
# structures the transcriber will actually meet are exercised.
# --------------------------------------------------------------------------


def variant_shaped_spec(variant: str) -> ModelSpec:
    """A fixture-width spec carrying one released variant's real structure.

    Widths stay tiny; layer count, attention period, KV-shared depth,
    double-wide flag, and KV head count come from schemas/gemma4_facts.json,
    so the fixture cannot drift from the contract.
    """
    contract = json.loads(CONTRACT_PATH.read_text())
    v = contract["variants"][variant]
    indices = v["layer_types_full_attention_indices"]
    return ModelSpec(
        num_layers=v["num_hidden_layers"],
        full_attention_period=indices[0] + 1,
        num_kv_shared_layers=v["num_kv_shared_layers"],
        double_wide=v["use_double_wide_mlp"],
        num_key_value_heads=v["num_key_value_heads"],
    )


@pytest.mark.parametrize("variant", ["E2B", "E4B"])
def test_variant_structure_final_layer_is_full_attention(transcribe, tmp_path, variant):
    """Every released variant ends on a full-attention layer.

    Upstream forces it (docs/models/gemma4-upstream.txt:3983-3987), and the
    contract asserts it for all four. The fixture defaults never produced it,
    so the last block carrying global_head_dim shapes went untranscribed.
    """
    spec = variant_shaped_spec(variant)
    assert spec.full_attention_indices()[-1] == spec.num_layers - 1
    d = specbuild.build_model_dir(tmp_path, spec)
    _, sections, _ = transcribe(SCRIPT, str(d))
    body = "\n".join(flat_lines(sections, "documented fields"))
    assert f"full_attention indices: {spec.full_attention_indices()}" in body, body
    assert "final layer type: full_attention" in body, body


@pytest.mark.parametrize(
    ("variant", "double_wide", "kv_heads"), [("E2B", True, 1), ("E4B", False, 2)]
)
def test_variant_structure_mlp_widths_and_kv_heads(
    transcribe, tmp_path, variant, double_wide, kv_heads
):
    """E2B doubles the MLP on its KV-shared tail; E4B does not, and has 2 KV heads."""
    spec = variant_shaped_spec(variant)
    assert spec.double_wide is double_wide
    assert spec.num_key_value_heads == kv_heads

    d = specbuild.build_model_dir(tmp_path, spec)
    _, sections, _ = transcribe(SCRIPT, str(d))

    counted = specbuild.parse_counted(flat_lines(sections, "distinct MLP widths"))
    widths = spec.expected_mlp_widths()
    assert counted == {str(w): widths.count(w) for w in set(widths)}, counted
    # E4B's flat stack must report exactly one width; E2B's must report two.
    assert len(set(widths)) == (2 if double_wide else 1)

    shapes = specbuild.parse_counted(
        flat_lines(sections, "attention projection shapes")
    )
    k_slide = [kv_heads * spec.head_dim, spec.hidden_size]
    n_full = len(spec.full_attention_indices())
    n_slide = spec.num_layers - n_full
    got = {k: v for k, v in shapes.items() if "k_proj" in k}
    assert sum(got.values()) == spec.num_layers, got
    assert any(str(k_slide) in k and v == n_slide for k, v in got.items()), got


# --------------------------------------------------------------------------
# Defect regressions (D1..D5 from the plan; D7 found against the real weights)
# --------------------------------------------------------------------------


def test_defect_d1_nested_indices_collapse_fully(transcribe, tmp_path):
    """Vision-tower names carry two numeric segments; both must collapse.

    Without /g on the sed, `...layers.0.attn.0...` and `...layers.0.attn.1...`
    stay distinct and the inventory reports two rows of 2 instead of one of 4.
    """
    spec = ModelSpec(include_vision_tower=True)
    d = specbuild.build_model_dir(tmp_path, spec)
    _, sections, _ = transcribe(SCRIPT, str(d))
    counted = specbuild.parse_counted(flat_lines(sections, "tensor inventory"))
    vision = {
        k: v for k, v in counted.items() if "vision_tower" in k and "linear.weight" in k
    }
    assert len(vision) == 1, f"nested indices did not collapse to one row: {vision}"
    assert next(iter(vision.values())) == 4, vision


def test_defect_d2_width_tally_holds_only_numbers(transcribe, tmp_path):
    """A gate_proj-adjacent tensor with no width must not enter the tally.

    The extraction sed passes non-matching lines through unchanged, so a
    zero-dimensional `mlp.gate_proj_scale` lands in the distinct-widths list
    as a whole tensor line.
    """
    spec = ModelSpec(
        extra_tensors=(
            ("model.language_model.layers.0.mlp.gate_proj_scale", (), "F32"),
        )
    )
    d = specbuild.build_model_dir(tmp_path, spec)
    _, sections, _ = transcribe(SCRIPT, str(d))
    counted = specbuild.parse_counted(flat_lines(sections, "distinct MLP widths"))
    bad = [k for k in counted if not k.isdigit()]
    assert not bad, f"non-numeric entries in the width tally: {bad}"


def test_defect_d3_mlp_listing_is_ordered_by_layer_index(transcribe, tmp_path):
    """Per-layer MLP order must be numeric, not lexicographic.

    `sort -t. -k4 -n` assumed the layer index was the fourth dot-field and
    sorted as text, so layer 10 landed between 1 and 2 — which is the order the
    double-wide boundary would then be read off. Twelve layers is the smallest
    count that separates the two orderings.

    This once used a short prefix to also exercise the field-depth assumption.
    It no longer can: the sections are scoped to the documented naming
    convention, so a short-prefix repository is reported by the naming guard
    (d4) and produces no listing at all. `sort_by_layer` extracts the index by
    regex rather than field position, which is the property that replaced it.
    """
    spec = ModelSpec(num_layers=12, num_kv_shared_layers=5)
    d = specbuild.build_model_dir(tmp_path, spec)
    _, sections, _ = transcribe(SCRIPT, str(d))
    order = layer_indices_in_order(flat_lines(sections, "MLP widths per layer"))
    assert order == sorted(order), f"layers listed out of numeric order: {order}"


def test_defect_d4_unrecognized_naming_fails_loudly(transcribe, tmp_path):
    """A repository whose tensors do not match the expected convention must
    not produce a clean, empty, exit-0 transcript.

    The variant name is deliberately one the document does not pin, so no
    blob-hash check runs and the only thing that can fail the script is the
    naming guard itself. Otherwise this passes on an unrelated mismatch.
    """
    spec = ModelSpec(name="gemma-4-XYZ", prefix="model", include_vision_tower=False)
    d = specbuild.build_model_dir(tmp_path, spec)
    result, _, _ = transcribe(SCRIPT, str(d))
    assert result.returncode != 0, (
        "script reported success on a repository it did not recognize:\n"
        f"{result.stdout}"
    )
    assert "[ERROR]" in result.stdout, result.stdout
    combined = result.stdout.lower()
    assert "naming" in combined or "convention" in combined, result.stdout


def test_defect_d7_config_describing_another_model_fails_loudly(transcribe, tmp_path):
    """config.json and the weights are separate downloads; disagreement is loud.

    The real E2B snapshot shipped E4B's config.json verbatim: 42 layers and
    width 2560 declared alongside weights holding 35 layers at width 1536. The
    blob-hash check saw only an unexpected hash, which reads as revision drift
    rather than as the wrong model's file.
    """
    spec = ModelSpec(name="gemma-4-XYZ", num_layers=7)
    d = specbuild.build_model_dir(tmp_path, spec)

    cfg_path = d / "config.json"
    cfg = json.loads(cfg_path.read_text())
    target = cfg["text_config"] if "text_config" in cfg else cfg
    target["num_hidden_layers"] = spec.num_layers + 5
    cfg_path.write_text(json.dumps(cfg))

    result, _, _ = transcribe(SCRIPT, str(d))
    assert result.returncode != 0, (
        f"script accepted a config describing a different model:\n{result.stdout}"
    )
    assert "[ERROR]" in result.stdout, result.stdout
    assert f"config says {spec.num_layers + 5}" in result.stdout, result.stdout
    assert f"weights hold {spec.num_layers}" in result.stdout, result.stdout


def test_config_agreeing_with_weights_does_not_trip_the_guard(transcribe, tmp_path):
    """Negative control: a matched config/weights pair must not fire the guard."""
    spec = ModelSpec(name="gemma-4-XYZ", num_layers=7)
    d = specbuild.build_model_dir(tmp_path, spec)
    result, _, _ = transcribe(SCRIPT, str(d))
    assert "different model than the weights" not in result.stdout, result.stdout
    assert f"Config vs weights: {spec.num_layers} layers, agrees" in result.stdout


def test_recognized_naming_does_not_trip_the_guard(transcribe, model_dir):
    """Negative control: the guard must not fire on a conventional repository.

    A guard that always fires would pass the test above while making every
    real run fail.
    """
    result, _, sections = transcribe(SCRIPT, str(model_dir))
    assert result.returncode == 0, result.stdout
    assert "[ERROR]" not in result.stdout, result.stdout
    assert "no tensor matched" not in result.stdout, result.stdout
    assert "Naming convention:" in result.stdout, result.stdout


def test_defect_d5_cited_doc_paths_exist(repo_root):
    """Every repo-relative doc path cited in either script must exist."""
    missing = []
    for script in ("gemma4_header_transcript.sh", "gguf_header_transcript.sh"):
        text = (repo_root / "scripts" / "models" / script).read_text()
        for path in re.findall(r"\bdocs/[\w/.-]+\.md", text):
            if not (repo_root / path).exists():
                missing.append(f"{script}: {path}")
    assert not missing, f"cited documentation paths do not exist: {missing}"


# --------------------------------------------------------------------------
# Mutation matrix: the transcript must depend on its input
# --------------------------------------------------------------------------


def test_mutation_lm_head_is_detected_and_fails(transcribe, tmp_path):
    clean = specbuild.build_model_dir(tmp_path / "a", ModelSpec())
    dirty = specbuild.build_model_dir(
        tmp_path / "b", mutate(ModelSpec(), include_lm_head=True)
    )
    ok, _, _ = transcribe(SCRIPT, str(clean))
    bad, _, bad_out = transcribe(SCRIPT, str(dirty))
    assert ok.returncode == 0, ok.stdout
    assert bad.returncode != 0, bad.stdout
    assert "lm_head" in bad_out.read_text()


def test_mutation_double_wide_boundary_moves(transcribe, tmp_path):
    base = ModelSpec()
    moved = mutate(base, num_kv_shared_layers=base.num_kv_shared_layers - 1)
    _, a, _ = transcribe(SCRIPT, str(specbuild.build_model_dir(tmp_path / "a", base)))
    _, b, _ = transcribe(SCRIPT, str(specbuild.build_model_dir(tmp_path / "b", moved)))
    wide = 2 * base.intermediate_size
    count_a = specbuild.parse_counted(flat_lines(a, "distinct MLP widths"))
    count_b = specbuild.parse_counted(flat_lines(b, "distinct MLP widths"))
    assert count_a[str(wide)] == base.num_kv_shared_layers, count_a
    assert count_b[str(wide)] == moved.num_kv_shared_layers, count_b
    assert count_a != count_b


def test_mutation_dtype_change_shows_in_tally(transcribe, tmp_path):
    base = ModelSpec()
    other = mutate(base, dtype="F32")
    _, a, _ = transcribe(SCRIPT, str(specbuild.build_model_dir(tmp_path / "a", base)))
    _, b, _ = transcribe(SCRIPT, str(specbuild.build_model_dir(tmp_path / "b", other)))
    ta = specbuild.parse_counted(flat_lines(a, "dtype distribution"))
    tb = specbuild.parse_counted(flat_lines(b, "dtype distribution"))
    assert ta == base.expected_dtype_tally()
    assert tb == other.expected_dtype_tally()
    assert ta != tb


def test_mutation_layer_count_changes_inventory(transcribe, tmp_path):
    base = ModelSpec()
    bigger = mutate(base, num_layers=base.num_layers + 3)
    _, a, _ = transcribe(SCRIPT, str(specbuild.build_model_dir(tmp_path / "a", base)))
    _, b, _ = transcribe(SCRIPT, str(specbuild.build_model_dir(tmp_path / "b", bigger)))
    ca = specbuild.parse_counted(flat_lines(a, "per-layer embedding path"))
    cb = specbuild.parse_counted(flat_lines(b, "per-layer embedding path"))
    assert max(ca.values()) == base.num_layers, ca
    assert max(cb.values()) == bigger.num_layers, cb


def test_mutation_shard_index_presence_is_reported(transcribe, tmp_path):
    d = specbuild.build_model_dir(tmp_path, ModelSpec())
    _, before, _ = transcribe(SCRIPT, str(d))
    assert "absent, as documented" in "\n".join(
        flat_lines(before, "shard index presence")
    )
    (d / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"a": "model-00001.safetensors"}})
    )
    result, after, _ = transcribe(SCRIPT, str(d))
    assert "EXISTS" in "\n".join(flat_lines(after, "shard index presence"))
    assert "[WARN]" in result.stdout


def test_mutation_config_change_flips_blob_hash(transcribe, tmp_path):
    """A doc-pinned variant name gets its config.json blob hash checked."""
    spec = ModelSpec(name="gemma-4-E2B")
    d = specbuild.build_model_dir(tmp_path, spec)
    result, _, out = transcribe(SCRIPT, str(d))
    body = out.read_text()
    assert "expected blob (from doc):" in body
    # The fixture is not the real config, so the hash must not match.
    assert result.returncode != 0
    assert "blob hash differs" in result.stdout


# --------------------------------------------------------------------------
# Robustness
# --------------------------------------------------------------------------


def test_missing_directory_reports_and_fails(transcribe, tmp_path):
    result, _, _ = transcribe(SCRIPT, str(tmp_path / "nope"))
    assert result.returncode != 0
    assert "[ERROR]" in result.stdout
    assert "not found" in result.stdout.lower()


def test_truncated_safetensors_reports_and_fails(transcribe, tmp_path):
    d = specbuild.build_model_dir(tmp_path, ModelSpec())
    raw = (d / "model.safetensors").read_bytes()
    (d / "model.safetensors").write_bytes(raw[: len(raw) // 3])
    result, _, _ = transcribe(SCRIPT, str(d))
    assert result.returncode != 0
    assert "[ERROR]" in result.stdout


def test_lfs_pointer_instead_of_weights_reports_and_fails(transcribe, tmp_path):
    d = specbuild.build_model_dir(tmp_path, ModelSpec())
    (d / "model.safetensors").write_text(
        "version https://git-lfs.github.com/spec/v1\n"
        "oid sha256:0000000000000000000000000000000000000000000000000000000000000000\n"
        "size 123\n"
    )
    result, _, _ = transcribe(SCRIPT, str(d))
    assert result.returncode != 0


def test_absurd_header_length_reports_and_fails(transcribe, tmp_path):
    d = specbuild.build_model_dir(tmp_path, ModelSpec())
    (d / "model.safetensors").write_bytes(struct.pack("<Q", 2**60) + b"{}")
    result, _, _ = transcribe(SCRIPT, str(d))
    assert result.returncode != 0


def test_missing_config_reports_and_fails(transcribe, tmp_path):
    d = specbuild.build_model_dir(tmp_path, ModelSpec())
    (d / "config.json").unlink()
    result, _, _ = transcribe(SCRIPT, str(d))
    assert result.returncode != 0
    assert "[ERROR]" in result.stdout


def test_missing_weights_skips_tensor_sections_and_fails(transcribe, tmp_path):
    d = specbuild.build_model_dir(tmp_path, ModelSpec())
    (d / "model.safetensors").unlink()
    result, sections, _ = transcribe(SCRIPT, str(d))
    assert result.returncode != 0
    assert any("MISSING" in title for title in sections)


def test_bad_flag_exits_two(run_script):
    assert run_script(SCRIPT, "--bogus").returncode == 2


# --------------------------------------------------------------------------
# Output contract
# --------------------------------------------------------------------------


def test_output_contract(transcribe, model_dir):
    result, _, _ = transcribe(SCRIPT, str(model_dir))
    assert not ANSI_RE.search(result.stdout), "ANSI emitted under NO_COLOR"
    tags = set(STATUS_RE.findall(result.stdout))
    assert tags <= ALLOWED_TAGS, f"unexpected status tags: {tags - ALLOWED_TAGS}"
    assert "=== Gemma 4 header transcript ===" in result.stdout


def test_transcript_names_the_document_it_verifies(transcribe, model_dir):
    _, _, out = transcribe(SCRIPT, str(model_dir))
    assert "docs/models/gemma-4.md" in out.read_text()


# --------------------------------------------------------------------------
# Differential check against the reference implementation
# --------------------------------------------------------------------------


@requires_safetensors
def test_matches_reference_safetensors_reader(transcribe, model_dir, spec):
    from safetensors import safe_open

    _, _, out = transcribe(SCRIPT, str(model_dir))
    text = out.read_text()
    with safe_open(str(model_dir / "model.safetensors"), framework="numpy") as f:
        keys = list(f.keys())
    assert f"tensor count: {len(keys)}" in text
    assert len(keys) == len(spec.tensors())


def test_defect_d10_attention_section_excludes_tower_projections(transcribe, tmp_path):
    """The head-dim split section must show language-model layers only.

    The grep matched a bare `self_attn\\.(q|k|v|o)_proj`, but the vision and
    audio towers carry the same projection names. On the real E2B checkpoint
    the section ran to 44 rows of which 36 were tower, burying the eight rows
    that show head_dim 256 against global_head_dim 512 -- and because the rows
    sort by descending count, the four full-attention rows landed last.

    The MLP tally immediately above was already anchored to the layer
    convention, with a comment explaining this exact contamination. The fix
    had been applied to one of the two sites.
    """
    spec = ModelSpec(
        extra_tensors=(
            (
                "model.vision_tower.encoder.layers.0.self_attn.q_proj.weight",
                (4, 4),
                "BF16",
            ),
            ("model.audio_tower.layers.0.self_attn.k_proj.weight", (4, 4), "BF16"),
        )
    )
    d = specbuild.build_model_dir(tmp_path, spec)
    _, sections, _ = transcribe(SCRIPT, str(d))
    rows = specbuild.parse_counted(flat_lines(sections, "attention projection shapes"))
    contaminants = [k for k in rows if "vision_tower" in k or "audio_tower" in k]
    assert not contaminants, (
        f"tower projections leaked into the section: {contaminants}"
    )
    assert rows, "the section is empty; the anchor matched nothing"
    assert all("language_model" in k for k in rows), sorted(rows)
