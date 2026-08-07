"""Tests for scripts/models/gguf_header_transcript.sh.

The GGUF reader in that script is hand-rolled, so the load-bearing test here
is not a round-trip against a fixture this suite also wrote — that would only
prove the writer and reader share a misunderstanding. It is
``test_reader_agrees_with_reference_parse_gguf``, which parses
``rust-harness/fixtures/tiny-gemma4.gguf`` (written by Rust,
``runtime/src/synth.rs::GgufWriter``) and compares against
``rust-harness/fixtures/reference_logits.py::parse_gguf`` (an independent
pure-Python reader). Neither shares an author with the script.

The spec-driven tests then cover what that fixture cannot reach: bool arrays,
BF16 tensors, and version 2, none of which any in-tree writer produces.

Which types the harness accepts is derived from the Rust source by
``specbuild.rust_loadable_tensor_types`` and ``rust_ok_array_element_types``
rather than restated, so this suite cannot drift from the loader the way the
script's hand-copied sets did.
"""

import importlib.util
import re
import struct

import pytest

from tests.models.specbuild import (
    GGML_BF16,
    GGML_F32,
    GGUF_ARRAY,
    GGUF_BOOL,
    GGUF_STRING,
    GGUF_UINT32,
    GgufSpec,
    gemma_gguf_spec,
    metadata_keys,
    parse_tensor_list,
    rust_loadable_tensor_types,
    rust_ok_array_element_types,
    write_gguf,
)

SCRIPT = "models/gguf_header_transcript.sh"

ANSI_RE = re.compile(r"\x1b\[")
STATUS_RE = re.compile(r"^\s*\[([A-Z]+)\]", re.MULTILINE)
ALLOWED_TAGS = {"OK", "WARN", "ERROR", "INFO", "RUN", "SKIP"}

requires_gguf = pytest.mark.skipif(
    importlib.util.find_spec("gguf") is None,
    reason="gguf package required for the differential check",
)


def build(tmp_path, spec: GgufSpec, name: str = "m.gguf", **kwargs):
    return write_gguf(tmp_path / name, spec, **kwargs)


# --------------------------------------------------------------------------
# Keystone: agreement with an independent reader on a Rust-written file
# --------------------------------------------------------------------------


def test_committed_rust_fixture_exists(repo_root):
    """Guards the keystone: if the fixture is gone, this suite has no
    independent cross-check and must say so rather than silently skip."""
    fixture = repo_root / "rust-harness" / "fixtures" / "tiny-gemma4.gguf"
    assert fixture.exists(), (
        f"{fixture} is missing; regenerate with "
        "`cargo run -p runtime --bin tiny-model -- fixtures/tiny-gemma4.gguf`"
    )


def test_reader_agrees_with_reference_parse_gguf(
    transcribe, repo_root, reference_parse_gguf
):
    """The script's reader and the in-tree independent reader must agree.

    This is the one check in the suite where neither side was written by the
    author of the other: the file comes from Rust, the reference reader from
    fixtures/reference_logits.py.
    """
    fixture = repo_root / "rust-harness" / "fixtures" / "tiny-gemma4.gguf"
    ref_meta, ref_tensors = reference_parse_gguf(fixture.read_bytes())

    _, _, out = transcribe(SCRIPT, str(fixture))
    text = out.read_text()
    got = parse_tensor_list(text)

    assert set(got) == set(ref_tensors), (
        "tensor name sets differ: "
        f"only ours {sorted(set(got) - set(ref_tensors))}, "
        f"only reference {sorted(set(ref_tensors) - set(got))}"
    )
    for name, (dims, _dtype) in got.items():
        # parse_gguf reshapes to reversed(dims); compare in its orientation.
        assert list(reversed(dims)) == list(ref_tensors[name].shape), (
            f"{name}: ours {dims}, reference {list(ref_tensors[name].shape)}"
        )

    ours = metadata_keys(text)
    ref_non_tokenizer = {k for k in ref_meta if not k.startswith("tokenizer.")}
    assert ref_non_tokenizer <= ours, (
        f"metadata keys the reference found but we did not: "
        f"{sorted(ref_non_tokenizer - ours)}"
    )
    assert f"tensor count: {len(ref_tensors)}" in text


def test_rust_fixture_reports_no_false_refusals(transcribe, repo_root):
    """Negative control: GgufWriter emits F32 and accepted array types only,
    so no REFUSED line may appear. A script that flags refusals
    unconditionally would pass every positive refusal test."""
    fixture = repo_root / "rust-harness" / "fixtures" / "tiny-gemma4.gguf"
    result, _, out = transcribe(SCRIPT, str(fixture))
    assert result.returncode == 0, result.stdout
    assert "REFUSED" not in out.read_text()
    assert "REFUSED" not in result.stdout


# --------------------------------------------------------------------------
# Round-trip against the spec
# --------------------------------------------------------------------------


def test_tensor_inventory_matches_spec(transcribe, tmp_path):
    spec = gemma_gguf_spec()
    path = build(tmp_path, spec)
    _, _, out = transcribe(SCRIPT, str(path))
    got = parse_tensor_list(out.read_text())
    assert set(got) == {n for n, _, _ in spec.tensors}
    for name, dims, _ in spec.tensors:
        assert got[name][0] == list(dims), f"{name}: {got[name][0]} != {list(dims)}"


def test_counts_match_spec(transcribe, tmp_path):
    spec = gemma_gguf_spec()
    _, _, out = transcribe(SCRIPT, str(build(tmp_path, spec)))
    text = out.read_text()
    assert f"tensor count: {len(spec.tensors)}" in text
    assert f"metadata kv count: {len(spec.kvs)}" in text
    assert f"gguf version: {spec.version}" in text


def test_all_metadata_keys_appear(transcribe, tmp_path):
    spec = gemma_gguf_spec()
    _, _, out = transcribe(SCRIPT, str(build(tmp_path, spec)))
    text = out.read_text()
    for key, _, _, _ in spec.kvs:
        assert key in text, f"metadata key {key} missing from transcript"


def test_chat_template_is_printed_whole(transcribe, tmp_path):
    """The chat template is the authoritative turn-marker source for Part B,
    so it must not be truncated."""
    spec = gemma_gguf_spec()
    _, _, out = transcribe(SCRIPT, str(build(tmp_path, spec)))
    text = out.read_text()
    assert "<|turn>" in text and "<turn|>" in text
    assert "--- begin ---" in text and "--- end ---" in text


def test_special_token_strings_are_listed_with_ids(transcribe, tmp_path):
    spec = gemma_gguf_spec()
    tokens = next(v for k, _, v, _ in spec.kvs if k == "tokenizer.ggml.tokens")
    _, _, out = transcribe(SCRIPT, str(build(tmp_path, spec)))
    text = out.read_text()
    for idx, tok in enumerate(tokens):
        if "turn" in tok or "tool" in tok:
            assert f"{idx}: {tok!r}" in text, f"{tok} not listed at id {idx}"


# --------------------------------------------------------------------------
# Mutation matrix
# --------------------------------------------------------------------------


def test_bool_array_is_accepted_not_refused(transcribe, tmp_path):
    """gguf.rs decodes bool arrays, so sliding_window_pattern is not a refusal.

    This test used to assert the opposite, because the script carried a stale
    copy of the accepted set. The Gemma 4 converter writes
    sliding_window_pattern as a bool array, so a false refusal here reads as
    "the harness cannot load this checkpoint".
    """
    spec = gemma_gguf_spec()
    assert GGUF_BOOL in rust_ok_array_element_types()
    assert spec.refused_array_keys() == [], spec.refused_array_keys()
    _, _, out = transcribe(SCRIPT, str(build(tmp_path, spec)))
    text = out.read_text()
    assert "gemma4.attention.sliding_window_pattern" in text
    assert "element type 7 (bool): 1 keys -- accepted" in text
    assert "REFUSED by gguf.rs" not in text


def test_mutation_unsupported_array_element_is_flagged_refused(transcribe, tmp_path):
    """An element type gguf.rs really does bail on must still be flagged.

    Keeps the positive side of the refusal check alive now that bool is
    accepted; without it, a script that never printed REFUSED would pass.
    """
    unsupported = GGUF_UINT32
    assert unsupported not in rust_ok_array_element_types()
    spec = gemma_gguf_spec()
    # kvs entries are (key, value type, value, array element type).
    spec.kvs = [
        (k, vt, v, unsupported) if et == GGUF_BOOL else (k, vt, v, et)
        for k, vt, v, et in spec.kvs
    ]
    assert spec.refused_array_keys() != []
    result, _, out = transcribe(SCRIPT, str(build(tmp_path, spec)))
    text = out.read_text()
    assert "REFUSED by gguf.rs" in text
    assert "[WARN]" in result.stdout


def test_mutation_accepted_arrays_only_produce_no_refusal(transcribe, tmp_path):
    """Negative control for the array check."""
    spec = gemma_gguf_spec()
    spec.kvs = [kv for kv in spec.kvs if kv[3] != GGUF_BOOL]
    assert spec.refused_array_keys() == []
    _, _, out = transcribe(SCRIPT, str(build(tmp_path, spec)))
    text = out.read_text()
    assert "REFUSED by gguf.rs" not in text
    assert "no array key uses a refused element type" in text


def test_bf16_tensors_are_loadable_not_refused(transcribe, tmp_path):
    """tensor.rs dequantizes BF16, the published storage type for Gemma 4.

    This test used to assert the opposite. BF16 is 318 of the 601 tensors in
    the E2B base checkpoint, so the stale set turned the majority of a real
    model into a reported refusal.
    """
    spec = gemma_gguf_spec()
    assert GGML_BF16 in rust_loadable_tensor_types()
    assert spec.refused_tensor_types() == []
    _, _, out = transcribe(SCRIPT, str(build(tmp_path, spec)))
    text = out.read_text()
    assert "BF16 (type 30)" in text
    assert "REFUSED by tensor.rs" not in text


def test_mutation_kquant_tensors_are_flagged_refused(transcribe, tmp_path):
    """A type tensor.rs really does bail on must still be flagged."""
    kquant = 12  # Q4_K, absent from tensor.rs
    assert kquant not in rust_loadable_tensor_types()
    spec = gemma_gguf_spec()
    spec.tensors = [(n, d, kquant) for n, d, _ in spec.tensors]
    assert spec.refused_tensor_types() == [kquant]
    result, _, out = transcribe(SCRIPT, str(build(tmp_path, spec)))
    assert "REFUSED by tensor.rs" in out.read_text()
    assert "[WARN]" in result.stdout


def test_mutation_all_f32_produces_no_refusal(transcribe, tmp_path):
    """Negative control for the tensor-type check."""
    spec = gemma_gguf_spec()
    spec.tensors = [(n, d, GGML_F32) for n, d, _ in spec.tensors]
    assert spec.refused_tensor_types() == []
    _, _, out = transcribe(SCRIPT, str(build(tmp_path, spec)))
    text = out.read_text()
    assert "REFUSED by tensor.rs" not in text
    assert "F32 (type 0)" in text and "loadable" in text


def test_mutation_version_two_is_flagged_as_harness_incompatible(transcribe, tmp_path):
    """rust-harness pins GGUF_VERSION = 3; a v2 file is one it will refuse."""
    spec = gemma_gguf_spec()
    spec.version = 2
    _, _, out = transcribe(SCRIPT, str(build(tmp_path, spec)))
    text = out.read_text()
    assert "gguf version: 2" in text
    assert "REFUSED by gguf.rs" in text


def test_mutation_version_three_is_not_flagged(transcribe, tmp_path):
    """Negative control for the version check."""
    _, _, out = transcribe(SCRIPT, str(build(tmp_path, gemma_gguf_spec())))
    assert "REFUSED by gguf.rs" not in out.read_text()


def test_mutation_tensor_count_changes_transcript(transcribe, tmp_path):
    small = gemma_gguf_spec()
    big = gemma_gguf_spec()
    big.tensors = big.tensors + [("blk.9.extra.weight", (4, 4), GGML_F32)]
    _, _, a = transcribe(SCRIPT, str(build(tmp_path, small, "a.gguf")))
    _, _, b = transcribe(SCRIPT, str(build(tmp_path, big, "b.gguf")))
    assert f"tensor count: {len(small.tensors)}" in a.read_text()
    assert f"tensor count: {len(big.tensors)}" in b.read_text()
    assert a.read_text() != b.read_text()


def test_nested_indices_collapse_fully(transcribe, tmp_path):
    """Names with two numeric segments must collapse to one inventory row."""
    spec = gemma_gguf_spec()
    spec.tensors = [
        (f"blk.{i}.attn.{j}.weight", (4, 4), GGML_F32)
        for i in range(2)
        for j in range(2)
    ]
    _, _, out = transcribe(SCRIPT, str(build(tmp_path, spec)))
    body = out.read_text().split("inventory, layer indices collapsed:", 1)[1]
    body = body.split("=== full tensor name list ===", 1)[0]
    rows = [ln for ln in body.splitlines() if "attn" in ln]
    assert len(rows) == 1, f"nested indices did not collapse: {rows}"
    assert rows[0].split()[0] == "4", rows


# --------------------------------------------------------------------------
# Robustness
# --------------------------------------------------------------------------


def test_wrong_magic_reports_and_fails(transcribe, tmp_path):
    path = build(tmp_path, gemma_gguf_spec(), magic=b"NOPE")
    result, _, out = transcribe(SCRIPT, str(path))
    assert result.returncode != 0
    assert "NOT A GGUF FILE" in out.read_text()


def test_truncated_file_reports_and_fails(transcribe, tmp_path):
    path = build(tmp_path, gemma_gguf_spec())
    raw = path.read_bytes()
    path.write_bytes(raw[: len(raw) // 4])
    result, _, _ = transcribe(SCRIPT, str(path))
    assert result.returncode != 0


def test_absurd_declared_length_names_the_cause(transcribe, tmp_path):
    """A length wider than the file must produce a diagnostic naming the
    offset, not an unexplained MemoryError."""
    path = tmp_path / "bad.gguf"
    path.write_bytes(
        b"GGUF"
        + struct.pack("<I", 3)
        + struct.pack("<Q", 1)
        + struct.pack("<Q", 1)
        + struct.pack("<Q", 2**60)
    )
    result, _, out = transcribe(SCRIPT, str(path))
    assert result.returncode != 0
    body = out.read_text()
    assert "exceeds file size" in body, body
    assert "MemoryError" not in body, body


def test_missing_file_reports_and_fails(transcribe, tmp_path):
    result, _, _ = transcribe(SCRIPT, str(tmp_path / "absent.gguf"))
    assert result.returncode != 0
    assert "[ERROR]" in result.stdout


def test_no_files_found_exits_one(run_script, tmp_path):
    result = run_script(SCRIPT, env={"MODEL_ROOT": str(tmp_path)})
    assert result.returncode == 1
    assert "No GGUF files found" in result.stdout


def test_unicode_and_newline_tokens_survive(transcribe, tmp_path):
    """Token strings are arbitrary bytes; neither must corrupt the transcript."""
    spec = gemma_gguf_spec()
    spec.kvs = [kv for kv in spec.kvs if kv[0] != "tokenizer.ggml.tokens"] + [
        (
            "tokenizer.ggml.tokens",
            GGUF_ARRAY,
            ["<|turn>", "naïve", "改行\nあり", "emoji \U0001f600"],
            GGUF_STRING,
        )
    ]
    result, _, out = transcribe(SCRIPT, str(build(tmp_path, spec)))
    assert result.returncode == 0, result.stdout
    text = out.read_text()
    assert "naïve" in text
    assert "vocab size: 4" in text


def test_bad_flag_exits_two(run_script):
    assert run_script(SCRIPT, "--bogus").returncode == 2


def test_multiple_files_all_transcribed(transcribe, tmp_path):
    a = build(tmp_path, gemma_gguf_spec(), "one.gguf")
    b = build(tmp_path, gemma_gguf_spec(), "two.gguf")
    result, _, out = transcribe(SCRIPT, str(a), str(b))
    text = out.read_text()
    assert str(a) in text and str(b) in text
    assert result.returncode == 0


def test_defect_d6_summary_attributes_every_refusal_to_its_file(transcribe, tmp_path):
    """Two files with identical refusals must produce two summary lines.

    The summary deduplicated with `sort -u` over lines that carry counts but
    not filenames, so files sharing a tensor-type tally collapsed into one
    line: six real GGUF files reported as four, and four separate bool-array
    keys reported as "1 keys". Under-reporting reads as a smaller problem
    rather than as a missing one.
    """

    def refusing_spec():
        # Both refusal paths at once, with types the loader really does reject:
        # Q4_K tensors and a uint32 array element. Two identical files must
        # still produce two summary lines each.
        spec = gemma_gguf_spec()
        spec.tensors = [(n, d, 12) for n, d, _ in spec.tensors]
        spec.kvs = [
            (k, vt, v, GGUF_UINT32) if et == GGUF_BOOL else (k, vt, v, et)
            for k, vt, v, et in spec.kvs
        ]
        return spec

    a = build(tmp_path, refusing_spec(), "one.gguf")
    b = build(tmp_path, refusing_spec(), "two.gguf")
    result, _, _ = transcribe(SCRIPT, str(a), str(b))

    summary = result.stdout.split("Summary:", 1)[1]
    for name in ("one.gguf", "two.gguf"):
        assert summary.count(f"{name}: Q4_K (type 12)") == 1, (
            f"{name} missing from the tensor-type summary:\n{summary}"
        )
        assert summary.count(f"{name}: element type 4 (uint32)") == 1, (
            f"{name} missing from the array-element summary:\n{summary}"
        )


# --------------------------------------------------------------------------
# Output contract
# --------------------------------------------------------------------------


def test_output_contract(transcribe, tmp_path):
    result, _, _ = transcribe(SCRIPT, str(build(tmp_path, gemma_gguf_spec())))
    assert not ANSI_RE.search(result.stdout), "ANSI emitted under NO_COLOR"
    tags = set(STATUS_RE.findall(result.stdout))
    assert tags <= ALLOWED_TAGS, f"unexpected status tags: {tags - ALLOWED_TAGS}"
    assert "=== GGUF header transcript ===" in result.stdout


# --------------------------------------------------------------------------
# Differential check against the reference implementation
# --------------------------------------------------------------------------


@requires_gguf
def test_matches_reference_gguf_reader(transcribe, repo_root):
    from gguf import GGUFReader

    fixture = repo_root / "rust-harness" / "fixtures" / "tiny-gemma4.gguf"
    reader = GGUFReader(str(fixture))
    _, _, out = transcribe(SCRIPT, str(fixture))
    got = parse_tensor_list(out.read_text())
    assert set(got) == {t.name for t in reader.tensors}


def test_defect_d8_verdicts_are_derived_from_the_rust_source(
    transcribe, tmp_path, repo_root
):
    """The accepted type sets are read from rust-harness, never restated.

    Both copies had drifted: the script declared tensor types {0, 1, 8} after
    tensor.rs gained BF16 (30), and array element types {5, 6, 8} after gguf.rs
    gained bool (7). The committed transcripts therefore carried 14 REFUSED
    verdicts, every one of them false, including "BF16 (type 30): 318 tensors
    -- REFUSED" against the E2B checkpoint the harness exists to load.
    """
    script = (
        repo_root / "scripts" / "models" / "gguf_header_transcript.sh"
    ).read_text()
    assert "RUST_HARNESS_LOADABLE = {" not in script, "type set restated in the script"
    assert "RUST_HARNESS_ARRAY_OK = {" not in script

    assert GGML_BF16 in rust_loadable_tensor_types()
    assert GGUF_BOOL in rust_ok_array_element_types()

    spec = gemma_gguf_spec()
    _, _, out = transcribe(SCRIPT, str(build(tmp_path, spec)))
    assert "REFUSED" not in out.read_text()


def test_defect_d9_per_layer_schedule_arrays_print_in_full(transcribe, tmp_path):
    """An array as long as block_count is a schedule and must not truncate.

    summarize() capped every array at 24 entries. sliding_window_pattern has
    one entry per layer, so on E4B's 42 layers the transcript showed 24 and
    hid 18 -- and the hidden tail is where the full-attention layers, the
    second RoPE regime, and the second head dimension live.
    """
    spec = gemma_gguf_spec()
    n_layers = 30  # longer than the old limit of 24
    spec.kvs = [
        (k, vt, n_layers, et) if k == "gemma4.block_count" else (k, vt, v, et)
        for k, vt, v, et in spec.kvs
    ]
    pattern = [i % 5 != 4 for i in range(n_layers)]
    spec.kvs = [
        (k, vt, pattern, et) if et == GGUF_BOOL else (k, vt, v, et)
        for k, vt, v, et in spec.kvs
    ]
    _, _, out = transcribe(SCRIPT, str(build(tmp_path, spec)))
    text = out.read_text()
    assert "more)" not in text, "a per-layer schedule was truncated"
    assert repr(pattern) in text
