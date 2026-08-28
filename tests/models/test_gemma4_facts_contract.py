"""Asserts the committed Gemma 4 artifacts against ``schemas/gemma4_facts.json``.

This module deliberately does what the rest of this suite forbids: it reads
recorded transcripts. The distinction is the point, and it is worth stating
plainly because the two rules look contradictory.

``test_gemma4_header_transcript.py`` and ``test_gguf_header_transcript.py`` test
**the script's behavior**, so they may never assert against a recorded run --
a golden transcript pasted from a previous run encodes that run's
misunderstanding (see ``specbuild`` module docstring). This module tests **the
committed artifact**: whether ``docs/models/gemma-4.md`` and the two transcripts
still agree with the architecture contract. A disagreement here means either the
artifact is stale or the contract is wrong. Both are findings; neither is a
false positive.

The contract is authored from the pinned ``config.json`` bytes, not from the
transcripts, so transcript-versus-contract is a genuine cross-check between two
independent derivations rather than a restatement. That property is what makes
these assertions worth running; if the contract is ever regenerated *from* a
transcript, they stop proving anything.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from tests.models.specbuild import (
    metadata_scalars,
    parse_counted,
    parse_tensor_list,
    read_sections,
    section_matching,
    split_gguf_files,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = REPO_ROOT / "schemas" / "gemma4_facts.json"
DOC = REPO_ROOT / "docs" / "models" / "gemma-4.md"
SAFETENSORS_TRANSCRIPT = REPO_ROOT / "docs" / "models" / "gemma4-safetensors.txt"
GGUF_TRANSCRIPT = REPO_ROOT / "docs" / "models" / "gemma4-gguf.txt"

# Variants with checkpoint headers collected. 12B and 31B are config-only, so no
# transcript assertion can name them; see notes.no_tensor_facts_for_12b_and_31b.
TENSOR_VARIANTS = ["E2B", "E4B"]


@pytest.fixture(scope="module")
def contract() -> dict:
    return json.loads(CONTRACT_PATH.read_text())


# --------------------------------------------------------------------------
# markdown table reading
# --------------------------------------------------------------------------


def _norm(cell: str):
    """Maps a markdown table cell to the JSON value it stands for."""
    text = cell.strip().strip("`").strip().strip('"').strip()
    if text in {"null", "None"}:
        return None
    # The doc writes JSON booleans; the safetensors transcript renders Python ones.
    if text in {"true", "True"}:
        return True
    if text in {"false", "False"}:
        return False
    if text in {"(unset)", "(absent)", ""}:
        return "<unset>"
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return text


def _table_after(heading: str) -> list[list[str]]:
    """Rows of the first markdown table following a ``## heading`` line."""
    lines = DOC.read_text().splitlines()
    start = next(
        (i for i, ln in enumerate(lines) if ln.strip() == heading),
        None,
    )
    assert start is not None, f"heading not found in {DOC.name}: {heading!r}"
    rows: list[list[str]] = []
    seen_table = False
    for line in lines[start + 1 :]:
        stripped = line.strip()
        if stripped.startswith("|"):
            seen_table = True
            cells = [c.strip() for c in stripped.strip("|").split("|")]
            if all(set(c) <= {"-", ":"} and c for c in cells):
                continue
            rows.append(cells)
        elif seen_table and not stripped:
            break
    assert rows, f"no table found after {heading!r}"
    return rows


# --------------------------------------------------------------------------
# 1. The prose document agrees with the contract
# --------------------------------------------------------------------------


def test_doc_invariants_table_matches_contract(contract):
    """Editing a number in gemma-4.md must break a test. This is that test."""
    rows = _table_after("## Invariants across all four variants")
    assert rows[0] == ["Field", "Value"], rows[0]
    documented = {r[0].strip("`"): _norm(r[1]) for r in rows[1:]}

    inv = contract["invariants"]
    checked = 0
    for field, value in inv.items():
        if field == "rope_parameters":
            continue
        if field not in documented:
            continue
        assert documented[field] == value, (
            f"{DOC.name} documents {field}={documented[field]!r}, "
            f"contract says {value!r}"
        )
        checked += 1
    # A parser that silently matched nothing would pass every assertion above.
    assert checked >= 12, f"only {checked} invariants cross-checked; parser drifted"


def test_doc_rope_rows_match_contract(contract):
    """The two RoPE rows are prose, so assert the numbers appear rather than parse."""
    rows = _table_after("## Invariants across all four variants")
    rope = contract["invariants"]["rope_parameters"]
    full = next(r[1] for r in rows if r[0].startswith("RoPE (full"))
    sliding = next(r[1] for r in rows if r[0].startswith("RoPE (sliding"))

    assert str(rope["full_attention"]["rope_theta"]) in full
    assert rope["full_attention"]["rope_type"] in full
    assert str(rope["full_attention"]["partial_rotary_factor"]) in full
    assert str(rope["sliding_attention"]["rope_theta"]) in sliding
    assert rope["sliding_attention"]["rope_type"] in sliding


def test_doc_per_variant_table_matches_contract(contract):
    rows = _table_after("## Per-variant text configuration")
    header = rows[0]
    assert header[0] == "Field", header
    columns = header[1:]
    assert columns == list(contract["variants"]), columns

    documented = {r[0].strip("`"): r[1:] for r in rows[1:]}
    checked = 0
    for field, cells in documented.items():
        for variant, cell in zip(columns, cells, strict=True):
            if field not in contract["variants"][variant]:
                continue
            want = contract["variants"][variant][field]
            got = _norm(cell)
            if got == "<unset>" and want is None:
                continue
            assert got == want, (
                f"{DOC.name} documents {field} for {variant} as {got!r}, "
                f"contract says {want!r}"
            )
            checked += 1
    assert checked >= 40, f"only {checked} per-variant cells cross-checked"


def test_doc_attention_schedule_matches_contract(contract):
    """The full-attention indices are explicit because E2B's period differs."""
    rows = _table_after("## Attention")
    header = rows[0]
    assert header[:4] == ["Model", "Layers", "Sliding", "Full"], header

    for row in rows[1:]:
        variant = row[0].strip("`")
        want = contract["variants"][variant]
        assert int(row[1]) == want["num_hidden_layers"], variant
        assert int(row[2]) == want["sliding_attention_layers"], variant
        assert int(row[3]) == want["full_attention_layers"], variant
        indices = [int(x) for x in row[4].split(",")]
        assert indices == want["layer_types_full_attention_indices"], variant


def test_final_layer_is_full_attention_in_every_variant(contract):
    """A cache or mask keyed off "the last layer is sliding" is wrong on all four."""
    for variant, facts in contract["variants"].items():
        last = facts["num_hidden_layers"] - 1
        assert last in facts["layer_types_full_attention_indices"], variant


# --------------------------------------------------------------------------
# 2. The safetensors transcript agrees with the contract
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def safetensors_sections() -> dict[str, list[str]]:
    return read_sections(SAFETENSORS_TRANSCRIPT)


@pytest.mark.parametrize("variant", TENSOR_VARIANTS)
def test_safetensors_documented_fields_match_contract(
    contract, safetensors_sections, variant
):
    body = section_matching(
        safetensors_sections, f"gemma-4-{variant} — config.json (documented fields)"
    )
    recorded = {}
    for line in body:
        m = re.match(r"^([a-z_]+): (.*)$", line.strip())
        if m:
            recorded[m.group(1)] = _norm(m.group(2))

    facts = {**contract["invariants"], **contract["variants"][variant]}
    checked = 0
    for field, value in facts.items():
        if field not in recorded or not isinstance(value, (int, float, str, bool)):
            continue
        assert recorded[field] == value, (
            f"{SAFETENSORS_TRANSCRIPT.name} records {variant} {field}="
            f"{recorded[field]!r}, contract says {value!r}"
        )
        checked += 1
    assert checked >= 20, f"only {checked} fields cross-checked for {variant}"


@pytest.mark.parametrize("variant", TENSOR_VARIANTS)
def test_safetensors_layer_schedule_matches_contract(
    contract, safetensors_sections, variant
):
    body = section_matching(
        safetensors_sections, f"gemma-4-{variant} — config.json (documented fields)"
    )
    text = "\n".join(body)
    want = contract["variants"][variant]
    assert f"layer_types length: {want['num_hidden_layers']}" in text
    indices = str(want["layer_types_full_attention_indices"])
    assert f"full_attention indices: {indices}" in text
    assert "final layer type: full_attention" in text


@pytest.mark.parametrize("variant", TENSOR_VARIANTS)
def test_safetensors_projection_shapes_match_contract(
    contract, safetensors_sections, variant
):
    """The head-dim split is visible in the weights, not inferred from config.

    Sliding layers use head_dim (256) and full-attention layers use
    global_head_dim (512), so each projection appears at two distinct shapes.
    """
    body = section_matching(
        safetensors_sections, f"gemma-4-{variant} — attention projection shapes"
    )
    counted = parse_counted(body)
    facts = contract["variants"][variant]
    shapes = facts["projection_shapes"]
    expected_counts = {
        "sliding": facts["sliding_attention_layers"],
        "full": facts["full_attention_layers"],
    }

    for kind, projections in shapes.items():
        for name, dims in projections.items():
            if name.endswith("_norm"):
                continue  # q_norm/k_norm are not in this section
            row = (
                f"model.language_model.layers.{{L}}.self_attn.{name}.weight "
                f"[{', '.join(str(d) for d in dims)}] BF16"
            )
            assert row in counted, (
                f"{variant} {kind} {name}: no row {row!r} in "
                f"{SAFETENSORS_TRANSCRIPT.name}"
            )
            assert counted[row] == expected_counts[kind], (
                f"{variant} {kind} {name}: transcript counts {counted[row]} layers, "
                f"contract says {expected_counts[kind]}"
            )


@pytest.mark.parametrize("variant", TENSOR_VARIANTS)
def test_safetensors_records_no_lm_head(contract, safetensors_sections, variant):
    """tie_word_embeddings is confirmed in the weights, not only in config."""
    assert contract["invariants"]["tie_word_embeddings"] is True
    body = section_matching(
        safetensors_sections, f"gemma-4-{variant} — lm_head presence"
    )
    assert body == ["no lm_head tensor, as documented"], body


# --------------------------------------------------------------------------
# 3. The GGUF transcript agrees with the contract
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def gguf_blocks() -> dict[str, str]:
    return split_gguf_files(GGUF_TRANSCRIPT)


def _base_block(blocks: dict[str, str], variant: str) -> str:
    """The base checkpoint block, excluding the mmproj and mtp companions."""
    hits = [
        text
        for path, text in blocks.items()
        if f"gemma-4-{variant}-it-BF16.gguf" in path
        and "mmproj" not in path
        and "mtp" not in path
    ]
    assert len(hits) == 1, f"expected one base block for {variant}, got {len(hits)}"
    return hits[0]


@pytest.mark.parametrize("variant", TENSOR_VARIANTS)
def test_gguf_metadata_matches_contract(contract, gguf_blocks, variant):
    """GGUF renames the config fields; gguf_key_aliases records the mapping.

    ``gemma4.attention.key_length`` is the trap: it carries global_head_dim
    (512), not head_dim (256), and rust-harness loads it into a field named
    ``head_dim``.
    """
    scalars = metadata_scalars(_base_block(gguf_blocks, variant))
    facts = {**contract["invariants"], **contract["variants"][variant]}

    expected = {
        "gemma4.block_count": facts["num_hidden_layers"],
        "gemma4.embedding_length": facts["hidden_size"],
        "gemma4.attention.head_count": facts["num_attention_heads"],
        "gemma4.attention.head_count_kv": facts["num_key_value_heads"],
        "gemma4.attention.key_length": facts["global_head_dim"],
        "gemma4.attention.sliding_window": facts["sliding_window"],
    }
    for key, want in expected.items():
        assert key in scalars, f"{variant}: {key} absent from GGUF transcript"
        assert _norm(scalars[key]) == want, (
            f"{variant}: GGUF {key}={scalars[key]!r}, contract says {want!r}"
        )


@pytest.mark.parametrize("variant", TENSOR_VARIANTS)
def test_gguf_shapes_are_the_reverse_of_safetensors(contract, gguf_blocks, variant):
    """GGUF stores dimensions reversed; the contract holds safetensors order."""
    tensors = parse_tensor_list(_base_block(gguf_blocks, variant))
    facts = contract["variants"][variant]
    gguf_name = {"q_proj": "attn_q", "k_proj": "attn_k", "v_proj": "attn_v"}

    for kind, indices in (
        ("sliding", None),
        ("full", facts["layer_types_full_attention_indices"]),
    ):
        layer = (
            indices[0]
            if indices
            else next(
                i
                for i in range(facts["num_hidden_layers"])
                if i not in facts["layer_types_full_attention_indices"]
            )
        )
        for hf, gg in gguf_name.items():
            want = list(reversed(facts["projection_shapes"][kind][hf]))
            key = f"blk.{layer}.{gg}.weight"
            assert key in tensors, f"{variant}: {key} absent"
            assert tensors[key][0] == want, (
                f"{variant} {key}: GGUF has {tensors[key][0]}, contract reversed "
                f"is {want}"
            )


@pytest.mark.parametrize("variant", TENSOR_VARIANTS)
def test_gguf_carries_kv_projections_on_every_block(contract, gguf_blocks, variant):
    """The absence of attn_k is not a shared-KV signature on these checkpoints.

    ``num_kv_shared_layers`` shares the cache, not the weights: every block
    carries its own attn_k and attn_v despite the modeling code constructing
    them only on non-shared layers. Pinned because a loader that treats a
    missing attn_k as the shared-KV marker reads these files wrongly.
    """
    tensors = parse_tensor_list(_base_block(gguf_blocks, variant))
    facts = contract["variants"][variant]
    assert facts["num_kv_shared_layers"] > 0
    for layer in range(facts["num_hidden_layers"]):
        for name in ("attn_k", "attn_v"):
            assert f"blk.{layer}.{name}.weight" in tensors, (
                f"{variant}: blk.{layer}.{name}.weight absent"
            )


# --------------------------------------------------------------------------
# The canonical chat template, extracted from the GGUF transcript
# --------------------------------------------------------------------------

CANONICAL_TEMPLATE = REPO_ROOT / "docs" / "models" / "gemma4-chat-template.jinja"


def _template_from_transcript(block: str) -> tuple[str, int]:
    """Returns (template text, the character count the transcript declares)."""
    lines = block.splitlines()
    i = next(
        n for n, ln in enumerate(lines) if ln.startswith("tokenizer.chat_template")
    )
    declared = int(lines[i].split("string, ")[1].split(" chars")[0])
    assert lines[i + 1].strip() == "--- begin ---"
    j = next(n for n in range(i + 2, len(lines)) if lines[n].strip() == "--- end ---")
    body = [ln[4:] if ln.startswith("    ") else ln for ln in lines[i + 2 : j]]
    return "\n".join(body) + "\n", declared


def test_canonical_template_matches_the_transcript(gguf_blocks):
    """The committed .jinja is an extraction, not a transcription.

    Asserted against the E2B block and cross-checked against E4B below, so an
    edit to the .jinja file that the checkpoints do not support fails here.
    """
    text, declared = _template_from_transcript(_base_block(gguf_blocks, "E2B"))
    assert len(text) == declared, (
        f"extracted {len(text)} chars, transcript declares {declared}"
    )
    assert CANONICAL_TEMPLATE.read_text() == text


def test_canonical_template_is_identical_across_e2b_and_e4b(gguf_blocks):
    """One template serves both checkpoints, so one committed copy is correct."""
    e2b, _ = _template_from_transcript(_base_block(gguf_blocks, "E2B"))
    e4b, _ = _template_from_transcript(_base_block(gguf_blocks, "E4B"))
    assert e2b == e4b


# --------------------------------------------------------------------------
# 4. The line-number citations still resolve
# --------------------------------------------------------------------------

# docs/models/README.md warns that gemma-4.md cites the
# transcripts by line number and that nothing verifies them: the documentation
# reference gate checks that a path exists, not that a line still says what a
# document claims. This table is that missing check. Each entry is
# (citing document, its line, cited transcript, cited line, required substring).
CITATIONS = [
    (
        # The measured layer_scalar value. The needle is the value rather than
        # the tensor name: "layer_scalar" also appears in the shape-inventory
        # rows above, so a name-anchored needle would pass against a transcript
        # that had lost the values section entirely.
        "docs/models/gemma-4.md",
        714,
        "docs/models/gemma4-safetensors.txt",
        741,
        "value 0.0185546875",
    ),
    (
        "docs/models/gemma-4.md",
        855,
        "docs/models/gemma4-safetensors.txt",
        414,
        "self_attn.k_proj.weight [256, 1536]",
    ),
    (
        "docs/models/gemma-4.md",
        855,
        "docs/models/gemma4-safetensors.txt",
        529,
        "self_attn.k_proj.weight [512, 1536]",
    ),
    (
        "docs/models/gemma-4.md",
        857,
        "docs/models/gemma4-gguf.txt",
        613,
        "blk.{L}.attn_k.weight [1536, 256]",
    ),
    (
        "docs/models/gemma-4.md",
        857,
        "docs/models/gemma4-gguf.txt",
        625,
        "blk.{L}.attn_k.weight [1536, 512]",
    ),
    (
        "docs/models/gemma-4.md",
        286,
        "docs/models/gemma4-gguf.txt",
        630,
        "rope_freqs.weight [256] F32",
    ),
    (
        "docs/models/gemma-4.md",
        286,
        "docs/models/gemma4-gguf.txt",
        639,
        "rope_freqs.weight [256] F32",
    ),
    (
        "docs/models/gemma-4.md",
        395,
        "docs/models/gemma4-gguf.txt",
        632,
        "per_layer_token_embd.weight [8960, 262144]",
    ),
    (
        "docs/models/gemma-4.md",
        395,
        "docs/models/gemma4-gguf.txt",
        641,
        "per_layer_token_embd.weight [8960, 262144]",
    ),
]


@pytest.mark.parametrize(
    ("citing_doc", "citing_line", "cited_file", "cited_line", "needle"),
    CITATIONS,
    ids=[f"{c[0].split('/')[-1]}:{c[1]}->{c[3]}" for c in CITATIONS],
)
def test_transcript_line_citations_resolve(
    citing_doc, citing_line, cited_file, cited_line, needle
):
    """A regenerated transcript shifts line numbers; this catches it.

    Previously the only defense was a note telling a human to re-check six
    citations by hand after regenerating.
    """
    citing = (REPO_ROOT / citing_doc).read_text().splitlines()
    assert cited_file in citing[citing_line - 1], (
        f"{citing_doc}:{citing_line} no longer cites {cited_file}; the citation "
        f"table in this test is stale"
    )
    cited = (REPO_ROOT / cited_file).read_text().splitlines()
    assert needle in cited[cited_line - 1], (
        f"{citing_doc}:{citing_line} cites {cited_file} line {cited_line} for "
        f"{needle!r}, but that line now reads: {cited[cited_line - 1]!r}"
    )


# --------------------------------------------------------------------------
# contract hygiene
# --------------------------------------------------------------------------


def test_contract_names_its_own_guards(contract):
    assert Path(__file__).name in " ".join(contract["guarded_by"])


def test_contract_paths_exist(contract):
    for entry in contract["writers"] + contract["readers"] + contract["guarded_by"]:
        candidate = entry.split()[0].split("(")[0]
        if "/" not in candidate or candidate.startswith("google/"):
            continue
        assert (REPO_ROOT / candidate).exists(), f"{candidate} does not exist"


def test_tensor_facts_absent_for_config_only_variants(contract):
    """12B and 31B carry no projection_shapes, by decision rather than oversight."""
    for variant in ("12B", "31B"):
        assert "projection_shapes" not in contract["variants"][variant]
    assert "no_tensor_facts_for_12b_and_31b" in contract["notes"]


# --------------------------------------------------------------------------
# 5. The upstream-source citations resolve
# --------------------------------------------------------------------------

UPSTREAM_TRANSCRIPT = REPO_ROOT / "docs" / "models" / "gemma4-upstream.txt"

# The "Resolved from upstream source" section settles five architecture
# questions against transformers and llama.cpp, citing file, commit SHA, and
# line. Those SHAs lived in prose and nowhere else: the analysis was sound but
# left no artifact, so nothing could re-check it and nobody re-ran it. Each
# entry is (upstream path, line, required substring).
UPSTREAM_CITATIONS = [
    (
        "src/transformers/modeling_rope_utils.py",
        193,
        "def _compute_proportional_rope_parameters(",
    ),
    ("src/transformers/models/gemma4/modeling_gemma4.py", 1371, "self.layer_scalar"),
    (
        "src/transformers/models/gemma4/modeling_gemma4.py",
        1444,
        "hidden_states *= self.layer_scalar",
    ),
    (
        "src/transformers/models/gemma4/modeling_gemma4.py",
        1523,
        "init.ones_(module.layer_scalar)",
    ),
    (
        "src/transformers/models/gemma4/modeling_gemma4.py",
        1179,
        "config.attention_k_eq_v",
    ),
    (
        "src/transformers/models/gemma4/modeling_gemma4.py",
        1241,
        "shared_kv_states[self.layer_type]",
    ),
    (
        "src/transformers/models/gemma4/modeling_gemma4.py",
        1259,
        "shared_kv_states[self.layer_type]",
    ),
    ("conversion/gemma.py", 682, "first_kv_shared_layer_idx"),
]


@pytest.fixture(scope="module")
def upstream_lines() -> dict[str, dict[int, str]]:
    """{upstream path: {line number: text}} from the committed transcript."""
    if not UPSTREAM_TRANSCRIPT.is_file():
        pytest.skip(f"upstream transcript absent: {UPSTREAM_TRANSCRIPT}")
    out: dict[str, dict[int, str]] = {}
    current: str | None = None
    for line in UPSTREAM_TRANSCRIPT.read_text().splitlines():
        m = re.match(r"^PATH:   (\S+)$", line)
        if m:
            current = m.group(1)
            out[current] = {}
            continue
        m = re.match(r"^\s*(\d+)  (.*)$", line)
        if m and current:
            out[current][int(m.group(1))] = m.group(2)
    return out


def test_upstream_transcript_covers_every_pinned_source(contract, upstream_lines):
    """Each revision the document pins must appear in the transcript."""
    doc = DOC.read_text()
    for path in {c[0] for c in UPSTREAM_CITATIONS}:
        assert path in upstream_lines, f"{path} absent from the upstream transcript"
        assert path.split("/")[-1] in doc


@pytest.mark.parametrize(
    ("path", "line", "needle"),
    UPSTREAM_CITATIONS,
    ids=[f"{c[0].split('/')[-1]}:{c[1]}" for c in UPSTREAM_CITATIONS],
)
def test_upstream_line_citations_resolve(upstream_lines, path, line, needle):
    text = upstream_lines[path].get(line)
    assert text is not None, f"{path} has no line {line} in the transcript"
    assert needle in text, (
        f"gemma-4.md cites {path} line {line} for {needle!r}, "
        f"but that line reads: {text!r}"
    )


def test_upstream_transcript_records_the_pinned_shas(contract):
    """The transcript's SHAs must match the table in gemma-4.md.

    A transcript fetched from a branch rather than a pin would drift silently.
    """
    transcript = UPSTREAM_TRANSCRIPT.read_text()
    doc = DOC.read_text()
    shas = set(re.findall(r"^COMMIT: ([0-9a-f]{40})$", transcript, re.MULTILINE))
    assert shas, "no pinned commits recorded in the upstream transcript"
    for sha in shas:
        assert sha in doc, f"{sha} is in the transcript but not pinned in {DOC.name}"


# --------------------------------------------------------------------------
# 6. No REFUSED verdict survives in the committed transcript
# --------------------------------------------------------------------------


def test_committed_transcripts_carry_no_false_refusals():
    """No REFUSED verdict may appear in the committed GGUF transcript.

    The 14 that once did were artifacts of hand-copied ggml type sets that had
    drifted from the Rust loader: BF16 is loadable and bool arrays are
    accepted. The sets are now derived from the Rust source, so a REFUSED here
    again means either a genuinely unsupported type in the checkpoint or a
    fresh drift in the derivation -- both worth failing on.
    """
    assert "REFUSED" not in GGUF_TRANSCRIPT.read_text()
