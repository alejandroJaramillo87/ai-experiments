# Model header-transcriber tests

Tests for the Gemma 4 header transcribers in `scripts/models/`. These scripts produce the
transcripts used to correct `docs/models/gemma-4.md`, which `rust-harness`, `training`, and
`interp` all build against, so a wrong transcript value does not stay in the transcript — it
becomes a documented fact and then a pipeline assumption.

The suite is therefore built against a specific failure mode rather than for coverage. The
danger is not a crash; it is **plausible, well-formatted output that does not depend on the
input**: a `grep` that matches nothing and writes an empty section reading as "no such
tensors", or a `sed` that passes a non-matching line through into a tally. Both exit 0.

## Design

**Spec-driven, never golden-string.** A golden transcript pasted from a previous run encodes
that run's misunderstanding. Every fixture is generated from a declarative `ModelSpec` or
`GgufSpec` in `specbuild.py`, and every assertion compares against a value recomputed from that
spec. No test asserts against a recorded transcript.

**One cross-check shares no author with the code under test.**
`test_reader_agrees_with_reference_parse_gguf` parses
`tests/models/fixtures/tiny-exporter-written.gguf` — written by the upstream `gguf` package
through `interp/steering/export.py` — and compares
against `tests/models/gguf_reference_reader.py::parse_gguf`, an independent pure-Python reader.
This is the check that keeps the script's hand-rolled GGUF reader honest, per the
independent-verification principle in `.styles/RUST_STYLE.md`. It must **run**, never skip: if
it skips, the committed fixture is missing and the suite has no independent check at all.

**Negative controls carry equal weight.** A script that prints `REFUSED` unconditionally passes
every positive refusal test. `test_mutation_all_f32_produces_no_refusal`,
`test_rust_fixture_reports_no_false_refusals`, `test_mutation_version_three_is_not_flagged`,
and `test_recognized_naming_does_not_trip_the_guard` exist to fail in that case.

## Test index

| Test file | Covers | Approach |
|-----------|--------|----------|
| `test_gemma4_header_transcript.py` | `scripts/models/gemma4_header_transcript.sh` | Spec round-trip, defect regressions, mutation matrix, robustness, output contract |
| `test_gguf_header_transcript.py` | `scripts/models/gguf_header_transcript.sh` | Independent-reader cross-check, spec round-trip, refusal detection with negative controls, robustness |
| `test_gemma4_facts_contract.py` | the committed artifacts, not the scripts | `schemas/gemma4_facts.json` against `docs/models/gemma-4.md`, both header transcripts, the canonical chat template, and every line-number citation |
| `test_audit_recording.py` | `rust-harness/fixtures/audit_recording.py`, the half that runs without a checkpoint | sequence reconstruction from the recorder's own line shapes, the two-branch and gap refusals, and the L2 measures on rows with known answers; the teacher-forced pass itself needs the workstation |
| `specbuild.py` | neither (helper) | Not named `test_*` so pytest does not collect it |

**`test_gemma4_facts_contract.py` is the one module that reads recorded transcripts**, which
the rule above forbids the other two. The distinction is deliberate: they test what the
scripts *do*, and a golden string would encode a past run's misunderstanding; it tests whether
the committed artifacts still *agree*, which no fixture can answer. The contract it asserts
against is authored from the pinned `config.json` bytes rather than from the transcripts, so
the comparison is between two independent derivations. Regenerating the contract *from* a
transcript would make every assertion in that module vacuous.

**Which GGML tensor types and GGUF array element types the harness accepts is derived from the
Rust source**, by `specbuild.rust_loadable_tensor_types` and `rust_ok_array_element_types`.
Three copies of those sets used to exist and two had drifted; see `d8` below.

## Defect regressions

Each was confirmed failing against the scripts as written, then fixed. Select with
`-k defect`.

| Test | Defect it pins |
|------|----------------|
| `test_defect_d1_nested_indices_collapse_fully` | `sed` lacked `/g`, so tower names with two indices reported as two rows instead of one |
| `test_defect_d2_width_tally_holds_only_numbers` | the width `sed` passed non-matching lines through, putting whole tensor lines into the tally |
| `test_defect_d3_mlp_listing_is_ordered_by_layer_index` | `sort -t. -k4 -n` assumed name depth; layer 10 sorted between 1 and 2 |
| `test_defect_d4_unrecognized_naming_fails_loudly` | unrecognized tensor naming produced an empty, clean, exit-0 transcript |
| `test_defect_d5_cited_doc_paths_exist` | a cited `docs/` path did not exist |
| `test_defect_d6_summary_attributes_every_refusal_to_its_file` | the GGUF summary deduplicated with `sort -u` over lines carrying counts but not filenames, so six files reported as four and four bool-array keys as "1 keys" |
| `test_defect_d7_config_describing_another_model_fails_loudly` | `config.json` and the weights were never cross-checked, so a snapshot carrying another variant's config passed with only a hash warning |
| `test_defect_d8_verdicts_are_derived_from_the_rust_source` | the accepted type sets were hand-copied from Rust and had drifted, so every BF16 tensor and bool array was reported `REFUSED` -- 14 false verdicts in the committed transcripts, including 318 of 601 tensors in the E2B base file |
| `test_defect_d9_per_layer_schedule_arrays_print_in_full` | arrays truncated at 24 entries, hiding 18 of E4B's 42 `sliding_window_pattern` values -- the half that says where the two RoPE regimes and head dimensions change over |
| `test_defect_d10_attention_section_excludes_tower_projections` | the attention grep was unanchored, so vision and audio tower projections filled the head-dim-split section; the anchor had been applied to the adjacent MLP tally only |

D6 and D7 were found by running the scripts against the real weights, not against
fixtures. D7 is the one that mattered: the downloaded `gemma-4-E2B` snapshot shipped
E4B's `config.json` verbatim — 42 layers at width 2560 declared beside weights holding
35 layers at width 1536 — and the blob-hash check reported only an unexpected hash,
which reads as revision drift rather than as the wrong model's file.

`d4` deliberately uses a variant name the document does not pin, so no blob-hash check runs and
the naming guard is the only thing that can fail the script. Without that, the test passes on an
unrelated mismatch.

## Fixture index

| Fixture | Consumer | Proves |
|---------|----------|--------|
| `ModelSpec` → `build_model_dir` | `test_gemma4_header_transcript.py` | tensor inventory, dtype tally, double-wide MLP boundary, head-dim split, and zero-dimensional companions are all read from the file rather than assumed |
| `ModelSpec(prefix="model")` | `d3`, `d4` | a naming convention the script does not expect is reported, not silently skipped |
| `GgufSpec` → `write_gguf` | `test_gguf_header_transcript.py` | bool arrays, BF16 tensors, and version 2 are transcribed rather than mis-parsed |
| `tests/models/fixtures/tiny-exporter-written.gguf` | keystone cross-check | the script's reader agrees with an independent reader on a file neither wrote |

`specbuild.py` hand-rolls its GGUF writer because nothing in this repository writes GGUF for it
to borrow. Nothing reads GGUF either, since R10 deleted the last reader, so the spec-driven
cases are the only source of bool arrays, BF16 tensors and version 2. The keystone cross-check
is what keeps that hand-rolled writer from encoding the same misunderstanding as the reader
under test.

Dimensions are small (hidden size 8, 7 layers). What the transcribers get wrong is structural,
and none of it depends on a dimension being 1536 rather than 8; small shapes keep fixtures
byte-exact and loadable by the reference `safetensors` and `gguf` packages.

## Optional dependencies

Two differential tests skip when their package is absent, matching
`tests/interp/test_probe_export.py`:

| Test | Requires |
|------|----------|
| `test_matches_reference_safetensors_reader` | `safetensors` |
| `test_matches_reference_gguf_reader` | `gguf` |
| `test_reader_agrees_with_reference_parse_gguf` | `numpy` (imported by `reference_logits.py`) |

Per `/AGENTS.md`, read the run for `[SKIP]` lines rather than trusting the exit code — a green
run with the keystone skipped is much weaker than it looks.

## Running

```bash
uv run --no-sync pytest tests/models -q -rs   # -rs shows skip reasons
uv run --no-sync pytest tests/models -k defect
uv run --no-sync pytest tests/models -k mutation
```
