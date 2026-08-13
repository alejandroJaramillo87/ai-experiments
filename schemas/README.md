# Cross-language artifact contracts

Six contracts. Five cross the boundary between the Rust harness
(`/rust-harness/`) and the Python training and interpretability pipelines
(`/interp/`, `/common/`, `/training/`). Four of those are artifact formats:
each has a writer and a reader in different languages, so no compiler and no
type system connects the two sides. The fifth, `gemma4_facts.json`, is not a
format but a set of architecture values that all three tiers held private
copies of, which is the same problem one level up. This directory holds the
contract each pair agrees on, and every contract names the tests that enforce
it.

The sixth, `alignment.json`, has a Rust writer and no reader in either
language yet. It is here rather than in prose beside the writer because the
longitudinal analysis it exists for will be Python, and a contract written
before the second side arrives is what keeps the format from being settled
unilaterally by whichever reader is written first. Its `readers` list is empty
and its notes say so; that is a stated gap, not the pattern to copy.

The files are plain JSON key documents, not JSON Schema. They record which keys
are required, which are emitted by one writer only, and which fields are
deliberately not comparable across implementations. Both sides compare against
them with set logic; no validator dependency is involved.

## Table of contents

- [Why these exist](#why-these-exist)
- [The contracts](#the-contracts)
- [Not the report envelope](#not-the-report-envelope)
- [How enforcement works](#how-enforcement-works)
- [Changing a contract](#changing-a-contract)

## Why these exist

Before these files, all three seams were specified in prose and enforced by
nothing.

The activation-capture manifest key list was typed out twice, once in
`/interp/probes/capture.py` and once in
`/rust-harness/runtime/src/bin/activation-capture.rs`, with no shared
definition. The Rust test read shards back through a bespoke `np.load` snippet
rather than through `load_capture`, the reader the probe toolkit actually uses,
so none of that reader's consistency guards had ever seen a Rust-written
directory.

The observability catalog line is formatted by hand in
`/rust-harness/harness/src/obs.rs` rather than through serde. It matched
`/common/obs/catalog.py` field for field, by coincidence of two people typing
the same six keys. `read_runs` skips a line it cannot parse, so drift would
have presented as an empty runs table rather than an error.

The chat template is the seam that had already failed. The training tier
corrected Gemma 4 from Gemma 3's `<start_of_turn>` framing to the official
`<|turn>` markers and bumped `CHAT_TEMPLATE_VERSION` to 2; the harness kept
the Gemma 3 constants. Those literals are absent from the Gemma 4 vocabulary,
so every turn boundary tokenized as a byte run instead of a trained id, and
nothing failed: output stayed fluent, and the only golden test was ignored
with an unimplemented body. Training and inference were rendering one
conversation to two different prompts.

The GGUF exporters had no coverage in the direction that matters at all:
`Probe::from_gguf` and `ControlVector::from_gguf` had only ever parsed bytes
written by `runtime::synth::GgufWriter`, never bytes written by the Python
exporters whose output they exist to read.

## The contracts

| File | Seam | Direction |
|------|------|-----------|
| `activation_capture.manifest.json` | Capture directory: `manifest.json`, `examples.jsonl`, `shard_NNNNN.npz` | Both write, Python reads |
| `gguf_artifacts.json` | Probe and control-vector GGUF images | Python writes, Rust reads |
| `obs_record.json` | Run-level JSONL: event file and run catalog | Both write, Python reads |
| `chat_template.json` | The Gemma 4 chat template: the prompt string a conversation renders to | Both write, both read |
| `gemma4_facts.json` | Gemma 4 architecture values: layer counts, head dimensions, the attention schedule, RoPE regimes | Google writes, all three tiers read |
| `alignment.json` | Per-position byte spans and probe scores for one run | Rust writes, no reader yet |

Each file carries a `guarded_by` list naming the tests that assert it, and a
`notes` object recording the divergences that are intentional. A field listed
under `notes` as not comparable across writers is a decision, not an oversight:
`dataset_sha256` hashes a different domain on each side, and
`tokenizer_fingerprint` is built from different parts. Tests pin those
differences, so a one-sided change to either fails.

## Not the report envelope

These contracts are unrelated to the JSON *report* envelope documented in
`/docs/observability/report_schema.md`, and confusing the two is the easiest mistake to make
here. The envelope is a Python-only convention — `schema_version`, a discriminator, a
timestamp, and `git_sha` — checked at runtime by `validate_report` in `/common/reports.py`.
These files are cross-language key inventories checked in tests, with no validator and no
runtime cost. There is no key in common between the two: `obs_record.json` has no
`schema_version` or `git_sha`, and the envelope has no `run_id`.

The one place they meet is the metrics file, whose first record is built with
`report_header("run_metrics")` and must satisfy the envelope. That record is **not** covered
by any file in this directory; `obs_record.json` covers the run event stream and the run
catalog only.

## How enforcement works

Each contract is asserted at both writers rather than at the reader, because
the readers are permissive by design and absorb drift silently.

- **Rust side.** Tests read the contract from `../../schemas/` relative to
  their crate root. When the file is absent — as in the container build, whose
  context is `rust-harness/` alone — they report `[SKIP]` and pass, rather than
  failing on an environment difference.
- **Python side.** Tests under `/tests/interp/` and `/tests/common/` assert the
  same key sets against the Python writers.
- **End to end.** Two tests run the real code on the other side of the seam:
  `python_exporter_contract.rs` invokes the actual `interp` exporters and parses
  the result with the Rust loaders, and `activation_capture.rs` pipes a
  Rust-written capture through `interp.probes.capture.load_capture`.

The end-to-end tests need a `python3` on `PATH` that can import numpy and gguf,
and need the parent tree present. They skip cleanly without either.

Adding a key to one writer alone now fails that writer's contract test.
Updating the contract to accept it fails the other writer's test until it
follows. That two-step failure is the enforcement; a single shared file that
only one side reads would not provide it.

## Changing a contract

1. Decide whether the change is compatible. A new optional key that no reader
   requires is compatible; a renamed, retyped, or removed key is not.
2. For an incompatible change, bump `schema_version` in the artifact and say
   what changed in the contract's `notes`.
3. Update both writers and the contract in one commit, so no commit leaves the
   two sides disagreeing.
4. Run `make -C rust-harness quality` and the Python test modules named in
   `guarded_by`.
