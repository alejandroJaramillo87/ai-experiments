# Cross-language artifact contracts

The contracts the Rust harness (`/rust-harness/`) and the Python tiers (`/interp/`,
`/common/`, `/training/`) agree on, one JSON file each, with the tests that enforce it
named inside. Eight cross the language boundary. Six of those are artifact formats:
each has a writer and a reader in different languages, so no compiler and no
type system connects the two sides. The seventh, `gemma4_facts.json`, is not a
format but a set of architecture values that all three tiers held private
copies of, which is the same problem one level up. The eighth,
`harness_chat_endpoint.json`, is not a file at all: it is the HTTP request and
response shape the harness serves under `--serve` and `common/serving.py`
calls, which is the same problem again with the bytes in flight rather than at
rest.

Two more, `alignment.json` and `attention_capture.manifest.json`, have a Rust
writer and no analysis reader yet. They are here rather than in prose beside
their writers because the analysis each exists for will be Python, and a
contract written before the second side arrives is what keeps the format from
being settled unilaterally by whichever reader is written first.
`alignment.json` has one Python reader so far, the hardware test that decodes
every row against the real vocabulary; `attention_capture.manifest.json` has
none, and its notes say so. That is a stated gap, not the pattern to copy. A
one-sided contract is the weaker kind, because only a second implementation
can catch a shared misunderstanding of the format.

The eleventh, `hf_export.manifest.json`, is written and read by Python alone.
It is here because the `model_hash` it records is the digest the Rust
preflight compares under `--deep` and the interp capture derives at capture
time: three writers of one value in two languages, which is this directory's
problem even though the file itself never crosses.

The files are plain JSON key documents, not JSON Schema. They record which keys
are required, which are emitted by one writer only, and which fields are
deliberately not comparable across implementations. Both sides compare against
them with set logic; no validator dependency is involved.

## Table of Contents

<!-- toc:start -->

1. [Why these exist](#why-these-exist)
2. [The contracts](#the-contracts)
3. [The report envelope, and where it meets these contracts](#the-report-envelope-and-where-it-meets-these-contracts)
4. [How enforcement works](#how-enforcement-works)
5. [Changing a contract](#changing-a-contract)

<!-- toc:end -->

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
`rust-harness/obs/src/lib.rs` rather than through serde. It matched
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

The interp exporters had no coverage in the direction that matters at all:
the probe and control-vector readers had only ever parsed bytes written by
`runtime::synth`, never bytes written by the Python exporters whose output
they exist to read. Both artifacts were GGUF then, and neither is now: the
probe moved to safetensors plus a sidecar at R7, the control vector followed
at R10 once retiring llama.cpp steering left this repository the only reader,
and each contract moved with its artifact rather than the seam being
re-specified in prose.

## The contracts

The table is rendered from the files: the seam is each contract's `description`, and the
direction is derived from its `writers` and `readers` lists, where a path under
`rust-harness/` is Rust, one under the Python tiers or `tests/` is Python, and anything
else (an upstream `config.json`, a transcript under `docs/`) is upstream. A contract that
gains a reader gets a true row without anyone editing prose.

<!-- inventory:contracts:start -->

| Contract | Seam | Direction |
|----------|------|-----------|
| [activation_capture.manifest.json](activation_capture.manifest.json) | manifest.json at the root of an activation-capture directory, beside examples.jsonl and shard_NNNNN.npz. | Rust and Python write, Python reads |
| [alignment.json](alignment.json) | alignment.jsonl written by --alignment-out: one header line, then one line per position of the run. | Rust writes, Python reads |
| [attention_capture.manifest.json](attention_capture.manifest.json) | manifest.json at the root of an attention-capture directory, beside attention.npz. | Rust writes, no reader yet |
| [chat_template.json](chat_template.json) | The Gemma 4 chat template: the byte string a conversation renders to before tokenization, shared by the training tier and the rust-harness prompt builders. | Rust and Python write, Rust and Python read |
| [control_vector_artifact.json](control_vector_artifact.json) | The control vector the Python interp exporter writes and the Rust runtime steers with: an artifact directory holding a safetensors image of per-block f32 directions, plus a manifest.json beside it. | Python writes, Rust and Python read |
| [gemma4_facts.json](gemma4_facts.json) | Gemma 4 architecture facts: the values rust-harness, interp, and training each hold private copies of. | Upstream writes, Rust and Python read |
| [harness_chat_endpoint.json](harness_chat_endpoint.json) | the OpenAI-shaped HTTP surface the rust harness serves under --serve, and that common/serving.py calls. | Rust writes, Python reads |
| [hf_export.manifest.json](hf_export.manifest.json) | The manifest python -m training export writes beside every exported checkpoint directory (<out_root>/<name>/manifest.json next to <out_root>/<name>/hf/): where the weights came from, how they were produced, and model_hash, the sha256 of the written model.safetensors. | Python writes, Python reads |
| [obs_record.json](obs_record.json) | Run-level observability JSONL under ${OBS_LOG_DIR}/<cli>/: one event file per run plus an appended run catalog. | Rust and Python write, Python reads |
| [probe_artifact.json](probe_artifact.json) | The probe the Python interp exporter writes and the Rust runtime scores with: a safetensors image of three f32 vectors, plus a JSON sidecar beside it at the same path with a .json extension. | Python writes, Rust reads |
| [run_metrics.json](run_metrics.json) | A run's per-step metrics file: one shared report-envelope header record followed by one record per step. | Rust and Python write, Python reads |

<!-- inventory:contracts:end -->

Each file carries a `guarded_by` list naming the tests that assert it, and a
`notes` object recording the divergences that are intentional. A field listed
under `notes` as not comparable across writers is a decision, not an oversight:
`dataset_sha256` hashes a different domain on each side, and
`tokenizer_fingerprint` is built from different parts. Tests pin those
differences, so a one-sided change to either fails.

## The report envelope, and where it meets these contracts

The JSON *report* envelope documented in `/docs/observability/report_schema.md` is a
different mechanism from these files, and confusing the two is the easiest mistake to make
here. The envelope — `schema_version`, a discriminator, a timestamp, and `git_sha` — is
checked at runtime by `validate_report` in `/common/reports.py`. These files are
cross-language key inventories checked in tests, with no validator and no runtime cost.

The two meet at the metrics file, whose first record is the envelope and whose remaining
records are per-step data. `run_metrics.json` covers that file; `obs_record.json` covers the
run event stream and the run catalog. They are separate contracts because they are separate
files with separate readers, and a run has one of each.

That split used to be a gap rather than a contract: the envelope was described here as "a
Python-only convention", built by a single call to `report_header()`, and the metrics record
was covered by nothing. The harness made it a cross-language shape — see
`run_metrics.json`'s `notes.why_this_file_exists` — at which point the argument for leaving
it uncovered expired.

Two related claims that were also true only while Python was the sole writer, and are not
now: `obs_record.json` does carry a `schema_version` (introduced at 2 by the `run_dir`
change), and the envelope's `git_sha` is produced on the Rust side by
`runtime::envelope::git_sha()`, which matches `/common/provenance.py` including its
`"unknown"` fallback.

## How enforcement works

Each contract is asserted at both writers rather than at the reader, because
the readers are permissive by design and absorb drift silently.

- **Rust side.** Tests read the contract from `../../schemas/` relative to
  their crate root. When the file is absent — as in the container build, whose
  context is `rust-harness/` alone — they report `[SKIP]` and pass, rather than
  failing on an environment difference.
- **Python side.** Tests under `/tests/interp/`, `/tests/common/`,
  `/tests/training/`, `/tests/models/` and `/tests/hardware/` assert the same
  key sets against the Python writers; each contract's `guarded_by` names its
  own.
- **End to end.** Two tests run the real code on the other side of the seam:
  `python_exporter_contract.rs` invokes the actual `interp` exporters and parses
  the result with the Rust loaders, and `activation_capture.rs` pipes a
  Rust-written capture through `interp.probes.capture.load_capture`.

The end-to-end tests need a `python3` on `PATH` that can import numpy and
safetensors, and need the parent tree present. They skip cleanly without
either.

`harness_chat_endpoint.json` is the one contract no test reads. Its
`guarded_by` tests pin the request and response shapes in code on both sides,
so the file is the document the two were checked against by hand rather than a
fixture either loads; a change to the endpoint edits it in the same commit for
the same reason as the rest.

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
