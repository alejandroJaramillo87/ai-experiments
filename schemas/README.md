# Cross-language artifact contracts

Ten contracts. Seven cross the boundary between the Rust harness
(`/rust-harness/`) and the Python training and interpretability pipelines
(`/interp/`, `/common/`, `/training/`). Five of those are artifact formats:
each has a writer and a reader in different languages, so no compiler and no
type system connects the two sides. The sixth, `gemma4_facts.json`, is not a
format but a set of architecture values that all three tiers held private
copies of, which is the same problem one level up. The seventh,
`harness_chat_endpoint.json`, is not a file at all: it is the HTTP request and
response shape the harness serves under `--serve` and `common/serving.py`
calls, which is the same problem again with the bytes in flight rather than at
rest. This directory holds the contract each pair agrees on, and every contract
names the tests that enforce it.

`bench_models.json` is the tenth, and it is here for `gemma4_facts.json`'s reason
rather than a cross-language one: it is not a format but a set of architecture values
that a benchmark report must carry and that nothing else in the tree states.
`training/models/families.py` covers LoRA targets and chat templates, and the only
machine-readable dense/MoE distinction before it was a refusal in
`training/calculator/components.py`. The cost of leaving it unstated is on record:
`docs/optimizations/README.md`'s headline CPU figure was measured against a
mixture-of-experts model and says so nowhere.

The remaining two, `alignment.json` and `attention_capture.manifest.json`,
have a Rust writer and no reader in either language yet. They are here rather
than in prose beside their writers because the analysis each exists for will
be Python, and a contract written before the second side arrives is what keeps
the format from being settled unilaterally by whichever reader is written
first. Both `readers` lists are empty and both sets of notes say so; that is a
stated gap, not the pattern to copy. A one-sided contract is the weaker kind,
because only a second implementation can catch a shared misunderstanding of
the format.

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

| File | Seam | Direction |
|------|------|-----------|
| `activation_capture.manifest.json` | Capture directory: `manifest.json`, `examples.jsonl`, `shard_NNNNN.npz` | Both write, Python reads |
| `control_vector_artifact.json` | A control vector: a safetensors image of per-block directions plus a sibling manifest | Python writes, Rust reads |
| `probe_artifact.json` | A probe: safetensors parameter vectors plus a JSON sidecar of scalars | Python writes, Rust reads |
| `obs_record.json` | Run-level JSONL: event file and run catalog | Both write, Python reads |
| `run_metrics.json` | A run's `metrics.jsonl`: the report-envelope header and the per-step records the dashboard charts | Both write, Python reads |
| `chat_template.json` | The Gemma 4 chat template: the prompt string a conversation renders to | Both write, both read |
| `gemma4_facts.json` | Gemma 4 architecture values: layer counts, head dimensions, the attention schedule, RoPE regimes | Google writes, all three tiers read |
| `harness_chat_endpoint.json` | The harness `--serve` HTTP surface: `/v1/chat/completions` and `/v1/health` | Rust serves, Python calls |
| `alignment.json` | Per-position byte spans and probe scores for one run | Rust writes, no reader yet |
| `attention_capture.manifest.json` | Attention-capture directory: `manifest.json` and `attention.npz` | Rust writes, no reader yet |
| `bench_models.json` | Architecture facts for the benchmarked models: dense or MoE, total and active parameters, expert counts, quantization | Hand-maintained, Python reads |

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
