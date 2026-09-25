# The fixtures

Every input under `rust-harness/fixtures/` that a test loads: what wrote it, what reads
it, whether it is committed, and what each kind of fixture can and cannot prove. The
directory's own README lists its subdirectories and generators from the tree; this page
is the reasoning a table cannot carry, and the argument for why the committed goldens
are change detectors rather than correctness claims.

## Table of Contents

<!-- toc:start -->

1. [The synthetic checkpoints](#the-synthetic-checkpoints)
2. [The transformers-authored checkpoint](#the-transformers-authored-checkpoint)
3. [The long-context golden](#the-long-context-golden)
4. [The interp artifacts](#the-interp-artifacts)
5. [The ladder reference](#the-ladder-reference)
6. [What the committed goldens are worth](#what-the-committed-goldens-are-worth)
7. [Related documentation](#related-documentation)

<!-- toc:end -->

## The synthetic checkpoints

`tiny-gemma4/` and `tiny-gemma4-e4b/` are Gemma 4 checkpoints at toy dimensions with
random weights, each a `config.json` and a `model.safetensors`, with a `*.tokenizer.json`
beside each and `tiny-bpe.tokenizer.json` for the tokenizer goldens. All are written by
the `tiny-model` binary from a `TinySpec` in `runtime/src/synth.rs`, deterministically in
the spec's seed, so regenerating reproduces identical bytes:

```bash
cd rust-harness
cargo run -p runtime --bin tiny-model -- fixtures/tiny-gemma4
cargo run -p runtime --bin tiny-model -- fixtures/tiny-gemma4-e4b e4b
cargo run -p runtime --bin tiny-model -- fixtures/tiny-bpe.tokenizer.json bpe
```

The uniform fixture declares none of Gemma 4's optional features and proves they are
absent-safe; the E4B-shaped one declares all seven (interleaved sliding and global
attention, the shared-KV tail, per-layer embeddings, MatFormer feed-forward widths,
dual RoPE, the norms and scalars) and proves they are right. `fixtures/reference_logits.py`
is an independent numpy forward pass over the same files and writes
`tiny-gemma4-logits.json` and `tiny-gemma4-e4b-logits.json`; `runtime/tests/cpu_reference.rs`
regenerates each checkpoint, runs the CPU backend and compares row by row, localizing a
mismatch to the first divergent position. `tokenizer_golden.rs` pins the BPE segmentation
and the special-literal allowlist over the BPE fixture with ids written out rather than
recomputed, so a `tokenizers` upgrade that changes segmentation fails with the before and
after. `fixture_writers.rs` reads each writer's output back through the consumer that
will read it for real, because an exit code cannot distinguish a checkpoint from a file
of the right name holding a truncated header. Committed: the checkpoints are kilobytes,
and `.gitignore` carves `rust-harness/fixtures/**/*.safetensors` back out of the
model-file ignore for them.

## The transformers-authored checkpoint

`tiny-hf-gemma4/` is the oracle nobody here wrote. `build_tiny_reference.py` supplies
shapes read off the released E4B `config.json` and seeded weights, and `transformers`
constructs the config from its own classes, instantiates its own Gemma 4, saves it
(`config.json`, `model.safetensors`, `generation_config.json`) and runs its own forward
pass, captured as `capture.json`. `manifest.json` records the sha256 of each of those,
of the generator itself, the seed (20260810) and the torch and `transformers` versions;
`runtime/tests/tiny_reference_manifest.rs` holds the fixture to its manifest. Workstation
only to write, since it needs torch; committed whole so the sandbox can measure against a
real Gemma 4 without a GPU or a download. Read by `ladder_tiny.rs` as the Tier T oracle
and by every offline test that needs a real Gemma 4 rather than a synthetic one:
`ladder_flags.rs`, `ladder_obs.rs`, `ladder_size.rs`, `layer_diff_numerics.rs`,
`capture_obs.rs`, `tracing_spans.rs`, and `runtime/src/checkpoint.rs`'s own tests. The
Python side reads it too: `reference_attention.py` and `reference_long_context.py`
prove their own windowing on it, and `tests/common/fixtures/` documents the copy the
training export test hands to the binaries.

## The long-context golden

`long-context-logits.json` is a third kind, and the distinction matters. It is not
compared against any reference: it holds rows this engine produced at 200 tokens over
the E4B-shaped spec with a window of 3, at six positions (both ends, both sides of the
first two group boundaries, one interior), recorded once by `long-context-golden` so
that a later change to how the KV cache is stored has to either reproduce them or
re-bless them deliberately. `runtime/src/conformance.rs::golden` is the one definition
of the spec, the token count and the positions, read by both the writer and
`runtime/tests/long_context.rs`, so the comparison cannot drift into two different
things. What keeps it from freezing a wrong answer is ordering: the window convention it
encodes was checked against `transformers` first, at 150 tokens over fifty crossings of
the Tier T fixture's window, and it was blessed only after that passed. It has done its
job once: capping sliding-layer KV storage at the window rewrote how the cache is stored
and indexed, and these bytes reproduced unchanged.

## The interp artifacts

`python-probe.json` with `python-probe.safetensors`, and `python-control-vector/`
(`manifest.json`, `vector.safetensors`), are written by the real exporters,
`interp/probes/export.py` and `interp/steering/export.py`, through
`export_interp_fixtures.py`, held to `schemas/probe_artifact.json` and
`schemas/control_vector_artifact.json`. Every other golden synthesizes bytes in Rust,
which checks a parser against a second Rust implementation rather than against the
writer it exists to read; these check it against the writer. The values are dyadic
rationals, exact in float32, so `runtime/tests/python_exporter_contract.rs` compares
for equality, and `activation_capture.rs` uses the probe end to end. Regenerate with
`python3 rust-harness/fixtures/export_interp_fixtures.py rust-harness/fixtures`; needs
numpy and safetensors, no GPU, no checkpoint, no network.

## The ladder reference

`rust-harness/fixtures/reference/corpus.txt` and `prompts.txt` beside it are committed and are the inputs to
L3 and L4; an `e4b/<size>/` directory under `reference/` holds the six Tier E artifacts, generated per run on
the workstation and gitignored because a capture at a 262144-wide vocabulary runs to tens
of megabytes. [correctness-ladder.md](correctness-ladder.md) owns the artifacts, the
corpus, the prompts and their provenance; the directory's README says where things land.

## What the committed goldens are worth

`fixtures/*-logits.json` pin the CPU backend against `fixtures/reference_logits.py`
on the tiny synthetic checkpoints. Both sides are read as a cross-language
check, and that is a real guard against a one-sided slip.

It is not a correctness claim, for two reasons. The fixtures hold random
weights, so a wrong arrangement produces noise that looks exactly like the
right arrangement's noise. And both implementations were transcribed from the
architecture documents by the same reader, so a misreading lands in both and
they agree about it.

That is not hypothetical. Two defects — a `1/sqrt(head_dim)` applied to
attention logits Gemma 4 leaves unscaled, and a missing scale-free norm on the
value states — were present in both implementations through nine work packages
of green gate, and were found by reading `docs/models/gemma4-upstream.txt`
after a real checkpoint produced noise. The second contributes no tensor to a
checkpoint, so no name-based audit of the weights could have found it.

`fixtures/long-context-logits.json` is a third kind, and the distinction
matters. It is not compared against any reference: it holds rows this engine
produced at 200 tokens, recorded once, so that a later package changing how the
KV cache is stored has to either reproduce them or re-bless them deliberately.
A regenerate-and-compare fixture cannot do that job, because regenerating from
the current engine and comparing against the current engine agrees by
definition. Written by `runtime/src/bin/long-context-golden.rs`, read by
`runtime/tests/long_context.rs`.

It is a change detector and not an oracle, for the reason this section already
gives. What keeps it from freezing a wrong answer is ordering: the window
convention it encodes was checked against `transformers` first, at 150 tokens
over fifty crossings of the Tier T fixture's window of 3, and it was blessed
only after that passed.

It has since done its job once. Capping sliding-layer KV storage at the window
rewrote how the cache is stored and indexed, and these bytes reproduced
unchanged — which is the outcome that says the rewrite preserved behaviour,
and the only construct in the workspace that could have said so.

Real-weight verification is `runtime/src/bin/layer-diff.rs` against
`fixtures/reference_capture.py`, which compares the residual stream layer by
layer to transformers itself. Until that reports clean, the goldens are
regression pins and nothing more.

## Related documentation

- [../../rust-harness/fixtures/README.md](../../rust-harness/fixtures/README.md) — the directory, its subdirectories and generators rendered from the tree
- [correctness-ladder.md](correctness-ladder.md) — the rungs the reference artifacts feed
- [conformance.md](conformance.md) — the invariants the long-context golden sits beside
- [testing.md](testing.md) — the tiers these fixtures serve
