# runtime

The inference engine as a library crate: the checkpoint reader, the tokenizer, the KV
cache, the sampler, grammar-constrained decoding, sessions, and the `Model` trait that
separates all of that from the forward pass. `src/model/cpu.rs` is the one backend, a
hand-rolled f32 forward pass over a Hugging Face Gemma 4 checkpoint; `src/model/mock.rs` is
the scripted stand-in everything above the trait is tested against. The crate carries
`#![forbid(unsafe_code)]`, which is why the run log's `flock` lives in `obs` instead.

Two modules hold verdict logic rather than machinery, so the binaries can stay drivers:
`ladder.rs` is the seven-rung correctness ladder against `transformers`, and
`conformance.rs` the invariants every backend owes. The seven utility binaries under
`src/bin/` share the flag parser in `cli.rs`; `checkpoint-info`, `activation-capture`,
`attention-capture`, `ladder`, `layer-diff`, `long-context-golden` and `tiny-model`
are each rostered below with what they do.

## Sources

<!-- inventory:rust:start -->

| File | Tests | Purpose |
|------|-------|--------|
| `src/align.rs` | 8 | Incremental detokenization with byte spans. |
| `src/bin/activation-capture.rs` | 5 | Write last-token residual activations in the interp capture shard format. |
| `src/bin/attention-capture.rs` | 6 | Write post-softmax attention weights for chosen layers and query heads over one prompt. |
| `src/bin/checkpoint-info.rs` | 0 | Dump a Hugging Face checkpoint's architecture and tensors. |
| `src/bin/ladder.rs` | 3 | Run the seven-rung correctness gate against a real checkpoint. |
| `src/bin/layer-diff.rs` | 0 | Compare the CPU backend's residual stream against transformers. |
| `src/bin/long-context-golden.rs` | 0 | Record the logits of a long prefill, as a tripwire. |
| `src/bin/tiny-model.rs` | 0 | Write the synthetic tiny-model fixtures to a path. |
| `src/checkpoint.rs` | 33 | Hugging Face checkpoint reader: config.json and model.safetensors. |
| `src/cli.rs` | 20 | The command line of a runtime binary, as data. |
| `src/conformance.rs` | 5 | Invariants every backend owes, stated once. |
| `src/digest.rs` | 5 | SHA-256, for artifacts that have to prove they were not edited. |
| `src/envelope.rs` | 1 | The provenance fields every written artifact carries. |
| `src/eval.rs` | 10 | Output-correctness evaluation: perplexity over a fixed corpus. |
| `src/grammar.rs` | 0 | Tool-call grammar compilation and token masking. |
| `src/kv.rs` | 14 | KV cache sizing for interleaved sliding-window/global attention. |
| `src/ladder.rs` | 51 | The seven-rung correctness gate against a reference implementation. |
| `src/lib.rs` | 2 | Inference engine library crate. |
| `src/metrics.rs` | 12 | Comparison arithmetic for checking our numbers against a reference. |
| `src/model/cpu.rs` | 99 | Hand-rolled CPU reference backend for Gemma-4-style checkpoints. |
| `src/model/mock.rs` | 9 | Scripted `Model` implementation for Tier 2 tests. |
| `src/model/mod.rs` | 0 | The seam between the harness stack and the forward pass. |
| `src/model/probe.rs` | 15 | In-band linear probe monitor over the residual-stream tap. |
| `src/model/record.rs` | 19 | Flight recorder over the `Model` trait. |
| `src/npz.rs` | 3 | Minimal writer for numpy .npy arrays inside a .npz (zip) container. |
| `src/pyenv.rs` | 3 | Interpreter resolution for the cross-language contract tests. |
| `src/reference.rs` | 17 | Reading the artifacts the ladder compares against. |
| `src/sampler.rs` | 21 | Temperature/top-k/top-p sampling with optional grammar masking. |
| `src/session.rs` | 27 | Conversation handle owning the KV cache. |
| `src/status.rs` | 7 | The six-tag status line, shared by every binary in the workspace. |
| `src/synth.rs` | 9 | Synthetic tiny-model checkpoint builder for tests and sandbox runs. |
| `src/tensor.rs` | 9 | Tensor payload access and dequantization to f32. |
| `src/testlog.rs` | 1 | Status lines for the cross-language contract tests. |
| `src/tokenizer.rs` | 19 | Thin wrapper over the Hugging Face `tokenizers` crate. |

<!-- inventory:rust:end -->

## Integration tests

The offline tier of the correctness ladder (`ladder_tiny.rs`, against the committed
`transformers`-authored fixture), the golden tests over the synthetic checkpoints, the
long-context and conformance suite, the fixture writers, every binary's command line
from outside the process, and the cross-language contracts that hand a written artifact
to the real Python reader. A `[SKIP]` line names a Python dependency the resolved
interpreter could not import.

<!-- inventory:rust_tests:start -->

| File | Tests | Covers |
|------|-------|--------|
| `tests/activation_capture.rs` | 8 | End-to-end coverage of the activation-capture binary: dataset load, the last-row tap, shard slicing, and manifest writing, driven against the committed tiny GGUF fixture. |
| `tests/attention_capture.rs` | 6 | End-to-end coverage of the attention-capture binary: the tap, the dense-matrix encoding, span_first, and the manifest, driven against the committed mixed sliding/global fixture. |
| `tests/binary_cli_surface.rs` | 6 | The seven binaries' command line, asserted from outside. |
| `tests/capture_obs.rs` | 5 | The three instrumented capture binaries' run records. |
| `tests/checkpoint_info_check.rs` | 5 | The --check flag's exit-code contract. |
| `tests/cpu_reference.rs` | 12 | Phase 3 golden test: the CPU backend's logits match an independent numpy implementation of the same forward pass, computed from the identical committed checkpoint. |
| `tests/fixture_writers.rs` | 7 | What tiny-model and long-context-golden actually wrote. |
| `tests/fork.rs` | 3 | Tier 2 tests for KV cache forking at the Session level. |
| `tests/grammar.rs` | 17 | Tier 2 tests for grammar-constrained tool-call decoding. |
| `tests/hf_format_smoke.rs` | 8 | Pins the safetensors and tokenizers API surface the checkpoint port depends on, so a dependency bump that changes behaviour fails here rather than inside the loader. |
| `tests/ladder_flags.rs` | 7 | The four tuning flags, and the claims their help text makes. |
| `tests/ladder_obs.rs` | 4 | The ladder's run records, measured against a real invocation. |
| `tests/ladder_reference_contract.rs` | 7 | The ladder's constants against the generators that have to agree with them. |
| `tests/ladder_size.rs` | 1 | --size reaches the run, and is recorded as the run used it. |
| `tests/ladder_tiny.rs` | 5 | The CPU backend against a Gemma 4 transformers authored. |
| `tests/layer_diff_numerics.rs` | 6 | The verdict layer-diff reaches, not the help it prints. |
| `tests/long_context.rs` | 18 | The forward pass at lengths where the window engages. |
| `tests/python_exporter_contract.rs` | 6 | The real Python interp exporters against the Rust loaders. |
| `tests/record_replay.rs` | 3 | Phase 1 acceptance: a recorded run, loaded back as a MockModel script, reproduces the original token sequence through a full Session generation. |
| `tests/recording_replay.rs` | 1 | A flight recording from a real run, loaded back as a MockModel script, states what the run did with it. |
| `tests/status_python_parity.rs` | 1 | The six-tag table is one format, not two. |
| `tests/tiny_reference_manifest.rs` | 3 | The Tier T fixture is what its manifest says. |
| `tests/tokenizer_golden.rs` | 6 | Token-id goldens over the committed BPE fixture. |
| `tests/tracing_spans.rs` | 9 | Phase 2 acceptance (runtime side): the span schema is asserted with a capturing subscriber over a full mock generation. |

<!-- inventory:rust_tests:end -->

## Related documentation

- [../README.md](../README.md) — the workspace
- [../../docs/rust-harness/testing.md](../../docs/rust-harness/testing.md) — the tiers,
  the gate and the interpreter resolution
- [../../docs/rust-harness/correctness-ladder.md](../../docs/rust-harness/correctness-ladder.md) —
  every rung, its measures and its tolerances
- [../../docs/rust-harness/conformance.md](../../docs/rust-harness/conformance.md) — the
  backend invariants and the cross-language contracts
- [../fixtures/README.md](../fixtures/README.md) — what the tests load
