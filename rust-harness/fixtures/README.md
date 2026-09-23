# fixtures

The committed inputs the workspace's tests load, and the generators that wrote them.
Everything here is small enough to commit and is committed, with one exception: the
Tier E ladder references, an `e4b/` directory under `reference/`, are captures of the real checkpoint,
generated on the workstation per run and gitignored.

Three kinds of fixture sit side by side. The synthetic checkpoints (`tiny-gemma4/`,
`tiny-gemma4-e4b/`, with their `*.tokenizer.json` and `*-logits.json`) are written by
the `tiny-model` binary and scored by `reference_logits.py`, an independent numpy
forward pass; they pin the CPU backend against a second implementation. The
`transformers`-authored checkpoint (`tiny-hf-gemma4/`) is written by
`build_tiny_reference.py` on the workstation and carries the reference's own forward
pass, so the sandbox can measure against Gemma 4 itself. The interp artifacts
(`python-probe.*`, `python-control-vector/`) are written by the real Python exporters
through `export_interp_fixtures.py`, so the Rust readers are checked against the
writers they exist to read rather than against a Rust approximation of them.

## Directories

<!-- inventory:dirs:start -->

| Directory | Holds |
|-----------|-------|
| [python-control-vector/](python-control-vector/README.md) | A control-vector artifact exactly as `interp/steering/export.py` writes it: the safetensors image of one direction per steered block and its JSON manifest, held to `schemas/control_vector_artifact.json`. |
| [reference/](reference/README.md) | Where the correctness ladder's reference artifacts live, and the two committed inputs they are generated from. |
| [tiny-gemma4/](tiny-gemma4/README.md) | A synthetic Gemma 4 checkpoint with random weights, written by the `tiny-model` binary from the default `TinySpec` in `runtime/src/synth.rs`. |
| [tiny-gemma4-e4b/](tiny-gemma4-e4b/README.md) | The synthetic checkpoint shaped like the released E4B config (interleaved sliding and global layers, shared KV, per-layer embeddings, MatFormer widths) at toy dimensions, written by the `tiny-model` binary and read by `runtime/tests/cpu_reference.rs`, `capture_obs.rs` and `attention_capture.rs`. |
| [tiny-hf-gemma4/](tiny-hf-gemma4/README.md) | The tiny Gemma 4 that `transformers` itself constructed and saved, with `capture.json` holding the reference forward pass over it and `manifest.json` the sha256 of every file, the generator, the seed and the torch and `transformers` versions. |

<!-- inventory:dirs:end -->

## Generators

Every `.py` here runs from the repository's Python environment; the `reference_*.py`
family needs torch and `transformers` and therefore the workstation.

<!-- inventory:scripts:start -->

| Script | Purpose |
|--------|---------|
| `audit_recording.py` | Audit a flight recording against transformers, token by token. |
| `build_tiny_reference.py` | Build the tiny Gemma 4 that transformers itself authors, and capture it. |
| `export_interp_fixtures.py` | Write the probe and control-vector fixtures with the real exporters. |
| `fetch_corpus.py` | Fetch and normalize the public-domain corpus L3 scores. |
| `reference_attention.py` | Capture transformers' post-softmax attention rows past the window, for L6. |
| `reference_capture.py` | Capture transformers' own residual stream and logits for one prompt. |
| `reference_common.py` | Facts every ladder reference artifact records, written once. |
| `reference_greedy.py` | Capture transformers' greedy continuations for L4, the ladder's top rung. |
| `reference_logits.py` | Independent reference forward pass for the tiny synthetic models. |
| `reference_long_context.py` | Capture transformers' logits past the sliding window, for L5. |
| `reference_perplexity.py` | Capture the reference implementation's per-token NLL for L3. |
| `reference_tokenizer.py` | Capture the reference tokenizer's ids for L0, the ladder's first rung. |

<!-- inventory:scripts:end -->

## Related documentation

- [../../docs/rust-harness/fixtures.md](../../docs/rust-harness/fixtures.md) — every
  file here: what writes it, what reads it, and what the goldens are worth
- [../../docs/rust-harness/correctness-ladder.md](../../docs/rust-harness/correctness-ladder.md) —
  the rungs the reference artifacts feed
- [../README.md](../README.md) — the workspace
