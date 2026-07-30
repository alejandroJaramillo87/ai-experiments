# Rust harness as an interpretability substrate

Design for running the write-back interpretability path (steering, probes,
monitoring) on the `rust-harness` runtime instead of llama.cpp. Bridges the
[interp platform](writeback.md) and the
[rust harness observability design](../inference/rust-harness-observability.md).

## Table of contents

- [Motivation](#motivation)
- [The tap seam](#the-tap-seam)
- [Steering](#steering)
- [Activation capture](#activation-capture)
- [In-band monitoring](#in-band-monitoring)
- [What stays in Python](#what-stays-in-python)
- [Validation](#validation)
- [Constraint on future backends](#constraint-on-future-backends)
- [Roadmap](#roadmap)

## Motivation

The llama.cpp write-back path works through a foreign binary, which forces
three workarounds: a logging reverse proxy to observe requests, activation
capture in a separate bf16 workbench (so probes train on activations from a
different numerical world than the quantized serving path), and interventions
limited to launch flags. The rust runtime is owned code behind narrow trait
seams, so the same three capabilities become in-band:

1. **Tap**: activations are recorded from the serving path itself, quantized
   numerics included, closing the bf16-versus-quantized capture gap by
   construction.
2. **Steer**: control vectors load from the same GGUF artifact
   `interp/steering` already exports, applied inside the forward pass.
3. **Monitor**: a probe dot-product per position is one `Tap` implementation,
   with no proxy or log-replay machinery.

No SAE artifacts are required: contrastive direction extraction and linear
probes need only captures produced by the harness itself. Published SAEs
(when they exist for the served model) would slot in as another source of
directions through the same control-vector format.

## The tap seam

`runtime/src/model/cpu.rs` defines the seam:

```rust
pub trait Tap: Send {
    fn on_residual(&mut self, layer: usize, position: usize, x: &[f32]);
}
```

The tap fires after each decoder layer's final residual add. This matches
the hidden state nnsight exposes as `block.output[0]`, which is what
`interp/probes/capture.py` records — probes trained on Python captures and
rust captures observe the same tensor. Taps are optional (`None` costs one
branch per layer), do not transfer across forks, and fire after steering is
applied, so captures observe the stream downstream layers actually consume.

## Steering

`ControlVector::from_gguf` reads the llama.cpp control-vector layout that
`interp/steering/export.py` writes: one f32 tensor per layer named
`direction.<layer>` (1-based), width equal to the embedding dimension.
`scale * direction` is added to the residual after each named layer.
Steering carries across forks, so a steered agent's subagents serve the same
steered model. Binary flags:

```bash
$ harness --model model.gguf --control-vector vector.gguf \
    --control-vector-scale 1.0 <task>
```

## Activation capture

The `activation-capture` binary writes the exact artifact directory
`interp/probes/capture.py` produces, so the Python probe toolkit (train,
registry, evaluate) consumes rust-served activations unchanged:

```bash
$ cargo run -p runtime --bin activation-capture -- \
    --model model.gguf --dataset examples.jsonl --layers 3,5 --out <dir>
```

- Shards `shard_00000.npz` and onward hold `layer_<L>` float16 rows (last
  prompt token per example, matching the Python capture's
  `position: "last"`) plus int64 `labels`, 2048 examples per shard.
- `examples.jsonl` and a schema-1 `manifest.json` complete the directory.
  The manifest records `capture_backend: "rust-harness-cpu"` and a
  `model_hash` over the GGUF file.
- The `tokenizer_fingerprint` derives from the GGUF vocabulary, not the
  Hugging Face tokenizer object, so it is not comparable to fingerprints
  from Python captures. It keeps rust-captured artifacts internally
  consistent across capture, train, and score, which is the property the
  fingerprint check protects.

Verified in the sandbox: `interp.probes.capture.load_capture()` loads a
rust-written capture of the synthetic fixture with correct shapes, dtypes,
labels, and texts.

## In-band monitoring

A trained linear probe runs inside the forward pass as one `Tap`
implementation (`runtime/src/model/probe.rs`). The exporter
`interp/probes/export.py` converts a registered probe artifact
(`probe.npz` plus manifest, hash-validated on load) into a single GGUF
file: architecture `probe`, f32 tensors `probe.weights`, `probe.mu`, and
`probe.sigma`, and KV metadata `probe.bias`, `probe.layer` (0-based),
`probe.threshold`, `probe.name`, and `probe.model_hash`. The rust side
scores with the same math as `interp/probes/score.py` —
`sigmoid(((x - mu) / sigma) . w + bias)`, flagged when strictly above the
trained threshold — and the parity is golden-tested against the numpy
reference.

```bash
$ harness --model model.gguf --probe probe.gguf \
    [--probe-out flags.jsonl] <task>
```

Every position of the probe's layer is scored, a granularity the Python
monitor (whole-text replay, last token per request) cannot reach. Flagged
positions are written as JSONL records (`probe`, `layer`, `position`,
`score`, `threshold`; default `probe-flags.jsonl` in the working
directory) and a one-line summary goes to stderr. The monitor observes
only: monitored and unmonitored runs produce bit-identical logits, which
the golden test asserts. Because taps do not transfer across forks, a
monitored agent's subagents run unmonitored; extending monitoring across
forks is future work.

## What stays in Python

The analysis half is unchanged and remains fed by capture directories:
probe fitting (`interp/probes/train.py`), the registry and evaluation,
direction extraction math, the steering scorecard and its LLM judge, and the
SAE ecosystem (Gemma Scope, SAEBench, Neuronpedia). These are offline
consumers of artifacts; they do not care which binary produced the shards.

## Validation

The numerics validation extends the phase 3 pattern
(`rust-harness/fixtures/reference_logits.py`): the numpy reference emits the
last-position per-layer residuals and the last-position logits under a fixed
synthetic control vector, and golden tests
(`runtime/tests/cpu_reference.rs`) assert the CPU backend matches both
within the established 2e-4 tolerance. Property tests assert taps do not
perturb logits, a zero control vector is bit-identical to no steering, and
steering is deterministic and carries across forks.

The artifact flow is also verified end to end in the sandbox: a control
vector written by `interp/steering/export.py` (`write_control_vector_gguf`)
loads through `--control-vector`, and the harness binary run over the
synthetic fixture produces output that is deterministic across repeats and
differs from the unsteered run. Repeat runs must use a fresh session root:
`session.jsonl` persists the transcript by design, so a second run in the
same directory continues the session rather than reproducing it.

## Constraint on future backends

The tap seam is part of the `Model` backend contract from now on: the CUDA
backend (roadmap phase A) must implement `Tap` delivery with the same
semantics (post-layer residual, steering applied first). On the GPU this
implies a device-to-host copy per tapped layer; taps stay optional so the
untapped serving path pays nothing.

## Roadmap

| Step | Status | Depends on |
|------|--------|------------|
| Tap seam, steering, capture, goldens (sandbox) | done | — |
| Real-model capture and contrastive extraction on the workstation | pending | E4B architecture reconciliation, CPU backend speed measurement |
| In-band probe monitor as a `Tap` implementation (sandbox) | done | — |
| Monitor with a probe trained on real-model rust captures | pending | real-model capture on the workstation |
| CUDA tap parity | pending | roadmap phase A |
