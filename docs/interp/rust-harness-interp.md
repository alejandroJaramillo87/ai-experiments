# Rust harness as an interpretability substrate

Design for running the write-back interpretability path (steering, probes,
monitoring) on the `rust-harness` runtime instead of llama.cpp. Bridges the
[interp platform](writeback.md) and the
[rust harness observability design](../rust-harness/observability.md).

## Table of contents

- [Motivation](#motivation)
- [The tap seam](#the-tap-seam)
- [Steering](#steering)
- [Activation capture](#activation-capture)
- [In-band monitoring](#in-band-monitoring)
- [What stays in Python](#what-stays-in-python)
- [Validation](#validation)
- [What per-layer embeddings change](#what-per-layer-embeddings-change)
- [Constraint on future backends](#constraint-on-future-backends)
- [Roadmap](#roadmap)

## Motivation

The llama.cpp write-back path works through a foreign binary, which forces
three workarounds: a logging reverse proxy to observe requests, activation
capture in a separate bf16 workbench (so probes train on activations from a
different numerical world than the quantized serving path), and interventions
limited to launch flags. The rust runtime is owned code behind narrow trait
seams, so the same three capabilities become in-band:

1. **Tap**: activations are recorded from the serving path itself, so the
   capture and the served forward pass are the same computation rather than two
   that resemble each other.
2. **Steer**: control vectors load from the same GGUF artifact
   `interp/steering` already exports, applied inside the forward pass.
3. **Monitor**: a probe dot-product per position is one `Tap` implementation,
   with no proxy or log-replay machinery.

**The capture gap dissolved rather than closed, and the difference matters.**
The original argument for the tap was that it would record quantized numerics,
so a probe would at last see the values the served model actually computes. That
is no longer what happens. The rust runtime reads the published checkpoint and
holds each weight at its published width, so there is no quantization on the
serving path to capture — the bf16 workbench and the rust serving path now
compute in the same numerical world, and the gap is absent instead of
instrumented. The tap is still worth having for the other two reasons: it is
in-band, and it is the same code path that serves. But a probe trained on rust
captures is no longer studying a different model from one trained on a bf16
workbench, and any claim resting on that distinction is void.

The llama.cpp path above remains quantized, so the gap is real there. It is the
reason not to treat the two backends' captures as interchangeable.

No SAE artifacts are required: contrastive direction extraction and linear
probes need only captures produced by the harness itself. Published SAEs
(when they exist for the served model) would slot in as another source of
directions through the same control-vector format.

## The tap seam

`runtime/src/model/cpu.rs` defines the seam:

```rust
#[non_exhaustive]
pub enum Site {
    LayerOutput,
}

pub trait Tap: Send {
    fn on_residual(&mut self, layer: usize, site: Site, position: usize, x: &[f32]);
}
```

`Site` names which point of the layer a row came from. One variant today, and
the seam fires only that one; the field exists because a tap keyed on the layer
alone records "layer 7" for a row whose meaning is fixed by this paragraph
rather than by the value, and a stored row carries that ambiguity into the
artifact. It is `#[non_exhaustive]`, so a second point is additive for a tap in
another crate. Implementations take the site they want by comparison
(`if site != Site::LayerOutput { return; }`); within `runtime` a `match` with a
wildcard arm is an unreachable pattern while the enum has one variant, and the
gate runs `-D warnings`.

The tap fires on the decoder layer's output, which under Gemma 4 is not the
final residual add. A layer applies, in order: the attention residual, the
feed-forward residual, the per-layer-embedding residual, then a multiply by
`layer_output_scale`. The scale is the module's last operation, so the tap
fires after it. Stored scales run 0.0186 to 0.879, a spread of roughly 54x
across the stack, so the distinction is not academic — tapping before the
multiply would report a differently scaled tensor at every layer.

This still matches the hidden state nnsight exposes as `block.output[0]`,
which is what `interp/probes/capture.py` records: the Python forward hook
receives the already-scaled tensor, so no correction factor is needed and
none is applied. Probes trained on Python captures and rust captures observe
the same tensor. Taps are optional (`None` costs one branch per layer), do
not transfer across forks, and fire after steering is applied, so captures
observe the stream downstream layers actually consume.

## Steering

`ControlVector::from_gguf` reads the llama.cpp control-vector layout that
`interp/steering/export.py` writes: one f32 tensor per layer named
`direction.<layer>`, width equal to the embedding dimension. `direction.N`
steers decoder block `N`, matching llama.cpp and the repeng training
semantics that produce these files, so the same `vector.gguf` steers the
same block here as under `llama-cpu` and `llama-gpu`. Numbering starts at
1 and block 0 is unsteerable, as upstream. `scale * direction` is added to
the named block's output, after that block's `layer_output_scale` multiply
and before the tap fires.
Steering carries across forks, so a steered agent's subagents serve the same
steered model. Binary flags:

```bash
$ harness --model <checkpoint-dir> --control-vector vector.gguf \
    --control-vector-scale 1.0 <task>
```

## Activation capture

The `activation-capture` binary writes the exact artifact directory
`interp/probes/capture.py` produces, so the Python probe toolkit (train,
registry, evaluate) consumes rust-served activations unchanged:

```bash
$ cargo run -p runtime --bin activation-capture -- \
    --model <checkpoint-dir> --dataset examples.jsonl --layers 3,5 --out <dir>
```

- Shards `shard_00000.npz` and onward hold `layer_<L>` float32 rows (last
  prompt token per example, matching the Python capture's
  `position: "last"`) plus int64 `labels`, 2048 examples per shard.
  float32 is exact for the bf16 models this tier captures from -- bf16 is
  the high half of an f32, so widening is a shift -- and the npz container
  forces the choice, because numpy has no bfloat16 dtype. That constraint is
  specific to npz and does not generalize: safetensors does carry bf16,
  though only `safetensors.torch` can read it back (`safetensors.numpy`
  raises `KeyError: 'BF16'`). The schema pins the value at
  `/schemas/activation_capture.manifest.json`.
- `examples.jsonl` and a schema-1 `manifest.json` complete the directory.
  The manifest records `site: "layer_output"` — which point of each layer the
  rows were taken at, written from `LastRowTap::SITE` so the stored text cannot
  drift from the value a reader parses back. The shard keys stay `layer_<L>`:
  with one site the key is unambiguous, and renaming it would break
  `load_capture` for no information gain. A second site is what makes the key
  ambiguous, and the schema note records the cost of that day. The manifest also
  records `capture_backend: "rust-harness-cpu"`,
  `model_revision: "safetensors"`, and a `model_hash` over
  `model.safetensors`.
- The `tokenizer_fingerprint` is a sha256 over the model path and vocabulary
  size with no chat template, so it is not comparable to fingerprints from
  Python captures. Both sides now read the same tokenizer.json, so the
  incomparability is a matter of formula rather than of vocabulary. It keeps
  rust-captured artifacts internally consistent across capture, train, and
  score, which is the property the fingerprint check protects.
- `dataset_sha256` is likewise per-backend: Rust hashes the file's raw bytes,
  Python hashes text-mode `f.read()`, which normalizes CRLF. The two agree on
  an LF dataset and differ on a CRLF one, and a test asserts both halves so a
  one-sided change to either hashing domain fails rather than silently
  reinterpreting the manifests already on disk.

The key set is not maintained twice by hand any more:
`schemas/activation_capture.manifest.json` declares the 15 shared keys plus
the Rust-only `capture_backend`, and both writers are asserted against it.

Verified in the sandbox (`runtime/tests/activation_capture.rs`): the binary
writes a capture of the synthetic fixture whose manifest fields, shard names,
and row counts match the format above; rows and labels read back aligned with
their source texts; and the whole directory loads through
`interp.probes.capture.load_capture`, so that reader's count, length, and
repeated-layer guards are exercised against Rust-written output. The tests
that reach the parent tree skip when it or numpy is unavailable; the
artifact-shape half always runs.

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
$ harness --model <checkpoint-dir> --probe probe.gguf \
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

The workload shape matters for what monitored positions mean. Agent tasks
are prefill-heavy: most scored positions are the model reading
externally authored text (task descriptions, tool output, file contents),
where a flag means the model built the probed representation while
ingesting input. Chat mode (`--chat`) is decode-heavy: scored positions
are the model producing its own tokens, which is where intent-like
attributes live. Probes are trained on last-token representations of
complete texts, so their transfer to mid-generation decode positions is
unmeasured; evaluating probe quality separately on prefill and decode
positions is pending real-model captures (roadmap below). The
prefill/decode asymmetry generalizes into a two-frame taxonomy — agent mode
as the ecological frame, chat mode as the controlled frame — recorded in
[observation-frames.md](../rust-harness/observation-frames.md).

Two context-layer mechanics shift what a flag's `position` field means.
Compaction splices a model-written summary over the middle of the
transcript, so the next generation re-prefills the summary region and the
probe scores those positions again; flag positions are KV-cache
positions, not transcript-lifetime positions, and positions recorded
before a compaction do not align with positions recorded after it. A
session restored from `session.kv` skips the prefill of the restored
prefix entirely, so restored positions are never re-scored — a monitored
run that must score the whole conversation should start without a
snapshot file.

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

The artifact flow is now verified in both directions, against the contracts in
`schemas/` (see `schemas/README.md`).

Rust-written captures go through `interp.probes.capture.load_capture`, the
reader the probe toolkit actually calls, rather than through a bespoke
`np.load` — so the reader's consistency guards, which join rows, labels, and
texts by position, are exercised against a Rust-written directory
(`runtime/tests/activation_capture.rs`).

In the other direction, `runtime/tests/python_exporter_contract.rs` loads
probe and control-vector GGUFs written by `interp/probes/export.py` and
`interp/steering/export.py` through `Probe::from_gguf` and
`ControlVector::from_gguf`. Two committed fixtures
(`fixtures/python-probe.gguf`, `fixtures/python-control-vector.gguf`) are
exporter output, so the parse assertions run wherever the workspace builds; a
further test re-runs the exporters and compares parsed content, which is what
notices an exporter that has drifted from the committed bytes. The comparison
is semantic rather than byte for byte, because a `gguf` version bump may
legitimately change padding.

The layer numbering is what made this worth closing. `probe.layer` is 0-based
and `direction.<N>` is 1-based; each side enforced its own rule and nothing
checked that the two compose. They do — a fixture writing `direction.1` and
`direction.3` lands at blocks 1 and 3 with block 2 unsteered.

Both cross-tree tests need `python3` with numpy and gguf plus the parent tree,
and report `[SKIP]` without them. Runbook entries RH.9c and RH.9d remain the
workstation check against a real checkpoint; what they no longer carry alone
is the format contract.

A separate practical note for those entries: repeat runs must use a fresh
session root. `session.jsonl` persists the transcript by design, so a second
run in the same directory continues the session rather than reproducing it.

## What per-layer embeddings change

Published descriptions of Gemma 4 E4B retain per-layer embeddings (PLE) and
drop AltUp; the
[architecture reconciliation](../rust-harness/runtime-design.md#architecture-reconciliation)
records the sources and confidence. Both findings matter here, in opposite
directions.

Dropping AltUp preserves the seam. AltUp maintains several parallel
hidden-state streams per layer, so `on_residual` would have no single
residual vector to deliver, and the signature, the capture shard format, and
the `block.output[0]` correspondence would all have to be renegotiated across
the language boundary. One residual stream survives, so none of that changes.

Retaining PLE shifts what a tapped vector means. Each layer sums a
token-identity embedding lookup with a projection of the input embedding and
adds the result to the residual stream after the feed-forward residual. The
tap point stays well defined, and it stays the same tensor the Python capture
path records, because the injection lands before both the
`layer_output_scale` multiply and the block's output. What changes
is the interpretation: a component of the tapped vector is a function of the
current token identity and the layer index alone, not of context. A probe
trained on that stream may therefore score partly on token identity, and a
flag raised at one position says less about the surrounding text than the same
flag would on a Gemma 3 residual.

The consequence is for probe evaluation, not for the contract. Probe quality
is already scheduled for separate evaluation on prefill and decode positions
(see the roadmap below); that evaluation should report token-identity baseline
performance alongside it, so a probe that has learned the PLE component rather
than a semantic direction is visible rather than assumed away.

## Constraint on future backends

The tap seam is part of the `Model` backend contract from now on: the CUDA
backend (roadmap phase A) must implement `Tap` delivery with the same
semantics: the layer output after the `layer_output_scale` multiply, with
steering applied first. On the GPU this
implies a device-to-host copy per tapped layer; taps stay optional so the
untapped serving path pays nothing. The full set of CUDA-backend
interpretability constraints (determinism requirement, provenance world,
revised live ladder) is recorded in
[interp-scale-reference.md](interp-scale-reference.md#the-cuda-backend).

## Roadmap

| Step | Status | Depends on |
|------|--------|------------|
| Tap seam, steering, capture, goldens (sandbox) | done | — |
| Real-model capture and contrastive extraction on the workstation | pending | E4B architecture reconciliation, CPU backend speed measurement |
| In-band probe monitor as a `Tap` implementation (sandbox) | done | — |
| Monitor with a probe trained on real-model rust captures | pending | real-model capture on the workstation |
| Probe quality evaluated separately on prefill and decode positions | pending | real-model capture on the workstation |
| CUDA tap parity | pending | roadmap phase A |
