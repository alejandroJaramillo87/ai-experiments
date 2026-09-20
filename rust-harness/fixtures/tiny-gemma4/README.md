# tiny-gemma4

A synthetic Gemma 4 checkpoint with random weights, written by the `tiny-model` binary
from the default `TinySpec` in `runtime/src/synth.rs`. Its logits are pinned in
`../tiny-gemma4-logits.json` by `../reference_logits.py`, and `runtime/tests/cpu_reference.rs`
regenerates the checkpoint and compares. A change detector on the CPU backend, not an
oracle: see `docs/rust-harness/fixtures.md`.
