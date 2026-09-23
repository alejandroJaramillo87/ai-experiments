# interp/steering

Control-vector steering (W1): extract a direction from a labeled capture, export it as the
safetensors image the Rust harness applies inside its forward pass, wire the harness to
serve with it, and score the three arms (unsteered, steered, and a judge) against the live
services. Weights are never modified; the vector is added to the residual stream at load.

`extract.py` offers two backends behind one interface and `export.py` writes the artifact:
a safetensors image of per-block f32 directions named `block.<index>` (0-based, so
`block.0` is legal) plus a `manifest.json` beside it carrying the model hash the harness
checks at load. That format replaced GGUF at R10, when retiring llama.cpp steering left
this repository the only reader; the contract is `schemas/control_vector_artifact.json`
and `rust-harness/runtime/tests/python_exporter_contract.rs` reads this package's real
output. The prompt sets in [prompts/](prompts/README.md) are committed for the same reason
the probe sets are: a capture's dataset hash must resolve to a commit.

## Modules

<!-- inventory:modules:start -->

| Module | Purpose |
|--------|---------|
| `export.py` | Control-vector safetensors export (W1 decision 2, format per R10). |
| `extract.py` | Direction extraction (W1 decision 1): two backends behind one interface. |
| `scorecard.py` | Three-arm steering scorecard (W1 decisions 4-6). |
| `wiring.py` | Service wiring helpers (W1 decision 3). |

<!-- inventory:modules:end -->

## Where the real docs live

- [../../docs/interp/writeback.md](../../docs/interp/writeback.md) — extract, serve and score, in order
- [../../docs/interp/rust-harness-interp.md](../../docs/interp/rust-harness-interp.md) — the R10 format change and the layer-numbering seam
- [../../site/docs/formats/control-vector-artifact.md](../../site/docs/formats/control-vector-artifact.md) — the artifact format for a reader outside this tree
