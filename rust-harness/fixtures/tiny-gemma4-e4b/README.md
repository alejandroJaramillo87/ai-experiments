# tiny-gemma4-e4b

The synthetic checkpoint shaped like the released E4B config (interleaved sliding and
global layers, shared KV, per-layer embeddings, MatFormer widths) at toy dimensions,
written by the `tiny-model` binary and read by `runtime/tests/cpu_reference.rs`,
`capture_obs.rs` and `attention_capture.rs`. Its logits are pinned in
`../tiny-gemma4-e4b-logits.json`.
