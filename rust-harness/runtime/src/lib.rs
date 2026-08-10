/*
 * runtime — inference engine library crate.
 *
 * Linked into the harness binary; owns model weights, the KV cache, sampling,
 * and grammar-constrained decoding. The harness talks to it through `Session`
 * handles over the `Model` trait, so everything above the forward pass builds
 * and tests without a GPU.
 */

// This crate produces the training and interp numerics, so memory safety here
// is not negotiable: forbid rather than deny, because forbid cannot be lifted
// by a module-level allow. Nothing in the crate needs unsafe — the GPU backend
// reaches hardware through Candle, which owns the FFI.
#![forbid(unsafe_code)]

pub mod checkpoint;
pub mod eval;
pub mod gguf;
pub mod grammar;
pub mod kv;
pub mod model;
pub mod npz;
pub mod pyenv;
pub mod sampler;
pub mod session;
pub mod synth;
pub mod tensor;
pub mod testlog;
pub mod tokenizer;
