/*
 * runtime — inference engine library crate.
 *
 * Linked into the harness binary; owns model weights, the KV cache, sampling,
 * and grammar-constrained decoding. The harness talks to it through `Session`
 * handles over the `Model` trait, so everything above the forward pass builds
 * and tests without a GPU.
 */

// This crate produces the training and interp numerics, so memory safety here
// is still the default: deny, and every exception is a written argument at the
// site. The one exception is `checkpoint::CheckpointFiles::read`, which
// memory-maps model.safetensors because `fs::read` of a 15 GB checkpoint costs
// a private copy of the whole file on top of the f32 expansion; `Mmap::map` is
// unsafe and there is no safe wrapper for it. The numerics themselves reach no
// further than safe Rust, and the GPU backend reaches hardware through Candle,
// which owns the FFI.
//
// This was `forbid` until the checkpoint loader needed the map. `deny` is the
// weaker guarantee on purpose: an `allow` can now be written, so the review
// question moves from "does this compile" to "is the SAFETY comment true".
#![deny(unsafe_code)]

pub mod checkpoint;
pub mod digest;
pub mod eval;
pub mod gguf;
pub mod grammar;
pub mod kv;
pub mod ladder;
pub mod metrics;
pub mod model;
pub mod npz;
pub mod pyenv;
pub mod reference;
pub mod sampler;
pub mod session;
pub mod synth;
pub mod tensor;
pub mod testlog;
pub mod tokenizer;
