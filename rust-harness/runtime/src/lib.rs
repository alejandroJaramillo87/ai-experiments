/*
 * runtime — inference engine library crate.
 *
 * Linked into the harness binary; owns model weights, the KV cache, sampling,
 * and grammar-constrained decoding. The harness talks to it through `Session`
 * handles over the `Model` trait, so everything above the forward pass builds
 * and tests without a GPU.
 */

pub mod eval;
pub mod gguf;
pub mod grammar;
pub mod kv;
pub mod model;
pub mod npz;
pub mod sampler;
pub mod session;
pub mod synth;
pub mod tensor;
pub mod tokenizer;
