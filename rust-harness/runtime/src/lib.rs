/*
 * runtime — inference engine library crate.
 *
 * Linked into the harness binary; owns model weights, the KV cache, sampling,
 * and grammar-constrained decoding. The harness talks to it through `Session`
 * handles over the `Model` trait, so everything above the forward pass builds
 * and tests without a GPU.
 */

pub mod grammar;
pub mod model;
pub mod sampler;
