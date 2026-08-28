/*
 * harness — library surface of the coding agent, exposed for integration
 * tests; the binary in main.rs is a thin entry point over these modules.
 */

// The run log's two `libc::flock` calls were the only unsafe this crate ever
// held, and they now live in the `obs` member. Stating the attribute rather
// than leaving the crate merely unsafe-free is what keeps the extraction from
// silently undoing itself: the next FFI call fails to compile here and has to
// argue for itself in `obs`, where the workspace's unsafe policy is written
// down. Both crate roots carry it -- main.rs is a separate crate and the
// attribute does not propagate.
#![forbid(unsafe_code)]

pub mod actionable;
pub mod agent;
pub mod alignment;
pub mod chat;
pub mod cli;
pub mod compact;
pub mod generator;
pub mod interplog;
pub mod permissions;
pub mod preflight;
pub mod runheader;
pub mod server;
pub mod stream;
pub mod subagents;
pub mod template;
pub mod tools;
pub mod transcript;
pub mod util;
