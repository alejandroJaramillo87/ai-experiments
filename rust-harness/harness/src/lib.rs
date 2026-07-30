/*
 * harness — library surface of the coding agent, exposed for integration
 * tests; the binary in main.rs is a thin entry point over these modules.
 */

pub mod agent;
pub mod chat;
pub mod generator;
pub mod obs;
pub mod permissions;
pub mod stream;
pub mod subagents;
pub mod template;
pub mod tools;
pub mod transcript;
pub mod util;
