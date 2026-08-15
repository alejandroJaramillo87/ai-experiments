/*
 * harness — library surface of the coding agent, exposed for integration
 * tests; the binary in main.rs is a thin entry point over these modules.
 */

pub mod agent;
pub mod alignment;
pub mod chat;
pub mod cli;
pub mod compact;
pub mod generator;
pub mod interplog;
pub mod obs;
pub mod permissions;
pub mod server;
pub mod stream;
pub mod subagents;
pub mod template;
pub mod tools;
pub mod transcript;
pub mod util;
