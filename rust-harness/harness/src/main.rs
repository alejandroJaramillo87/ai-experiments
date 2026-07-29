/*
 * harness — single-binary local coding agent.
 *
 * Wires the full stack: permissions from permissions.yaml, a durable JSONL
 * transcript, the agent loop, and a SessionGenerator over the runtime. The
 * generator requires the cuda backend (phase A); until it lands the binary
 * reports that and exits, everything above it being already testable.
 */

use std::io::Write;
use std::path::Path;
use std::process::ExitCode;

use anyhow::{Context, Result};

use harness::agent::Agent;
use harness::generator::SessionGenerator;
use harness::permissions::Permissions;
use harness::tools::ToolCall;
use harness::transcript::Transcript;

/// Interactive Ask resolution: prompt on stderr, read one line from stdin.
fn ask_stdin(call: &ToolCall) -> bool {
    eprint!("allow {} {}? [y/N] ", call.name(), call.target());
    let _ = std::io::stderr().flush();
    let mut line = String::new();
    if std::io::stdin().read_line(&mut line).is_err() {
        return false;
    }
    matches!(line.trim(), "y" | "yes")
}

/// Construct the generator over the real model. Requires the phase A cuda
/// backend; there is no mock fallback in the shipping binary.
fn build_generator() -> Result<SessionGenerator> {
    anyhow::bail!(
        "inference backend not built: the cuda backend (roadmap phase A) has not landed; \
         the harness stack above it is complete and runs under cargo test"
    )
}

fn run(root: &Path, task: &str) -> Result<String> {
    let permissions = match std::fs::read_to_string(root.join("permissions.yaml")) {
        Ok(yaml) => Permissions::from_yaml(&yaml).context("permissions.yaml")?,
        // No config: every call goes through the interactive Ask path.
        Err(_) => Permissions::from_yaml("rules: []").expect("empty rule set"),
    };
    let transcript = Transcript::open(&root.join("session.jsonl")).context("session.jsonl")?;
    let mut agent = Agent::new(root.to_path_buf(), permissions, transcript);
    let mut generator = build_generator()?;
    agent.run_task(&mut generator, task, 64, ask_stdin)
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        eprintln!("usage: harness <task description>");
        return ExitCode::from(2);
    }
    let task = args.join(" ");
    let root = match std::env::current_dir() {
        Ok(d) => d,
        Err(e) => {
            eprintln!("[ERROR] Working directory: {e}");
            return ExitCode::from(1);
        }
    };
    match run(&root, &task) {
        Ok(answer) => {
            println!("{answer}");
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("[ERROR] Task failed: {e:#}");
            ExitCode::from(1)
        }
    }
}
