/*
 * agent — the core loop: render transcript, generate, dispatch tools.
 *
 * The loop is generic over a text-level Generator so Tier 2 tests script
 * whole model turns; the real implementation wraps a runtime Session plus
 * tokenizer. Tool results and permission denials both return to the model as
 * tool entries, so a denial is recoverable rather than fatal.
 */

use std::path::PathBuf;

use anyhow::{bail, Result};

use crate::permissions::{Decision, Permissions};
use crate::template::{render_entry, MODEL_TURN_OPEN};
use crate::tools::ToolCall;
use crate::transcript::{Role, Transcript};

pub trait Generator {
    fn complete(&mut self, prompt: &str) -> Result<String>;
}

pub struct Agent {
    pub root: PathBuf,
    pub permissions: Permissions,
    pub transcript: Transcript,
    // Incremental prompt state: `rendered` holds every closed turn for
    // `transcript.entries()[..rendered_upto]`, so each loop iteration appends
    // only the new entries instead of re-rendering the whole transcript.
    rendered: String,
    rendered_upto: usize,
}

impl Agent {
    pub fn new(root: PathBuf, permissions: Permissions, transcript: Transcript) -> Self {
        Self {
            root,
            permissions,
            transcript,
            rendered: String::new(),
            rendered_upto: 0,
        }
    }

    /// Prompt for the next model turn, rendered incrementally.
    fn prompt(&mut self) -> &str {
        for e in &self.transcript.entries()[self.rendered_upto..] {
            self.rendered.push_str(&render_entry(e));
        }
        self.rendered_upto = self.transcript.entries().len();
        &self.rendered
    }

    /// Run one task to a final text answer. `on_ask` resolves Ask decisions
    /// (interactive prompt in the binary, scripted in tests).
    pub fn run_task(
        &mut self,
        generator: &mut dyn Generator,
        task: &str,
        max_turns: usize,
        mut on_ask: impl FnMut(&ToolCall) -> bool,
    ) -> Result<String> {
        self.transcript.append(Role::User, task)?;
        // Circuit breaker: a model repeating the exact failing call makes no
        // progress; abort instead of silently burning the turn budget.
        const MAX_REPEATED_FAILURES: usize = 3;
        let mut last_failure: Option<ToolCall> = None;
        let mut repeats = 0usize;
        for _ in 0..max_turns {
            self.prompt();
            // Borrow the rendered buffer with the open model turn appended,
            // then trim it back: the closed-turn prefix stays reusable.
            self.rendered.push_str(MODEL_TURN_OPEN);
            let output = generator.complete(&self.rendered);
            self.rendered
                .truncate(self.rendered.len() - MODEL_TURN_OPEN.len());
            let output = output?;
            self.transcript.append(Role::Model, output.as_str())?;
            match ToolCall::parse(&output) {
                Ok(None) => return Ok(output),
                Ok(Some(call)) => {
                    let allowed = match self.permissions.evaluate(&call) {
                        Decision::Allow => true,
                        Decision::Deny => false,
                        Decision::Ask => on_ask(&call),
                    };
                    // Tool failures return to the model as text; only
                    // harness failures (transcript IO) abort the loop.
                    let (result, failed) = if allowed {
                        match call.execute(&self.root) {
                            Ok(out) => (out, false),
                            Err(e) => (format!("tool error: {e:#}"), true),
                        }
                    } else {
                        (
                            format!("permission denied: {} {}", call.name(), call.target()),
                            true,
                        )
                    };
                    if failed && last_failure.as_ref() == Some(&call) {
                        repeats += 1;
                        if repeats >= MAX_REPEATED_FAILURES {
                            bail!(
                                "aborting: tool call {} {} failed identically {repeats} times",
                                call.name(),
                                call.target()
                            );
                        }
                    } else {
                        last_failure = failed.then(|| call.clone());
                        repeats = usize::from(failed);
                    }
                    self.transcript.append(Role::Tool, result)?;
                }
                // A malformed tool call goes back to the model as an error to
                // correct, never out to the user as a final answer.
                Err(e) => {
                    self.transcript
                        .append(Role::Tool, format!("tool error: {e:#}"))?;
                }
            }
        }
        bail!("no final answer within {max_turns} turns")
    }
}
