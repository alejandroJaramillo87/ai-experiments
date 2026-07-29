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
use crate::template::render;
use crate::tools::ToolCall;
use crate::transcript::{Role, Transcript};

pub trait Generator {
    fn complete(&mut self, prompt: &str) -> Result<String>;
}

pub struct Agent {
    pub root: PathBuf,
    pub permissions: Permissions,
    pub transcript: Transcript,
}

impl Agent {
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
        for _ in 0..max_turns {
            let prompt = render(self.transcript.entries());
            let output = generator.complete(&prompt)?;
            self.transcript.append(Role::Model, output.clone())?;
            let Some(call) = ToolCall::parse(&output)? else {
                return Ok(output);
            };
            let allowed = match self.permissions.evaluate(&call) {
                Decision::Allow => true,
                Decision::Deny => false,
                Decision::Ask => on_ask(&call),
            };
            let result = if allowed {
                // Tool failures return to the model as text; only harness
                // failures (transcript IO) abort the loop.
                call.execute(&self.root)
                    .unwrap_or_else(|e| format!("tool error: {e:#}"))
            } else {
                format!("permission denied: {} {}", call.name(), call.target())
            };
            self.transcript.append(Role::Tool, result)?;
        }
        bail!("no final answer within {max_turns} turns")
    }
}
