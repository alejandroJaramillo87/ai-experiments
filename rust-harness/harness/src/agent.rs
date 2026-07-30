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
use crate::transcript::{Entry, Role, Transcript};

pub trait Generator {
    fn complete(&mut self, prompt: &str) -> Result<String>;

    /// Token count of `text` under the generator's tokenizer. The default is
    /// a crude bytes-per-token heuristic so scripted test generators need no
    /// tokenizer; the real generator overrides it with an exact encode.
    fn count_tokens(&mut self, text: &str) -> Result<usize> {
        Ok(text.len() / 4)
    }

    /// The context budget in tokens, when the generator has one. `None`
    /// disables threshold-triggered compaction (scripted test generators).
    fn context_budget(&self) -> Option<usize> {
        None
    }
}

fn count_compactions(transcript: &Transcript) -> usize {
    transcript
        .entries()
        .iter()
        .filter(|e| e.role == Role::Compaction)
        .count()
}

pub struct Agent {
    pub root: PathBuf,
    pub permissions: Permissions,
    pub transcript: Transcript,
    // Automatic compaction threshold as a fraction of the context budget.
    pub compact_at: f32,
    // Rendered memory turn, kept separately so the prompt can be rebuilt
    // from scratch after a compaction record lands.
    memory_rendered: String,
    // Incremental prompt state: `rendered` holds the memory turn plus every
    // closed turn for `transcript.effective()[..rendered_upto]`, so each loop
    // iteration appends only the new entries instead of re-rendering the
    // whole transcript. A new compaction record invalidates the whole thing
    // (`compactions_seen` detects that) and forces a rebuild.
    rendered: String,
    rendered_upto: usize,
    compactions_seen: usize,
}

impl Agent {
    /// `memory` is the HARNESS.md content: a synthetic leading turn rendered
    /// before every persisted entry and never written to the transcript, so
    /// it stays stable within a process and the KV prefix stays maximal.
    pub fn new(
        root: PathBuf,
        permissions: Permissions,
        transcript: Transcript,
        memory: Option<&str>,
    ) -> Self {
        // Seeding `rendered` with rendered_upto = 0 places the memory turn
        // ahead of the persisted entries without persisting it.
        let memory_rendered = match memory {
            Some(content) => render_entry(&Entry {
                role: Role::System,
                content: content.to_string(),
                replaces: None,
            }),
            None => String::new(),
        };
        let compactions_seen = count_compactions(&transcript);
        Self {
            root,
            permissions,
            transcript,
            compact_at: 0.8,
            rendered: memory_rendered.clone(),
            memory_rendered,
            rendered_upto: 0,
            compactions_seen,
        }
    }

    /// Prompt for the next model turn, rendered incrementally over the
    /// effective transcript. A compaction record since the last render
    /// invalidates the incremental state and forces a full rebuild.
    fn prompt(&mut self) -> &str {
        let compactions = count_compactions(&self.transcript);
        if compactions != self.compactions_seen {
            self.compactions_seen = compactions;
            self.rendered = self.memory_rendered.clone();
            self.rendered_upto = 0;
        }
        let effective = self.transcript.effective();
        for e in &effective[self.rendered_upto..] {
            self.rendered.push_str(&render_entry(e));
        }
        self.rendered_upto = effective.len();
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
            // Automatic compaction: when the rendered prompt crosses the
            // threshold, summarize the middle of the transcript and rebuild
            // the prompt over the spliced version before generating.
            if crate::compact::should_compact(
                generator,
                &self.transcript,
                &self.rendered,
                self.compact_at,
            )? && crate::compact::compact(generator, &mut self.transcript)?
            {
                self.prompt();
            }
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
