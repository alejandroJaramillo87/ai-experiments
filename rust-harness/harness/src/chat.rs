/*
 * chat — interactive REPL over the generator, without the agent loop.
 *
 * Each input line becomes a user turn; the reply is generated over the full
 * rendered transcript (delta prefill keeps the per-turn cost proportional to
 * the new turn within one process) and appended as a model turn. No tools,
 * permissions, or grammar are involved. The transcript persists to the same
 * session.jsonl the agent loop uses, so a restarted chat resumes the
 * conversation.
 */

use std::io::{BufRead, Write};

use anyhow::{Context, Result};

use crate::agent::Generator;
use crate::template::{render_entry, MODEL_TURN_OPEN, TURN_CLOSE};
use crate::transcript::{Role, Transcript};

/// Strip a trailing turn delimiter. The session stops at the tokenizer EOS,
/// but an instruction-tuned Gemma may close its turn with a distinct
/// <end_of_turn> token that then survives into the decoded text.
fn trim_reply(output: &str) -> &str {
    let trimmed = output.trim_end();
    trimmed
        .strip_suffix(TURN_CLOSE)
        .map(str::trim_end)
        .unwrap_or(trimmed)
}

/// Run the chat loop until end of input or a `/quit` line. Returns the
/// number of model turns generated.
pub fn run_chat(
    generator: &mut dyn Generator,
    transcript: &mut Transcript,
    input: impl BufRead,
    mut out: impl Write,
) -> Result<usize> {
    // Closed-turn prefix, extended per turn — the same incremental shape as
    // the agent loop, so delta prefill sees an append-only prompt.
    let mut rendered = String::new();
    for e in transcript.entries() {
        rendered.push_str(&render_entry(e));
    }
    let mut turns = 0usize;
    for line in input.lines() {
        let line = line.context("read chat input")?;
        let text = line.trim();
        if text.is_empty() {
            continue;
        }
        if text == "/quit" {
            break;
        }
        transcript.append(Role::User, text)?;
        rendered.push_str(&render_entry(
            transcript.entries().last().expect("entry just appended"),
        ));
        rendered.push_str(MODEL_TURN_OPEN);
        let output = generator.complete(&rendered);
        rendered.truncate(rendered.len() - MODEL_TURN_OPEN.len());
        let reply = trim_reply(&output?).to_string();
        transcript.append(Role::Model, reply.as_str())?;
        rendered.push_str(&render_entry(
            transcript.entries().last().expect("entry just appended"),
        ));
        writeln!(out, "{reply}").context("write chat output")?;
        out.flush().context("flush chat output")?;
        turns += 1;
    }
    Ok(turns)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Scripted generator recording every prompt it is handed.
    struct Scripted {
        replies: Vec<&'static str>,
        prompts: Vec<String>,
    }

    impl Generator for Scripted {
        fn complete(&mut self, prompt: &str) -> Result<String> {
            self.prompts.push(prompt.to_string());
            Ok(self.replies.remove(0).to_string())
        }
    }

    fn transcript_in(dir: &tempfile::TempDir) -> Transcript {
        Transcript::open(&dir.path().join("session.jsonl")).unwrap()
    }

    #[test]
    fn prompts_grow_append_only_across_turns() {
        let dir = tempfile::tempdir().unwrap();
        let mut t = transcript_in(&dir);
        let mut g = Scripted {
            replies: vec!["hello", "again"],
            prompts: vec![],
        };
        let turns = run_chat(&mut g, &mut t, "hi\nmore\n".as_bytes(), Vec::new()).unwrap();
        assert_eq!(turns, 2);
        assert_eq!(g.prompts.len(), 2);
        // Turn 2's prompt extends turn 1's closed-turn prefix.
        let prefix = g.prompts[0].strip_suffix(MODEL_TURN_OPEN).unwrap();
        assert!(g.prompts[1].starts_with(prefix));
        assert!(g.prompts[1].contains("<start_of_turn>model\nhello<end_of_turn>"));
        assert!(g.prompts[1].ends_with(MODEL_TURN_OPEN));
    }

    #[test]
    fn transcript_persists_and_a_second_chat_resumes() {
        let dir = tempfile::tempdir().unwrap();
        {
            let mut t = transcript_in(&dir);
            let mut g = Scripted {
                replies: vec!["four"],
                prompts: vec![],
            };
            run_chat(&mut g, &mut t, "2+2?\n".as_bytes(), Vec::new()).unwrap();
        }
        let mut t = transcript_in(&dir);
        assert_eq!(t.entries().len(), 2);
        let mut g = Scripted {
            replies: vec!["eight"],
            prompts: vec![],
        };
        run_chat(&mut g, &mut t, "double it\n".as_bytes(), Vec::new()).unwrap();
        // The resumed turn's prompt replays the persisted turns first.
        assert!(g.prompts[0].contains("2+2?"));
        assert!(g.prompts[0].contains("four"));
        assert_eq!(t.entries().len(), 4);
    }

    #[test]
    fn trailing_end_of_turn_is_trimmed_from_the_reply() {
        let dir = tempfile::tempdir().unwrap();
        let mut t = transcript_in(&dir);
        let mut g = Scripted {
            replies: vec!["fine, thanks<end_of_turn>"],
            prompts: vec![],
        };
        let mut out = Vec::new();
        run_chat(&mut g, &mut t, "how are you?\n".as_bytes(), &mut out).unwrap();
        assert_eq!(String::from_utf8(out).unwrap(), "fine, thanks\n");
        assert_eq!(t.entries()[1].content, "fine, thanks");
    }

    #[test]
    fn quit_and_blank_lines_generate_nothing() {
        let dir = tempfile::tempdir().unwrap();
        let mut t = transcript_in(&dir);
        let mut g = Scripted {
            replies: vec![],
            prompts: vec![],
        };
        let turns = run_chat(
            &mut g,
            &mut t,
            "\n  \n/quit\nignored\n".as_bytes(),
            Vec::new(),
        );
        assert_eq!(turns.unwrap(), 0);
        assert!(t.entries().is_empty());
    }
}
