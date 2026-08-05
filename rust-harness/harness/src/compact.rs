/*
 * compact — transcript compaction via model-written summaries.
 *
 * When a conversation approaches the context budget, the middle of the
 * effective transcript is summarized by the model itself and replaced with
 * the summary. Persistence is append-only: a Compaction record naming the
 * replaced range joins session.jsonl and Transcript::effective folds it in
 * on every read, so the raw history survives for debugging and the
 * transcript's crash recovery applies unchanged.
 *
 * The splice shortens the rendered prompt's common prefix, so the next
 * Session::sync truncates the KV cache to the kept head and prefills only
 * the summary and tail — the same divergence path every re-rendered prompt
 * takes.
 */

use anyhow::{Context, Result};

use crate::agent::Generator;
use crate::template::{render_entry, MODEL_TURN_OPEN};
use crate::transcript::{Entry, Role, Transcript};

/// Effective entries always kept at the head: the opening turn that states
/// the task (agent mode) or starts the conversation (chat mode).
pub const KEEP_PREFIX: usize = 1;
/// Most recent effective entries always kept verbatim.
pub const KEEP_LAST: usize = 4;

const SUMMARY_MARKER: &str = "Summary of earlier conversation:";

const INSTRUCTION: &str = "Summarize the conversation above in a short paragraph. \
     Keep every fact, decision, file name, and open question needed to \
     continue the work. Reply with the summary only.";

/// Whether the automatic trigger should fire: the rendered prompt has
/// crossed `threshold` of the budget, there is a middle to compact, and at
/// least KEEP_LAST entries arrived since the last compaction (thrash guard —
/// without it a summary that stays over the threshold would re-trigger
/// every turn on an ever-smaller middle).
pub fn should_compact(
    generator: &mut dyn Generator,
    transcript: &Transcript,
    rendered: &str,
    threshold: f32,
) -> Result<bool> {
    let Some(budget) = generator.context_budget() else {
        return Ok(false);
    };
    if transcript.effective().len() <= KEEP_PREFIX + KEEP_LAST
        || transcript.entries_since_compaction() < KEEP_LAST
    {
        return Ok(false);
    }
    let tokens = generator.count_tokens(rendered)?;
    Ok(tokens as f32 > threshold * budget as f32)
}

/// Summarize the middle of the effective transcript and append the
/// compaction record. Returns false when there is nothing to compact. The
/// summarization turn itself is transient: it goes through the session (so
/// the KV cache re-primes on the next real prompt) but is never persisted.
pub fn compact(generator: &mut dyn Generator, transcript: &mut Transcript) -> Result<bool> {
    let effective = transcript.effective();
    if effective.len() <= KEEP_PREFIX + KEEP_LAST {
        return Ok(false);
    }
    let start = KEEP_PREFIX;
    let end = effective.len() - KEEP_LAST;
    // A fresh sequence rather than a continuation, so it carries its own BOS.
    let mut prompt = String::from(crate::template::BOS);
    for e in &effective[start..end] {
        prompt.push_str(&render_entry(e));
    }
    prompt.push_str(&render_entry(&Entry {
        role: Role::User,
        content: INSTRUCTION.to_string(),
        replaces: None,
    }));
    prompt.push_str(MODEL_TURN_OPEN);
    let summary = generator
        .complete(&prompt)
        .context("summarize for compaction")?;
    let summary = summary.trim();
    anyhow::ensure!(!summary.is_empty(), "compaction produced an empty summary");
    transcript.append_compaction(format!("{SUMMARY_MARKER}\n{summary}"), (start, end))?;
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Scripted {
        replies: Vec<&'static str>,
        prompts: Vec<String>,
        budget: Option<usize>,
    }

    impl Generator for Scripted {
        fn complete(&mut self, prompt: &str) -> Result<String> {
            self.prompts.push(prompt.to_string());
            Ok(self.replies.remove(0).to_string())
        }
        fn context_budget(&self) -> Option<usize> {
            self.budget
        }
    }

    fn transcript_with_turns(dir: &tempfile::TempDir, n: usize) -> Transcript {
        let mut t = Transcript::open(&dir.path().join("session.jsonl")).unwrap();
        t.append(Role::User, "the task").unwrap();
        for i in 0..n {
            t.append(Role::Model, format!("model turn {i}")).unwrap();
            t.append(Role::Tool, format!("tool result {i}")).unwrap();
        }
        t
    }

    #[test]
    fn the_compaction_prompt_carries_its_own_bos() {
        // The summarization prompt is a fresh sequence rather than a
        // continuation of the conversation, so it needs its own BOS.
        let dir = tempfile::tempdir().unwrap();
        let mut t = transcript_with_turns(&dir, 5);
        let mut g = Scripted {
            replies: vec!["they went back and forth"],
            prompts: vec![],
            budget: None,
        };
        assert!(compact(&mut g, &mut t).unwrap());
        let prompt = &g.prompts[0];
        assert!(prompt.starts_with(crate::template::BOS), "{prompt}");
        assert_eq!(prompt.matches(crate::template::BOS).count(), 1, "{prompt}");
    }

    #[test]
    fn compact_splices_the_middle_and_keeps_head_and_tail() {
        let dir = tempfile::tempdir().unwrap();
        let mut t = transcript_with_turns(&dir, 5);
        let before = t.effective().len(); // 11 entries
        let mut g = Scripted {
            replies: vec!["they went back and forth"],
            prompts: vec![],
            budget: None,
        };
        assert!(compact(&mut g, &mut t).unwrap());
        let after = t.effective();
        // Head, one summary, tail.
        assert_eq!(after.len(), KEEP_PREFIX + 1 + KEEP_LAST);
        assert!(after.len() < before);
        assert_eq!(after[0].content, "the task");
        assert_eq!(after[1].role, Role::System);
        assert!(after[1].content.starts_with(SUMMARY_MARKER));
        assert!(after[1].content.contains("they went back and forth"));
        assert_eq!(after.last().unwrap().content, "tool result 4");
        // The summarization prompt covered the middle only.
        assert!(!g.prompts[0].contains("the task"));
        assert!(g.prompts[0].contains("model turn 0"));
        assert!(!g.prompts[0].contains("tool result 4"));
    }

    #[test]
    fn compaction_survives_reopen_and_raw_history_is_intact() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("session.jsonl");
        {
            let mut t = transcript_with_turns(&dir, 5);
            let mut g = Scripted {
                replies: vec!["summary text"],
                prompts: vec![],
                budget: None,
            };
            compact(&mut g, &mut t).unwrap();
        }
        let t = Transcript::open(&path).unwrap();
        // Raw entries: 11 conversation entries plus the compaction record.
        assert_eq!(t.entries().len(), 12);
        assert_eq!(t.effective().len(), KEEP_PREFIX + 1 + KEEP_LAST);
        assert!(t.effective()[1].content.contains("summary text"));
    }

    #[test]
    fn second_compaction_folds_over_the_first() {
        let dir = tempfile::tempdir().unwrap();
        let mut t = transcript_with_turns(&dir, 5);
        let mut g = Scripted {
            replies: vec!["first summary", "second summary"],
            prompts: vec![],
            budget: None,
        };
        compact(&mut g, &mut t).unwrap();
        for i in 0..4 {
            t.append(Role::Model, format!("late turn {i}")).unwrap();
        }
        assert!(compact(&mut g, &mut t).unwrap());
        let eff = t.effective();
        assert_eq!(eff.len(), KEEP_PREFIX + 1 + KEEP_LAST);
        // The second summary subsumes the first: the first summary and the
        // old tail were inside the second compaction's range.
        assert!(eff[1].content.contains("second summary"));
        assert!(eff.iter().all(|e| !e.content.contains("first summary")));
        assert!(g.prompts[1].contains("first summary"));
    }

    #[test]
    fn nothing_to_compact_is_a_clean_no_op() {
        let dir = tempfile::tempdir().unwrap();
        let mut t = transcript_with_turns(&dir, 2); // 5 entries = head + tail
        let mut g = Scripted {
            replies: vec![],
            prompts: vec![],
            budget: None,
        };
        assert!(!compact(&mut g, &mut t).unwrap());
        assert_eq!(t.entries().len(), 5);
    }

    #[test]
    fn threshold_gates_the_automatic_trigger() {
        let dir = tempfile::tempdir().unwrap();
        let t = transcript_with_turns(&dir, 5);
        let mut g = Scripted {
            replies: vec![],
            prompts: vec![],
            budget: Some(100), // default count_tokens: len/4
        };
        let long = "x".repeat(400); // 100 tokens > 0.8 * 100
        let short = "x".repeat(100); // 25 tokens
        assert!(should_compact(&mut g, &t, &long, 0.8).unwrap());
        assert!(!should_compact(&mut g, &t, &short, 0.8).unwrap());
        // No budget: never triggers.
        g.budget = None;
        assert!(!should_compact(&mut g, &t, &long, 0.8).unwrap());
    }

    #[test]
    fn thrash_guard_holds_until_new_turns_accumulate() {
        let dir = tempfile::tempdir().unwrap();
        let mut t = transcript_with_turns(&dir, 5);
        let mut g = Scripted {
            replies: vec!["summary"],
            prompts: vec![],
            budget: Some(10),
        };
        compact(&mut g, &mut t).unwrap();
        let long = "x".repeat(400);
        // Over threshold, but no new turns since the compaction record.
        assert!(!should_compact(&mut g, &t, &long, 0.8).unwrap());
        for i in 0..KEEP_LAST {
            t.append(Role::Model, format!("new {i}")).unwrap();
        }
        assert!(should_compact(&mut g, &t, &long, 0.8).unwrap());
    }

    #[test]
    fn empty_summary_is_an_error_and_appends_nothing() {
        let dir = tempfile::tempdir().unwrap();
        let mut t = transcript_with_turns(&dir, 5);
        let mut g = Scripted {
            replies: vec!["   "],
            prompts: vec![],
            budget: None,
        };
        assert!(compact(&mut g, &mut t)
            .unwrap_err()
            .to_string()
            .contains("empty summary"));
        assert_eq!(t.entries().len(), 11);
    }
}
