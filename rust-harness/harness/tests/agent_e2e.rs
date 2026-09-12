/*
 * agent_e2e.rs — Tier 2 acceptance test for the phase D harness loop.
 *
 * Acceptance check (design doc, phase D): a scripted coding task — read a
 * file, edit it, run its test — completes end to end. Model turns come from
 * a scripted generator; every other layer (tools, permissions, transcript,
 * template rendering) is the real implementation.
 */

use std::collections::VecDeque;

use harness::agent::{Agent, Generator};
use harness::permissions::Permissions;
use harness::transcript::{Role, Transcript};

struct ScriptedGenerator {
    turns: VecDeque<String>,
    prompts: Vec<String>,
}

impl ScriptedGenerator {
    fn new(turns: &[&str]) -> Self {
        Self {
            turns: turns.iter().map(|s| s.to_string()).collect(),
            prompts: Vec::new(),
        }
    }
}

impl Generator for ScriptedGenerator {
    fn complete(&mut self, prompt: &str) -> anyhow::Result<String> {
        self.prompts.push(prompt.to_string());
        self.turns
            .pop_front()
            .ok_or_else(|| anyhow::anyhow!("script exhausted"))
    }
}

const PERMISSIONS: &str = "
rules:
  - tool: bash
    pattern: \"sh check.sh**\"
    action: allow
  - tool: bash
    action: deny
  - tool: \"*\"
    action: allow
";

#[test]
fn scripted_coding_task_completes_end_to_end() {
    let dir = tempfile::tempdir().unwrap();
    // A "codebase": a value file with a bug and a check script asserting the fix.
    std::fs::write(dir.path().join("value.txt"), "answer=41\n").unwrap();
    std::fs::write(
        dir.path().join("check.sh"),
        "grep -q 'answer=42' value.txt && echo CHECK PASSED\n",
    )
    .unwrap();

    let mut generator = ScriptedGenerator::new(&[
        r#"{"tool": "read", "args": {"path": "value.txt"}}"#,
        r#"{"tool": "edit", "args": {"path": "value.txt", "old": "answer=41", "new": "answer=42"}}"#,
        r#"{"tool": "bash", "args": {"command": "sh check.sh"}}"#,
        "Fixed: value.txt now sets answer=42 and check.sh passes.",
    ]);

    let transcript_path = dir.path().join("session.jsonl");
    let mut agent = Agent::new(
        dir.path().to_path_buf(),
        Permissions::from_yaml(PERMISSIONS).unwrap(),
        Transcript::open(&transcript_path).unwrap(),
        None,
    );

    let answer = agent
        .run_task(&mut generator, "Fix the answer in value.txt", 8, |_| false)
        .unwrap();
    assert!(answer.starts_with("Fixed:"));

    // The edit really happened and the check really ran.
    let value = std::fs::read_to_string(dir.path().join("value.txt")).unwrap();
    assert_eq!(value, "answer=42\n");
    let tool_results: Vec<&str> = agent
        .transcript
        .entries()
        .iter()
        .filter(|e| e.role == Role::Tool)
        .map(|e| e.content.as_str())
        .collect();
    assert_eq!(tool_results.len(), 3);
    assert_eq!(tool_results[0], "answer=41\n"); // read result
    assert!(tool_results[2].contains("CHECK PASSED")); // bash result

    // Prompts grow append-only across turns (delta-prefill prerequisite).
    for pair in generator.prompts.windows(2) {
        let closed = pair[0].trim_end_matches(harness::template::MODEL_TURN_OPEN);
        assert!(pair[1].starts_with(closed), "prompt not append-only");
    }

    // The transcript is durable: resuming from disk sees the whole session.
    let resumed = Transcript::open(&transcript_path).unwrap();
    assert_eq!(resumed.entries().len(), agent.transcript.entries().len());
}

#[test]
fn denied_tool_call_returns_to_the_model_and_the_loop_recovers() {
    let dir = tempfile::tempdir().unwrap();
    let mut generator = ScriptedGenerator::new(&[
        r#"{"tool": "bash", "args": {"command": "curl evil.example"}}"#,
        "Understood, stopping.",
    ]);
    let mut agent = Agent::new(
        dir.path().to_path_buf(),
        Permissions::from_yaml(PERMISSIONS).unwrap(),
        Transcript::open(&dir.path().join("s.jsonl")).unwrap(),
        None,
    );
    let answer = agent
        .run_task(&mut generator, "do something", 4, |_| false)
        .unwrap();
    assert_eq!(answer, "Understood, stopping.");
    let denial = &agent.transcript.entries()[2];
    assert_eq!(denial.role, Role::Tool);
    assert!(denial.content.contains("permission denied: bash"));
}

#[test]
fn ask_decision_is_resolved_by_the_handler() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("f.txt"), "data").unwrap();
    // No matching rule for read under an empty rule set: Ask, approved.
    let mut generator =
        ScriptedGenerator::new(&[r#"{"tool": "read", "args": {"path": "f.txt"}}"#, "done"]);
    let mut agent = Agent::new(
        dir.path().to_path_buf(),
        Permissions::from_yaml("rules: []").unwrap(),
        Transcript::open(&dir.path().join("s.jsonl")).unwrap(),
        None,
    );
    let mut asked = Vec::new();
    agent
        .run_task(&mut generator, "read it", 4, |call| {
            asked.push(call.name().to_string());
            true
        })
        .unwrap();
    assert_eq!(asked, vec!["read"]);
    assert_eq!(agent.transcript.entries()[2].content, "data");
}

#[test]
fn malformed_tool_call_returns_to_the_model_instead_of_ending_the_task() {
    let dir = tempfile::tempdir().unwrap();
    // Turn 1: a truncated tool call, the footprint of a cut-off generation.
    let mut generator = ScriptedGenerator::new(&[
        r#"{"tool": "bash", "args": {"comm"#,
        "Recovered after the malformed call.",
    ]);
    let mut agent = Agent::new(
        dir.path().to_path_buf(),
        Permissions::from_yaml("rules: [{tool: \"*\", action: allow}]").unwrap(),
        Transcript::open(&dir.path().join("s.jsonl")).unwrap(),
        None,
    );
    let answer = agent
        .run_task(&mut generator, "do something", 4, |_| false)
        .unwrap();
    assert_eq!(answer, "Recovered after the malformed call.");
    let feedback = &agent.transcript.entries()[2];
    assert_eq!(feedback.role, Role::Tool);
    assert!(feedback.content.contains("malformed tool call"));
}

#[test]
fn tool_paths_cannot_escape_the_workspace_root() {
    let dir = tempfile::tempdir().unwrap();
    let mut generator = ScriptedGenerator::new(&[
        r#"{"tool": "read", "args": {"path": "/etc/passwd"}}"#,
        r#"{"tool": "write", "args": {"path": "../outside.txt", "content": "x"}}"#,
        "done",
    ]);
    let mut agent = Agent::new(
        dir.path().to_path_buf(),
        Permissions::from_yaml("rules: [{tool: \"*\", action: allow}]").unwrap(),
        Transcript::open(&dir.path().join("s.jsonl")).unwrap(),
        None,
    );
    agent
        .run_task(&mut generator, "escape", 4, |_| false)
        .unwrap();
    let tool_results: Vec<&str> = agent
        .transcript
        .entries()
        .iter()
        .filter(|e| e.role == Role::Tool)
        .map(|e| e.content.as_str())
        .collect();
    assert!(tool_results[0].contains("escapes workspace root"));
    assert!(tool_results[1].contains("escapes workspace root"));
    assert!(!dir.path().parent().unwrap().join("outside.txt").exists());
}

#[test]
fn repeated_identical_tool_failures_trip_the_breaker() {
    let dir = tempfile::tempdir().unwrap();
    let bad = r#"{"tool": "read", "args": {"path": "missing.txt"}}"#;
    let mut generator = ScriptedGenerator::new(&[bad, bad, bad, "never reached"]);
    let mut agent = Agent::new(
        dir.path().to_path_buf(),
        Permissions::from_yaml("rules: [{tool: \"*\", action: allow}]").unwrap(),
        Transcript::open(&dir.path().join("s.jsonl")).unwrap(),
        None,
    );
    let err = agent
        .run_task(&mut generator, "loop on failure", 16, |_| false)
        .unwrap_err();
    assert!(err.to_string().contains("failed identically"), "{err}");
}

#[test]
fn alternating_failures_do_not_trip_the_breaker() {
    let dir = tempfile::tempdir().unwrap();
    let bad_a = r#"{"tool": "read", "args": {"path": "a.txt"}}"#;
    let bad_b = r#"{"tool": "read", "args": {"path": "b.txt"}}"#;
    let mut generator = ScriptedGenerator::new(&[bad_a, bad_b, bad_a, bad_b, "gave up"]);
    let mut agent = Agent::new(
        dir.path().to_path_buf(),
        Permissions::from_yaml("rules: [{tool: \"*\", action: allow}]").unwrap(),
        Transcript::open(&dir.path().join("s.jsonl")).unwrap(),
        None,
    );
    let answer = agent
        .run_task(&mut generator, "flail", 16, |_| false)
        .unwrap();
    assert_eq!(answer, "gave up");
}

#[test]
fn success_between_failures_resets_the_breaker() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("ok.txt"), "fine").unwrap();
    let bad = r#"{"tool": "read", "args": {"path": "missing.txt"}}"#;
    let good = r#"{"tool": "read", "args": {"path": "ok.txt"}}"#;
    let mut generator = ScriptedGenerator::new(&[bad, bad, good, bad, bad, "done"]);
    let mut agent = Agent::new(
        dir.path().to_path_buf(),
        Permissions::from_yaml("rules: [{tool: \"*\", action: allow}]").unwrap(),
        Transcript::open(&dir.path().join("s.jsonl")).unwrap(),
        None,
    );
    let answer = agent
        .run_task(&mut generator, "mixed", 16, |_| false)
        .unwrap();
    assert_eq!(answer, "done");
}

#[test]
fn exceeding_max_turns_is_an_error_not_a_hang() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("f.txt"), "x").unwrap();
    let loop_call = r#"{"tool": "read", "args": {"path": "f.txt"}}"#;
    let mut generator = ScriptedGenerator::new(&[loop_call, loop_call, loop_call]);
    let mut agent = Agent::new(
        dir.path().to_path_buf(),
        Permissions::from_yaml("rules: [{tool: \"*\", action: allow}]").unwrap(),
        Transcript::open(&dir.path().join("s.jsonl")).unwrap(),
        None,
    );
    let err = agent
        .run_task(&mut generator, "loop", 3, |_| false)
        .unwrap_err();
    assert!(err.to_string().contains("no final answer"));
    // Typed, so the binary can attach the remedy that names the checkpoint:
    // a message alone would have to be matched on its wording.
    let budget = err
        .downcast_ref::<harness::agent::TurnBudgetExhausted>()
        .unwrap_or_else(|| panic!("not a TurnBudgetExhausted: {err:#}"));
    assert_eq!(budget.turns, 3, "{err:#}");
}

#[test]
fn memory_leads_the_agent_prompt_and_never_persists() {
    let dir = tempfile::tempdir().unwrap();
    let mut generator = ScriptedGenerator::new(&["done"]);
    let mut agent = Agent::new(
        dir.path().to_path_buf(),
        Permissions::from_yaml("rules: [{tool: \"*\", action: allow}]").unwrap(),
        Transcript::open(&dir.path().join("s.jsonl")).unwrap(),
        Some("prefer small diffs"),
    );
    agent
        .run_task(&mut generator, "task", 2, |_| false)
        .unwrap();
    // The tool contract leads, then memory. Both are System entries, and
    // Gemma 4 has a native system role, so they render as system turns
    // rather than the user turn Gemma 3 forced.
    let expected = format!(
        "{bos}{open}system\n{contract}{close}\n{open}system\nprefer small diffs{close}\n",
        bos = harness::template::BOS,
        open = harness::template::TURN_OPEN,
        close = harness::template::TURN_CLOSE,
        contract = harness::tools::contract(),
    );
    assert!(
        generator.prompts[0].starts_with(&expected),
        "{}",
        generator.prompts[0]
    );
    // BOS is written once, by the template tier, and never repeated as the
    // incremental buffer grows or is rebuilt after a compaction.
    for (i, p) in generator.prompts.iter().enumerate() {
        assert_eq!(
            p.matches(harness::template::BOS).count(),
            1,
            "prompt {i}: {p}"
        );
    }
    // The memory turn is synthetic: the persisted transcript holds only the
    // task and the answer.
    let resumed = Transcript::open(&dir.path().join("s.jsonl")).unwrap();
    assert_eq!(resumed.entries().len(), 2);
    assert!(resumed
        .entries()
        .iter()
        .all(|e| e.content != "prefer small diffs"));
}
