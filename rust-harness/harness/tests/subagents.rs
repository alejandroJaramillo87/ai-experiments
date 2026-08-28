/*
 * subagents.rs — Tier 2 acceptance test for phase E fork and interleave.
 *
 * Acceptance check (design doc, phase E): a two-subagent scripted task shows
 * (a) spawn without prefix re-prefill and (b) wall-clock overlap of one
 * agent's tool execution with the other's generation. Time is tokio's paused
 * test clock, so the overlap is measured, not slept for.
 */

use std::cell::RefCell;
use std::time::Duration;

use harness::subagents::{generate_interleaved, gpu_lock, spawn};
use runtime::grammar::Vocab;
use runtime::model::mock::MockModel;
use runtime::sampler::{Sampler, SamplingParams};
use runtime::session::Session;
use tokio::time::Instant;

fn vocab() -> Vocab {
    let tokens: Vec<String> = (0..31).map(|i| format!("t{i}")).collect();
    let mut tokens = tokens;
    tokens.push("</s>".to_string());
    Vocab::new(tokens, 31)
}

/// Script whose argmax is always token 1, long enough that no generation
/// here ever reaches EOS.
fn steady_script(vocab_len: usize) -> Vec<Vec<f32>> {
    let mut l = vec![0.0; vocab_len];
    l[1] = 1.0;
    vec![l; 256]
}

fn greedy() -> Sampler {
    Sampler::new(SamplingParams {
        temperature: 0.0,
        top_p: 1.0,
        top_k: None,
        seed: 0,
    })
}

#[tokio::test(start_paused = true)]
async fn two_subagents_fork_without_reprefill_and_overlap_tool_with_generation() {
    let v = vocab();
    let mock = MockModel::new(v.len(), steady_script(v.len()));
    let recorder = mock.recorder();
    let mut parent = Session::new(Box::new(mock), 4096).unwrap();

    // Shared parent context: system prompt plus transcript.
    let prefix: Vec<u32> = (2..7).collect();
    parent.sync(&prefix).unwrap();
    assert_eq!(recorder.total_prefilled(), 5);

    // Spawn both subagents; each pays prefill for its task prompt only.
    let mut task_a = prefix.clone();
    task_a.extend([10, 11]);
    let (mut agent_a, delta_a) = spawn(&parent, &task_a).unwrap();
    let mut task_b = prefix.clone();
    task_b.extend([12, 13, 14]);
    let (mut agent_b, delta_b) = spawn(&parent, &task_b).unwrap();
    assert_eq!(
        (delta_a, delta_b),
        (2, 3),
        "spawn prefilled beyond the task prompt"
    );
    assert_eq!(
        recorder.total_prefilled(),
        10,
        "prefix re-prefilled on spawn"
    );

    let gpu = gpu_lock();
    let step = Duration::from_millis(10);
    let events: RefCell<Vec<(&'static str, Instant)>> = RefCell::new(Vec::new());
    let log = |name: &'static str| events.borrow_mut().push((name, Instant::now()));

    let a_run = async {
        // Generate, then a 50ms tool call outside the GPU lock, then resume.
        let g1 = generate_interleaved(
            &gpu,
            &mut agent_a,
            &task_a,
            &v,
            None,
            &mut greedy(),
            3,
            step,
            |_| log("a_token"),
        )
        .await
        .unwrap();
        log("a_tool_start");
        tokio::time::sleep(Duration::from_millis(50)).await;
        log("a_tool_end");
        let mut resumed = task_a.clone();
        resumed.extend(&g1.tokens);
        resumed.push(15); // tool result token
        generate_interleaved(
            &gpu,
            &mut agent_a,
            &resumed,
            &v,
            None,
            &mut greedy(),
            2,
            step,
            |_| log("a_token"),
        )
        .await
        .unwrap();
    };
    let b_run = async {
        generate_interleaved(
            &gpu,
            &mut agent_b,
            &task_b,
            &v,
            None,
            &mut greedy(),
            8,
            step,
            |_| log("b_token"),
        )
        .await
        .unwrap();
    };
    tokio::join!(a_run, b_run);

    let events = events.into_inner();
    // Single generation in flight at any instant, across all sessions.
    assert_eq!(recorder.max_in_flight(), 1);

    // B generated while A was inside its tool call.
    let at = |name| events.iter().find(|(n, _)| *n == name).unwrap().1;
    let (tool_start, tool_end) = (at("a_tool_start"), at("a_tool_end"));
    let b_during_tool = events
        .iter()
        .filter(|(n, t)| *n == "b_token" && *t > tool_start && *t < tool_end)
        .count();
    assert!(
        b_during_tool > 0,
        "no generation overlapped the tool call: {events:?}"
    );

    // Both agents produced their scripted token counts.
    let count = |name| events.iter().filter(|(n, _)| *n == name).count();
    assert_eq!(count("a_token"), 5);
    assert_eq!(count("b_token"), 8);
}
