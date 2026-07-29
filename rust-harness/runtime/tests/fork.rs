/*
 * fork.rs — Tier 2 tests for KV cache forking at the Session level.
 *
 * Phase E runtime half: a subagent spawn forks the parent's cache and
 * prefills only its task prompt; the shared prefix is never re-prefilled.
 */

use runtime::model::mock::MockModel;
use runtime::session::Session;

#[test]
fn spawn_prefills_only_the_task_prompt() {
    let mock = MockModel::new(32, vec![]);
    let recorder = mock.recorder();
    let mut parent = Session::new(Box::new(mock), 4096).unwrap();

    // Parent context: system prompt, tool schemas, transcript so far.
    let prefix: Vec<u32> = (0..10).collect();
    parent.sync(&prefix).unwrap();
    assert_eq!(recorder.total_prefilled(), 10);

    // Spawn: fork, then sync to prefix + task prompt.
    let mut child = parent.fork().unwrap();
    let mut child_transcript = prefix.clone();
    child_transcript.extend([20, 21, 22]);
    let delta = child.sync(&child_transcript).unwrap();

    assert_eq!(delta, 3, "spawn must prefill only the task prompt");
    assert_eq!(
        recorder.total_prefilled(),
        13,
        "prefix re-prefilled on spawn"
    );
    let last = recorder.records().pop().unwrap();
    assert_eq!(last.start, 10);
    assert_eq!(last.tokens, vec![20, 21, 22]);
}

#[test]
fn sibling_spawns_share_the_prefix_cost_once() {
    let mock = MockModel::new(32, vec![]);
    let recorder = mock.recorder();
    let mut parent = Session::new(Box::new(mock), 4096).unwrap();
    let prefix: Vec<u32> = (0..8).collect();
    parent.sync(&prefix).unwrap();

    for task_tok in [20u32, 21, 22] {
        let mut child = parent.fork().unwrap();
        let mut t = prefix.clone();
        t.push(task_tok);
        assert_eq!(child.sync(&t).unwrap(), 1);
    }
    // 8 prefix tokens once, plus one task token per sibling.
    assert_eq!(recorder.total_prefilled(), 11);
}

#[test]
fn child_divergence_does_not_disturb_the_parent() {
    let mock = MockModel::new(32, vec![]);
    let mut parent = Session::new(Box::new(mock), 4096).unwrap();
    let prefix: Vec<u32> = (0..6).collect();
    parent.sync(&prefix).unwrap();

    let mut child = parent.fork().unwrap();
    // Child rewrites history from position 3.
    let mut diverged: Vec<u32> = prefix[..3].to_vec();
    diverged.extend([30, 31]);
    assert_eq!(child.sync(&diverged).unwrap(), 2);
    assert_eq!(child.cached_len(), 5);
    assert_eq!(parent.cached_len(), 6);

    // Parent continues unaffected: appending is still a pure delta.
    let mut extended = prefix.clone();
    extended.push(7);
    assert_eq!(parent.sync(&extended).unwrap(), 1);
}
