/*
 * recording_replay — a flight recording from a real run, loaded back as a
 * MockModel script, states what the run did with it.
 *
 * `record_replay.rs` proves the format on recordings this test writes itself.
 * This one takes a recording written by `harness --record` against the real
 * checkpoint, which cannot be committed: a row is the full vocabulary, about
 * 2 MB of JSON per generated token. The path arrives in HARNESS_RECORDING,
 * and `make replay-check RECORDING=<file>` sets it. Under `make quality` the
 * variable is unset and the test skips, so the gate is not conditioned on a
 * workstation artifact.
 *
 * What is asserted is the shape any recording of a sampled run must have,
 * not the parameters of one particular run: the runbook entry that drives
 * this (docs/verification/rust-harness/rust-harness.md, RH.5) reads the
 * printed sampling line against the command it ran.
 */

use std::path::Path;

use runtime::model::record::replay_file_with_selections;
use runtime::{status, testlog};

#[test]
fn a_flight_recording_replays_as_a_script_and_states_its_selections() {
    let Ok(path) = std::env::var("HARNESS_RECORDING") else {
        testlog::skip(
            "flight recording replay",
            "HARNESS_RECORDING is not set; record a run with harness --record and \
             pass its path, or run make replay-check RECORDING=<file>",
        );
        return;
    };
    let (script, recorded) =
        replay_file_with_selections(Path::new(&path)).unwrap_or_else(|e| panic!("{path}: {e:#}"));

    let sampling = recorded
        .sampling
        .expect("a recording of a sampled run names its sampling parameters in the header");
    let selections = &recorded.selections;
    assert!(!selections.is_empty(), "{path}: no token was selected");

    // The draw count rises by exactly one per stochastic step. That is what
    // makes the run's position in its own random stream recoverable, rather
    // than merely stated; a replay that re-rolled a step would have nothing
    // to disagree with otherwise.
    let draws: Vec<u64> = selections.iter().map(|s| s.draws).collect();
    for w in draws.windows(2) {
        assert_eq!(w[1], w[0] + 1, "draw count did not rise by one: {draws:?}");
    }
    let positions: Vec<usize> = selections.iter().map(|s| s.position).collect();
    for w in positions.windows(2) {
        assert!(
            w[1] > w[0],
            "selection positions are not increasing: {positions:?}"
        );
    }
    // Every selection was drawn from the row at the position before it, and
    // that row has to be in the script or a replay would draw from zeros.
    for s in selections {
        assert!(
            s.position >= 1 && s.position - 1 < script.script_len(),
            "selection at {} drew from a row the recording did not script ({} rows)",
            s.position,
            script.script_len()
        );
    }

    status::info(&format!("sampling: {sampling:?}"));
    status::info(&format!(
        "script: {} positions, vocab {}",
        script.script_len(),
        script.vocab_size()
    ));
    status::info(&format!(
        "selections: {} at positions {}..={}, draws {}..={}",
        selections.len(),
        positions[0],
        positions[positions.len() - 1],
        draws[0],
        draws[draws.len() - 1]
    ));
}
