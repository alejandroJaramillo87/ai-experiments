/*
 * sampling_effect — the sampling flags change what is sampled, not only what
 * the parser accepts.
 *
 * `sampling_flags.rs` holds --top-p to its range and `runtime::sampler`
 * holds the truncations to their arithmetic, and neither says a value on the
 * command line reaches the draw: a run with `--top-k 1` that sampled from
 * the full support would pass both. The flight recording is what closes
 * that gap. It scripts every logits row the model produced and records
 * every token selected from it, so a selection can be compared with its own
 * row's argmax after the fact, one process apart from the sampler.
 *
 * Three arms degenerate to the argmax by three different routes: a zero
 * temperature takes the greedy path and draws nothing; `--top-k 1` and a
 * nucleus too small to hold a second token both keep one candidate on the
 * stochastic path, which still draws. The draw counts tell those routes
 * apart, so a sampler that took the greedy path whenever one candidate
 * remained would fail here too.
 */

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use runtime::model::mock::MockModel;
use runtime::model::record::{replay_file_with_selections, Recorded};
use runtime::model::{Model, Token};

fn fixture_model() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../fixtures")
        .join("tiny-gemma4")
}

/// The token budget of every recorded run. Long enough that a run whose
/// cutoff was dropped has many draws in which to leave the argmax.
const MAX_TOKENS: &str = "12";

/// One recorded task run over the fixture with `extra` on the command line,
/// replayed as the script it wrote and the selections it made.
fn recorded_run(extra: &[&str]) -> (MockModel, Recorded) {
    let dir = tempfile::tempdir().expect("a scratch working directory");
    let recording = dir.path().join("flight.jsonl");
    let model = fixture_model();
    let out = Command::new(env!("CARGO_BIN_EXE_harness"))
        .args([
            "--model",
            model.to_str().unwrap(),
            "--record",
            recording.to_str().unwrap(),
            "--seed",
            "0",
            "--max-tokens",
            MAX_TOKENS,
        ])
        .args(extra)
        .arg("say hi")
        .current_dir(dir.path())
        .env("OBS_LOG_DIR", dir.path().join("obs"))
        .stdin(Stdio::null())
        .output()
        .expect("the harness builds as a test binary");
    assert!(
        out.status.success(),
        "{extra:?} failed the run: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let replayed = replay_file_with_selections(&recording)
        .unwrap_or_else(|e| panic!("{}: {e:#}", recording.display()));
    // The temporary directory drops here; the recording has been read.
    replayed
}

/// The index of the largest logit, first index on a tie, as the sampler's
/// greedy path and torch both choose.
fn argmax(row: &[f32]) -> Token {
    let mut best = 0;
    for (i, &v) in row.iter().enumerate() {
        if v > row[best] {
            best = i;
        }
    }
    best as Token
}

/// The tokens selected, paired with the argmax of the row each was drawn
/// from. A selection at position p drew from the row at p - 1, which the
/// script serves once p tokens have been prefilled; the selections come in
/// increasing position, so each prefill advances the mock by the gap, and
/// the mock ignores the tokens' values.
fn selections_against_argmax(script: &mut MockModel, recorded: &Recorded) -> Vec<(Token, Token)> {
    assert!(!recorded.selections.is_empty(), "no token was selected");
    recorded
        .selections
        .iter()
        .map(|s| {
            let gap = s
                .position
                .checked_sub(script.cached_len())
                .filter(|g| *g >= 1)
                .unwrap_or_else(|| {
                    panic!(
                        "selection at {} is not past the {} positions already served",
                        s.position,
                        script.cached_len()
                    )
                });
            let row = script
                .prefill(&vec![0; gap])
                .unwrap_or_else(|e| panic!("row at {}: {e:#}", s.position))
                .0;
            (s.token, argmax(&row))
        })
        .collect()
}

/// The draw count recorded at each selection.
fn draws(recorded: &Recorded) -> Vec<u64> {
    recorded.selections.iter().map(|s| s.draws).collect()
}

#[test]
fn a_zero_temperature_selects_every_argmax_without_drawing() {
    let (mut script, recorded) = recorded_run(&["--temperature", "0"]);
    let pairs = selections_against_argmax(&mut script, &recorded);
    for (token, best) in &pairs {
        assert_eq!(
            token, best,
            "greedy selected {token} over the argmax {best}: {pairs:?}"
        );
    }
    let draws = draws(&recorded);
    assert!(
        draws.iter().all(|&d| d == 0),
        "the greedy path consumed draws: {draws:?}"
    );
}

#[test]
fn a_top_k_of_one_selects_every_argmax_while_still_drawing() {
    let (mut script, recorded) = recorded_run(&["--temperature", "1", "--top-k", "1"]);
    let pairs = selections_against_argmax(&mut script, &recorded);
    for (token, best) in &pairs {
        assert_eq!(
            token, best,
            "--top-k 1 selected {token} over the argmax {best}: {pairs:?}"
        );
    }
    let draws = draws(&recorded);
    for w in draws.windows(2) {
        assert_eq!(
            w[1],
            w[0] + 1,
            "the stochastic path did not draw once per step: {draws:?}"
        );
    }
}

#[test]
fn a_nucleus_too_small_for_a_second_token_selects_every_argmax() {
    // The prefix covering the mass always holds the most probable token, so
    // a nucleus below any single probability keeps that token alone.
    let (mut script, recorded) = recorded_run(&["--temperature", "1", "--top-p", "0.0001"]);
    let pairs = selections_against_argmax(&mut script, &recorded);
    for (token, best) in &pairs {
        assert_eq!(
            token, best,
            "--top-p 0.0001 selected {token} over the argmax {best}: {pairs:?}"
        );
    }
}

/// The counterpart that makes the three arms above evidence: the same seed
/// over the full support leaves the argmax at least once, so agreeing with
/// it under a cutoff is the cutoff's doing and not the fixture's.
#[test]
fn the_full_support_at_the_same_seed_leaves_the_argmax() {
    let (mut script, recorded) = recorded_run(&["--temperature", "1", "--top-p", "1"]);
    let pairs = selections_against_argmax(&mut script, &recorded);
    assert!(
        pairs.iter().any(|(token, best)| token != best),
        "every draw over the full support landed on the argmax, so the cutoff \
         arms cannot tell a live cutoff from a dropped one: {pairs:?}"
    );
}
