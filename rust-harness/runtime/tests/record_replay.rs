/*
 * record_replay — phase 1 acceptance: a recorded run, loaded back as a
 * MockModel script, reproduces the original token sequence through a full
 * Session generation. Exercises the file path end to end (write, reopen,
 * replay) rather than an in-memory buffer.
 */

use runtime::grammar::Vocab;
use runtime::model::mock::MockModel;
use runtime::model::record::{replay_file, replay_file_with_selections, RecordingModel};
use runtime::model::Model;
use runtime::sampler::{Sampler, SamplingParams};
use runtime::session::Session;

fn vocab() -> Vocab {
    let mut tokens: Vec<String> = (b'a'..=b'z').map(|b| (b as char).to_string()).collect();
    tokens.push("</s>".to_string());
    let eos = (tokens.len() - 1) as u32;
    Vocab::new(tokens, eos)
}

/// Script that emits `output` after a prompt of length `start`, then EOS.
fn script_emitting(vocab_len: usize, start: usize, output: &[u32], eos: u32) -> Vec<Vec<f32>> {
    let mut script = vec![vec![0.0; vocab_len]; start + output.len()];
    for (i, &tok) in output.iter().enumerate() {
        script[start - 1 + i][tok as usize] = 1.0;
    }
    script[start - 1 + output.len()][eos as usize] = 1.0;
    script
}

fn greedy() -> Sampler {
    Sampler::new(SamplingParams {
        temperature: 0.0,
        top_p: 1.0,
        top_k: None,
        seed: 0,
    })
}

#[test]
fn recorded_run_replays_to_the_same_token_sequence() {
    let v = vocab();
    let prompt = [0u32, 1, 2];
    let output = [7u32, 8, 9, 4];
    let script = script_emitting(v.len(), prompt.len(), &output, v.eos());
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("run.rec.jsonl");

    // Recorded run against the "real" (here: scripted) model.
    let recorded = {
        let model = RecordingModel::to_file(Box::new(MockModel::new(v.len(), script)), &path, None)
            .unwrap();
        let mut s = Session::new(Box::new(model), 4096).unwrap();
        s.generate(&prompt, &v, None, &mut greedy(), 32, |_| {})
            .unwrap()
    };
    assert_eq!(recorded.tokens, output);

    // Replay: the recording alone drives an identical generation, no
    // original script in sight.
    let mut s = Session::new(Box::new(replay_file(&path).unwrap()), 4096).unwrap();
    let replayed = s
        .generate(&prompt, &v, None, &mut greedy(), 32, |_| {})
        .unwrap();
    assert_eq!(replayed.tokens, recorded.tokens);
    assert_eq!(replayed.stopped_at_eos, recorded.stopped_at_eos);
}

/// A vocabulary of `n` single-letter tokens with no EOS in reach, so a
/// stochastic run generates to its budget instead of stopping early.
fn endless_vocab(n: usize) -> Vocab {
    let mut tokens: Vec<String> = (0..n)
        .map(|i| ((b'a' + i as u8) as char).to_string())
        .collect();
    tokens.push("</s>".to_string());
    let eos = (tokens.len() - 1) as u32;
    Vocab::new(tokens, eos)
}

/// Flat logits over the non-EOS tokens: every choice is the sampler's, so a
/// replay that re-rolls the dice diverges immediately rather than being
/// carried to the right answer by a peaked distribution.
fn flat_script(vocab_len: usize, rows: usize) -> Vec<Vec<f32>> {
    (0..rows)
        .map(|_| {
            let mut row = vec![1.0; vocab_len];
            // Finite, not -inf: JSON has no infinity, so serde writes null
            // and the recording will not parse back. A large negative is out
            // of reach of a top_k of 4 either way.
            *row.last_mut().unwrap() = -1000.0;
            row
        })
        .collect()
}

#[test]
fn a_stochastic_run_replays_to_the_same_tokens_from_its_recorded_seed() {
    // The property the selection lines exist for. Before they did, a
    // recording held the distribution and a replay drew from it again: the
    // logits matched, the transcript did not, and nothing said so.
    let v = endless_vocab(8);
    let params = SamplingParams {
        temperature: 1.0,
        top_p: 1.0,
        top_k: Some(4),
        seed: 20260812,
    };
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("stochastic.rec.jsonl");

    let recorded = {
        let model = RecordingModel::to_file(
            Box::new(MockModel::new(v.len(), flat_script(v.len(), 64))),
            &path,
            Some(params.into()),
        )
        .unwrap();
        let mut s = Session::new(Box::new(model), 4096).unwrap();
        s.generate(&[0, 1], &v, None, &mut Sampler::new(params), 16, |_| {})
            .unwrap()
    };
    assert_eq!(recorded.tokens.len(), 16);

    let (script, selections) = replay_file_with_selections(&path).unwrap();
    // The recording states its own parameters, so the replay is built from
    // the file rather than from the constant above -- which is what a reader
    // handed only the file can do.
    let replay_params: SamplingParams = selections.sampling.expect("header states them").into();
    let mut sampler = Sampler::new(replay_params);
    let mut s = Session::new(Box::new(script), 4096).unwrap();
    let replayed = s
        .generate(&[0, 1], &v, None, &mut sampler, 16, |_| {})
        .unwrap();
    assert_eq!(
        replayed.tokens, recorded.tokens,
        "replay from the recorded seed diverged"
    );
    assert_eq!(
        sampler.draws(),
        selections.selections.last().unwrap().draws,
        "replay ended at a different point in the draw stream"
    );
    assert_eq!(
        selections
            .selections
            .iter()
            .map(|s| s.token)
            .collect::<Vec<_>>(),
        recorded.tokens
    );

    // The negative, without which the assertions above would pass on a
    // recorder that wrote no selections at all: a different seed over the
    // same script produces a different transcript, so reproducing it is the
    // seed's doing and not the distribution's.
    let mut other = Sampler::new(SamplingParams {
        seed: replay_params.seed + 1,
        ..replay_params
    });
    let mut s = Session::new(Box::new(replay_file(&path).unwrap()), 4096).unwrap();
    let diverged = s
        .generate(&[0, 1], &v, None, &mut other, 16, |_| {})
        .unwrap();
    assert_ne!(
        diverged.tokens, recorded.tokens,
        "a different seed reproduced the transcript, so the seed is not what drove it"
    );
}

#[test]
fn recording_survives_truncation_and_reprefill() {
    let v = vocab();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("run.rec.jsonl");
    {
        let script = script_emitting(v.len(), 3, &[5], v.eos());
        let model = RecordingModel::to_file(Box::new(MockModel::new(v.len(), script)), &path, None)
            .unwrap();
        let mut s = Session::new(Box::new(model), 4096).unwrap();
        s.sync(&[0, 1, 2]).unwrap();
        // Divergent transcript: truncate to the common prefix, re-prefill.
        s.sync(&[0, 1, 9, 9]).unwrap();
    }
    // Replay is keyed by absolute position, so the re-prefilled positions
    // simply overwrite and the recording still loads cleanly.
    let mut replayed = replay_file(&path).unwrap();
    assert_eq!(replayed.cached_len(), 0);
    replayed.prefill(&[0, 1, 9, 9]).unwrap();
    assert_eq!(replayed.cached_len(), 4);
}
