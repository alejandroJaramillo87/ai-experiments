/*
 * record_replay — phase 1 acceptance: a recorded run, loaded back as a
 * MockModel script, reproduces the original token sequence through a full
 * Session generation. Exercises the file path end to end (write, reopen,
 * replay) rather than an in-memory buffer.
 */

use runtime::grammar::Vocab;
use runtime::model::mock::MockModel;
use runtime::model::record::{replay_file, RecordingModel};
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
        let model =
            RecordingModel::to_file(Box::new(MockModel::new(v.len(), script)), &path).unwrap();
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

#[test]
fn recording_survives_truncation_and_reprefill() {
    let v = vocab();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("run.rec.jsonl");
    {
        let script = script_emitting(v.len(), 3, &[5], v.eos());
        let model =
            RecordingModel::to_file(Box::new(MockModel::new(v.len(), script)), &path).unwrap();
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
