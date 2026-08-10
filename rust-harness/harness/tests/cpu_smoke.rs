/*
 * cpu_smoke — phase 3 sandbox acceptance: the full harness stack (agent
 * loop, chat template, tokenizer, session, sampler) runs end to end over
 * the real CPU backend with synthetic weights. Output is gibberish by
 * construction; the assertion is that the real code path completes and is
 * deterministic, not that the text means anything.
 */

use harness::agent::Agent;
use harness::generator::SessionGenerator;
use harness::permissions::Permissions;
use harness::transcript::Transcript;
use runtime::model::cpu::CpuModel;
use runtime::sampler::SamplingParams;
use runtime::session::Session;
use runtime::synth::{build_tiny_config_json, build_tiny_safetensors, TinySpec};
use runtime::tokenizer::Tokenizer;

/// The synthetic checkpoint, in memory. Built once per test and borrowed by
/// each generator, since two generators over the same weights is what the
/// determinism assertion below compares.
fn tiny_checkpoint() -> runtime::checkpoint::CheckpointFiles {
    let spec = TinySpec::default();
    runtime::checkpoint::CheckpointFiles {
        config: build_tiny_config_json(&spec).unwrap(),
        weights: build_tiny_safetensors(&spec).unwrap(),
        dir: std::path::PathBuf::from("<memory>"),
    }
}

fn generator(files: &runtime::checkpoint::CheckpointFiles) -> SessionGenerator {
    let ck = runtime::checkpoint::parse(files).unwrap();
    let cpu = CpuModel::from_checkpoint(&ck).unwrap();
    let max_context = cpu.config().context_length;
    let session = Session::new(Box::new(cpu), max_context).unwrap();
    let tokenizer = Tokenizer::from_json(
        &runtime::synth::build_tiny_tokenizer_json(&TinySpec::default()).unwrap(),
    )
    .unwrap();
    // Short generation bound: random weights rarely emit EOS, and the
    // context budget must hold prompt + output.
    SessionGenerator::new(session, tokenizer, SamplingParams::default(), 48).unwrap()
}

#[test]
fn harness_runs_a_task_end_to_end_on_the_cpu_backend() {
    let image = tiny_checkpoint();
    let dir = tempfile::tempdir().unwrap();
    let transcript = Transcript::open(&dir.path().join("session.jsonl")).unwrap();
    let permissions = Permissions::from_yaml("rules: []").unwrap();
    let mut agent = Agent::new(dir.path().to_path_buf(), permissions, transcript, None);
    let answer = agent
        .run_task(&mut generator(&image), "say hello", 4, |_| false)
        .unwrap();
    // Random weights produce arbitrary but non-empty deterministic text.
    assert!(!answer.is_empty());

    // Determinism: an identical fresh stack yields the identical answer.
    let dir2 = tempfile::tempdir().unwrap();
    let transcript2 = Transcript::open(&dir2.path().join("session.jsonl")).unwrap();
    let permissions2 = Permissions::from_yaml("rules: []").unwrap();
    let mut agent2 = Agent::new(dir2.path().to_path_buf(), permissions2, transcript2, None);
    let answer2 = agent2
        .run_task(&mut generator(&image), "say hello", 4, |_| false)
        .unwrap();
    assert_eq!(answer, answer2);
}
