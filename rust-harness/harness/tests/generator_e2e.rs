/*
 * generator_e2e.rs — Tier 2 test for the text↔token seam.
 *
 * Drives the real agent loop through SessionGenerator: prompt rendering →
 * tokenizer encode → Session (MockModel) → decode → tool dispatch. This is
 * the path the real backend slots into; only the forward pass is scripted.
 */

use harness::agent::Agent;
use harness::generator::SessionGenerator;
use harness::permissions::Permissions;
use harness::template::{render_entry, MODEL_TURN_OPEN};
use harness::transcript::{Entry, Role, Transcript};
use runtime::model::mock::MockModel;
use runtime::model::Token;
use runtime::sampler::SamplingParams;
use runtime::session::Session;
use runtime::tokenizer::Tokenizer;

/// Byte-fallback-only tokenizer: 3 control specials then one token per byte,
/// so any prompt text encodes and token id = byte value + 3.
fn byte_tokenizer() -> Tokenizer {
    let mut pieces = vec![
        "<pad>".to_string(),
        "<bos>".to_string(),
        "<eos>".to_string(),
    ];
    let mut types = vec![3i32; 3];
    for b in 0..=255u8 {
        pieces.push(format!("<0x{b:02X}>"));
        types.push(6);
    }
    let scores = vec![0.0; pieces.len()];
    Tokenizer::from_parts(pieces, scores, &types, 1, 2).unwrap()
}

fn byte_token(b: u8) -> Token {
    b as Token + 3
}

#[test]
fn agent_loop_runs_through_tokenizer_and_session() {
    let dir = tempfile::tempdir().unwrap();
    let tokenizer = byte_tokenizer();
    let vocab_len = tokenizer.vocab_len();

    // The exact prompt the first model turn sees: rendered user turn plus
    // the open model turn, with a BOS token in front.
    let task = "say ok";
    let prompt = format!(
        "{}{}",
        render_entry(&Entry {
            role: Role::User,
            content: task.into(),
        }),
        MODEL_TURN_OPEN
    );
    // BOS + byte tokens; spaces expand to the 3-byte SentencePiece marker,
    // so derive the length from the tokenizer rather than the byte count.
    let prompt_len = 1 + tokenizer.encode(&prompt).unwrap().len();

    // Script: after the prompt, emit "ok" then EOS. Position p selects the
    // token at position p + 1.
    let mut script = vec![vec![0.0f32; vocab_len]; prompt_len + 2];
    script[prompt_len - 1][byte_token(b'o') as usize] = 1.0;
    script[prompt_len][byte_token(b'k') as usize] = 1.0;
    script[prompt_len + 1][2] = 1.0; // EOS

    let session = Session::new(Box::new(MockModel::new(vocab_len, script)), 4096).unwrap();
    let mut generator = SessionGenerator::new(
        session,
        tokenizer,
        SamplingParams {
            temperature: 0.0,
            top_p: 1.0,
            seed: 0,
        },
        64,
    )
    .unwrap();

    let mut agent = Agent::new(
        dir.path().to_path_buf(),
        Permissions::from_yaml("rules: [{tool: \"*\", action: allow}]").unwrap(),
        Transcript::open(&dir.path().join("s.jsonl")).unwrap(),
    );
    let answer = agent.run_task(&mut generator, task, 4, |_| false).unwrap();
    assert_eq!(answer, "ok");
    // The KV cache holds the full prompt plus the two generated tokens.
    assert_eq!(generator.session().cached_len(), prompt_len + 2);
}
