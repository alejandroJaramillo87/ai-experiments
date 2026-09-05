/*
 * generator_e2e.rs — Tier 2 test for the text↔token seam.
 *
 * Drives the real agent loop through SessionGenerator: prompt rendering →
 * tokenizer encode → Session (MockModel) → decode → tool dispatch. This is
 * the path the real backend slots into; only the forward pass is scripted.
 */

use harness::agent::{Agent, Generator};
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
    for b in 0..=255u8 {
        pieces.push(format!("<0x{b:02X}>"));
    }
    // No content pieces and no merges, so byte fallback covers everything and
    // the id of a byte is its value plus the three specials ahead of it.
    let json = runtime::synth::tokenizer_json_from_parts(&pieces, &[], "<pad>").unwrap();
    Tokenizer::from_json(&json).unwrap()
}

fn byte_token(b: u8) -> Token {
    b as Token + 3
}

#[test]
fn agent_loop_runs_through_tokenizer_and_session() {
    let dir = tempfile::tempdir().unwrap();
    let tokenizer = byte_tokenizer();
    let vocab_len = tokenizer.vocab_len();

    // The exact prompt the first model turn sees: the tool contract as a
    // system turn, the rendered user turn, then the open model turn, with a
    // BOS token in front.
    let task = "say ok";
    let prompt = format!(
        "{}{}{}",
        render_entry(&Entry {
            role: Role::System,
            content: harness::tools::contract(),
            replaces: None,
        }),
        render_entry(&Entry {
            role: Role::User,
            content: task.into(),
            replaces: None,
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
            top_k: None,
            seed: 0,
        },
        64,
    )
    .unwrap();

    let mut agent = Agent::new(
        dir.path().to_path_buf(),
        Permissions::from_yaml("rules: [{tool: \"*\", action: allow}]").unwrap(),
        Transcript::open(&dir.path().join("s.jsonl")).unwrap(),
        None,
    );
    let answer = agent.run_task(&mut generator, task, 4, |_| false).unwrap();
    assert_eq!(answer, "ok");
    // The KV cache holds the full prompt plus the two generated tokens.
    assert_eq!(generator.session().cached_len(), prompt_len + 2);
}

#[test]
fn count_tokens_counts_what_complete_will_encode() {
    // compact::should_compact divides this against the context budget, and
    // it is the only production token counter. Every test that reaches
    // should_compact drives a Scripted generator inheriting the trait's
    // text.len()/4 default, so without this the real body was unexercised:
    // replacing it with Ok(0) left the whole suite green.
    let tokenizer = byte_tokenizer();
    let bos_id = tokenizer.bos();
    let vocab_len = tokenizer.vocab_len();
    let session = Session::new(Box::new(MockModel::new(vocab_len, vec![])), 4096).unwrap();
    let mut generator = SessionGenerator::new(
        session,
        byte_tokenizer(),
        SamplingParams {
            temperature: 0.0,
            top_p: 1.0,
            top_k: None,
            seed: 0,
        },
        64,
    )
    .unwrap();

    let prompt = format!(
        "{}{}{MODEL_TURN_OPEN}",
        harness::template::BOS,
        render_entry(&Entry {
            role: Role::User,
            content: "count me".to_string(),
            replaces: None,
        })
    );
    // Of the template's literals only BOS is in this fixture's vocabulary,
    // so BOS collapses to one id and the rest byte-expands. Derived from the
    // tokenizer rather than written as a literal, since the byte count of
    // the turn framing is not the property under test.
    let expected = tokenizer
        .encode(&prompt.replace(harness::template::BOS, ""))
        .unwrap()
        .len()
        + 1;

    assert_eq!(
        generator.count_tokens(&prompt).unwrap(),
        expected,
        "count_tokens disagrees with the encoding complete performs"
    );
    // The prompt carries its own BOS, so no id is added back on top of it.
    assert_eq!(
        tokenizer
            .encode_prompt(&prompt, &[harness::template::BOS])
            .unwrap()
            .iter()
            .filter(|&&id| id == bos_id)
            .count(),
        1,
        "the prompt resolved to more or fewer than one BOS id"
    );
}

#[test]
fn compaction_style_splice_reprefills_only_from_the_splice_point() {
    // A compaction rewrites the middle of the prompt while keeping the head.
    // The session must truncate the KV cache to the kept head and prefill
    // only the summary and tail — never the whole prompt again.
    let tokenizer = byte_tokenizer();
    let vocab_len = tokenizer.vocab_len();
    let mock = MockModel::new(vocab_len, vec![]);
    let recorder = mock.recorder();
    let mut session = Session::new(Box::new(mock), 4096).unwrap();

    let enc = |s: &str| tokenizer.encode(s).unwrap();
    let head = enc("<|turn>user\nthe task<turn|>\n");
    let middle = enc("<|turn>model\nlong exchange<turn|>\n");
    let tail = enc("<|turn>user\nlatest<turn|>\n");
    let summary = enc("<|turn>user\nSummary of earlier conversation:\nshort<turn|>\n");

    let before: Vec<Token> = [head.clone(), middle, tail.clone()].concat();
    session.sync(&before).unwrap();
    let prefilled_before = recorder.total_prefilled();

    let after: Vec<Token> = [head.clone(), summary.clone(), tail.clone()].concat();
    session.sync(&after).unwrap();
    // At most the spliced region was prefilled — the head stayed cached.
    // (The delta can be smaller still: token-level prefix matching extends
    // into the spliced region wherever the summary happens to share leading
    // tokens with what it replaced, e.g. the turn delimiter.)
    let delta = recorder.total_prefilled() - prefilled_before;
    assert!(delta <= summary.len() + tail.len(), "head was re-prefilled");
    assert!(delta >= tail.len());
    assert_eq!(session.cached_len(), after.len());
}
