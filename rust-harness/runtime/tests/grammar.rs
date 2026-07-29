/*
 * grammar.rs — Tier 2 tests for grammar-constrained tool-call decoding.
 *
 * Acceptance check (design doc, phase C): zero malformed tool calls over 200
 * consecutive forced tool-call generations, with the mock model's logits
 * adversarially concentrated on tokens the grammar must exclude.
 */

use runtime::grammar::{Grammar, GrammarState, ParamType, ToolSchema, Vocab};
use runtime::model::mock::MockModel;
use runtime::model::Model;
use runtime::sampler::{Sampler, SamplingParams};

fn edit_schema() -> ToolSchema {
    ToolSchema {
        name: "edit".to_string(),
        params: vec![
            ("path".to_string(), ParamType::String),
            ("line".to_string(), ParamType::Int),
            ("create".to_string(), ParamType::Bool),
        ],
    }
}

/// Character-level vocabulary plus a few multi-character tokens and an EOS.
fn test_vocab() -> Vocab {
    let mut tokens: Vec<String> = (32u8..127).map(|b| (b as char).to_string()).collect();
    tokens.push("{\"tool\"".to_string()); // multi-char token crossing skeleton chars
    tokens.push("true".to_string());
    tokens.push("false".to_string());
    tokens.push("</s>".to_string());
    let eos = (tokens.len() - 1) as u32;
    Vocab::new(tokens, eos)
}

#[test]
fn grammar_accepts_valid_tool_call() {
    let g = Grammar::compile(&edit_schema());
    let s = r#"{"tool": "edit", "args": {"path": "src/a.rs", "line": 42, "create": false}}"#;
    let mut st = g.start();
    for c in s.chars() {
        st = g.advance(&st, c).expect("valid call rejected");
    }
    assert!(g.is_complete(&st));
}

#[test]
fn grammar_rejects_wrong_tool_name_and_malformed_json() {
    let g = Grammar::compile(&edit_schema());
    let advance_all = |s: &str| -> Option<GrammarState> {
        let mut st = g.start();
        for c in s.chars() {
            st = g.advance(&st, c)?;
        }
        Some(st)
    };
    assert!(advance_all(r#"{"tool": "bash""#).is_none()); // wrong tool name
    assert!(advance_all(r#"{"tool": "edit", "args": {"line""#).is_none()); // param order fixed
    assert!(advance_all(r#"{"tool": "edit" "args""#).is_none()); // missing comma
    assert!(advance_all(r#"{"tool": "edit", "args": {"path": 42"#).is_none()); // int for string
    assert!(advance_all(r#"{"tool": "edit", "args": {"path": "a", "line": 4.2"#).is_none());
    assert!(advance_all(r#"{"tool": "edit", "args": {"path": "a", "line": 042"#).is_none());
    assert!(advance_all(r#"{"tool": "edit", "args": {"path": "a", "line": -,"#).is_none());
    // string values may not contain a raw quote or newline
    assert!(advance_all("{\"tool\": \"edit\", \"args\": {\"path\": \"a\nb").is_none());
}

#[test]
fn escaped_quotes_and_backslashes_are_legal_in_string_values() {
    let g = Grammar::compile(&edit_schema());
    let s = r#"{"tool": "edit", "args": {"path": "a\"b\\c", "line": 0, "create": true}}"#;
    let mut st = g.start();
    for c in s.chars() {
        st = g.advance(&st, c).expect("escaped string rejected");
    }
}

#[test]
fn mask_permits_exactly_the_grammar_allowed_tokens() {
    let g = Grammar::compile(&edit_schema());
    let vocab = test_vocab();
    // At the start, only tokens whose entire string is a valid prefix of the
    // grammar are allowed: `{` and the multi-char `{"tool"` token.
    let mask = g.mask(&g.start(), &vocab);
    let allowed: Vec<&str> = (0..vocab.len())
        .filter(|&t| mask[t])
        .map(|t| vocab.text(t as u32))
        .collect();
    assert_eq!(allowed, vec!["{", "{\"tool\""]);

    // Inside an int value, digits stay allowed and letters do not.
    let prefix = r#"{"tool": "edit", "args": {"path": "a", "line": 1"#;
    let mut st = g.start();
    for c in prefix.chars() {
        st = g.advance(&st, c).unwrap();
    }
    let mask = g.mask(&st, &vocab);
    let idx = |s: &str| vocab.iter().position(|t| t == s).unwrap();
    assert!(mask[idx("0")] && mask[idx("9")]);
    assert!(mask[idx(",")]); // value may end here
    assert!(!mask[idx("x")] && !mask[idx("\"")]);
}

#[test]
fn eos_is_allowed_only_at_completion() {
    let g = Grammar::compile(&edit_schema());
    let vocab = test_vocab();
    let full = r#"{"tool": "edit", "args": {"path": "a", "line": 1, "create": true}}"#;
    let mut st = g.start();
    for c in full.chars() {
        let mask = g.mask(&st, &vocab);
        assert!(!mask[vocab.eos() as usize], "EOS allowed before completion");
        st = g.advance(&st, c).unwrap();
    }
    assert!(g.is_complete(&st));
    let mask = g.mask(&st, &vocab);
    assert!(mask[vocab.eos() as usize]);
    assert_eq!(
        mask.iter().filter(|&&b| b).count(),
        1,
        "only EOS at completion"
    );
}

#[test]
fn random_grammar_walks_always_parse_as_schema_valid_json() {
    // Property test with a seeded LCG: any path through the mask is valid.
    let g = Grammar::compile(&edit_schema());
    let vocab = test_vocab();
    let mut seed: u64 = 0x5eed;
    for _ in 0..50 {
        let mut st = g.start();
        let mut out = String::new();
        loop {
            let mask = g.mask(&st, &vocab);
            let allowed: Vec<u32> = (0..vocab.len() as u32)
                .filter(|&t| mask[t as usize])
                .collect();
            assert!(!allowed.is_empty(), "grammar dead end after: {out}");
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let tok = allowed[(seed >> 33) as usize % allowed.len()];
            if tok == vocab.eos() {
                break;
            }
            for c in vocab.text(tok).chars() {
                st = g.advance(&st, c).unwrap();
            }
            out.push_str(vocab.text(tok));
        }
        let v: serde_json::Value = serde_json::from_str(&out).expect("walk produced invalid JSON");
        assert_eq!(v["tool"], "edit");
        assert!(v["args"]["path"].is_string());
        // Walks can emit integers wider than 64 bits; serde parses those as
        // f64, so assert number-ness rather than integer width.
        assert!(v["args"]["line"].is_number());
        assert!(v["args"]["create"].is_boolean());
    }
}

#[test]
fn test_200_forced_tool_calls_produce_zero_malformed_calls() {
    let g = Grammar::compile(&edit_schema());
    let vocab = test_vocab();
    // Adversarial script: every position puts all probability mass on token 0
    // (the space character), which the grammar mostly forbids. Sampling must
    // still be forced onto grammar-legal tokens by the mask alone.
    let mut adversarial = vec![0.0f32; vocab.len()];
    adversarial[0] = 100.0;
    let script = vec![adversarial; 512];
    let params = SamplingParams {
        temperature: 0.8,
        top_p: 0.95,
        seed: 7,
    };

    for run in 0..200 {
        let mut model = MockModel::new(vocab.len(), script.clone());
        let mut sampler = Sampler::new(SamplingParams {
            seed: params.seed + run,
            ..params
        });
        model.prefill(&[1, 2, 3]).unwrap(); // pretend prompt
        let mut st = g.start();
        let mut out = String::new();
        let mut logits = model.prefill(&[0]).unwrap();
        loop {
            let mask = g.mask(&st, &vocab);
            let tok = sampler.sample(&logits, Some(mask.as_slice())).unwrap();
            if tok == vocab.eos() {
                break;
            }
            for c in vocab.text(tok).chars() {
                st = g
                    .advance(&st, c)
                    .expect("sampler emitted grammar-illegal token");
            }
            out.push_str(vocab.text(tok));
            logits = model.prefill(&[tok]).unwrap();
        }
        let v: serde_json::Value = serde_json::from_str(&out)
            .unwrap_or_else(|e| panic!("run {run}: malformed: {e}\n{out}"));
        assert_eq!(v["tool"], "edit", "run {run}");
        assert!(v["args"].is_object(), "run {run}");
    }
}
