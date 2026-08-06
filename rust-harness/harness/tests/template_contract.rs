/*
 * template_contract — the chat template seam between the tiers.
 *
 * The training tier renders a conversation to a prompt string and trains on
 * it; the harness renders the same conversation and generates against it. No
 * compiler connects the two, and they drifted once already: Python corrected
 * Gemma 4 to the <|turn> framing at CHAT_TEMPLATE_VERSION 2 while the harness
 * kept emitting Gemma 3's <start_of_turn>, which is not in the Gemma 4
 * vocabulary and so tokenized as a byte run rather than failing.
 *
 * Two assertions guard it. The first compares the Rust constants against
 * schemas/chat_template.json, so a one-sided edit fails on this side. The
 * second runs the real Python renderer and compares byte for byte, which is
 * the property that actually matters and the one a re-implementation of
 * render_segments in Rust would not establish.
 */

use std::path::{Path, PathBuf};

use harness::template::{self, render_entry};
use harness::transcript::{Entry, Role};

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// The parsed contract, or None when it is absent — as in the container
/// build, whose context is rust-harness/ alone.
fn contract() -> Option<serde_json::Value> {
    let text = std::fs::read_to_string(repo_root().join("schemas/chat_template.json")).ok()?;
    Some(serde_json::from_str(&text).expect("contract is valid JSON"))
}

/// The interpreter that can run the real renderer, or None when the parent
/// tree or the renderer's own imports are absent.
///
/// Probing the imports the script actually performs, rather than running an
/// empty script, is the point: an interpreter that exists but cannot import
/// `training.data.sft` used to pass this guard and then fail the test on the
/// dependency, reporting a template defect where there was an environment gap.
fn python() -> Option<PathBuf> {
    if !repo_root().join("training/data/sft.py").is_file() {
        return None;
    }
    runtime::pyenv::interpreter(
        &repo_root(),
        &["training.models.families", "training.data.sft"],
    )
}

fn entry(role: Role, content: &str) -> Entry {
    Entry {
        role,
        content: content.into(),
        replaces: None,
    }
}

/// The conversation both tiers render, as (harness role, python role, text).
/// A system turn is included because the native system role is the change
/// that version 2 introduced.
/// Content is whitespace-edged on purpose. The prose messages alone left the
/// two tiers' trimming rules untested, and they differ: Python strips by
/// str.isspace, which covers the C0 separators U+001C-U+001F that Rust's
/// str::trim leaves in place. A tool result carrying one reaches here.
const CONVERSATION: &[(Role, &str, &str)] = &[
    (Role::System, "system", "you are a helpful assistant"),
    (Role::User, "user", "  what is the capital of france\n"),
    (Role::Model, "assistant", "the capital of france is paris"),
    (Role::User, "user", "\u{1c}and of spain\u{1f}"),
    (
        Role::Model,
        "assistant",
        "\tthe capital of spain is madrid  ",
    ),
];

#[test]
fn rust_constants_match_the_shared_chat_template_contract() {
    let Some(contract) = contract() else {
        runtime::testlog::skip(
            "contract assertions",
            "schemas/chat_template.json is absent",
        );
        return;
    };
    let spec = &contract["spec"];
    assert_eq!(spec["bos"], template::BOS);
    assert_eq!(spec["eot"], template::TURN_CLOSE);
    assert_eq!(
        contract["chat_template_version"],
        template::CHAT_TEMPLATE_VERSION
    );
    // turn_prefix carries a {role} placeholder, so it is compared through
    // the rendering rather than against TURN_OPEN directly.
    let prefix = spec["turn_prefix"]
        .as_str()
        .expect("turn_prefix is a string");
    assert_eq!(prefix.replace("{role}", "model"), template::MODEL_TURN_OPEN);
    assert!(
        prefix.starts_with(template::TURN_OPEN),
        "turn_prefix {prefix} does not open with {}",
        template::TURN_OPEN
    );
    assert_eq!(spec["trim_content"], true);
    assert_eq!(spec["assistant_role"], "model");

    // The neutralized set is pinned by equality, not containment: a marker
    // added to MARKERS but not the contract is as much a drift as the
    // reverse, and the contract is what the Python side reads. Sorted
    // because neither side's order is meaningful.
    let mut declared: Vec<&str> = contract["markers"]["neutralized_in_content"]
        .as_array()
        .expect("neutralized_in_content is an array")
        .iter()
        .map(|v| v.as_str().expect("marker is a string"))
        .collect();
    let mut ours: Vec<&str> = template::MARKERS.to_vec();
    declared.sort_unstable();
    ours.sort_unstable();
    assert_eq!(
        ours, declared,
        "template::MARKERS and the contract disagree"
    );

    // What the renderer emits must itself be neutralized in content, or
    // untrusted text could forge the one structure the prompt relies on.
    for marker in contract["markers"]["rendered"]
        .as_array()
        .expect("rendered is an array")
    {
        let marker = marker.as_str().expect("marker is a string");
        assert!(
            template::MARKERS.contains(&marker),
            "rendered marker {marker} is not neutralized in content"
        );
    }
}

#[test]
fn python_render_segments_reproduces_the_rust_rendering() {
    let Some(python) = python() else {
        runtime::testlog::skip(
            "python template parity",
            "no interpreter imports training.data.sft (the parent tree is absent, or \
             the environment lacks the training group — `make install`, or set \
             HARNESS_PYTHON)",
        );
        return;
    };
    // The real renderer and the real spec, not a transcription of either:
    // a Rust copy of render_segments would agree with a Rust bug.
    const SCRIPT: &str = "import json, sys\n\
         from training.models.families import family_spec, CHAT_TEMPLATE_VERSION\n\
         from training.data.sft import render_segments\n\
         messages = json.load(sys.stdin)\n\
         spec = family_spec('gemma4').chat_template\n\
         text = ''.join(t for t, _ in render_segments(messages, spec))\n\
         json.dump({'text': text, 'version': CHAT_TEMPLATE_VERSION}, sys.stdout)\n";
    let messages: Vec<serde_json::Value> = CONVERSATION
        .iter()
        .map(|(_, role, content)| serde_json::json!({"role": role, "content": content}))
        .collect();

    let mut child = std::process::Command::new(&python)
        .args(["-c", SCRIPT])
        .env("PYTHONPATH", repo_root())
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .expect("python3 runs");
    serde_json::to_writer(child.stdin.take().expect("stdin piped"), &messages).unwrap();
    let out = child.wait_with_output().expect("python3 completes");
    assert!(
        out.status.success(),
        "the python renderer failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let got: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();

    // Rust renders closed turns and then opens a model turn for generation;
    // Python renders closed conversations only, so the comparison stops at
    // the last closed turn.
    let mut rust = String::from(template::BOS);
    for (role, _, content) in CONVERSATION {
        rust.push_str(&render_entry(&entry(role.clone(), content)));
    }
    assert_eq!(
        got["text"].as_str().expect("text is a string"),
        rust,
        "the two tiers render one conversation to different bytes"
    );
    assert_eq!(got["version"], template::CHAT_TEMPLATE_VERSION);
}
