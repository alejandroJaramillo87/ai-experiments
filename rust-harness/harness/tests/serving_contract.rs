/*
 * serving_contract — the endpoint against the Python client that will call it.
 *
 * harness/tests/server.rs asserts the wire contract against a request this
 * workspace also wrote, which is the failure mode .styles/RUST_STYLE.md names:
 * a check written in the same language by the same author reproduces the
 * author's misunderstanding on both sides. If we misread what
 * common/serving.py sends, our client sends the same wrong thing and agrees.
 *
 * So this drives a live server through common/serving.py's own `wait_healthy`
 * and `chat_completion` -- the exact functions interp/steering/scorecard.py
 * and interp/commands/organism.py reach the model through. It is the test
 * that decides whether the endpoint can replace llama-server for interp, and
 * the only one here that could discover a field we never thought about.
 *
 * The model is scripted, so this says nothing about output quality; the
 * workstation run against real E4B is what settles that.
 */

use std::io::Write;
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use harness::generator::SessionGenerator;
use harness::server::{serve, ServerState};
use runtime::model::mock::MockModel;
use runtime::sampler::SamplingParams;
use runtime::session::Session;
use runtime::tokenizer::Tokenizer;

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// The interpreter that can import the client under test, or None when the
/// parent tree is absent -- as in the container build, whose context is
/// rust-harness/ alone.
fn python() -> Option<PathBuf> {
    if !repo_root().join("common/serving.py").is_file() {
        return None;
    }
    runtime::pyenv::interpreter(&repo_root(), &["common.serving"])
}

/// A greedy server over a scripted model, on an ephemeral port.
fn start() -> u16 {
    let mut pieces = vec![
        "<pad>".to_string(),
        "<bos>".to_string(),
        "<eos>".to_string(),
    ];
    for b in 0..=255u8 {
        pieces.push(format!("<0x{b:02X}>"));
    }
    let json = runtime::synth::tokenizer_json_from_parts(&pieces, &[], "<pad>").unwrap();
    let tokenizer = Tokenizer::from_json(&json).unwrap();
    let vocab_len = tokenizer.vocab_len();

    // One dominant letter, so greedy decoding gives a predictable reply.
    let mut row = vec![f32::NEG_INFINITY; vocab_len];
    row[b'y' as usize + 3] = 1.0;
    let script = vec![row; 4096];

    let params = SamplingParams {
        temperature: 0.0,
        top_p: 1.0,
        top_k: None,
        seed: 0,
    };
    let session = Session::new(Box::new(MockModel::new(vocab_len, script)), 4096).unwrap();
    let generator = SessionGenerator::new(session, tokenizer, params, 8).unwrap();
    let state = Arc::new(ServerState::new(
        generator,
        "scripted".to_string(),
        params,
        8,
    ));
    let listener = TcpListener::bind(("127.0.0.1", 0)).unwrap();
    let port = listener.local_addr().unwrap().port();
    std::thread::spawn(move || {
        let _ = serve(listener, state);
    });
    port
}

/// Run a script against the live server with the repository on PYTHONPATH.
fn drive(python: &Path, script: &str, port: u16) -> std::process::Output {
    // Written to a file rather than passed with -c: the scripts below are
    // multi-line and quoting them through the shell would be the fragile part
    // of a test whose subject is elsewhere.
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("drive.py");
    let mut file = std::fs::File::create(&path).unwrap();
    file.write_all(script.as_bytes()).unwrap();
    file.flush().unwrap();
    std::process::Command::new(python)
        .arg(&path)
        .arg(port.to_string())
        .env("PYTHONPATH", repo_root())
        .output()
        .expect("python3 runs")
}

#[test]
fn the_python_client_completes_a_chat_against_the_endpoint() {
    let Some(python) = python() else {
        runtime::testlog::skip(
            "serving client assertions",
            "no interpreter imports common.serving (the parent tree is absent, or the \
             environment is unprovisioned -- `make install`, or set HARNESS_PYTHON)",
        );
        return;
    };
    let port = start();

    // A ServiceSpec built here rather than read from SERVICES, so the test
    // exercises the client against an ephemeral port. The registry entry that
    // names the real port is asserted separately, in the Python suite.
    const SCRIPT: &str = "import json, sys\n\
         from common import serving\n\
         spec = serving.ServiceSpec('harness', int(sys.argv[1]), 'HARNESS_MODEL',\n\
         '/v1/health', concurrency=1, start_budget_s=30)\n\
         serving.wait_healthy(spec)\n\
         out = serving.chat_completion(spec, [{'role': 'user', 'content': 'hi'}],\n\
         {'temperature': 0.0, 'max_tokens': 4}, timeout=60)\n\
         json.dump({'content': out['content'], 'usage': out['usage']}, sys.stdout)\n";
    let out = drive(&python, SCRIPT, port);
    assert!(
        out.status.success(),
        "the python client failed against the endpoint\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    let parsed: serde_json::Value = serde_json::from_slice(&out.stdout)
        .unwrap_or_else(|e| panic!("client output is not JSON ({e}): {:?}", out.stdout));

    // chat_completion raises ServingError unless choices[0].message.content
    // is present, so reaching here already proves the shape. What is asserted
    // is that the content is the reply and the usage block survived the trip.
    assert_eq!(
        parsed["content"], "yyyy",
        "greedy decoding should spell the scripted letter: {parsed}"
    );
    assert_eq!(
        parsed["usage"]["completion_tokens"], 4,
        "usage did not reach the client intact: {parsed}"
    );
}

#[test]
fn the_python_client_classifies_a_bad_request_as_terminal() {
    let Some(python) = python() else {
        runtime::testlog::skip(
            "serving error-classification assertions",
            "no interpreter imports common.serving (the parent tree is absent, or the \
             environment is unprovisioned -- `make install`, or set HARNESS_PYTHON)",
        );
        return;
    };
    let port = start();

    // post_chat maps 4xx to ServingError and 5xx to ServingTransientError, so
    // the status this endpoint chooses decides whether a caller retries. A
    // refusal the client retried forever would be worse than the refusal.
    const SCRIPT: &str = "import sys\n\
         from common import serving\n\
         url = f'http://127.0.0.1:{sys.argv[1]}/v1/chat/completions'\n\
         try:\n\
         \x20   serving.post_chat(url, {'messages': [{'role': 'x', 'content': 'hi'}]})\n\
         except serving.ServingTransientError as e:\n\
         \x20   sys.exit(f'retryable, should be terminal: {e}')\n\
         except serving.ServingError as e:\n\
         \x20   print('terminal', e)\n\
         else:\n\
         \x20   sys.exit('an unknown role was accepted')\n";
    let out = drive(&python, SCRIPT, port);
    assert!(
        out.status.success(),
        "stdout: {}\nstderr: {}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
}
