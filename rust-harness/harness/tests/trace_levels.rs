/*
 * trace_levels — --trace-level selects how deep a Chrome trace goes.
 *
 * `trace_structure.rs` asserts the span relations at whatever level a trace
 * was written, and `cli_surface.rs` asserts that the flag needs
 * `--trace-chrome` and names its three levels on a typo. Neither passes a
 * level and reads what changed, so a filter wired to a constant would leave
 * both green: every trace would be a debug trace and the flag a word the
 * parser accepted. What is asserted here is the flag's own claim, that the
 * three levels are nested and each adds the spans its help text names.
 */

use std::collections::BTreeSet;
use std::path::Path;
use std::process::{Command, Stdio};

use serde_json::Value;

/// The names of every span a task run opened, traced at `level`, or at the
/// default when `level` is `None`.
///
/// One task rather than a chat: the level decides which callsites are
/// enabled, and one generation exercises every callsite the three levels
/// tell apart. Two tokens keep the trace short; the fixture's weights are
/// random and no token is read.
fn span_names(level: Option<&str>) -> BTreeSet<String> {
    // Held to the end: the trace is read back from it.
    let dir = tempfile::tempdir().expect("a scratch working directory");
    let fixtures = Path::new(env!("CARGO_MANIFEST_DIR")).join("../fixtures");
    let trace_path = dir.path().join("trace.json");
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_harness"));
    cmd.arg("--model")
        .arg(fixtures.join("tiny-gemma4"))
        .arg("--text-only")
        .arg("--trace-chrome")
        .arg(&trace_path)
        .args(["--max-tokens", "2", "--seed", "0"]);
    if let Some(level) = level {
        cmd.args(["--trace-level", level]);
    }
    let out = cmd
        .arg("say hi")
        .env("OBS_LOG_DIR", dir.path().join("obs"))
        .current_dir(dir.path())
        .stdin(Stdio::null())
        .output()
        .expect("the harness builds as a test binary");
    assert!(
        out.status.success(),
        "--trace-level {level:?} failed the run: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let text = std::fs::read_to_string(&trace_path)
        .unwrap_or_else(|e| panic!("{}: {e}", trace_path.display()));
    let value: Value =
        serde_json::from_str(&text).unwrap_or_else(|e| panic!("{}: {e}", trace_path.display()));
    let events = match value {
        Value::Array(events) => events,
        Value::Object(mut o) => match o.remove("traceEvents") {
            Some(Value::Array(events)) => events,
            _ => panic!("{}: no traceEvents array", trace_path.display()),
        },
        other => panic!("{}: unexpected top-level {other}", trace_path.display()),
    };
    events
        .iter()
        .filter(|e| e.get("ph").and_then(Value::as_str) == Some("B"))
        .filter_map(|e| e.get("name").and_then(Value::as_str))
        .map(str::to_string)
        .collect()
}

/// The three levels are strictly nested, and each carries the spans its help
/// text promises: info the session-level spans and the load, debug one span
/// per layer and block, trace one per kernel.
#[test]
fn each_trace_level_adds_the_spans_its_help_text_names() {
    let info = span_names(Some("info"));
    let debug = span_names(Some("debug"));
    let trace = span_names(Some("trace"));

    for name in [
        "session.sync",
        "session.step",
        "sampler.sample",
        "model.forward",
        "model.load",
    ] {
        assert!(info.contains(name), "info lacks {name}: {info:?}");
    }
    assert!(
        !info.iter().any(|n| n.starts_with("model.layer")),
        "info carries a per-layer span: {info:?}"
    );
    assert!(
        debug.contains("model.layer") && debug.contains("model.layer.attn"),
        "debug lacks the per-layer spans: {debug:?}"
    );
    assert!(
        !debug.iter().any(|n| n.starts_with("kernel.")),
        "debug carries a kernel span: {debug:?}"
    );
    assert!(
        trace.contains("kernel.attention") && trace.contains("kernel.matmul"),
        "trace lacks the kernel spans: {trace:?}"
    );
    // Strict nesting: every span a shallower level opened, the deeper one
    // opened too, and the deeper one opened more.
    assert!(
        info.is_subset(&debug) && info.len() < debug.len(),
        "info {info:?} is not strictly inside debug {debug:?}"
    );
    assert!(
        debug.is_subset(&trace) && debug.len() < trace.len(),
        "debug {debug:?} is not strictly inside trace {trace:?}"
    );
}

/// The default is `debug`, as the flag table states, and not either
/// neighbour: a default that drifted to `trace` would cost every traced run
/// a span per kernel, and one that drifted to `info` would lose the layer
/// spans RH.6 reads.
#[test]
fn the_default_trace_level_is_debug() {
    let unnamed = span_names(None);
    let debug = span_names(Some("debug"));
    assert_eq!(
        unnamed, debug,
        "the default level opened a different span set"
    );
}
