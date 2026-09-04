/*
 * tracing_spans — phase 2 acceptance (harness side): tool.exec and
 * agent.fork spans, and a Chrome trace that loads as JSON with the
 * documented span names present. The trace tests build the layer the way
 * main.rs does, filter included, so the level `--trace-level` selects is
 * shown to reach the file: the interior spans at the default, none of them
 * at info.
 */

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use harness::subagents;
use harness::tools::ToolCall;
use runtime::model::cpu::CpuModel;
use runtime::model::mock::MockModel;
use runtime::session::Session;
use tracing::field::{Field, Visit};
use tracing::span::{Attributes, Id, Record};
use tracing::Subscriber;
use tracing_subscriber::filter::LevelFilter;
use tracing_subscriber::layer::{Context, Layer, SubscriberExt};
use tracing_subscriber::registry::Registry;

#[derive(Debug, Clone)]
struct CapturedSpan {
    name: String,
    fields: Vec<(String, String)>,
}

#[derive(Default)]
struct CaptureState {
    spans: Vec<CapturedSpan>,
    // Registry span ids are reused after close; map only the currently open
    // ids to their chronological slot for the `on_record` merge.
    open: HashMap<u64, usize>,
}

#[derive(Clone, Default)]
struct Capture(Arc<Mutex<CaptureState>>);

struct FieldVisitor(Vec<(String, String)>);

impl Visit for FieldVisitor {
    fn record_debug(&mut self, field: &Field, value: &dyn std::fmt::Debug) {
        self.0
            .push((field.name().to_string(), format!("{value:?}")));
    }
}

impl<S: Subscriber> Layer<S> for Capture {
    fn on_new_span(&self, attrs: &Attributes<'_>, id: &Id, _ctx: Context<'_, S>) {
        let mut visitor = FieldVisitor(Vec::new());
        attrs.record(&mut visitor);
        let mut state = self.0.lock().unwrap();
        let idx = state.spans.len();
        state.spans.push(CapturedSpan {
            name: attrs.metadata().name().to_string(),
            fields: visitor.0,
        });
        state.open.insert(id.into_u64(), idx);
    }

    fn on_record(&self, id: &Id, values: &Record<'_>, _ctx: Context<'_, S>) {
        let mut visitor = FieldVisitor(Vec::new());
        values.record(&mut visitor);
        let mut state = self.0.lock().unwrap();
        if let Some(&idx) = state.open.get(&id.into_u64()) {
            state.spans[idx].fields.extend(visitor.0);
        }
    }
}

impl Capture {
    fn named(&self, name: &str) -> Vec<CapturedSpan> {
        self.0
            .lock()
            .unwrap()
            .spans
            .iter()
            .filter(|s| s.name == name)
            .cloned()
            .collect()
    }
}

fn field<'a>(span: &'a CapturedSpan, name: &str) -> Option<&'a str> {
    span.fields
        .iter()
        .find(|(k, _)| k == name)
        .map(|(_, v)| v.as_str())
}

#[test]
fn tool_exec_span_carries_tool_target_and_output_bytes() {
    let capture = Capture::default();
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("hello.txt"), "hello\n").unwrap();
    let subscriber = Registry::default().with(capture.clone());
    tracing::subscriber::with_default(subscriber, || {
        let call = ToolCall::parse(r#"{"tool": "read", "args": {"path": "hello.txt"}}"#)
            .unwrap()
            .unwrap();
        call.execute(dir.path()).unwrap();
    });
    let spans = capture.named("tool.exec");
    assert_eq!(spans.len(), 1);
    assert_eq!(field(&spans[0], "tool"), Some("\"read\""));
    assert_eq!(field(&spans[0], "target"), Some("\"hello.txt\""));
    assert_eq!(field(&spans[0], "output_bytes"), Some("6"));
}

#[test]
fn agent_fork_span_carries_parent_cache_and_task_delta() {
    let capture = Capture::default();
    let subscriber = Registry::default().with(capture.clone());
    tracing::subscriber::with_default(subscriber, || {
        let mut parent = Session::new(Box::new(MockModel::new(8, vec![])), 4096).unwrap();
        parent.sync(&[1, 2, 3]).unwrap();
        // Child transcript: parent prefix plus a two-token task prompt.
        let (_child, delta) = subagents::spawn(&parent, &[1, 2, 3, 4, 5]).unwrap();
        assert_eq!(delta, 2);
    });
    let spans = capture.named("agent.fork");
    assert_eq!(spans.len(), 1);
    assert_eq!(field(&spans[0], "parent_cached_len"), Some("3"));
    assert_eq!(field(&spans[0], "delta_len"), Some("2"));
}

/// The span names a Chrome trace carries after a mock fork and a real
/// two-layer load and sync, written through the same builder and filter
/// main.rs installs for `--trace-level <level>`.
fn chrome_trace_names(level: LevelFilter) -> Vec<String> {
    let dir = tempfile::tempdir().unwrap();
    let trace_path = dir.path().join("trace.json");
    {
        let (layer, guard) = tracing_chrome::ChromeLayerBuilder::new()
            .file(&trace_path)
            .include_args(true)
            .build();
        let subscriber = Registry::default().with(layer.with_filter(level));
        tracing::subscriber::with_default(subscriber, || {
            let mut parent = Session::new(Box::new(MockModel::new(8, vec![])), 4096).unwrap();
            parent.sync(&[1, 2, 3]).unwrap();
            subagents::spawn(&parent, &[1, 2, 3, 4]).unwrap();
            let tiny =
                std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../fixtures/tiny-gemma4");
            let (model, _) = CpuModel::load(&tiny, false).unwrap();
            let mut real = Session::new(Box::new(model), 64).unwrap();
            real.sync(&[1, 2, 3]).unwrap();
        });
        drop(guard); // flushes the trace file
    }
    let text = std::fs::read_to_string(&trace_path).unwrap();
    let events: serde_json::Value = serde_json::from_str(&text).unwrap();
    let events = events.as_array().expect("chrome trace is a JSON array");
    events
        .iter()
        .filter_map(|e| e.get("name").and_then(|n| n.as_str()))
        .map(str::to_string)
        .collect()
}

#[test]
fn chrome_trace_of_a_mock_run_loads_as_json_with_the_schema_names() {
    // DEBUG is the default `--trace-level`.
    let names = chrome_trace_names(LevelFilter::DEBUG);
    let has = |n: &str| names.iter().any(|x| x == n);
    assert!(has("session.sync"), "{names:?}");
    assert!(has("agent.fork"), "{names:?}");
    assert!(has("model.forward"), "{names:?}");
    assert!(has("model.load"), "{names:?}");
    assert!(has("model.layer"), "{names:?}");
    assert!(has("model.layer.attn"), "{names:?}");
    assert!(has("model.lm_head"), "{names:?}");
    assert!(
        !has("kernel.matmul"),
        "kernels are TRACE, above the default: {names:?}"
    );
}

#[test]
fn an_info_trace_carries_no_layer_spans() {
    let names = chrome_trace_names(LevelFilter::INFO);
    let has = |n: &str| names.iter().any(|x| x == n);
    assert!(has("session.sync"), "{names:?}");
    assert!(has("model.forward"), "{names:?}");
    assert!(has("model.load"), "{names:?}");
    assert!(!has("model.layer"), "{names:?}");
    assert!(!has("model.prelude"), "{names:?}");
    assert!(!has("checkpoint.read"), "{names:?}");
    assert!(!has("kernel.matmul"), "{names:?}");
}
