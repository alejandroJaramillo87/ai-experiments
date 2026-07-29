/*
 * tracing_spans — phase 2 acceptance (harness side): tool.exec and
 * agent.fork spans, and a mock-run Chrome trace that loads as JSON with the
 * documented span names present.
 */

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use harness::subagents;
use harness::tools::ToolCall;
use runtime::model::mock::MockModel;
use runtime::session::Session;
use tracing::field::{Field, Visit};
use tracing::span::{Attributes, Id, Record};
use tracing::Subscriber;
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

#[test]
fn chrome_trace_of_a_mock_run_loads_as_json_with_the_schema_names() {
    let dir = tempfile::tempdir().unwrap();
    let trace_path = dir.path().join("trace.json");
    {
        let (layer, guard) = tracing_chrome::ChromeLayerBuilder::new()
            .file(&trace_path)
            .include_args(true)
            .build();
        let subscriber = Registry::default().with(layer);
        tracing::subscriber::with_default(subscriber, || {
            let mut parent = Session::new(Box::new(MockModel::new(8, vec![])), 4096).unwrap();
            parent.sync(&[1, 2, 3]).unwrap();
            subagents::spawn(&parent, &[1, 2, 3, 4]).unwrap();
        });
        drop(guard); // flushes the trace file
    }
    let text = std::fs::read_to_string(&trace_path).unwrap();
    let events: serde_json::Value = serde_json::from_str(&text).unwrap();
    let events = events.as_array().expect("chrome trace is a JSON array");
    let names: Vec<&str> = events
        .iter()
        .filter_map(|e| e.get("name").and_then(|n| n.as_str()))
        .collect();
    assert!(names.contains(&"session.sync"), "{names:?}");
    assert!(names.contains(&"agent.fork"), "{names:?}");
}
