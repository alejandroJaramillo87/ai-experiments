/*
 * tracing_spans — phase 2 acceptance (runtime side): the span schema is
 * asserted with a capturing subscriber over a full mock generation. Span
 * names and fields are the stable vocabulary shared with debugging and
 * benchmark analysis, so a rename here is a schema change, not a detail.
 *
 * The second half (U10) asserts the interior spans on the CPU backend: one
 * per layer under `model.forward` with attention, MLP and PLE children, the
 * prelude and LM head, the load stages, the tokenizer, and the kernels at
 * TRACE. Nesting is asserted by parent name, and a level filter is shown to
 * drop exactly the tiers below it. The last test is the one that matters
 * most: the logits are bit-identical with and without a subscriber.
 */

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use runtime::grammar::{Grammar, ParamType, ToolSchema, Vocab};
use runtime::model::cpu::CpuModel;
use runtime::model::mock::MockModel;
use runtime::model::Model;
use runtime::sampler::{Sampler, SamplingParams};
use runtime::session::Session;
use runtime::tokenizer::Tokenizer;
use tracing::field::{Field, Visit};
use tracing::span::{Attributes, Id, Record};
use tracing::Subscriber;
use tracing_subscriber::filter::LevelFilter;
use tracing_subscriber::layer::{Context, Layer, SubscriberExt};
use tracing_subscriber::registry::{LookupSpan, Registry};

/// One captured span: its name, the name of the span it was opened under,
/// and every field recorded on it, at creation or later via `Span::record`.
#[derive(Debug, Clone)]
struct CapturedSpan {
    name: String,
    parent: Option<String>,
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

impl<S> Layer<S> for Capture
where
    S: Subscriber + for<'a> LookupSpan<'a>,
{
    fn on_new_span(&self, attrs: &Attributes<'_>, id: &Id, ctx: Context<'_, S>) {
        let mut visitor = FieldVisitor(Vec::new());
        attrs.record(&mut visitor);
        // The contextual parent, which is what the code under test relies on:
        // no span here names its parent explicitly.
        let parent = ctx
            .span(id)
            .and_then(|s| s.parent())
            .map(|p| p.name().to_string());
        let mut state = self.0.lock().unwrap();
        let idx = state.spans.len();
        state.spans.push(CapturedSpan {
            name: attrs.metadata().name().to_string(),
            parent,
            fields: visitor.0,
        });
        state.open.insert(id.into_u64(), idx);
    }

    fn on_close(&self, id: Id, _ctx: Context<'_, S>) {
        // Registry ids are reused after close; drop the mapping so a later
        // span with the same id cannot merge fields into an earlier slot.
        self.0.lock().unwrap().open.remove(&id.into_u64());
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

fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../fixtures")
}

/// The committed two-layer fixture: no shared KV, no per-layer embeddings.
fn tiny() -> CpuModel {
    CpuModel::load(&fixtures().join("tiny-gemma4"), false)
        .unwrap()
        .0
}

/// The Tier T fixture: four layers, the last two reading a donor's KV cache,
/// and per-layer embeddings. Absent means the generator has not been run on
/// a workstation, the same guard as `ladder_flags.rs`.
fn tiny_hf() -> Option<CpuModel> {
    let dir = fixtures().join("tiny-hf-gemma4");
    if !dir.join("capture.json").exists() {
        runtime::testlog::skip(
            "interior span assertions on the shared-KV fixture",
            "fixtures/tiny-hf-gemma4 has not been generated; \
             run fixtures/build_tiny_reference.py on the workstation",
        );
        return None;
    }
    Some(CpuModel::load(&dir, false).unwrap().0)
}

/// Runs `f` with `layer` installed for the current thread only, so the level
/// question can be asked in-process without a second global subscriber.
fn under<L, T>(layer: L, f: impl FnOnce() -> T) -> T
where
    L: Layer<Registry> + Send + Sync + 'static,
{
    tracing::subscriber::with_default(Registry::default().with(layer), f)
}

/// Two forwards: a three-token prefill and a one-token step.
fn two_forwards(model: &mut CpuModel) {
    model.prefill(&[1, 2, 3]).unwrap();
    model.prefill(&[4]).unwrap();
}

fn count(capture: &Capture, name: &str) -> usize {
    capture.named(name).len()
}

/// Serializes the two tests in this binary. `tracing` caches per-callsite
/// interest process-globally, so a test driving these callsites with no
/// subscriber can race another test's subscriber registration and leave a
/// stale "never" interest behind, silently dropping spans for that run.
/// Cargo runs tests in one process on multiple threads, so the lock is the
/// fix rather than an ordering assumption.
static CALLSITE_LOCK: Mutex<()> = Mutex::new(());

/// Grammar-constrained generation over a character vocab, driving every
/// instrumented runtime path: sync, step, sample, and grammar mask.
fn traced_generation(capture: &Capture) {
    let mut tokens: Vec<String> = "{}\"argstol: ,ping123"
        .chars()
        .map(|c| c.to_string())
        .collect();
    tokens.push("</s>".to_string());
    let eos = (tokens.len() - 1) as u32;
    let vocab = Vocab::new(tokens, eos);
    let grammar = Grammar::compile(&ToolSchema {
        name: "ping".into(),
        params: vec![("n".into(), ParamType::Int)],
    });
    let script = vec![vec![1.0; vocab.len()]; 512];
    let subscriber = Registry::default().with(capture.clone());
    tracing::subscriber::with_default(subscriber, || {
        let mut session =
            Session::new(Box::new(MockModel::new(vocab.len(), script)), 4096).unwrap();
        let mut sampler = Sampler::new(SamplingParams {
            temperature: 1.0,
            top_p: 1.0,
            top_k: None,
            seed: 5,
        });
        session
            .generate(&[0, 1], &vocab, Some(&grammar), &mut sampler, 256, |_| {})
            .unwrap();
    });
}

#[test]
fn generation_emits_the_documented_span_schema() {
    let _serial = CALLSITE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let capture = Capture::default();
    traced_generation(&capture);

    // session.sync: the prompt prefill with its delta and cache lengths.
    let syncs = capture.named("session.sync");
    assert_eq!(syncs.len(), 1);
    assert_eq!(field(&syncs[0], "cached_len"), Some("0"));
    assert_eq!(field(&syncs[0], "delta_len"), Some("2"));

    // session.step: one per generated token (plus the EOS step), positions
    // advancing from the prompt boundary, each with a chosen token.
    let steps = capture.named("session.step");
    assert!(steps.len() > 2, "expected a multi-token generation");
    assert_eq!(field(&steps[0], "position"), Some("2"));
    assert!(field(&steps[0], "token").is_some());

    // sampler.sample: one per step, carrying the configured temperature and
    // a recorded candidate count.
    let samples = capture.named("sampler.sample");
    assert_eq!(samples.len(), steps.len());
    assert_eq!(field(&samples[0], "temperature"), Some("1.0"));
    assert!(field(&samples[0], "candidates").is_some());

    // grammar.advance: one mask per step, with the automaton state and the
    // allowed-token count under the mask.
    let masks = capture.named("grammar.advance");
    assert_eq!(masks.len(), steps.len());
    assert!(field(&masks[0], "state").is_some());
    let allowed: usize = field(&masks[0], "allowed").unwrap().parse().unwrap();
    assert!(allowed >= 1);
}

#[test]
fn no_subscriber_means_no_capture_and_no_failure() {
    let _serial = CALLSITE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    // Spans compile to disabled no-ops without a subscriber; a plain run
    // must behave identically.
    let vocab = Vocab::new(vec!["a".into(), "</s>".into()], 1);
    let script = vec![vec![1.0, 0.0]; 4];
    let mut session = Session::new(Box::new(MockModel::new(2, script)), 16).unwrap();
    let mut sampler = Sampler::new(SamplingParams {
        temperature: 0.0,
        top_p: 1.0,
        top_k: None,
        seed: 0,
    });
    let generation = session
        .generate(&[0], &vocab, None, &mut sampler, 2, |_| {})
        .unwrap();
    assert_eq!(generation.tokens.len(), 2);
}

#[test]
fn layer_spans_nest_under_model_forward_once_per_layer() {
    let _serial = CALLSITE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let Some(mut model) = tiny_hf() else { return };
    let capture = Capture::default();
    // The load happened outside the subscriber; only the forwards are traced.
    under(capture.clone(), || two_forwards(&mut model));

    let forwards = count(&capture, "model.forward");
    assert_eq!(forwards, 2);
    let layers = capture.named("model.layer");
    assert_eq!(layers.len(), 4 * forwards, "{layers:#?}");
    for span in &layers {
        assert_eq!(span.parent.as_deref(), Some("model.forward"), "{span:?}");
    }
    let indices: Vec<&str> = layers.iter().filter_map(|s| field(s, "layer")).collect();
    assert_eq!(indices, ["0", "1", "2", "3", "0", "1", "2", "3"]);
    // Layers 2 and 3 are the shared-KV tail of this fixture: they read a
    // donor below them rather than owning a cache.
    for span in &layers[..4] {
        let i: usize = field(span, "layer").unwrap().parse().unwrap();
        let src: usize = field(span, "kv_source").unwrap().parse().unwrap();
        let owns = field(span, "owns_kv").unwrap();
        if i < 2 {
            assert_eq!((src, owns), (i, "true"), "{span:?}");
        } else {
            assert!(src < i, "{span:?}");
            assert_eq!(owns, "false", "{span:?}");
        }
    }
}

#[test]
fn block_spans_appear_once_per_layer_and_ple_only_where_configured() {
    let _serial = CALLSITE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let Some(mut model) = tiny_hf() else { return };
    let capture = Capture::default();
    under(capture.clone(), || two_forwards(&mut model));

    let layers = count(&capture, "model.layer");
    let forwards = count(&capture, "model.forward");
    for name in ["model.layer.attn", "model.layer.mlp", "model.layer.ple"] {
        let spans = capture.named(name);
        assert_eq!(spans.len(), layers, "{name}");
        for span in &spans {
            assert_eq!(span.parent.as_deref(), Some("model.layer"), "{span:?}");
        }
    }
    assert_eq!(count(&capture, "model.prelude"), forwards);
    let heads = capture.named("model.lm_head");
    assert_eq!(heads.len(), forwards);
    // `prefill` asks for the last row only, on both calls.
    assert_eq!(field(&heads[1], "rows"), Some("1"), "{heads:?}");
    assert_eq!(field(&heads[1], "n"), Some("1"), "{heads:?}");

    // The committed fixture has no per-layer embeddings, so no PLE span.
    let plain = Capture::default();
    let mut model = tiny();
    under(plain.clone(), || two_forwards(&mut model));
    assert_eq!(count(&plain, "model.layer"), 2 * 2);
    assert_eq!(count(&plain, "model.layer.ple"), 0);
    assert_eq!(count(&plain, "model.layer.mlp"), 4);
}

#[test]
fn kernel_spans_are_children_of_the_block_that_called_them() {
    let _serial = CALLSITE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let Some(mut model) = tiny_hf() else { return };
    let capture = Capture::default();
    under(capture.clone(), || two_forwards(&mut model));

    let matmuls = capture.named("kernel.matmul");
    let parents = |name: &str| {
        matmuls
            .iter()
            .filter(|s| s.parent.as_deref() == Some(name))
            .count()
    };
    let layers = count(&capture, "model.layer");
    // q and o on every layer, k and v on the two that own a cache.
    assert_eq!(
        parents("model.layer.attn"),
        2 * layers + 2 * (layers / 2),
        "{matmuls:#?}"
    );
    // gate, up, down.
    assert_eq!(parents("model.layer.mlp"), 3 * layers);
    // gate and projection.
    assert_eq!(parents("model.layer.ple"), 2 * layers);
    assert_eq!(count(&capture, "kernel.attention"), layers);
    let attention = capture.named("kernel.attention");
    assert_eq!(attention[0].parent.as_deref(), Some("model.layer.attn"));
    let heads = capture.named("kernel.lm_head");
    assert_eq!(heads.len(), count(&capture, "model.lm_head"));
    assert_eq!(heads[0].parent.as_deref(), Some("model.lm_head"));
}

#[test]
fn a_level_filter_drops_exactly_the_tiers_below_it() {
    let _serial = CALLSITE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let mut model = tiny();
    let full = Capture::default();
    under(full.clone(), || two_forwards(&mut model));
    let layers = count(&full, "model.layer");
    assert!(layers > 0);
    assert!(count(&full, "kernel.matmul") > 0);

    // DEBUG keeps the layers and drops the kernels.
    let mut model = tiny();
    let debug = Capture::default();
    under(debug.clone().with_filter(LevelFilter::DEBUG), || {
        two_forwards(&mut model)
    });
    assert_eq!(count(&debug, "model.layer"), layers);
    assert_eq!(count(&debug, "kernel.matmul"), 0);
    assert_eq!(count(&debug, "kernel.attention"), 0);
    assert_eq!(count(&debug, "kernel.lm_head"), 0);

    // INFO is the schema the first test in this file asserts, and nothing
    // below it.
    let mut model = tiny();
    let info = Capture::default();
    under(info.clone().with_filter(LevelFilter::INFO), || {
        two_forwards(&mut model)
    });
    assert_eq!(count(&info, "model.forward"), 2);
    assert_eq!(count(&info, "model.layer"), 0);
    assert_eq!(count(&info, "model.prelude"), 0);
    assert_eq!(count(&info, "model.lm_head"), 0);
}

#[test]
fn model_load_emits_read_parse_build_and_one_span_per_layer() {
    let _serial = CALLSITE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let capture = Capture::default();
    under(capture.clone(), || {
        CpuModel::load(&fixtures().join("tiny-gemma4"), false).unwrap();
    });
    let loads = capture.named("model.load");
    assert_eq!(loads.len(), 1);
    assert_eq!(field(&loads[0], "text_only"), Some("false"));
    for name in [
        "checkpoint.read",
        "checkpoint.parse",
        "model.from_checkpoint",
        "checkpoint.fingerprint",
    ] {
        let spans = capture.named(name);
        assert_eq!(spans.len(), 1, "{name}");
        assert_eq!(spans[0].parent.as_deref(), Some("model.load"), "{name}");
    }
    let layers = capture.named("model.load.layer");
    assert_eq!(layers.len(), 2, "{layers:#?}");
    for span in &layers {
        assert_eq!(span.parent.as_deref(), Some("model.from_checkpoint"));
        assert_eq!(field(span, "owns_kv"), Some("true"));
    }
}

#[test]
fn tokenizer_encode_records_byte_and_token_counts() {
    let _serial = CALLSITE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let tokenizer = Tokenizer::from_file(&fixtures().join("tiny-gemma4.tokenizer.json")).unwrap();
    let capture = Capture::default();
    let text = "abc cab";
    let ids = under(capture.clone(), || tokenizer.encode(text).unwrap());
    let spans = capture.named("tokenizer.encode");
    assert_eq!(spans.len(), 1, "{spans:#?}");
    assert_eq!(
        field(&spans[0], "bytes"),
        Some(text.len().to_string().as_str())
    );
    assert_eq!(
        field(&spans[0], "tokens"),
        Some(ids.len().to_string().as_str())
    );
    assert!(!ids.is_empty());
}

#[test]
fn spans_do_not_change_a_single_logit() {
    let _serial = CALLSITE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let prompt = [1, 2, 3];
    let mut quiet = tiny();
    let plain = quiet.prefill(&prompt).unwrap().0;

    let mut model = tiny();
    let traced = under(Capture::default(), || model.prefill(&prompt).unwrap().0);

    assert_eq!(plain.len(), traced.len());
    // Exact equality is the claim: a span touches no float, and this is the
    // assertion that keeps it so.
    let first = plain
        .iter()
        .zip(&traced)
        .position(|(a, b)| a.to_bits() != b.to_bits());
    assert_eq!(
        first,
        None,
        "logit {} differs: {} untraced against {} traced",
        first.unwrap(),
        plain[first.unwrap()],
        traced[first.unwrap()]
    );
}
