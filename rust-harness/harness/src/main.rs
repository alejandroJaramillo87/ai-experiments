/*
 * harness — single-binary local coding agent.
 *
 * Wires the full stack: permissions from permissions.yaml, a durable JSONL
 * transcript, the agent loop, and a SessionGenerator over the runtime. The
 * generator runs the hand-rolled CPU backend against the checkpoint passed
 * via --model; the cuda backend (phase A) will slot in at the same seam.
 */

use std::io::Write;
use std::path::Path;
use std::process::ExitCode;

use anyhow::{Context, Result};

use harness::agent::Agent;
use harness::generator::SessionGenerator;
use harness::permissions::Permissions;
use harness::tools::ToolCall;
use harness::transcript::Transcript;

/// Interactive Ask resolution: prompt on stderr, read one line from stdin.
fn ask_stdin(call: &ToolCall) -> bool {
    eprint!("allow {} {}? [y/N] ", call.name(), call.target());
    let _ = std::io::stderr().flush();
    let mut line = String::new();
    if std::io::stdin().read_line(&mut line).is_err() {
        return false;
    }
    matches!(line.trim(), "y" | "yes")
}

/// A live probe monitor: the loaded probe plus the shared state its tap
/// fills during the run. Drained by `report_probe` after the task.
struct MonitorHandle {
    probe: std::sync::Arc<runtime::model::probe::Probe>,
    state: std::sync::Arc<std::sync::Mutex<runtime::model::probe::MonitorState>>,
}

/// Sentinel an interp artifact carries when the weights it was fitted on
/// have no recorded identity (`interp/probes/capture.py` defaults
/// `model_hash` to it, and the exporters pass it through).
const UNKNOWN_MODEL_HASH: &str = "unknown";

/// FIPS 180-4 SHA-256, hex. Duplicated from
/// `runtime/src/bin/activation-capture.rs`, which writes the capture
/// manifest's `model_hash` with it: items of a binary target cannot be
/// imported, and the two digests must stay the same function or the
/// identity check below compares nothing.
fn sha256_hex(bytes: &[u8]) -> String {
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
        0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
        0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
        0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
        0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
        0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2,
    ];
    let mut h: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
        0x5be0cd19,
    ];
    let mut msg = bytes.to_vec();
    let bit_len = (bytes.len() as u64) * 8;
    msg.push(0x80);
    while msg.len() % 64 != 56 {
        msg.push(0);
    }
    msg.extend_from_slice(&bit_len.to_be_bytes());
    for block in msg.chunks_exact(64) {
        let mut w = [0u32; 64];
        for (i, c) in block.chunks_exact(4).enumerate() {
            w[i] = u32::from_be_bytes(c.try_into().unwrap());
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }
        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut hh] = h;
        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ (!e & g);
            let t1 = hh
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K[i])
                .wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let t2 = s0.wrapping_add(maj);
            hh = g;
            g = f;
            f = e;
            e = d.wrapping_add(t1);
            d = c;
            c = b;
            b = a;
            a = t1.wrapping_add(t2);
        }
        for (s, v) in h.iter_mut().zip([a, b, c, d, e, f, g, hh]) {
            *s = s.wrapping_add(v);
        }
    }
    h.iter().map(|v| format!("{v:08x}")).collect()
}

/// Refuse an interp artifact fitted on weights other than the served ones.
/// A probe scored against a different model, or a control vector added to
/// its residual stream, produces meaningless output indistinguishable from
/// a genuine flag or steer, so the pairing is refused the way every Python
/// consumer refuses it (`interp/probes/registry.py` `require_match`,
/// `interp/steering/wiring.py`). Only two real hashes can disagree: the
/// `unknown` sentinel downgrades to a skipped check, which is what keeps
/// Python-captured artifacts loading. The served digest arrives as a
/// closure because producing it hashes multi-gigabyte weights, a cost a
/// skipped check must not pay.
fn check_model_identity(
    artifact: &str,
    artifact_hash: &str,
    served_sha256: impl FnOnce() -> String,
) -> Result<()> {
    if artifact_hash == UNKNOWN_MODEL_HASH {
        eprintln!(
            "[WARN] Model identity: {artifact} records model hash \
             \"{UNKNOWN_MODEL_HASH}\", check skipped"
        );
        return Ok(());
    }
    let served = served_sha256();
    anyhow::ensure!(
        artifact_hash == served,
        "{artifact} was trained on a different model: {artifact} model_hash \
         {artifact_hash}, served model sha256 {served}"
    );
    Ok(())
}

/// The model hash recorded beside a control vector, when its artifact
/// directory is intact. The vector GGUF itself carries only
/// `controlvector.model_hint` — a model type, not an identity — so the
/// sibling `manifest.json` the exporter writes (`interp/steering/export.py`)
/// is the only hash available. A vector copied away from its manifest, or
/// one whose manifest is unreadable, stays loadable and unchecked.
fn control_vector_model_hash(path: &Path) -> Option<String> {
    let text = std::fs::read_to_string(path.parent()?.join("manifest.json")).ok()?;
    let manifest: serde_json::Value = serde_json::from_str(&text).ok()?;
    manifest.get("model_hash")?.as_str().map(str::to_string)
}

/// Construct the generator over the CPU backend for the checkpoint at
/// `model_path`, wrapping the model in the flight recorder when `record`
/// names a path. There is no mock fallback in the shipping binary.
#[allow(clippy::too_many_arguments)]
fn build_generator(
    model_path: Option<&Path>,
    tokenizer_path: Option<&Path>,
    record: Option<&Path>,
    control_vector: Option<&Path>,
    control_scale: f32,
    probe: Option<&Path>,
    sampling: runtime::sampler::SamplingParams,
    max_tokens: usize,
    context: Option<usize>,
    snapshot: Option<&Path>,
) -> Result<(SessionGenerator, Option<MonitorHandle>, u64)> {
    use runtime::model::{
        cpu::{ControlVector, CpuModel},
        probe::{Probe, ProbeMonitor},
        record::RecordingModel,
        Model,
    };
    let Some(model_path) = model_path else {
        anyhow::bail!(
            "no model: pass --model <checkpoint-dir> to run on the CPU backend; \
             the cuda backend (roadmap phase A) has not landed"
        )
    };
    let files = runtime::checkpoint::CheckpointFiles::read(model_path)
        .with_context(|| format!("read model {}", model_path.display()))?;
    let ck = runtime::checkpoint::parse(&files).context("parse checkpoint")?;
    let mut cpu = CpuModel::from_checkpoint(&ck).context("load CPU model")?;
    // Model identity for the interp artifacts and the snapshot header. The
    // weights alone, not config.json: a checkpoint is a directory, and the
    // weights are what an artifact was fitted on.
    let data = &files.weights;
    if let Some(path) = control_vector {
        let cv_data = std::fs::read(path)
            .with_context(|| format!("read control vector {}", path.display()))?;
        let cv_file = runtime::gguf::parse(&cv_data).context("parse control-vector GGUF")?;
        match control_vector_model_hash(path) {
            Some(hash) => check_model_identity("control vector", &hash, || sha256_hex(data))?,
            None => eprintln!(
                "[WARN] Model identity: no readable manifest.json beside {}, \
                 control vector unchecked",
                path.display()
            ),
        }
        let cv = ControlVector::from_gguf(&cv_file, &cv_data, control_scale)
            .context("load control vector")?;
        cpu.set_control_vector(cv).context("apply control vector")?;
    }
    // The monitor observes the steered stream: set_control_vector first,
    // then the tap, matching the documented tap semantics.
    let monitor = match probe {
        Some(path) => {
            let p_data =
                std::fs::read(path).with_context(|| format!("read probe {}", path.display()))?;
            let p_file = runtime::gguf::parse(&p_data).context("parse probe GGUF")?;
            let p = std::sync::Arc::new(Probe::from_gguf(&p_file, &p_data).context("load probe")?);
            check_model_identity("probe", &p.model_hash, || sha256_hex(data))?;
            anyhow::ensure!(
                p.width() == cpu.config().embed_dim,
                "probe width {} does not match model embedding dimension {}",
                p.width(),
                cpu.config().embed_dim
            );
            anyhow::ensure!(
                p.layer < cpu.config().n_layers,
                "probe layer {} out of range for a {}-layer model",
                p.layer,
                cpu.config().n_layers
            );
            let (tap, state) = ProbeMonitor::new(std::sync::Arc::clone(&p));
            cpu.set_tap(Box::new(tap));
            Some(MonitorHandle { probe: p, state })
        }
        None => None,
    };
    // --context caps the budget below the model's trained window; raising
    // it above the checkpoint's max_position_embeddings is refused because
    // the model was not trained there. The KV estimate warns before a budget the CPU backend cannot
    // realistically hold in memory (it stores f32 K/V for the full
    // sequence, so windowed layers are estimated at their window size and
    // the true cost is an upper bound of context length for every layer).
    let trained = cpu.config().context_length;
    let max_context = match context {
        Some(n) => {
            anyhow::ensure!(
                n <= trained,
                "--context {n} exceeds the model's trained context length {trained}"
            );
            n
        }
        None => trained,
    };
    {
        let c = cpu.config();
        let n_kv_heads = c.n_kv_heads;
        // Geometry per layer, taken from the loaded spec rather than from one
        // model-wide head width. Gemma 4's global layers are twice as wide as
        // its sliding layers, so a single width silently under-reports the
        // budget across every global layer -- enough to keep the warning below
        // from firing on a checkpoint that would exceed it.
        //
        // Shared-KV layers are skipped rather than listed: they store nothing,
        // and counting them over-reports by the whole shared tail -- 18 of
        // E4B's 42 layers, 20 of E2B's 35.
        let layers: Vec<_> = cpu
            .layer_specs()
            .iter()
            .enumerate()
            .filter(|(i, l)| l.kv_source == *i)
            .map(|(_, l)| {
                (
                    l.kind,
                    runtime::kv::KvGeometry {
                        n_kv_heads,
                        head_dim: l.head_dim,
                        bytes_per_element: 4, // CPU backend caches f32
                    },
                )
            })
            .collect();
        let bytes = runtime::kv::budget_bytes(&layers, max_context);
        let gib = bytes as f64 / (1024.0 * 1024.0 * 1024.0);
        eprintln!("[INFO] KV budget: {max_context} tokens, estimated {gib:.2} GiB f32");
        if bytes > 2 << 30 {
            eprintln!(
                "[WARN] KV estimate above 2 GiB: consider --context to cap the budget \
                 (compaction keeps long sessions inside it)"
            );
        }
    }
    let model: Box<dyn Model> = match record {
        Some(path) => Box::new(RecordingModel::to_file(Box::new(cpu), path)?),
        None => Box::new(cpu),
    };
    let mut session = runtime::session::Session::new(model, max_context)?;
    // Snapshot restore: resume the KV cache from the previous clean exit.
    // A header or weight-hash mismatch is loud and the stale file is
    // removed; a missing file is the normal first run. Whether the restored
    // tokens still match the transcript is settled later by sync's prefix
    // matching, which degrades to a full re-prefill at worst.
    let model_hash = runtime::session::fingerprint(data);
    if let Some(path) = snapshot {
        match std::fs::File::open(path) {
            Ok(f) => {
                let mut reader = std::io::BufReader::new(f);
                match session.restore(&mut reader, model_hash) {
                    Ok(()) => eprintln!(
                        "[INFO] Session restored: {} cached tokens from {}",
                        session.cached_len(),
                        path.display()
                    ),
                    Err(e) => {
                        let _ = std::fs::remove_file(path);
                        anyhow::bail!(
                            "restore session snapshot {}: {e:#} (stale file removed; rerun)",
                            path.display()
                        );
                    }
                }
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => return Err(e).with_context(|| format!("open snapshot {}", path.display())),
        }
    }
    let tokenizer_path = runtime::tokenizer::resolve_tokenizer(model_path, tokenizer_path)?;
    let tokenizer =
        runtime::tokenizer::Tokenizer::from_file(&tokenizer_path).context("load tokenizer")?;
    let generator = SessionGenerator::new(session, tokenizer, sampling, max_tokens)?;
    Ok((generator, monitor, model_hash))
}

/// Drain the monitor state: write one JSONL record per flagged position to
/// `out`, print a one-line summary to stderr. A missing parent directory is
/// created rather than treated as a failure, since the flags exist only in
/// memory and are lost the moment this returns. Observation only, so this
/// runs after the task and never affects its result: the caller folds a
/// failure here into a warning through `fold_probe_report`.
fn report_probe(monitor: &MonitorHandle, out: &Path) -> Result<()> {
    let state = monitor.state.lock().expect("monitor state lock");
    if let Some(parent) = out.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("create probe output directory {}", parent.display()))?;
    }
    let mut file = std::fs::File::create(out)
        .with_context(|| format!("create probe output {}", out.display()))?;
    for flag in &state.flags {
        let record = serde_json::json!({
            "probe": monitor.probe.name,
            "layer": monitor.probe.layer,
            "position": flag.position,
            "score": flag.score,
            "threshold": monitor.probe.threshold,
        });
        writeln!(file, "{record}").context("write probe flag record")?;
    }
    eprintln!(
        "[INFO] Probe {}: {} positions scored, {} flagged (threshold {}), flags at {}",
        monitor.probe.name,
        state.scored,
        state.flags.len(),
        monitor.probe.threshold,
        out.display()
    );
    Ok(())
}

/// Fold a probe report's outcome into the task's. Observation is not part
/// of the task, so a report failure leaves the task result alone and
/// surfaces as a warning line; propagating it instead turned a completed,
/// transcript-recorded task into exit 1 with the answer never printed.
/// Returning the line rather than printing it keeps the policy testable
/// without a live monitor.
fn fold_probe_report(
    task: Result<String>,
    reported: Result<()>,
) -> (Result<String>, Option<String>) {
    let warning = reported
        .err()
        .map(|e| format!("[WARN] Probe flags not written: {e:#}"));
    (task, warning)
}

#[allow(clippy::too_many_arguments)]
fn run(
    root: &Path,
    task: &str,
    model: Option<&Path>,
    tokenizer: Option<&Path>,
    record: Option<&Path>,
    control_vector: Option<&Path>,
    control_scale: f32,
    probe: Option<&Path>,
    probe_out: Option<&Path>,
    sampling: runtime::sampler::SamplingParams,
    max_tokens: usize,
    context: Option<usize>,
    compact_at: f32,
    chat: bool,
) -> Result<()> {
    let transcript = Transcript::open(&root.join("session.jsonl")).context("session.jsonl")?;
    // HARNESS.md: persistent memory read once at startup and injected as a
    // synthetic leading turn. A missing file is fine; mid-session edits take
    // effect on the next start (the prompt prefix must stay stable so KV
    // delta prefill keeps working).
    let memory = std::fs::read_to_string(root.join("HARNESS.md")).ok();
    let snapshot_path = root.join("session.kv");
    let (mut generator, monitor, model_hash) = build_generator(
        model,
        tokenizer,
        record,
        control_vector,
        control_scale,
        probe,
        sampling,
        max_tokens,
        context,
        Some(&snapshot_path),
    )?;
    // The task's outcome is held rather than propagated: the bookkeeping
    // below runs either way, so flags collected before a failure survive it.
    let result = if chat {
        // Chat mode: REPL over the generator, no tools or permissions. The
        // probe monitor and control vector are already installed on the
        // model, so interp flags apply to chat turns unchanged.
        let mut transcript = transcript;
        let stdin = std::io::stdin();
        harness::chat::run_chat(
            &mut generator,
            &mut transcript,
            memory.as_deref(),
            compact_at,
            stdin.lock(),
            std::io::stdout(),
        )
        .map(|turns| {
            // Replies were already streamed to stdout; the summary is status.
            eprintln!("[INFO] Chat ended: {turns} model turns");
            String::new()
        })
    } else {
        let permissions = match std::fs::read_to_string(root.join("permissions.yaml")) {
            Ok(yaml) => Permissions::from_yaml(&yaml).context("permissions.yaml")?,
            // No config: every call goes through the interactive Ask path.
            Err(_) => Permissions::from_yaml("rules: []").expect("empty rule set"),
        };
        let mut agent = Agent::new(
            root.to_path_buf(),
            permissions,
            transcript,
            memory.as_deref(),
        );
        agent.compact_at = compact_at;
        agent.run_task(&mut generator, task, 64, ask_stdin)
    };
    if let Ok(answer) = &result {
        // The answer is the task's product: print it before the persistence
        // and observation bookkeeping, so neither can keep a completed
        // task's answer off stdout.
        if !answer.is_empty() {
            println!("{answer}");
        }
        // Clean-exit snapshot: save the KV cache beside the transcript so the
        // next start resumes without a full re-prefill. A save failure degrades
        // to a warning — a completed task must not fail over persistence.
        let saved = std::fs::File::create(&snapshot_path)
            .map_err(anyhow::Error::from)
            .and_then(|f| {
                let mut w = std::io::BufWriter::new(f);
                generator.session().save(&mut w, model_hash)?;
                std::io::Write::flush(&mut w)?;
                Ok(())
            });
        match saved {
            Ok(()) => eprintln!(
                "[INFO] Session snapshot: {} cached tokens at {}",
                generator.session().cached_len(),
                snapshot_path.display()
            ),
            Err(e) => eprintln!("[WARN] Session snapshot not saved: {e:#}"),
        }
    }
    let result = match &monitor {
        Some(monitor) => {
            let default_out = root.join("probe-flags.jsonl");
            let reported = report_probe(monitor, probe_out.unwrap_or(&default_out));
            let (result, warning) = fold_probe_report(result, reported);
            if let Some(warning) = warning {
                eprintln!("{warning}");
            }
            result
        }
        None => result,
    };
    result?;
    Ok(())
}

/// Refuse a flag value that is itself a flag, naming both. The extractor
/// takes the word after a flag as its value, so an omitted value swallows
/// the flag that follows: the swallowed flag never applies, its own value
/// falls into the task text, and the run exits 0 having done something the
/// user never asked for. Only the `--` spelling this parser recognizes is
/// refused; a negative number or a path starting with `-` is a value.
fn flag_value_error(flag: &str, value: &str) -> Option<String> {
    value
        .starts_with("--")
        .then(|| format!("[ERROR] {flag}: expected a value, found the flag {value}"))
}

fn main() -> ExitCode {
    const USAGE: &str = "usage: harness --model <checkpoint-dir> [--tokenizer <path.json>] \
         [--record <path>] \
         [--trace-chrome <path>] [--control-vector <path.gguf>] \
         [--control-vector-scale <f>] [--probe <path.gguf>] \
         [--probe-out <path>] [--temperature <f>] [--top-p <f>] \
         [--seed <n>] [--max-tokens <n>] [--context <n>] \
         [--compact-at <f>] (--chat | <task description>)";
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    // Snapshot before flag extraction mutates args: cmd.start records the
    // invocation as given, program name excluded (the shape common/obs logs).
    let argv_full = args.clone();
    // --model <checkpoint-dir>: config.json and model.safetensors for the
    // CPU backend. A path to the weights file resolves to its directory.
    // --record <path>: wrap the model in the flight recorder, appending every
    // forward call to a JSONL file replayable as a MockModel script.
    // --trace-chrome <path>: write a Perfetto-loadable trace of the span
    // timeline. No subscriber is installed otherwise, so spans cost nothing.
    // --control-vector <path.gguf>: steering directions in the llama.cpp
    // control-vector layout, added to the residual stream per layer.
    // --control-vector-scale <f>: multiplier on the directions (default 1.0).
    // --probe <path.gguf>: linear probe (interp/probes/export.py layout)
    // scored in-band at every position of its layer through the tap seam.
    // --probe-out <path>: JSONL file for flagged positions (default
    // probe-flags.jsonl in the working directory).
    // --temperature <f> / --top-p <f> / --seed <n>: sampling parameters
    // (defaults 0.8 / 0.95 / 0; seed 0 replays a run exactly).
    // --max-tokens <n>: per-turn generation budget (default 256).
    // --context <n>: cap the context budget below the model's trained
    // window (never above it; max_position_embeddings is the default and
    // the maximum).
    // --compact-at <f>: automatic-compaction threshold as a fraction of the
    // context budget (default 0.8, must be in (0, 1]).
    let mut model: Option<std::path::PathBuf> = None;
    let mut tokenizer: Option<std::path::PathBuf> = None;
    let mut record: Option<std::path::PathBuf> = None;
    let mut trace_chrome: Option<std::path::PathBuf> = None;
    let mut control_vector: Option<std::path::PathBuf> = None;
    let mut control_scale_arg: Option<std::path::PathBuf> = None;
    let mut probe: Option<std::path::PathBuf> = None;
    let mut probe_out: Option<std::path::PathBuf> = None;
    let mut temperature_arg: Option<std::path::PathBuf> = None;
    let mut top_p_arg: Option<std::path::PathBuf> = None;
    let mut seed_arg: Option<std::path::PathBuf> = None;
    let mut max_tokens_arg: Option<std::path::PathBuf> = None;
    let mut context_arg: Option<std::path::PathBuf> = None;
    let mut compact_at_arg: Option<std::path::PathBuf> = None;
    for (flag, slot) in [
        ("--model", &mut model),
        ("--record", &mut record),
        ("--trace-chrome", &mut trace_chrome),
        ("--control-vector", &mut control_vector),
        ("--control-vector-scale", &mut control_scale_arg),
        ("--probe", &mut probe),
        ("--tokenizer", &mut tokenizer),
        ("--probe-out", &mut probe_out),
        ("--temperature", &mut temperature_arg),
        ("--top-p", &mut top_p_arg),
        ("--seed", &mut seed_arg),
        ("--max-tokens", &mut max_tokens_arg),
        ("--context", &mut context_arg),
        ("--compact-at", &mut compact_at_arg),
    ] {
        if let Some(i) = args.iter().position(|a| a == flag) {
            if i + 1 >= args.len() {
                eprintln!("{USAGE}");
                return ExitCode::from(2);
            }
            if let Some(error) = flag_value_error(flag, &args[i + 1]) {
                eprintln!("{error}");
                return ExitCode::from(2);
            }
            *slot = Some(std::path::PathBuf::from(args.remove(i + 1)));
            args.remove(i);
        }
    }
    let control_scale = match control_scale_arg {
        Some(v) => match v.to_string_lossy().parse::<f32>() {
            Ok(s) if s.is_finite() => s,
            _ => {
                eprintln!("[ERROR] --control-vector-scale: not a finite number");
                return ExitCode::from(2);
            }
        },
        None => 1.0,
    };
    // Sampler::new treats out-of-range values as config bugs and panics, so
    // user input is range-checked here and rejected as a usage error instead.
    let mut sampling = runtime::sampler::SamplingParams::default();
    if let Some(v) = temperature_arg {
        match v.to_string_lossy().parse::<f32>() {
            Ok(t) if t.is_finite() && t >= 0.0 => sampling.temperature = t,
            _ => {
                eprintln!("[ERROR] --temperature: expected a finite number >= 0");
                return ExitCode::from(2);
            }
        }
    }
    if let Some(v) = top_p_arg {
        match v.to_string_lossy().parse::<f32>() {
            Ok(p) if p > 0.0 && p <= 1.0 => sampling.top_p = p,
            _ => {
                eprintln!("[ERROR] --top-p: expected a number in (0, 1]");
                return ExitCode::from(2);
            }
        }
    }
    if let Some(v) = seed_arg {
        match v.to_string_lossy().parse::<u64>() {
            Ok(s) => sampling.seed = s,
            _ => {
                eprintln!("[ERROR] --seed: expected an unsigned integer");
                return ExitCode::from(2);
            }
        }
    }
    let max_tokens = match max_tokens_arg {
        Some(v) => match v.to_string_lossy().parse::<usize>() {
            Ok(n) if n >= 1 => n,
            _ => {
                eprintln!("[ERROR] --max-tokens: expected an integer >= 1");
                return ExitCode::from(2);
            }
        },
        None => 256,
    };
    let context = match context_arg {
        Some(v) => match v.to_string_lossy().parse::<usize>() {
            Ok(n) if n >= 1 => Some(n),
            _ => {
                eprintln!("[ERROR] --context: expected an integer >= 1");
                return ExitCode::from(2);
            }
        },
        None => None,
    };
    let compact_at = match compact_at_arg {
        Some(v) => match v.to_string_lossy().parse::<f32>() {
            Ok(f) if f > 0.0 && f <= 1.0 => f,
            _ => {
                eprintln!("[ERROR] --compact-at: expected a number in (0, 1]");
                return ExitCode::from(2);
            }
        },
        None => 0.8,
    };
    // --chat: interactive REPL instead of a one-shot task; the two are
    // mutually exclusive, so leftover positional args are a usage error.
    let chat = if let Some(i) = args.iter().position(|a| a == "--chat") {
        args.remove(i);
        true
    } else {
        false
    };
    if args.is_empty() != chat {
        eprintln!("{USAGE}");
        return ExitCode::from(2);
    }
    // The guard flushes the trace file on drop at the end of main.
    let _trace_guard = match trace_chrome {
        Some(path) => {
            use tracing_subscriber::layer::SubscriberExt;
            let (layer, guard) = tracing_chrome::ChromeLayerBuilder::new()
                .file(path)
                .include_args(true)
                .build();
            tracing::subscriber::set_global_default(
                tracing_subscriber::registry::Registry::default().with(layer),
            )
            .expect("first and only subscriber installation");
            Some(guard)
        }
        None => None,
    };
    let task = args.join(" ");
    let root = match std::env::current_dir() {
        Ok(d) => d,
        Err(e) => {
            eprintln!("[ERROR] Working directory: {e}");
            return ExitCode::from(1);
        }
    };
    // Run-level obs records in the shared common/obs contract. Usage errors
    // above return before a run id exists, matching the Python dispatch;
    // without a writable OBS_LOG_DIR the log degrades to a silent no-op.
    let mut run_log = harness::obs::RunLog::begin(if chat { "chat" } else { "task" }, &argv_full);
    match run(
        &root,
        &task,
        model.as_deref(),
        tokenizer.as_deref(),
        record.as_deref(),
        control_vector.as_deref(),
        control_scale,
        probe.as_deref(),
        probe_out.as_deref(),
        sampling,
        max_tokens,
        context,
        compact_at,
        chat,
    ) {
        // The answer, if any, was printed by `run` before its bookkeeping.
        Ok(()) => {
            run_log.end(0, &root);
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("[ERROR] Task failed: {e:#}");
            run_log.fail(&e, 1);
            run_log.end(1, &root);
            ExitCode::from(1)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sha256_matches_the_fips_test_vectors() {
        // The same vectors the capture writer's copy is pinned to, so the
        // two implementations cannot drift into different digests.
        assert_eq!(
            sha256_hex(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn model_identity_accepts_an_artifact_hash_equal_to_the_served_digest() {
        let served = sha256_hex(b"gguf bytes");
        check_model_identity("probe", &served, || sha256_hex(b"gguf bytes"))
            .expect("an artifact fitted on the served weights is the matching case");
    }

    #[test]
    fn model_identity_refuses_a_mismatch_and_names_both_hashes() {
        let artifact = sha256_hex(b"weights the probe was trained on");
        let served = sha256_hex(b"weights actually loaded");
        let err = check_model_identity("probe", &artifact, || served.clone())
            .expect_err("two real hashes that disagree are a refusal")
            .to_string();
        assert!(err.contains(&artifact), "{err}");
        assert!(err.contains(&served), "{err}");
    }

    #[test]
    fn model_identity_skips_the_unknown_sentinel_without_hashing_the_model() {
        let mut hashed = false;
        check_model_identity("probe", UNKNOWN_MODEL_HASH, || {
            hashed = true;
            String::new()
        })
        .expect("unknown is the capture default, not a mismatch");
        assert!(!hashed, "a skipped check must not hash the model");
    }

    #[test]
    fn a_probe_report_failure_warns_and_leaves_a_completed_task_completed() {
        let (result, warning) = fold_probe_report(
            Ok("the answer".to_string()),
            Err(anyhow::anyhow!("create probe output /absent/flags.jsonl")),
        );
        assert_eq!(
            result.expect("a write failure must not fail the task"),
            "the answer"
        );
        let warning = warning.expect("the discarded flags must still be reported");
        assert!(warning.starts_with("[WARN] "), "{warning}");
        assert!(warning.contains("create probe output"), "{warning}");
    }

    #[test]
    fn a_probe_report_never_rewrites_the_task_result_it_folds_into() {
        let (ok, warning) = fold_probe_report(Ok("kept".to_string()), Ok(()));
        assert_eq!(ok.expect("a clean report changes nothing"), "kept");
        assert_eq!(warning, None);
        // A failed task stays failed with its own error, not the report's.
        let (err, _) = fold_probe_report(
            Err(anyhow::anyhow!("task error")),
            Err(anyhow::anyhow!("report error")),
        );
        assert_eq!(
            err.expect_err("a failed task stays failed").to_string(),
            "task error"
        );
    }

    /// Minimal probe GGUF in the `interp/probes/export.py` layout, enough
    /// for `Probe::from_gguf`; the scoring math is covered by the runtime's
    /// reference test, so the values here are arbitrary.
    fn probe_gguf() -> Vec<u8> {
        let mut w = runtime::synth::GgufWriter::new();
        w.kv_str("general.architecture", "probe");
        w.kv_str("probe.name", "report-probe");
        w.kv_str("probe.model_hash", UNKNOWN_MODEL_HASH);
        w.kv_u32("probe.layer", 0);
        w.kv_f32("probe.bias", 0.0);
        w.kv_f32("probe.threshold", 0.5);
        w.tensor_f32("probe.weights", &[2], &[1.0, 1.0]);
        w.tensor_f32("probe.mu", &[2], &[0.0, 0.0]);
        w.tensor_f32("probe.sigma", &[2], &[1.0, 1.0]);
        w.finish()
    }

    #[test]
    fn probe_flags_are_written_under_a_parent_directory_that_does_not_exist_yet() {
        let image = probe_gguf();
        let file = runtime::gguf::parse(&image).expect("parse probe GGUF");
        let probe = std::sync::Arc::new(
            runtime::model::probe::Probe::from_gguf(&file, &image).expect("load probe"),
        );
        let (_tap, state) = runtime::model::probe::ProbeMonitor::new(std::sync::Arc::clone(&probe));
        let monitor = MonitorHandle { probe, state };
        let dir = tempfile::tempdir().expect("tempdir");
        let out = dir.path().join("absent").join("flags.jsonl");
        report_probe(&monitor, &out).expect("a missing parent directory is created, not a failure");
        assert!(out.is_file(), "{}", out.display());
    }

    #[test]
    fn a_flag_value_that_is_itself_a_flag_is_refused_naming_the_flag_and_the_value() {
        // `--probe-out --max-tokens 4 hello` without a path: the old parser
        // stored "--max-tokens" as the probe output, dropped the token
        // budget, and pushed "4" into the task text, all at exit 0.
        let error = flag_value_error("--probe-out", "--max-tokens")
            .expect("an omitted value must not swallow the following flag");
        assert!(error.starts_with("[ERROR] "), "{error}");
        assert!(error.contains("--probe-out"), "{error}");
        assert!(error.contains("--max-tokens"), "{error}");
    }

    #[test]
    fn a_flag_value_that_only_looks_flag_adjacent_is_accepted() {
        // Refusing more than the `--` spelling would reject legitimate
        // values: a negative scale, or a path whose name starts with a dash.
        assert_eq!(flag_value_error("--probe-out", "flags.jsonl"), None);
        assert_eq!(flag_value_error("--control-vector-scale", "-1.5"), None);
        assert_eq!(flag_value_error("--model", "./-weights.gguf"), None);
    }

    #[test]
    fn control_vector_model_hash_reads_the_manifest_beside_the_vector() {
        let dir = tempfile::tempdir().expect("tempdir");
        let vector = dir.path().join("vector.gguf");
        assert_eq!(control_vector_model_hash(&vector), None);
        std::fs::write(
            dir.path().join("manifest.json"),
            r#"{"artifact": "control_vector", "model_hash": "deadbeef"}"#,
        )
        .expect("write manifest");
        assert_eq!(
            control_vector_model_hash(&vector),
            Some("deadbeef".to_string())
        );
    }
}
