/*
 * obs — run-level observability records, contract-compatible with the
 * repository's Python obs stack (common/obs).
 *
 * A workspace member rather than a module of `harness` because the seven
 * binaries in `runtime` need to write these records too, and `runtime`
 * carries `#![forbid(unsafe_code)]`. This crate is where that attribute
 * cannot go: it holds the only `unsafe` in the workspace, two `libc::flock`
 * calls on a descriptor `Inner` owns, whose obligation the `// SAFETY:`
 * comment below reasons about. Confining them here is what lets every other
 * member forbid unsafe outright rather than weaken to `deny`, which is the
 * trade `.styles/RUST_STYLE.md` argues for after the memmap2 removal.
 *
 * The harness writes the same JSONL shapes the Python CLIs write, so the
 * existing stateless readers (runs catalog, retention pruner, dashboard)
 * consume harness runs unchanged: an event file at
 * ${OBS_LOG_DIR}/harness/<run_id>.jsonl and a catalog line appended to
 * ${OBS_LOG_DIR}/harness/runs.jsonl at run end. Every obs failure degrades
 * silently to a no-op, matching the Python behavior in environments without
 * the log root (agent sandbox).
 */

use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use std::os::unix::io::AsRawFd;
use std::path::{Path, PathBuf};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

const CLI_NAME: &str = "harness";
const DEFAULT_LOG_DIR: &str = "/mnt/ai-data/logs";
// Caps mirror common/obs/core.py so fail records stay bounded on disk
const EXC_MSG_CAP: usize = 4096;
/// The metrics file inside a run directory, named as the Python tier names it
/// so `training/commands/report.py --run <dir>` finds it without a flag.
const METRICS_FILE: &str = "metrics.jsonl";

/// Round to three decimals, the precision `common/obs/core.py::phase` uses for
/// `duration_s`. Shared so the two writers agree digit for digit.
///
/// Public because a caller writing its own duration into a metrics record --
/// `ladder` does, from the wall clock `climb` already stamps per rung -- must
/// round it the same way `phase.end` rounds its own, or the same interval
/// reads as two different numbers depending on which record you looked at.
pub fn round3(seconds: f64) -> f64 {
    (seconds * 1000.0).round() / 1000.0
}

/// UTC civil date from days since 1970-01-01 (Howard Hinnant's algorithm).
pub fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    (if m <= 2 { y + 1 } else { y }, m as u32, d as u32)
}

/// ISO 8601 UTC timestamp with millisecond precision and explicit offset.
///
/// The offset is always +00:00 rather than Z: the Python readers pin the
/// pattern `...[+-]HH:MM`, and emitting UTC sidesteps local-offset
/// detection, which pure std cannot do portably.
fn format_ts(t: SystemTime) -> String {
    let dur = t.duration_since(UNIX_EPOCH).unwrap_or_default();
    let secs = dur.as_secs() as i64;
    let millis = dur.subsec_millis();
    let days = secs.div_euclid(86_400);
    let sod = secs.rem_euclid(86_400);
    let (year, month, day) = civil_from_days(days);
    format!(
        "{year:04}-{month:02}-{day:02}T{:02}:{:02}:{:02}.{millis:03}+00:00",
        sod / 3600,
        (sod % 3600) / 60,
        sod % 60
    )
}

/// Random 6-hex-char suffix: /dev/urandom, falling back to a hash of
/// (pid, epoch nanos) if the read fails.
fn random_suffix() -> String {
    let mut buf = [0u8; 3];
    let ok = File::open("/dev/urandom")
        .and_then(|mut f| f.read_exact(&mut buf))
        .is_ok();
    if !ok {
        use std::hash::{Hash, Hasher};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        std::process::id().hash(&mut h);
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .subsec_nanos()
            .hash(&mut h);
        let v = h.finish();
        buf = [(v >> 16) as u8, (v >> 8) as u8, v as u8];
    }
    format!("{:02x}{:02x}{:02x}", buf[0], buf[1], buf[2])
}

/// Run id in the shared format R-YYYYMMDD-HHMMSS-xxxxxx (UTC stamp).
pub fn mint_run_id() -> String {
    let dur = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    let secs = dur.as_secs() as i64;
    let days = secs.div_euclid(86_400);
    let sod = secs.rem_euclid(86_400);
    let (year, month, day) = civil_from_days(days);
    format!(
        "R-{year:04}{month:02}{day:02}-{:02}{:02}{:02}-{}",
        sod / 3600,
        (sod % 3600) / 60,
        sod % 60,
        random_suffix()
    )
}

fn json_str(s: &str) -> String {
    serde_json::to_string(s).unwrap_or_else(|_| "\"\"".to_string())
}

/// The base keys every event line carries, in the order `write_event` emits
/// them. A caller-supplied field may not displace one of these.
///
/// `common/obs/core.py:121-124` enforces the same rule on the Python side, and
/// `test_a_caller_field_cannot_displace_a_base_key` asserts it there. Rust has
/// had no such guard because `write_event` was private and all three callers
/// were in this file -- privacy was doing the work. `event` below is public, so
/// the rule has to be stated rather than assumed.
const BASE_KEYS: [&str; 8] = [
    "ts", "level", "run_id", "cli", "command", "pid", "event", "msg",
];

/// The report envelope's discriminator for a metrics file. `read_metrics`
/// takes the first record carrying this value as the header, and
/// `common/reports.py::validate_report` requires exactly one discriminator.
const METRICS_REPORT: &str = "run_metrics";

/// Keys `metrics_header` writes itself. As with `BASE_KEYS`, a caller field
/// naming one of these is skipped rather than emitted: a duplicate key is
/// legal JSON whose meaning depends on which one the reader keeps, and
/// displacing `report` or `run_id` breaks the envelope or the event-file join
/// rather than just the record.
const METRICS_BASE_KEYS: [&str; 6] = [
    "schema_version",
    "report",
    "timestamp",
    "run_id",
    "cli",
    "command",
];

/// Append one per-step record. `step` is written as a JSON number, not a
/// string: `common/obs/viewer.js:109` drops any record whose `step` is not
/// `typeof "number"`, silently and without a parse error.
fn write_metric(file: &mut File, step: u64, fields: &[(&str, Value)]) {
    let mut line = format!(
        "{{\"step\":{step},\"ts\":{}",
        json_str(&format_ts(SystemTime::now()))
    );
    for (k, v) in fields {
        if *k == "step" || *k == "ts" {
            continue;
        }
        line.push_str(&format!(",{}:{}", json_str(k), v.to_json()));
    }
    line.push_str("}\n");
    // One write_all per whole line, as for the catalog: independent O_APPEND
    // handles (see `RunLog::metrics_sink`) then interleave records rather
    // than tearing one.
    let _ = file.write_all(line.as_bytes());
    let _ = file.flush();
}

/// An open span, closing with `phase.end` when dropped. From `RunLog::phase`.
pub struct Phase<'a> {
    log: &'a mut RunLog,
    name: String,
    started: Instant,
}

impl Drop for Phase<'_> {
    fn drop(&mut self) {
        let duration_s = round3(self.started.elapsed().as_secs_f64());
        let name = std::mem::take(&mut self.name);
        self.log.event(
            "info",
            "phase.end",
            "",
            &[
                ("phase", Value::Str(name)),
                ("duration_s", Value::Num(duration_s)),
            ],
        );
    }
}

/// An independent append handle onto a run's metrics file, from
/// `RunLog::metrics_sink`. Every method is infallible; a disabled sink
/// silently drops records, matching the rest of this crate.
pub struct MetricsSink {
    file: Option<File>,
}

impl MetricsSink {
    /// A sink that drops everything, for callers with no run log at all.
    pub fn disabled() -> Self {
        MetricsSink { file: None }
    }

    /// Append one per-step record.
    pub fn metric(&mut self, step: u64, fields: &[(&str, Value)]) {
        if let Some(file) = self.file.as_mut() {
            write_metric(file, step, fields);
        }
    }
}

/// A field value for `RunLog::event`, serialized by this module.
///
/// Deliberately not a pre-serialized `String`, which is what `write_event`
/// takes internally. A caller passing a bare `"gemma".to_string()` there emits
/// `"model":gemma`, and both `read_events` and `read_runs` *skip* a line they
/// cannot parse -- so the event would vanish with no error anywhere, which is
/// the exact failure `schemas/obs_record.json`'s `notes.silent_skip` says the
/// contract exists to prevent. Escaping lives in one place in this crate.
pub enum Value {
    Str(String),
    Int(u64),
    Num(f64),
    /// A single-precision number, rendered at single precision.
    ///
    /// Not folded into `Num`: widening an f32 to f64 exposes the binary
    /// representation, so `temperature: 0.8` records as
    /// `0.800000011920929`. That is unreadable, and worse than unreadable in
    /// a record whose purpose is cross-run comparison -- two runs at the same
    /// temperature would look as though they might differ. `f32`'s Display
    /// gives the shortest decimal that round-trips to the stored value, which
    /// is the honest rendering of what the sampler actually holds.
    Num32(f32),
    Bool(bool),
}

impl Value {
    /// Render as a JSON fragment.
    ///
    /// A non-finite float has no JSON spelling; `null` is the honest one,
    /// since the alternative (`NaN`) is what makes a reader skip the line.
    fn to_json(&self) -> String {
        match self {
            Value::Str(s) => json_str(s),
            Value::Int(n) => n.to_string(),
            Value::Num(f) if f.is_finite() => {
                let s = format!("{f}");
                match s.contains(['.', 'e', 'E']) {
                    true => s,
                    false => format!("{s}.0"),
                }
            }
            Value::Num(_) => "null".to_string(),
            Value::Num32(f) if f.is_finite() => {
                let s = format!("{f}");
                match s.contains(['.', 'e', 'E']) {
                    true => s,
                    false => format!("{s}.0"),
                }
            }
            Value::Num32(_) => "null".to_string(),
            Value::Bool(b) => b.to_string(),
        }
    }
}

struct Inner {
    // Held open (and flocked LOCK_SH) for the run's duration so the Python
    // retention pruner treats the file as live and skips it
    events: File,
    dir: PathBuf,
    run_id: String,
    command: String,
    started: Instant,
    // ${OBS_LOG_DIR}/<cli>/<run_id>/ -- the run's artifact directory, and the
    // path recorded as `run_dir` in the catalog. Computed here but created
    // only when metrics are first written: an eagerly created directory would
    // leave one empty directory per run behind forever, since the retention
    // pruner globs `R-*.jsonl` and a directory named `<run_id>` never matches
    // it. The path is recorded either way, so the catalog carries one meaning
    // -- "where this run's metrics are" -- rather than two.
    run_dir: PathBuf,
    metrics: Option<File>,
}

/// Run-level record writer. Construct with `begin`, close with `end`.
/// All methods are infallible: any IO error silently drops the record,
/// matching the Python obs degradation contract.
pub struct RunLog {
    inner: Option<Inner>,
}

impl RunLog {
    /// No-op handle (log root unavailable).
    fn disabled() -> Self {
        RunLog { inner: None }
    }

    /// Open the run log under `${OBS_LOG_DIR}/harness/` and write cmd.start.
    /// `argv` is the argument list without the program name. When OBS_RUN_ID
    /// is set by a parent process it is recorded as `parent_run_id`.
    pub fn begin(command: &str, argv: &[String]) -> Self {
        let root = std::env::var_os("OBS_LOG_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from(DEFAULT_LOG_DIR));
        let parent = std::env::var("OBS_RUN_ID").ok();
        Self::begin_at(&root, command, argv, parent.as_deref())
    }

    /// `begin` with the log root and parent run id injected (testable form).
    pub fn begin_at(
        log_root: &Path,
        command: &str,
        argv: &[String],
        parent_run_id: Option<&str>,
    ) -> Self {
        let dir = log_root.join(CLI_NAME);
        if std::fs::create_dir_all(&dir).is_err() {
            return Self::disabled();
        }
        let run_id = mint_run_id();
        let events = match OpenOptions::new()
            .create(true)
            .append(true)
            .open(dir.join(format!("{run_id}.jsonl")))
        {
            Ok(f) => f,
            Err(_) => return Self::disabled(),
        };
        // Shared lock signals liveness to the retention pruner; best-effort
        // SAFETY: `events` owns the descriptor and stays alive past this call,
        // so the fd is open and valid for the whole of it; flock takes an fd
        // and flag bits, no pointers, so there is nothing to alias. The result
        // is discarded on purpose — the lock is advisory and only a hint to
        // the pruner, so a filesystem that cannot honor it (NFS without a
        // lock daemon) must not stop the run from logging.
        unsafe {
            libc::flock(events.as_raw_fd(), libc::LOCK_SH);
        }
        let run_dir = dir.join(&run_id);
        let mut log = RunLog {
            inner: Some(Inner {
                events,
                dir,
                run_id,
                command: command.to_string(),
                started: Instant::now(),
                run_dir,
                metrics: None,
            }),
        };
        let args_json = format!(
            "[{}]",
            argv.iter()
                .map(|a| json_str(a))
                .collect::<Vec<_>>()
                .join(",")
        );
        let mut extras = vec![("argv", args_json)];
        if let Some(p) = parent_run_id {
            extras.push(("parent_run_id", json_str(p)));
        }
        log.write_event("info", "cmd.start", &extras);
        log
    }

    /// Append one event with caller-supplied fields.
    ///
    /// The public door onto the event file. Fields are serialized here rather
    /// than by the caller (see `Value`), and any field whose key is in
    /// `BASE_KEYS` is skipped rather than emitted -- a duplicate key is legal
    /// JSON whose meaning depends on which one the reader keeps, so the event
    /// would still parse and quietly mean something else.
    ///
    /// Skipping rather than erroring matches this module's degradation
    /// contract: every obs failure here is a no-op, never a run failure.
    pub fn event(&mut self, level: &str, event: &str, msg: &str, fields: &[(&str, Value)]) {
        let extras: Vec<(&str, String)> = fields
            .iter()
            .filter(|(k, _)| !BASE_KEYS.contains(k))
            .map(|(k, v)| (*k, v.to_json()))
            .collect();
        self.write_event_with_msg(level, event, msg, &extras);
    }

    /// Assemble and append one event line. Base keys in the shared order:
    /// ts, level, run_id, cli, command, pid, event, msg, then extras.
    /// Built by hand (not serde maps) so key order survives serialization.
    fn write_event(&mut self, level: &str, event: &str, extras: &[(&str, String)]) {
        self.write_event_with_msg(level, event, "", extras);
    }

    /// `write_event` with the `msg` base key filled in.
    ///
    /// `msg` was hardcoded empty while every caller was in this file. Widening
    /// a public emitter's surface later is the awkward direction, so `event`
    /// takes it from the start and the existing callers keep passing "".
    fn write_event_with_msg(
        &mut self,
        level: &str,
        event: &str,
        msg: &str,
        extras: &[(&str, String)],
    ) {
        let Some(inner) = self.inner.as_mut() else {
            return;
        };
        let mut line = format!(
            "{{\"ts\":{},\"level\":{},\"run_id\":{},\"cli\":{},\"command\":{},\"pid\":{},\"event\":{},\"msg\":{}",
            json_str(&format_ts(SystemTime::now())),
            json_str(level),
            json_str(&inner.run_id),
            json_str(CLI_NAME),
            json_str(&inner.command),
            std::process::id(),
            json_str(event),
            json_str(msg),
        );
        for (k, v) in extras {
            line.push_str(&format!(",{}:{v}", json_str(k)));
        }
        line.push_str("}\n");
        // Single write of a whole line so concurrent readers never see a torn
        // record; errors are silently dropped per the degradation contract
        let _ = inner.events.write_all(line.as_bytes());
        let _ = inner.events.flush();
    }

    /// Bracket a coarse span, emitting `phase.start` now and `phase.end` when
    /// the returned guard drops.
    ///
    /// Two-sided with `common/obs/core.py::phase`: same event names, same
    /// `phase` field, same `duration_s` rounded to three decimals, same empty
    /// `msg`. File-only on both sides -- the command's own status lines stay
    /// the human view of progress.
    ///
    /// `phase.end` fires from `Drop`, so a span left by `?` or a panic unwind
    /// still records its duration, matching the Python `finally`.
    ///
    /// The guard borrows the log mutably, so nothing else can write an event
    /// while a phase is open. That is deliberate for events -- phases do not
    /// nest here -- but it is also why per-step metrics go through
    /// `metrics_sink` rather than through `self`.
    pub fn phase(&mut self, name: &str) -> Phase<'_> {
        let name = name.to_string();
        self.event(
            "info",
            "phase.start",
            "",
            &[("phase", Value::Str(name.clone()))],
        );
        Phase {
            log: self,
            name,
            started: Instant::now(),
        }
    }

    /// The run's artifact directory, when the log is active.
    ///
    /// The directory exists on disk only once `metrics_header` has been
    /// called; before that this is the path one would be created at.
    pub fn run_dir(&self) -> Option<&Path> {
        self.inner.as_ref().map(|i| i.run_dir.as_path())
    }

    /// Create the run directory and write the metrics file's header record.
    ///
    /// The header is the shared report envelope (`common/reports.py`), so
    /// `validate_report` accepts the file and `common/obs/dashboard.py`
    /// recognizes it: `report` is the discriminator, and `run_id` plus `cli`
    /// are what `default_events_path` joins to find this run's event file.
    ///
    /// `git_sha` is **not** supplied here and must come from `fields`, or the
    /// envelope fails `validate_report`. This crate cannot call
    /// `runtime::envelope::git_sha` -- `runtime` is what will depend on this
    /// crate, not the reverse -- and re-deriving the shell-out here would put
    /// a second implementation of provenance in the tree.
    ///
    /// The timestamp is `format_ts`, not the envelope's `%z` spelling: one
    /// shape for every timestamp the Rust obs path writes. See
    /// `schemas/run_metrics.json`'s `notes.timestamp_shape`.
    pub fn metrics_header(&mut self, fields: &[(&str, Value)]) {
        let Some(inner) = self.inner.as_mut() else {
            return;
        };
        if std::fs::create_dir_all(&inner.run_dir).is_err() {
            return;
        }
        let file = match OpenOptions::new()
            .create(true)
            .append(true)
            .open(inner.run_dir.join(METRICS_FILE))
        {
            Ok(f) => f,
            Err(_) => return,
        };
        let mut line = format!(
            "{{\"schema_version\":1,\"report\":{},\"timestamp\":{},\"run_id\":{},\"cli\":{},\"command\":{}",
            json_str(METRICS_REPORT),
            json_str(&format_ts(SystemTime::now())),
            json_str(&inner.run_id),
            json_str(CLI_NAME),
            json_str(&inner.command),
        );
        for (k, v) in fields {
            if METRICS_BASE_KEYS.contains(k) {
                continue;
            }
            line.push_str(&format!(",{}:{}", json_str(k), v.to_json()));
        }
        line.push_str("}\n");
        let mut file = file;
        let _ = file.write_all(line.as_bytes());
        let _ = file.flush();
        inner.metrics = Some(file);
    }

    /// Append one per-step metrics record.
    ///
    /// A no-op until `metrics_header` has run: a metrics file whose first
    /// record is not the envelope is one `validate_report` rejects and
    /// `read_metrics` reports with an empty header.
    pub fn metric(&mut self, step: u64, fields: &[(&str, Value)]) {
        let Some(inner) = self.inner.as_mut() else {
            return;
        };
        let Some(file) = inner.metrics.as_mut() else {
            return;
        };
        write_metric(file, step, fields);
    }

    /// An independent append handle onto this run's metrics file.
    ///
    /// Exists because the writers of per-step metrics cannot hold `&mut
    /// RunLog`: the served generator lives behind `Arc<ServerState>`, and
    /// during a `phase` the log is already mutably borrowed by the guard.
    /// Handing out a second O_APPEND descriptor is the same concurrency
    /// contract the catalog already relies on -- one `write_all` per whole
    /// line, so concurrent writers interleave records and never tear one.
    ///
    /// Disabled (a silent no-op) when the log is disabled or
    /// `metrics_header` has not run.
    pub fn metrics_sink(&self) -> MetricsSink {
        let Some(inner) = self.inner.as_ref() else {
            return MetricsSink { file: None };
        };
        if inner.metrics.is_none() {
            return MetricsSink { file: None };
        }
        MetricsSink {
            file: OpenOptions::new()
                .create(true)
                .append(true)
                .open(inner.run_dir.join(METRICS_FILE))
                .ok(),
        }
    }

    /// Record a run failure. Written before cmd.end, like the Python stack.
    pub fn fail(&mut self, err: &anyhow::Error, exit_code: i32) {
        let mut msg = format!("{err:#}");
        if msg.len() > EXC_MSG_CAP {
            let mut cut = EXC_MSG_CAP;
            while !msg.is_char_boundary(cut) {
                cut -= 1;
            }
            msg.truncate(cut);
            msg.push_str("...[truncated]");
        }
        let extras = vec![
            ("exc_type", json_str("anyhow")),
            ("exc_msg", json_str(&msg)),
            ("exit_code", exit_code.to_string()),
        ];
        self.write_event("error", "fail", &extras);
    }

    /// Write cmd.end and append the catalog line, then release the log.
    ///
    /// The catalog's `run_dir` is this run's own directory (`run_dir`), not a
    /// caller-supplied path. It used to be the harness's working directory,
    /// which named the session's files but never the run's metrics -- so
    /// `training report --run` could not open it. A caller that can pass a
    /// different path is a caller that can lie about it, which is why the
    /// parameter is gone rather than defaulted.
    pub fn end(mut self, exit_code: i32) {
        let Some(inner) = self.inner.as_ref() else {
            return;
        };
        let duration_s = round3(inner.started.elapsed().as_secs_f64());
        let extras = vec![
            ("exit_code", exit_code.to_string()),
            ("duration_s", format!("{duration_s}")),
        ];
        self.write_event("info", "cmd.end", &extras);

        let inner = self.inner.take().expect("checked above");
        let line = format!(
            "{{\"run_id\":{},\"command\":{},\"timestamp\":{},\"duration_s\":{duration_s},\"exit_code\":{exit_code},\"run_dir\":{}}}\n",
            json_str(&inner.run_id),
            json_str(&inner.command),
            json_str(&format_ts(SystemTime::now())),
            json_str(&inner.run_dir.to_string_lossy()),
        );
        // One write_all on an O_APPEND fd: concurrent writers interleave
        // whole lines (the contract the Python catalog reader relies on)
        if let Ok(mut f) = OpenOptions::new()
            .create(true)
            .append(true)
            .open(inner.dir.join("runs.jsonl"))
        {
            let _ = f.write_all(line.as_bytes());
        }
        // inner drops here, closing the event fd and releasing the flock
    }

    /// Run id of this log, when active.
    pub fn run_id(&self) -> Option<&str> {
        self.inner.as_ref().map(|i| i.run_id.as_str())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::BufRead;

    fn read_lines(path: &Path) -> Vec<serde_json::Value> {
        let f = File::open(path).unwrap();
        std::io::BufReader::new(f)
            .lines()
            .map(|l| serde_json::from_str(&l.unwrap()).unwrap())
            .collect()
    }

    fn base_key_order(line: &str) -> Vec<String> {
        // serde_json reorders maps, so key order is asserted on the raw text
        let mut keys = Vec::new();
        let mut rest = line;
        while let Some(i) = rest.find('"') {
            let tail = &rest[i + 1..];
            let j = tail.find('"').unwrap();
            let candidate = &tail[..j];
            let after = &tail[j + 1..];
            if let Some(value) = after.strip_prefix(':') {
                keys.push(candidate.to_string());
                // skip past a string value so its content is not taken as a key
                if let Some(v) = value.strip_prefix('"') {
                    let mut k = 0;
                    let bytes = v.as_bytes();
                    while k < bytes.len() {
                        if bytes[k] == b'\\' {
                            k += 2;
                            continue;
                        }
                        if bytes[k] == b'"' {
                            break;
                        }
                        k += 1;
                    }
                    rest = &v[k + 1..];
                    continue;
                }
            }
            rest = after;
        }
        keys
    }

    #[test]
    fn run_id_format() {
        let id = mint_run_id();
        let parts: Vec<&str> = id.split('-').collect();
        assert_eq!(parts.len(), 4);
        assert_eq!(parts[0], "R");
        assert_eq!(parts[1].len(), 8);
        assert!(parts[1].chars().all(|c| c.is_ascii_digit()));
        assert_eq!(parts[2].len(), 6);
        assert!(parts[2].chars().all(|c| c.is_ascii_digit()));
        assert_eq!(parts[3].len(), 6);
        assert!(parts[3]
            .chars()
            .all(|c| c.is_ascii_hexdigit() && !c.is_uppercase()));
        assert_ne!(mint_run_id(), id);
    }

    #[test]
    fn ts_format_matches_pinned_regex() {
        // reader pin: \d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}[+-]\d{2}:\d{2}
        let ts = format_ts(UNIX_EPOCH + std::time::Duration::from_millis(1_753_875_296_123));
        assert_eq!(ts, "2025-07-30T11:34:56.123+00:00");
        assert_eq!(format_ts(UNIX_EPOCH), "1970-01-01T00:00:00.000+00:00");
    }

    #[test]
    fn start_and_end_records() {
        let tmp = tempfile::tempdir().unwrap();
        let argv = vec!["--model".to_string(), "m.gguf".to_string()];
        let log = RunLog::begin_at(tmp.path(), "task", &argv, Some("R-20260101-000000-abcdef"));
        let run_id = log.run_id().unwrap().to_string();
        log.end(0);

        let events_path = tmp.path().join("harness").join(format!("{run_id}.jsonl"));
        let raw = std::fs::read_to_string(&events_path).unwrap();
        let events = read_lines(&events_path);
        assert_eq!(events.len(), 2);

        let start = &events[0];
        assert_eq!(start["event"], "cmd.start");
        assert_eq!(start["level"], "info");
        assert_eq!(start["cli"], "harness");
        assert_eq!(start["command"], "task");
        assert_eq!(start["msg"], "");
        assert_eq!(start["argv"], serde_json::json!(["--model", "m.gguf"]));
        assert_eq!(start["parent_run_id"], "R-20260101-000000-abcdef");
        assert!(start["pid"].is_u64());

        let end = &events[1];
        assert_eq!(end["event"], "cmd.end");
        assert!(end["exit_code"].is_i64());
        assert_eq!(end["exit_code"], 0);
        assert!(end["duration_s"].is_number());

        // Order, which the contract does not constrain. The authoritative
        // key set lives in schemas/obs_record.json and is asserted against
        // both writers by harness/tests/obs_contract.rs and
        // tests/common/test_obs_record_contract.py; this keeps the base keys
        // leading every line so the JSONL stays readable by eye.
        let first_line = raw.lines().next().unwrap();
        let keys = base_key_order(first_line);
        assert_eq!(
            &keys[..8],
            &["ts", "level", "run_id", "cli", "command", "pid", "event", "msg"]
        );

        let catalog = read_lines(&tmp.path().join("harness").join("runs.jsonl"));
        assert_eq!(catalog.len(), 1);
        let row = &catalog[0];
        assert_eq!(row["run_id"], run_id.as_str());
        assert_eq!(row["command"], "task");
        assert_eq!(row["exit_code"], 0);
        // The run's own directory, not the caller's cwd -- what
        // `training report --run` can open. See `end`.
        assert_eq!(
            row["run_dir"],
            tmp.path()
                .join("harness")
                .join(&run_id)
                .to_string_lossy()
                .as_ref()
        );
        // runs table renders timestamp[:19] as YYYY-MM-DDTHH:MM:SS
        let ts = row["timestamp"].as_str().unwrap();
        assert_eq!(ts.as_bytes()[10], b'T');
        assert!(ts.ends_with("+00:00"));
    }

    #[test]
    fn fail_record_and_catalog_on_failure() {
        let tmp = tempfile::tempdir().unwrap();
        let mut log = RunLog::begin_at(tmp.path(), "task", &[], None);
        let run_id = log.run_id().unwrap().to_string();
        let long = "x".repeat(EXC_MSG_CAP + 100);
        log.fail(&anyhow::anyhow!(long), 1);
        log.end(1);

        let events = read_lines(&tmp.path().join("harness").join(format!("{run_id}.jsonl")));
        assert_eq!(events.len(), 3);
        let fail = &events[1];
        assert_eq!(fail["event"], "fail");
        assert_eq!(fail["level"], "error");
        assert_eq!(fail["exc_type"], "anyhow");
        assert_eq!(fail["exit_code"], 1);
        let msg = fail["exc_msg"].as_str().unwrap();
        assert!(msg.ends_with("...[truncated]"));
        assert_eq!(msg.len(), EXC_MSG_CAP + "...[truncated]".len());
        assert!(events[0].get("parent_run_id").is_none());

        let catalog = read_lines(&tmp.path().join("harness").join("runs.jsonl"));
        assert_eq!(catalog[0]["exit_code"], 1);
    }

    #[test]
    fn catalog_appends_without_truncating() {
        let tmp = tempfile::tempdir().unwrap();
        RunLog::begin_at(tmp.path(), "task", &[], None).end(0);
        RunLog::begin_at(tmp.path(), "chat", &[], None).end(1);
        let catalog = read_lines(&tmp.path().join("harness").join("runs.jsonl"));
        assert_eq!(catalog.len(), 2);
        assert_eq!(catalog[0]["command"], "task");
        assert_eq!(catalog[1]["command"], "chat");
    }

    #[test]
    fn degrades_silently_when_root_unusable() {
        let tmp = tempfile::tempdir().unwrap();
        // a file where the log root should be makes create_dir_all fail
        let blocked = tmp.path().join("blocked");
        std::fs::write(&blocked, b"").unwrap();
        let mut log = RunLog::begin_at(&blocked, "task", &[], None);
        assert!(log.run_id().is_none());
        log.fail(&anyhow::anyhow!("boom"), 1);
        log.end(1);
        assert!(std::fs::read_dir(tmp.path()).unwrap().count() == 1);
    }

    #[test]
    fn phase_brackets_the_span_and_ends_on_drop() {
        let tmp = tempfile::tempdir().unwrap();
        let mut log = RunLog::begin_at(tmp.path(), "task", &[], None);
        let run_id = log.run_id().unwrap().to_string();
        {
            let _span = log.phase("model.load");
        }
        log.end(0);

        let events = read_lines(&tmp.path().join("harness").join(format!("{run_id}.jsonl")));
        let start = &events[1];
        assert_eq!(start["event"], "phase.start");
        assert_eq!(start["phase"], "model.load");
        // msg is empty on both sides: common/obs/core.py::phase passes ""
        assert_eq!(start["msg"], "");
        assert_eq!(start["level"], "info");
        assert!(start.get("duration_s").is_none());

        let end = &events[2];
        assert_eq!(end["event"], "phase.end");
        assert_eq!(end["phase"], "model.load");
        assert_eq!(end["msg"], "");
        assert!(end["duration_s"].is_number());
    }

    #[test]
    fn phase_end_fires_on_an_early_return() {
        // The Python contextmanager ends the phase from `finally`, so a span
        // left early still records its duration. Drop is what makes that true
        // here, and a span that only closed on the happy path would report
        // nothing for exactly the runs worth investigating.
        fn bail(log: &mut RunLog) -> Option<()> {
            let _span = log.phase("prefill");
            None?;
            Some(())
        }
        let tmp = tempfile::tempdir().unwrap();
        let mut log = RunLog::begin_at(tmp.path(), "task", &[], None);
        let run_id = log.run_id().unwrap().to_string();
        assert!(bail(&mut log).is_none());
        log.end(0);

        let events = read_lines(&tmp.path().join("harness").join(format!("{run_id}.jsonl")));
        let ends: Vec<_> = events
            .iter()
            .filter(|e| e["event"] == "phase.end")
            .collect();
        assert_eq!(ends.len(), 1);
        assert_eq!(ends[0]["phase"], "prefill");
    }

    #[test]
    fn the_run_directory_is_created_only_by_the_metrics_header() {
        // Eager creation would leave one empty directory per run behind
        // forever: retention.py::prune globs R-*.jsonl, which a directory
        // named <run_id> never matches.
        let tmp = tempfile::tempdir().unwrap();
        let mut log = RunLog::begin_at(tmp.path(), "task", &[], None);
        let dir = log.run_dir().unwrap().to_path_buf();
        assert!(!dir.exists(), "no metrics written yet");

        log.metrics_header(&[("git_sha", Value::Str("abc123".into()))]);
        assert!(dir.is_dir());
        assert!(dir.join("metrics.jsonl").is_file());

        // The catalog records the path either way, so run_dir has one meaning
        log.end(0);
        let catalog = read_lines(&tmp.path().join("harness").join("runs.jsonl"));
        assert_eq!(catalog[0]["run_dir"], dir.to_string_lossy().as_ref());
    }

    #[test]
    fn the_metrics_header_is_a_conforming_report_envelope() {
        let tmp = tempfile::tempdir().unwrap();
        let mut log = RunLog::begin_at(tmp.path(), "task", &[], None);
        let run_id = log.run_id().unwrap().to_string();
        let dir = log.run_dir().unwrap().to_path_buf();
        log.metrics_header(&[
            ("git_sha", Value::Str("abc123".into())),
            ("model", Value::Str("gemma-4-E4B".into())),
        ]);
        let records = read_lines(&dir.join("metrics.jsonl"));
        let header = &records[0];
        // common/reports.py::validate_report's four requirements
        assert_eq!(header["schema_version"], 1);
        assert_eq!(header["report"], "run_metrics");
        assert!(header["timestamp"].as_str().unwrap().ends_with("+00:00"));
        assert_eq!(header["git_sha"], "abc123");
        // dashboard.py::default_events_path joins these two to find the events
        assert_eq!(header["run_id"], run_id.as_str());
        assert_eq!(header["cli"], "harness");
        assert_eq!(header["model"], "gemma-4-E4B");
    }

    #[test]
    fn a_caller_field_cannot_displace_a_metrics_header_key() {
        let tmp = tempfile::tempdir().unwrap();
        let mut log = RunLog::begin_at(tmp.path(), "task", &[], None);
        let dir = log.run_dir().unwrap().to_path_buf();
        log.metrics_header(&[
            ("report", Value::Str("not_run_metrics".into())),
            ("cli", Value::Str("training".into())),
            ("git_sha", Value::Str("abc123".into())),
        ]);
        let raw = std::fs::read_to_string(dir.join("metrics.jsonl")).unwrap();
        // Skipped, not emitted: a duplicate key is legal JSON whose meaning
        // depends on which one the reader keeps
        assert_eq!(raw.matches("\"report\"").count(), 1);
        assert_eq!(raw.matches("\"cli\"").count(), 1);
        let header = &read_lines(&dir.join("metrics.jsonl"))[0];
        assert_eq!(header["report"], "run_metrics");
        assert_eq!(header["cli"], "harness");
    }

    #[test]
    fn a_metric_before_the_header_is_dropped() {
        // A metrics file whose first record is not the envelope is one
        // validate_report rejects and read_metrics reports with no header.
        let tmp = tempfile::tempdir().unwrap();
        let mut log = RunLog::begin_at(tmp.path(), "task", &[], None);
        let dir = log.run_dir().unwrap().to_path_buf();
        log.metric(0, &[("tokens_per_s", Value::Num(1.0))]);
        assert!(!dir.exists());
        assert!(log.metrics_sink().file.is_none());

        log.metrics_header(&[("git_sha", Value::Str("abc123".into()))]);
        log.metric(1, &[("tokens_per_s", Value::Num(2.0))]);
        let records = read_lines(&dir.join("metrics.jsonl"));
        assert_eq!(records.len(), 2);
        assert_eq!(records[1]["step"], 1);
    }

    #[test]
    fn step_is_a_number_and_a_sink_interleaves_whole_lines() {
        let tmp = tempfile::tempdir().unwrap();
        let mut log = RunLog::begin_at(tmp.path(), "task", &[], None);
        let dir = log.run_dir().unwrap().to_path_buf();
        log.metrics_header(&[("git_sha", Value::Str("abc123".into()))]);

        let mut a = log.metrics_sink();
        let mut b = log.metrics_sink();
        for i in 0..4u64 {
            a.metric(i * 2, &[("tokens_per_s", Value::Num(1.5))]);
            b.metric(i * 2 + 1, &[("rss_gb", Value::Num(2.25))]);
        }
        let records = read_lines(&dir.join("metrics.jsonl"));
        assert_eq!(records.len(), 9, "header plus eight whole records");
        for (i, rec) in records[1..].iter().enumerate() {
            // viewer.js:109 drops a record whose step is not typeof "number"
            assert!(rec["step"].is_number(), "step must be a JSON number");
            assert_eq!(rec["step"], i as u64);
            assert!(rec["ts"].as_str().unwrap().ends_with("+00:00"));
        }
    }

    #[test]
    fn a_disabled_sink_drops_everything() {
        let mut sink = MetricsSink::disabled();
        sink.metric(0, &[("tokens_per_s", Value::Num(1.0))]);
    }

    #[test]
    fn event_file_flocked_while_live() {
        let tmp = tempfile::tempdir().unwrap();
        let log = RunLog::begin_at(tmp.path(), "task", &[], None);
        let run_id = log.run_id().unwrap().to_string();
        let path = tmp.path().join("harness").join(format!("{run_id}.jsonl"));

        // the retention pruner's liveness probe: LOCK_EX|LOCK_NB must fail
        let probe = File::open(&path).unwrap();
        // SAFETY: `probe` owns the descriptor and outlives the call, so the fd
        // is valid throughout; flock takes no pointers, so nothing aliases.
        let live = unsafe { libc::flock(probe.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) };
        assert_eq!(live, -1, "live event file must appear locked");

        log.end(0);
        let probe = File::open(&path).unwrap();
        // SAFETY: as for the first probe — a freshly opened descriptor, owned
        // by `probe` and valid for the whole call, and no pointer arguments.
        let done = unsafe { libc::flock(probe.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) };
        assert_eq!(done, 0, "lock must release after end");
    }
}
