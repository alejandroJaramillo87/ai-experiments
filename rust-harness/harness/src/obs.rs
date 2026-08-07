/*
 * obs.rs — run-level observability records, contract-compatible with the
 * repository's Python obs stack (common/obs).
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

/// UTC civil date from days since 1970-01-01 (Howard Hinnant's algorithm).
fn civil_from_days(z: i64) -> (i64, u32, u32) {
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

struct Inner {
    // Held open (and flocked LOCK_SH) for the run's duration so the Python
    // retention pruner treats the file as live and skips it
    events: File,
    dir: PathBuf,
    run_id: String,
    command: String,
    started: Instant,
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
        let mut log = RunLog {
            inner: Some(Inner {
                events,
                dir,
                run_id,
                command: command.to_string(),
                started: Instant::now(),
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

    /// Assemble and append one event line. Base keys in the shared order:
    /// ts, level, run_id, cli, command, pid, event, msg, then extras.
    /// Built by hand (not serde maps) so key order survives serialization.
    fn write_event(&mut self, level: &str, event: &str, extras: &[(&str, String)]) {
        let Some(inner) = self.inner.as_mut() else {
            return;
        };
        let mut line = format!(
            "{{\"ts\":{},\"level\":{},\"run_id\":{},\"cli\":{},\"command\":{},\"pid\":{},\"event\":{},\"msg\":\"\"",
            json_str(&format_ts(SystemTime::now())),
            json_str(level),
            json_str(&inner.run_id),
            json_str(CLI_NAME),
            json_str(&inner.command),
            std::process::id(),
            json_str(event),
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
    pub fn end(mut self, exit_code: i32, run_dir: &Path) {
        let Some(inner) = self.inner.as_ref() else {
            return;
        };
        let duration_s = (inner.started.elapsed().as_secs_f64() * 1000.0).round() / 1000.0;
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
            json_str(&run_dir.to_string_lossy()),
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
        log.end(0, Path::new("/work"));

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
        assert_eq!(row["run_dir"], "/work");
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
        log.end(1, Path::new("/work"));

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
        RunLog::begin_at(tmp.path(), "task", &[], None).end(0, Path::new("/a"));
        RunLog::begin_at(tmp.path(), "chat", &[], None).end(1, Path::new("/b"));
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
        log.end(1, Path::new("/work"));
        assert!(std::fs::read_dir(tmp.path()).unwrap().count() == 1);
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

        log.end(0, Path::new("/work"));
        let probe = File::open(&path).unwrap();
        // SAFETY: as for the first probe — a freshly opened descriptor, owned
        // by `probe` and valid for the whole call, and no pointer arguments.
        let done = unsafe { libc::flock(probe.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) };
        assert_eq!(done, 0, "lock must release after end");
    }
}
