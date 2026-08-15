/*
 * interplog.rs — the served-request interp log.
 *
 * One JSONL record per completed `/v1/chat/completions`, appended to
 * `<root>/<service>/YYYY-MM-DD.jsonl`. This is the artifact interp's probe
 * monitor reads.
 *
 * It exists because a long-lived server has no end of run. The one-shot paths
 * drain `MonitorState` once, after the task, and write `probe-flags.jsonl`
 * (main.rs `report_probe`); under `--serve` that drain is unreachable, so the
 * flags accumulated in memory for the life of the process and were never
 * written. Flushing per request instead fixes both halves of that: the records
 * reach disk, and the buffer stops growing without bound.
 *
 * The record carries the probe scores taken in the process that answered the
 * request. That is the difference from the retired request-logging proxy,
 * which could only record the request and the reply and left the scoring to an
 * offline replay through a second copy of the model. Scores here come from the
 * weights that produced the text, so there is no replay and no precision
 * caveat to state.
 *
 * Degradation follows obs.rs: every method is infallible, an IO error drops
 * the record, and observation never fails a served request.
 */

use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::Serialize;

use runtime::model::probe::{MonitorState, Probe};

use obs::civil_from_days;

/// A live probe monitor: the loaded probe plus the shared state its tap fills.
///
/// One-shot runs drain it once, after the task (`report_probe` in main.rs);
/// a served run drains it after every request, through `drain`.
pub struct MonitorHandle {
    pub probe: std::sync::Arc<Probe>,
    pub state: std::sync::Arc<std::sync::Mutex<MonitorState>>,
}

impl MonitorHandle {
    /// Take everything the tap recorded since the last drain and describe it
    /// as one request's worth of observation.
    ///
    /// Draining is what bounds the buffer. `MonitorState.flags` and `.scores`
    /// only grow, and under `--serve` nothing used to empty them, so a
    /// long-lived server accumulated every scored position for the life of the
    /// process. The caller drains while holding the generation lock, which is
    /// also what makes the result attributable: generation is serialized, so
    /// everything in the buffer belongs to the request that just finished.
    pub fn drain(&self) -> ProbeRecord {
        let mut state = self.state.lock().expect("monitor state lock");
        let flags = std::mem::take(&mut state.flags);
        let scores = std::mem::take(&mut state.scores);
        let scored = std::mem::replace(&mut state.scored, 0);
        // Highest score over the request, which is the per-request statistic
        // the monitor builds its distribution from -- the same granularity
        // the retired replay produced, one number per logged request.
        //
        // `scores` is populated only when the monitor keeps clean scores,
        // which serve mode asks for. Without it the maximum would be the
        // maximum over flagged positions alone, which is a maximum over
        // "already above threshold" and describes nothing.
        let max_score = scores
            .iter()
            .map(|f| f.score)
            .fold(None::<f32>, |acc, s| Some(acc.map_or(s, |a| a.max(s))));
        // `scores` when the monitor keeps them, `flags` otherwise -- either
        // is a set of positions the tap reported during this request.
        let position_first = scores.iter().chain(flags.iter()).map(|f| f.position).min();
        ProbeRecord {
            name: self.probe.name.clone(),
            layer: self.probe.layer,
            site: self.probe.site.to_string(),
            threshold: self.probe.threshold,
            scored,
            position_first,
            max_score,
            flags: flags
                .into_iter()
                .map(|f| FlagRecord {
                    position: f.position,
                    score: f.score,
                })
                .collect(),
        }
    }
}

/// One over-threshold position, at its absolute cache position.
///
/// Absolute, not request-relative: the tap reports the position in the shared
/// session cache. Attribution comes from the drain being per request, not
/// from the number itself.
#[derive(Serialize)]
pub struct FlagRecord {
    pub position: usize,
    pub score: f32,
}

/// What the probe did during one request.
#[derive(Serialize)]
pub struct ProbeRecord {
    pub name: String,
    pub layer: usize,
    pub site: String,
    pub threshold: f32,
    /// Positions of the probe's layer scored during this request.
    pub scored: usize,
    /// Lowest position the tap reported during this request, or `None` if it
    /// scored nothing.
    ///
    /// Derived from what the tap reported, never from the session's cached
    /// length before generating. Those are not the same number: a served
    /// request whose prompt diverges from the cached one re-prefills from
    /// where it diverged, which can be position 0, while the cached length
    /// still reflects the previous request. Taking the cache length as the
    /// base labelled a request that scored positions 0..56 as based at 56.
    pub position_first: Option<usize>,
    /// Highest score over the request's positions, which is the per-request
    /// statistic the monitor's distribution is built from. `None` when the
    /// request scored no positions at the probe's layer.
    pub max_score: Option<f32>,
    pub flags: Vec<FlagRecord>,
}

/// One completed request. The field names through `error` are the ones
/// interp/probes/monitor.py has always read, so the monitor's window
/// reconstruction, retention and skip accounting are unchanged by the move
/// off the proxy; `probe` is the new field, and the reason the replay stage
/// is no longer needed.
#[derive(Serialize)]
pub struct RequestRecord {
    pub timestamp: String,
    pub target_service: String,
    pub model: String,
    pub messages: serde_json::Value,
    pub response_text: Option<String>,
    pub usage: Option<serde_json::Value>,
    pub latency_s: f64,
    pub status_code: u16,
    pub error: Option<String>,
    pub probe: Option<ProbeRecord>,
}

struct Inner {
    dir: PathBuf,
    /// Day the open file covers, so a server running past midnight rotates
    /// rather than appending the next day's traffic to yesterday's file. The
    /// monitor's retention pass keys off the filename date.
    day: String,
    file: File,
}

/// Append-only writer for the served-request log.
pub struct InterpLog {
    inner: Option<Inner>,
}

impl InterpLog {
    /// Disabled handle: every append is a no-op. This is what `--serve`
    /// without `--interp-log` gets, and what an unwritable directory
    /// degrades to.
    pub fn disabled() -> InterpLog {
        InterpLog { inner: None }
    }

    /// Open `<root>/<service>/` for appending. A directory that cannot be
    /// created disables the log rather than failing the server: the endpoint's
    /// job is to answer requests, and observation is not part of that
    /// contract.
    pub fn open(root: &Path, service: &str) -> InterpLog {
        let dir = root.join(service);
        if std::fs::create_dir_all(&dir).is_err() {
            return InterpLog::disabled();
        }
        let day = utc_day(SystemTime::now());
        match open_day(&dir, &day) {
            Some(file) => InterpLog {
                inner: Some(Inner { dir, day, file }),
            },
            None => InterpLog::disabled(),
        }
    }

    pub fn is_enabled(&self) -> bool {
        self.inner.is_some()
    }

    /// Append one record, rotating first if the day changed.
    pub fn append(&mut self, record: &RequestRecord) {
        let Some(inner) = self.inner.as_mut() else {
            return;
        };
        let day = utc_day(SystemTime::now());
        if day != inner.day {
            // A rotation that fails keeps writing to yesterday's file rather
            // than dropping the record: a misfiled record is recoverable, a
            // lost one is not.
            if let Some(file) = open_day(&inner.dir, &day) {
                inner.file = file;
                inner.day = day;
            }
        }
        let Ok(mut line) = serde_json::to_string(record) else {
            return;
        };
        line.push('\n');
        // One write of a whole line, so a reader tailing the file never sees
        // a torn record; flushed per record because the process is killed by
        // a signal and never gets to close the file.
        let _ = inner.file.write_all(line.as_bytes());
        let _ = inner.file.flush();
    }
}

fn open_day(dir: &Path, day: &str) -> Option<File> {
    OpenOptions::new()
        .create(true)
        .append(true)
        .open(dir.join(format!("{day}.jsonl")))
        .ok()
}

/// `YYYY-MM-DD` in UTC, the monitor's filename convention.
fn utc_day(t: SystemTime) -> String {
    let secs = t.duration_since(UNIX_EPOCH).unwrap_or_default().as_secs() as i64;
    let (y, m, d) = civil_from_days(secs.div_euclid(86_400));
    format!("{y:04}-{m:02}-{d:02}")
}

/// `%Y-%m-%dT%H:%M:%S.%fZ`, which is the pattern
/// interp/probes/monitor.py `read_log_window` pins with a strict `strptime`.
///
/// Deliberately not obs.rs's `format_ts`: that one ends `+00:00`, because the
/// obs readers pin an explicit offset, and this one must end `Z`. Two readers,
/// two pinned patterns, and a record that misses either is silently skipped
/// rather than rejected loudly -- so the formats are kept apart rather than
/// shared and bent.
pub fn monitor_ts(t: SystemTime) -> String {
    let dur = t.duration_since(UNIX_EPOCH).unwrap_or_default();
    let secs = dur.as_secs() as i64;
    let millis = dur.subsec_millis();
    let (y, m, d) = civil_from_days(secs.div_euclid(86_400));
    let sod = secs.rem_euclid(86_400);
    format!(
        "{y:04}-{m:02}-{d:02}T{:02}:{:02}:{:02}.{millis:03}Z",
        sod / 3600,
        (sod % 3600) / 60,
        sod % 60
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    fn at(epoch_secs: u64) -> SystemTime {
        UNIX_EPOCH + Duration::from_secs(epoch_secs)
    }

    fn record(text: &str) -> RequestRecord {
        RequestRecord {
            timestamp: monitor_ts(at(1_700_000_000)),
            target_service: "harness".to_string(),
            model: "test".to_string(),
            messages: serde_json::json!([{"role": "user", "content": text}]),
            response_text: Some("reply".to_string()),
            usage: None,
            latency_s: 0.5,
            status_code: 200,
            error: None,
            probe: None,
        }
    }

    #[test]
    fn the_timestamp_matches_the_pattern_the_monitor_pins() {
        // %Y-%m-%dT%H:%M:%S.%fZ -- millisecond fraction, trailing Z, no
        // offset. A drift here is invisible: the monitor counts an unparsable
        // record as skipped rather than failing.
        assert_eq!(monitor_ts(at(1_700_000_000)), "2023-11-14T22:13:20.000Z");
    }

    #[test]
    fn the_timestamp_is_not_the_obs_format() {
        let ts = monitor_ts(at(1_700_000_000));
        assert!(ts.ends_with('Z'), "{ts}");
        assert!(!ts.contains("+00:00"), "{ts}");
    }

    #[test]
    fn a_day_boundary_names_the_next_file() {
        // 2023-11-14T23:59:59Z and one second later.
        assert_eq!(utc_day(at(1_700_006_399)), "2023-11-14");
        assert_eq!(utc_day(at(1_700_006_400)), "2023-11-15");
    }

    #[test]
    fn records_append_under_the_service_directory() {
        let dir = tempfile::tempdir().expect("tempdir");
        let mut log = InterpLog::open(dir.path(), "harness");
        assert!(log.is_enabled());
        log.append(&record("one"));
        log.append(&record("two"));
        let path = dir
            .path()
            .join("harness")
            .join(format!("{}.jsonl", utc_day(SystemTime::now())));
        let body = std::fs::read_to_string(&path).expect("log file");
        let lines: Vec<&str> = body.lines().collect();
        assert_eq!(lines.len(), 2, "appended, not truncated: {body}");
        for line in lines {
            let parsed: serde_json::Value = serde_json::from_str(line).expect("json line");
            assert_eq!(parsed["target_service"], "harness");
        }
    }

    #[test]
    fn a_disabled_log_drops_records_without_panicking() {
        let mut log = InterpLog::disabled();
        assert!(!log.is_enabled());
        log.append(&record("one"));
    }

    #[test]
    fn the_record_carries_the_keys_the_monitor_reads() {
        // reconstruct_text needs messages[].content; read_log_window needs
        // timestamp and target_service. The monitor skips a record missing
        // any of them, so this is the contract that keeps a window non-empty.
        let json = serde_json::to_value(record("hello")).expect("serialize");
        for key in [
            "timestamp",
            "target_service",
            "messages",
            "response_text",
            "usage",
            "latency_s",
            "error",
        ] {
            assert!(json.get(key).is_some(), "missing {key} in {json}");
        }
        assert_eq!(json["messages"][0]["content"], "hello");
    }

    #[test]
    fn the_probe_block_carries_the_per_request_statistic() {
        let mut rec = record("hello");
        rec.probe = Some(ProbeRecord {
            name: "deception".to_string(),
            layer: 12,
            site: "layer_output".to_string(),
            threshold: 0.5,
            scored: 3,
            position_first: Some(100),
            max_score: Some(0.9),
            flags: vec![FlagRecord {
                position: 102,
                score: 0.9,
            }],
        });
        let json = serde_json::to_value(rec).expect("serialize");
        // Widened, not rounded: an f32 0.9 serializes as its exact f64 value.
        // Compared with a tolerance because the alternative is pinning that
        // decimal expansion, which tests the float formatter rather than this
        // record.
        let max = json["probe"]["max_score"].as_f64().expect("max_score");
        assert!((max - 0.9).abs() < 1e-6, "{max}");
        assert_eq!(json["probe"]["position_first"], 100);
        // Absolute cache position, so the offset within the request is the
        // difference from position_first.
        assert_eq!(json["probe"]["flags"][0]["position"], 102);
    }
}
