/*
 * serve_interp_log — the served-request log, end to end through the binary.
 *
 * What is under test is the thing a served run could not do before: reach
 * disk. The probe monitor was always installed under --serve, and its scores
 * always accumulated, but the only drain ran after the task and serving has
 * no after. So this spawns the real binary with --serve, drives requests over
 * a socket, and reads the file back — a unit test of the writer cannot fail
 * for the reason that mattered, which was that nothing called it.
 *
 * The fixture model's weights are random, so nothing here says a score is
 * correct. What it says is that every completed request produced one record,
 * that the record carries the fields interp/probes/monitor.py reads, and that
 * consecutive requests are attributable — the tap reports absolute cache
 * positions, so a per-request drain that lost the base would report positions
 * belonging to whatever ran before.
 */

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

fn fixture_model() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../fixtures")
        .join("tiny-gemma4")
}

fn model_width() -> usize {
    let config = std::fs::read_to_string(fixture_model().join("config.json")).unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&config).unwrap();
    parsed["text_config"]["hidden_size"].as_u64().unwrap() as usize
}

/// A probe in the interp/probes/export.py layout, as probe_identity.rs writes
/// one.
///
/// `threshold` is a parameter because the two useful settings test different
/// things: above every attainable sigmoid nothing flags, and at zero every
/// scored position does. The second is what makes a missing drain visible --
/// flags carry positions, and a record that was not drained carries the
/// previous request's.
fn write_probe(dir: &Path, model_hash: &str, threshold: f32) -> PathBuf {
    let width = model_width();
    let weights: Vec<f32> = (0..width).map(|j| 0.1 * (j as f32 - 4.0)).collect();
    let mu: Vec<f32> = (0..width).map(|j| 0.01 * j as f32).collect();
    let sigma: Vec<f32> = (0..width).map(|j| 1.0 + 0.1 * j as f32).collect();
    let path = dir.join("probe.safetensors");
    std::fs::write(
        &path,
        runtime::synth::build_f32_safetensors(&[
            ("weights", &weights),
            ("mu", &mu),
            ("sigma", &sigma),
        ])
        .unwrap(),
    )
    .unwrap();
    std::fs::write(
        path.with_extension("json"),
        format!(
            r#"{{"schema_version": 1, "artifact": "probe",
               "timestamp": "2026-01-01T00:00:00+0000", "git_sha": "abcdef0",
               "name": "served-probe", "model_hash": "{model_hash}",
               "layer": 0, "bias": 0.1, "threshold": {threshold}, "width": {width}}}"#
        ),
    )
    .unwrap();
    path
}

/// A port nothing is listening on. Bound and released rather than guessed:
/// two test binaries running concurrently must not collide.
fn free_port() -> u16 {
    TcpListener::bind(("127.0.0.1", 0))
        .unwrap()
        .local_addr()
        .unwrap()
        .port()
}

/// A spawned harness that dies with the test, however the test ends.
///
/// `std::process::Child` does not kill on drop, and every assertion below runs
/// between the spawn and the explicit kill. A tripped assertion therefore used
/// to orphan a server holding the fixture model, and orphans accumulate: each
/// one competes for the cores the next run needs to answer /v1/health inside
/// its deadline, so one failure here made the next failure likelier. That is a
/// feedback loop rather than flakiness, and it cost four sessions before the
/// cause was the process table rather than the code under test.
struct Server {
    child: std::process::Child,
    port: u16,
}

impl Server {
    /// Spawn the binary with `--model` and `--serve` already supplied, plus
    /// whatever the caller adds. stderr is discarded rather than piped: nothing
    /// reads it, and a full pipe would block the server mid-request.
    fn spawn(dir: &Path, extra: &[&std::ffi::OsStr]) -> Server {
        let port = free_port();
        let child = std::process::Command::new(env!("CARGO_BIN_EXE_harness"))
            .current_dir(dir)
            .arg("--model")
            .arg(fixture_model())
            .args(extra)
            .arg("--serve")
            .arg(port.to_string())
            .stdin(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .spawn()
            .expect("spawn harness");
        Server { child, port }
    }

    /// Stop the server. Idempotent, so a test may end the process at the point
    /// its assertions need it dead and still leave the drop guard in place.
    fn kill(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        self.kill();
    }
}

/// One request, with the transport errors returned rather than panicked on.
///
/// Separate from `request` because a server that is still starting refuses or
/// resets a connection, and during startup that is the expected answer rather
/// than a failure.
fn try_request(port: u16, method: &str, path: &str, body: Option<&str>) -> std::io::Result<String> {
    let mut stream = TcpStream::connect(("127.0.0.1", port))?;
    let head = match body {
        Some(b) => format!(
            "{method} {path} HTTP/1.1\r\nHost: localhost\r\nContent-Length: {}\r\n\r\n{b}",
            b.len()
        ),
        None => format!("{method} {path} HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n"),
    };
    stream.write_all(head.as_bytes())?;
    stream.flush()?;
    let mut out = String::new();
    stream.read_to_string(&mut out)?;
    Ok(out)
}

/// One request against a server already known to be up, where a transport
/// error is a test failure.
fn request(port: u16, method: &str, path: &str, body: Option<&str>) -> String {
    try_request(port, method, path, body).expect("request")
}

/// Poll /v1/health until it answers. Loading the fixture checkpoint is fast,
/// but not instant, and a fixed sleep is either flaky or slow.
///
/// A transport error retries rather than failing. The loop used to probe with
/// a throwaway `TcpStream::connect` and then issue the real request over a
/// second connection, which asks the wrong question: a listener that accepted
/// one connection a moment ago need not accept the next, so the request could
/// be refused or reset while the server was still coming up. That raced at
/// about one run in five, and it presented as an unrelated test being flaky
/// under load, because load is what decides how often the race is lost.
fn wait_healthy(server: &mut Server) {
    let port = server.port;
    let deadline = Instant::now() + Duration::from_secs(60);
    while Instant::now() < deadline {
        if let Ok(Some(status)) = server.child.try_wait() {
            panic!("harness exited before serving: {status}");
        }
        if let Ok(response) = try_request(port, "GET", "/v1/health", None) {
            if response.contains(" 200 ") {
                return;
            }
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    // No kill here: the drop guard runs on the panic below, which is also what
    // covers every assertion the callers make after this returns.
    panic!("harness never answered /v1/health on {port}");
}

fn chat(port: u16, text: &str) -> String {
    let body =
        format!(r#"{{"messages":[{{"role":"user","content":"{text}"}}],"max_tokens":2,"seed":7}}"#);
    request(port, "POST", "/v1/chat/completions", Some(&body))
}

#[test]
fn every_served_request_lands_in_the_interp_log_with_its_probe_scores() {
    let dir = tempfile::tempdir().unwrap();
    // "unknown" is the sentinel interp/probes/capture.py defaults to, which
    // downgrades the identity check to a warning. The check itself is
    // probe_identity.rs's subject, not this test's.
    // Threshold 0: every scored position is above it, so `flags` is
    // non-empty and carries positions -- which is what makes the drain
    // observable from the file.
    let probe = write_probe(dir.path(), "unknown", 0.0);
    let logs = dir.path().join("logs");

    let mut server = Server::spawn(
        dir.path(),
        &[
            "--probe".as_ref(),
            probe.as_os_str(),
            "--interp-log".as_ref(),
            logs.as_os_str(),
        ],
    );
    let port = server.port;

    wait_healthy(&mut server);
    for text in ["first question", "second question"] {
        let response = chat(port, text);
        assert!(response.contains(" 200 "), "chat failed: {response}");
    }
    // Killed, not signalled gracefully: this is how the service actually
    // ends, and the records must already be on disk. A writer that flushed
    // at shutdown would pass a gentler test and lose every record here. The
    // kill is explicit for that reason -- the drop guard would run too late,
    // after the assertions that read the file.
    server.kill();

    let day_files: Vec<PathBuf> = std::fs::read_dir(logs.join("harness"))
        .expect("log directory")
        .map(|e| e.unwrap().path())
        .collect();
    assert_eq!(day_files.len(), 1, "one day file: {day_files:?}");
    let body = std::fs::read_to_string(&day_files[0]).unwrap();
    let records: Vec<serde_json::Value> = body
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str(l).expect("json record"))
        .collect();
    assert_eq!(records.len(), 2, "one record per request: {body}");

    for (record, text) in records.iter().zip(["first question", "second question"]) {
        assert_eq!(record["target_service"], "harness");
        assert_eq!(record["status_code"], 200);
        assert_eq!(record["messages"][0]["content"], text);
        assert!(
            record["response_text"].is_string(),
            "response recorded: {record}"
        );
        assert!(record["usage"]["total_tokens"].as_u64().unwrap() > 0);
        assert!(record["error"].is_null());
        let probe = &record["probe"];
        assert_eq!(probe["name"], "served-probe");
        assert_eq!(probe["layer"], 0);
        assert_eq!(probe["site"], "layer_output");
        assert!(
            probe["scored"].as_u64().unwrap() > 0,
            "the tap ran during the request: {record}"
        );
        assert!(
            probe["max_score"].is_number(),
            "max over the request: {record}"
        );
        let flags = probe["flags"].as_array().unwrap();
        // Threshold 0: a sigmoid is always above it, so every scored position
        // flagged. Equality is the check that `scored` counts this request's
        // positions and not the process's.
        assert_eq!(
            flags.len() as u64,
            probe["scored"].as_u64().unwrap(),
            "every scored position flagged: {record}"
        );
        // The load-bearing assertion. The probe scores one position per token
        // the forward pass moved, so a request cannot have scored more
        // positions than its own usage counts. A record that was not drained
        // carries every earlier request's positions too and breaks this.
        assert!(
            probe["scored"].as_u64().unwrap() <= record["usage"]["total_tokens"].as_u64().unwrap(),
            "scored positions belong to this request alone: {record}"
        );
        assert!(probe["position_first"].is_number(), "{record}");
    }

    // position_first is read off what the tap reported, not off the session's
    // cached length before generating. The two disagree here: each request
    // re-prefills its own prompt from 0, while the cached length still holds
    // the previous request's total. Pinning this is what keeps the derivation
    // from being "simplified" back to the cache length.
    assert_eq!(records[0]["probe"]["position_first"], 0);
    assert_eq!(records[1]["probe"]["position_first"], 0);
}

#[test]
fn serving_without_the_flag_writes_no_log() {
    let dir = tempfile::tempdir().unwrap();
    let mut server = Server::spawn(dir.path(), &[]);
    wait_healthy(&mut server);
    assert!(chat(server.port, "hello").contains(" 200 "));
    server.kill();
    // Recording traffic is opt-in. An endpoint that wrote a transcript of
    // every request to the working directory by default would be a surprise.
    assert!(
        !dir.path().join("harness").exists(),
        "no log without --interp-log"
    );
}

/// The bind address the container depends on.
///
/// Loopback is the default and is right for a harness run on a workstation,
/// but a container published port forwards to the container's external
/// interface: a server on the container's own 127.0.0.1 accepts nothing from
/// the host, refusing the connection with no log line to say why. The
/// entrypoint therefore passes --serve-addr 0.0.0.0, and this holds that flag
/// to binding what it is given. Connecting over 127.0.0.1 is the assertion
/// that matters -- 0.0.0.0 must include loopback, or the compose healthcheck,
/// which probes 127.0.0.1 from inside the container, would fail instead.
#[test]
fn serve_addr_binds_the_address_it_is_given() {
    let dir = tempfile::tempdir().unwrap();
    let mut server = Server::spawn(dir.path(), &["--serve-addr".as_ref(), "0.0.0.0".as_ref()]);
    wait_healthy(&mut server);
    assert!(chat(server.port, "hello").contains(" 200 "));
    server.kill();
}

/// A bind address without a server to bind is a mistake worth naming: the
/// flag is silently inert everywhere else, so a run that meant to serve and
/// forgot --serve would otherwise look like it worked.
#[test]
fn serve_addr_without_serve_is_refused() {
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_harness"))
        .arg("--model")
        .arg(fixture_model())
        .arg("--serve-addr")
        .arg("0.0.0.0")
        .arg("say hello")
        .output()
        .expect("run harness");
    assert_eq!(output.status.code(), Some(2));
    assert!(
        String::from_utf8_lossy(&output.stderr).contains("--serve-addr"),
        "the error names the flag"
    );
}
