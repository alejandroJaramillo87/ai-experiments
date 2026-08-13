/*
 * server — the OpenAI-shaped loopback endpoint the interp pipeline calls.
 *
 * Two paths, because two are all any caller uses: `GET /v1/health` and
 * `POST /v1/chat/completions`. The clients are common/serving.py's
 * `chat_completion` and interp/steering/scorecard.py's `chat`, both blocking
 * urllib POSTs of a Content-Length JSON body. Neither streams, asks for
 * logprobs, or sends more than one completion's worth of request, so none of
 * that is served; `stream: true` is refused rather than answered with a whole
 * body, which would be a lie about what happened.
 *
 * HTTP is hand-rolled over std::net rather than taken from a framework. The
 * surface is one verb over two paths with a length-delimited body, and
 * .styles/RUST_STYLE.md bounds the adopt-over-build exception to checkpoint
 * formats and forbids pulling a network client into an offline gate. So this
 * costs no dependency and no Cargo.lock movement. The cost it does carry:
 * no keep-alive (every response closes the connection) and no chunked request
 * body (refused, not mis-parsed). urllib sends neither.
 *
 * Concurrency is the one place the shape is forced. `Model` is Send but not
 * Sync and every generation path takes `&mut self`, so exactly one generation
 * runs at a time. A sequential accept loop would also queue health probes
 * behind it, and a CPU-backend E4B reply takes minutes against a 3-second
 * poll and a 10-second compose healthcheck -- a busy service would read as a
 * dead one. So: a thread per connection, with the generator behind a mutex
 * that the health handler never takes.
 */

use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Instant, SystemTime};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use runtime::sampler::SamplingParams;

use crate::generator::SessionGenerator;
use crate::interplog::{monitor_ts, InterpLog, MonitorHandle, ProbeRecord, RequestRecord};
use crate::template;
use crate::transcript::{Entry, Role};

/// The name this endpoint records itself under, which is both the log
/// subdirectory and the `target_service` field interp reads. It is the
/// compose service name so a record says which container answered.
pub const SERVICE_NAME: &str = "harness";

/// Connections served at once. Generation serializes on the mutex regardless;
/// this bounds the threads waiting for it, so a client that opens sockets
/// without sending cannot exhaust the process. Beyond it, 503.
const MAX_CONNECTIONS: usize = 8;

/// Request body cap. A prompt far past the context budget is a client error,
/// and reading it into memory before saying so is the wrong order.
const MAX_BODY_BYTES: usize = 8 << 20;

/// What the endpoint serves, shared across connection threads.
pub struct ServerState {
    /// The generator, and the reason chat requests serialize. Held apart from
    /// everything else so `GET /v1/health` can answer without waiting on a
    /// generation that may run for minutes.
    generator: Mutex<SessionGenerator>,
    /// Reported back as the `model` field. Callers do not match on it -- the
    /// two request sites send `"default"` and nothing respectively -- but a
    /// recorded transcript should name what answered.
    model: String,
    /// Sampling used for whatever a request leaves unset.
    defaults: SamplingParams,
    max_tokens: usize,
    /// Completion ids. A counter rather than random bytes: an RNG crate is
    /// prohibited, and for a research endpoint a reproducible id is a feature.
    next_id: AtomicU64,
    /// Where a completed request is recorded, and the probe monitor drained
    /// into it. Behind its own mutex, never the generator's: a served request
    /// holds the generator for minutes, and the log write must not extend
    /// that, nor must a slow disk serialize behind generation.
    ///
    /// Absent by default. `--serve` without `--interp-log` serves without
    /// recording, which is what every non-interp caller wants.
    observation: Mutex<Observation>,
}

/// The interp side of a served request: the sink, and the monitor drained
/// into it.
struct Observation {
    log: InterpLog,
    monitor: Option<MonitorHandle>,
}

impl ServerState {
    pub fn new(
        generator: SessionGenerator,
        model: String,
        defaults: SamplingParams,
        max_tokens: usize,
    ) -> ServerState {
        ServerState {
            generator: Mutex::new(generator),
            model,
            defaults,
            max_tokens,
            next_id: AtomicU64::new(0),
            observation: Mutex::new(Observation {
                log: InterpLog::disabled(),
                monitor: None,
            }),
        }
    }

    /// Record every completed request to `log`, draining `monitor` into each
    /// record. Separate from `new` because only the interp path asks for it:
    /// an agent-facing endpoint should not write a transcript of its traffic
    /// to disk as a side effect of being served.
    pub fn observing(self, log: InterpLog, monitor: Option<MonitorHandle>) -> ServerState {
        ServerState {
            observation: Mutex::new(Observation { log, monitor }),
            ..self
        }
    }
}

/// A parsed request line and body. Headers beyond `Content-Length` are read
/// and dropped: no caller sends one this endpoint acts on.
#[derive(Debug, PartialEq, Eq)]
pub struct Request {
    pub method: String,
    pub path: String,
    pub body: Vec<u8>,
}

/// A response ready to write. `body` is always JSON; every caller parses it,
/// including on the error paths.
#[derive(Debug, PartialEq, Eq)]
pub struct Response {
    pub status: u16,
    pub body: String,
}

impl Response {
    fn json(status: u16, body: String) -> Response {
        Response { status, body }
    }

    /// A 4xx the client must fix. `post_chat` treats 4xx as terminal and 5xx
    /// as retryable, so the split decides whether a caller retries.
    fn bad_request(message: &str) -> Response {
        Response::json(400, error_body(message))
    }

    fn reason(&self) -> &'static str {
        match self.status {
            200 => "OK",
            400 => "Bad Request",
            404 => "Not Found",
            405 => "Method Not Allowed",
            413 => "Payload Too Large",
            500 => "Internal Server Error",
            503 => "Service Unavailable",
            _ => "Unknown",
        }
    }

    fn write_to(&self, out: &mut impl Write) -> std::io::Result<()> {
        write!(
            out,
            "HTTP/1.1 {} {}\r\n\
             Content-Type: application/json\r\n\
             Content-Length: {}\r\n\
             Connection: close\r\n\
             \r\n{}",
            self.status,
            self.reason(),
            self.body.len(),
            self.body
        )
    }
}

/// The error envelope every non-2xx carries. `post_chat` reads the status
/// rather than the body, but a person reading a failed run needs the reason.
fn error_body(message: &str) -> String {
    serde_json::json!({ "error": { "message": message } }).to_string()
}

#[derive(Deserialize)]
struct ChatRequest {
    messages: Vec<ChatMessage>,
    /// Ignored, and optional twice over: common/serving.py sends the literal
    /// "default" and scorecard.py omits the field. The served checkpoint is a
    /// command-line argument, so there is nothing here to select.
    #[serde(default)]
    #[allow(dead_code)]
    model: Option<String>,
    #[serde(default)]
    temperature: Option<f32>,
    #[serde(default)]
    top_p: Option<f32>,
    #[serde(default)]
    top_k: Option<usize>,
    #[serde(default)]
    max_tokens: Option<usize>,
    #[serde(default)]
    seed: Option<u64>,
    #[serde(default)]
    stream: Option<bool>,
}

/// `Serialize` as well as `Deserialize` because the interp log records the
/// messages verbatim: the monitor rebuilds the scored text from them.
#[derive(Deserialize, Serialize)]
struct ChatMessage {
    role: String,
    content: String,
}

#[derive(Serialize)]
struct ChatResponse {
    id: String,
    object: &'static str,
    created: u64,
    model: String,
    choices: Vec<Choice>,
    usage: Usage,
}

#[derive(Serialize)]
struct Choice {
    index: usize,
    message: ResponseMessage,
    finish_reason: &'static str,
}

#[derive(Serialize)]
struct ResponseMessage {
    role: &'static str,
    content: String,
}

#[derive(Serialize)]
struct Usage {
    prompt_tokens: usize,
    completion_tokens: usize,
    total_tokens: usize,
}

/// Serve until the listener fails. One thread per accepted connection; the
/// generator mutex is what makes generation sequential.
pub fn serve(listener: TcpListener, state: Arc<ServerState>) -> Result<()> {
    let live = Arc::new(AtomicU64::new(0));
    for stream in listener.incoming() {
        let stream = match stream {
            Ok(s) => s,
            // One refused connection is not a reason to stop serving.
            Err(e) => {
                eprintln!("[WARN] Connection failed: {e}");
                continue;
            }
        };
        if live.load(Ordering::SeqCst) as usize >= MAX_CONNECTIONS {
            let mut stream = stream;
            let _ = Response::json(503, error_body("too many concurrent connections"))
                .write_to(&mut stream);
            continue;
        }
        live.fetch_add(1, Ordering::SeqCst);
        let state = Arc::clone(&state);
        let live = Arc::clone(&live);
        std::thread::spawn(move || {
            if let Err(e) = converse(stream, &state) {
                eprintln!("[WARN] Request failed: {e:#}");
            }
            live.fetch_sub(1, Ordering::SeqCst);
        });
    }
    Ok(())
}

/// Read one request, answer it, close. A parse failure still gets a response:
/// a client that sent something malformed deserves the status, not a reset.
fn converse(mut stream: TcpStream, state: &ServerState) -> Result<()> {
    let response = match read_request(&mut stream) {
        Ok(request) => handle(&request, state),
        Err(e) => Response::bad_request(&format!("{e:#}")),
    };
    response.write_to(&mut stream).context("writing response")?;
    stream.flush().context("flushing response")?;
    Ok(())
}

/// Parse a request line, headers, and a `Content-Length` body.
pub fn read_request(stream: &mut impl Read) -> Result<Request> {
    let mut reader = BufReader::new(stream);
    let mut line = String::new();
    reader
        .read_line(&mut line)
        .context("reading the request line")?;
    let mut parts = line.split_whitespace();
    let method = parts
        .next()
        .context("empty request line")?
        .trim()
        .to_string();
    let path = parts
        .next()
        .with_context(|| format!("request line has no path: {}", line.trim()))?
        .to_string();

    let mut length: Option<usize> = None;
    let mut chunked = false;
    loop {
        let mut header = String::new();
        let read = reader.read_line(&mut header).context("reading a header")?;
        // End of stream without a blank line: the client hung up mid-request.
        anyhow::ensure!(read != 0, "headers ended without a blank line");
        let header = header.trim_end();
        if header.is_empty() {
            break;
        }
        let Some((name, value)) = header.split_once(':') else {
            anyhow::bail!("header has no colon: {header}");
        };
        let name = name.trim().to_ascii_lowercase();
        let value = value.trim();
        if name == "content-length" {
            length = Some(
                value
                    .parse::<usize>()
                    .with_context(|| format!("content-length is not a number: {value}"))?,
            );
        }
        if name == "transfer-encoding" && value.to_ascii_lowercase().contains("chunked") {
            chunked = true;
        }
    }
    // Refused rather than mis-parsed. Reading a chunked body as if it were
    // length-delimited would hand the JSON parser the frame headers.
    anyhow::ensure!(!chunked, "chunked request bodies are not served");

    let length = length.unwrap_or(0);
    anyhow::ensure!(
        length <= MAX_BODY_BYTES,
        "request body of {length} bytes exceeds the {MAX_BODY_BYTES}-byte limit"
    );
    let mut body = vec![0u8; length];
    reader
        .read_exact(&mut body)
        .with_context(|| format!("body shorter than the declared {length} bytes"))?;
    Ok(Request { method, path, body })
}

/// Route one request. Separated from the socket so the whole surface is
/// testable without one.
pub fn handle(request: &Request, state: &ServerState) -> Response {
    match (request.method.as_str(), request.path.as_str()) {
        // Answered without touching the generator mutex: a health probe
        // during a long generation has to return now, not after it.
        ("GET", "/v1/health") => Response::json(200, r#"{"status":"ok"}"#.to_string()),
        ("POST", "/v1/chat/completions") => chat_completion(&request.body, state),
        ("GET", _) | ("POST", _) => {
            Response::json(404, error_body(&format!("no such path: {}", request.path)))
        }
        _ => Response::json(
            405,
            error_body(&format!("method not served: {}", request.method)),
        ),
    }
}

fn chat_completion(body: &[u8], state: &ServerState) -> Response {
    let request: ChatRequest = match serde_json::from_slice(body) {
        Ok(r) => r,
        Err(e) => return Response::bad_request(&format!("request body is not valid JSON: {e}")),
    };
    if request.stream == Some(true) {
        return Response::bad_request(
            "stream is not served; omit it or send false for a whole response",
        );
    }
    if request.messages.is_empty() {
        return Response::bad_request("messages is empty");
    }
    let entries = match entries_of(&request.messages) {
        Ok(e) => e,
        Err(message) => return Response::bad_request(&message),
    };
    let (params, max_tokens) = match sampling_of(&request, state.defaults, state.max_tokens) {
        Ok(p) => p,
        Err(message) => return Response::bad_request(&message),
    };

    // render sanitizes every message through render_entry, so content
    // carrying a turn marker stays text instead of forging a boundary.
    let prompt = template::render(&entries);
    let started = Instant::now();
    let (completion, probe) = {
        let mut generator = match state.generator.lock() {
            Ok(g) => g,
            // A panic in an earlier generation left the session in an unknown
            // state. Saying so beats serving from it.
            Err(e) => {
                return Response::json(
                    500,
                    error_body(&format!("generator is poisoned by an earlier failure: {e}")),
                )
            }
        };
        let completion = generator.complete_sampled(&prompt, params, max_tokens);
        // Drained while the generation lock is still held. Generation is
        // serialized on it, so everything the tap recorded belongs to this
        // request; draining after releasing would race the next one.
        let probe = state
            .observation
            .lock()
            .ok()
            .and_then(|o| o.monitor.as_ref().map(|m| m.drain()));
        (completion, probe)
    };
    let latency_s = started.elapsed().as_secs_f64();
    let completion = match completion {
        Ok(c) => c,
        // 5xx: post_chat classifies it as transient, which is right -- the
        // request was well formed and a retry may land.
        Err(e) => {
            let message = format!("generation failed: {e:#}");
            // Recorded too. A window in which the failures are invisible
            // would show a drop in flagged traffic that never happened.
            record_request(
                state,
                &request.messages,
                None,
                None,
                latency_s,
                500,
                Some(message.clone()),
                probe,
            );
            return Response::json(500, error_body(&message));
        }
    };

    let response = ChatResponse {
        id: format!("chatcmpl-{}", state.next_id.fetch_add(1, Ordering::SeqCst)),
        object: "chat.completion",
        created: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0),
        model: state.model.clone(),
        choices: vec![Choice {
            index: 0,
            message: ResponseMessage {
                role: "assistant",
                content: completion.text,
            },
            finish_reason: if completion.stopped_at_eos {
                "stop"
            } else {
                "length"
            },
        }],
        usage: Usage {
            prompt_tokens: completion.prompt_tokens,
            completion_tokens: completion.completion_tokens,
            total_tokens: completion.prompt_tokens + completion.completion_tokens,
        },
    };
    record_request(
        state,
        &request.messages,
        Some(response.choices[0].message.content.clone()),
        Some(&response.usage),
        latency_s,
        200,
        None,
        probe,
    );
    match serde_json::to_string(&response) {
        Ok(body) => Response::json(200, body),
        Err(e) => Response::json(500, error_body(&format!("encoding the response: {e}"))),
    }
}

/// Append one completed request to the interp log, if one is open.
///
/// Everything here is best-effort by construction: a serialization failure
/// drops the record and an absent log does nothing. Observation must not
/// change what the caller receives, so this returns nothing to act on.
#[allow(clippy::too_many_arguments)]
fn record_request(
    state: &ServerState,
    messages: &[ChatMessage],
    response_text: Option<String>,
    usage: Option<&Usage>,
    latency_s: f64,
    status_code: u16,
    error: Option<String>,
    probe: Option<ProbeRecord>,
) {
    let Ok(mut observation) = state.observation.lock() else {
        return;
    };
    if !observation.log.is_enabled() {
        return;
    }
    let record = RequestRecord {
        timestamp: monitor_ts(SystemTime::now()),
        target_service: SERVICE_NAME.to_string(),
        model: state.model.clone(),
        messages: serde_json::to_value(messages).unwrap_or(serde_json::Value::Null),
        response_text,
        usage: usage.and_then(|u| serde_json::to_value(u).ok()),
        latency_s,
        status_code,
        error,
        probe,
    };
    observation.log.append(&record);
}

/// Map OpenAI roles onto transcript entries.
///
/// An unrecognized role is refused rather than folded into `user`: the roles
/// render to different prompt text, so guessing would silently answer a
/// different conversation than the one asked about.
fn entries_of(messages: &[ChatMessage]) -> Result<Vec<Entry>, String> {
    messages
        .iter()
        .map(|m| {
            let role = match m.role.as_str() {
                "system" => Role::System,
                "user" => Role::User,
                "assistant" => Role::Model,
                other => return Err(format!("unknown role: {other}")),
            };
            Ok(Entry {
                role,
                content: m.content.clone(),
                replaces: None,
            })
        })
        .collect()
}

/// Sampling for one request, falling back to the server's defaults.
///
/// Range-checked here because `Sampler::new` panics on an out-of-range value:
/// what is a configuration bug on a command line is ordinary bad input over
/// HTTP, and it has to come back as a 400 naming the value.
fn sampling_of(
    request: &ChatRequest,
    defaults: SamplingParams,
    default_max_tokens: usize,
) -> Result<(SamplingParams, usize), String> {
    let mut params = defaults;
    if let Some(t) = request.temperature {
        if !t.is_finite() || t < 0.0 {
            return Err(format!("temperature must be a finite number >= 0, got {t}"));
        }
        params.temperature = t;
    }
    if let Some(p) = request.top_p {
        if !(p > 0.0 && p <= 1.0) {
            return Err(format!("top_p must be in (0, 1], got {p}"));
        }
        params.top_p = p;
    }
    if let Some(k) = request.top_k {
        // Zero is refused for the reason the command line refuses it: it
        // reads as either "keep nothing" or "keep everything".
        if k < 1 {
            return Err("top_k must be >= 1; omit it for no cutoff".to_string());
        }
        params.top_k = Some(k);
    }
    if let Some(seed) = request.seed {
        params.seed = seed;
    }
    let max_tokens = match request.max_tokens {
        Some(n) if n >= 1 => n,
        Some(n) => return Err(format!("max_tokens must be >= 1, got {n}")),
        None => default_max_tokens,
    };
    Ok((params, max_tokens))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A request as bytes on the wire, so the parser is tested against the
    /// framing rather than against a struct we built ourselves.
    fn wire(headers: &str, body: &str) -> Vec<u8> {
        format!("{headers}\r\n\r\n{body}").into_bytes()
    }

    #[test]
    fn a_post_with_a_content_length_body_parses() {
        let raw = wire(
            "POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Length: 7",
            "1234567",
        );
        let request = read_request(&mut raw.as_slice()).unwrap();
        assert_eq!(request.method, "POST", "method from {raw:?}");
        assert_eq!(request.path, "/v1/chat/completions", "path from {raw:?}");
        assert_eq!(request.body, b"1234567", "body from {raw:?}");
    }

    #[test]
    fn a_get_without_a_content_length_has_an_empty_body() {
        let raw = wire("GET /v1/health HTTP/1.1\r\nHost: x", "");
        let request = read_request(&mut raw.as_slice()).unwrap();
        assert_eq!(request.method, "GET", "method from {raw:?}");
        assert!(request.body.is_empty(), "body was {:?}", request.body);
    }

    #[test]
    fn a_body_shorter_than_its_content_length_is_an_error() {
        let raw = wire("POST /x HTTP/1.1\r\nContent-Length: 99", "short");
        let error = read_request(&mut raw.as_slice()).unwrap_err();
        assert!(
            format!("{error:#}").contains("99"),
            "error should name the declared length, got: {error:#}"
        );
    }

    #[test]
    fn a_chunked_body_is_refused_rather_than_mis_parsed() {
        let raw = wire(
            "POST /x HTTP/1.1\r\nTransfer-Encoding: chunked",
            "5\r\nhello\r\n0\r\n",
        );
        let error = read_request(&mut raw.as_slice()).unwrap_err();
        assert!(
            format!("{error:#}").contains("chunked"),
            "error should name chunked encoding, got: {error:#}"
        );
    }

    #[test]
    fn a_content_length_past_the_cap_is_refused_before_allocating() {
        let header = format!("POST /x HTTP/1.1\r\nContent-Length: {}", MAX_BODY_BYTES + 1);
        let raw = wire(&header, "");
        let error = read_request(&mut raw.as_slice()).unwrap_err();
        assert!(
            format!("{error:#}").contains("exceeds"),
            "error should name the limit, got: {error:#}"
        );
    }

    #[test]
    fn a_header_name_is_matched_case_insensitively() {
        let raw = wire("POST /x HTTP/1.1\r\ncOnTeNt-LeNgTh: 2", "ab");
        let request = read_request(&mut raw.as_slice()).unwrap();
        assert_eq!(request.body, b"ab", "body from {raw:?}");
    }

    #[test]
    fn the_three_openai_roles_map_onto_transcript_roles() {
        let messages = vec![
            ChatMessage {
                role: "system".into(),
                content: "s".into(),
            },
            ChatMessage {
                role: "user".into(),
                content: "u".into(),
            },
            ChatMessage {
                role: "assistant".into(),
                content: "a".into(),
            },
        ];
        let entries = entries_of(&messages).unwrap();
        let roles: Vec<Role> = entries.iter().map(|e| e.role.clone()).collect();
        assert_eq!(
            roles,
            vec![Role::System, Role::User, Role::Model],
            "roles were {roles:?}"
        );
    }

    #[test]
    fn an_unknown_role_is_refused_and_named() {
        let messages = vec![ChatMessage {
            role: "toolbox".into(),
            content: "x".into(),
        }];
        let error = entries_of(&messages).unwrap_err();
        assert!(
            error.contains("toolbox"),
            "error should name the role, got: {error}"
        );
    }

    /// The request the sampling tests vary one field of.
    fn request_with(json: &str) -> ChatRequest {
        serde_json::from_str(json).expect("test request should parse")
    }

    /// Resolve sampling for a request written as JSON. `sampling_of` reads
    /// only the two defaults off the state, so it takes them directly and the
    /// tests reach the real function rather than a copy of its rules -- a
    /// second implementation would agree with its own bugs.
    fn resolve(
        json: &str,
        defaults: SamplingParams,
        max_tokens: usize,
    ) -> Result<(SamplingParams, usize), String> {
        sampling_of(&request_with(json), defaults, max_tokens)
    }

    #[test]
    fn a_request_omitting_the_model_field_parses() {
        // scorecard.py sends no model field at all.
        let request = request_with(r#"{"messages":[],"max_tokens":8,"temperature":0.0}"#);
        assert!(request.model.is_none(), "model should be absent");
    }

    #[test]
    fn an_absent_sampling_field_falls_back_to_the_server_default() {
        let defaults = SamplingParams {
            temperature: 0.7,
            top_p: 0.9,
            top_k: None,
            seed: 11,
        };
        let (params, max_tokens) = resolve(r#"{"messages":[]}"#, defaults, 42).unwrap();
        assert_eq!(params.temperature, 0.7, "temperature was {params:?}");
        assert_eq!(params.seed, 11, "seed was {params:?}");
        assert_eq!(max_tokens, 42, "max_tokens was {max_tokens}");
    }

    #[test]
    fn a_request_seed_overrides_the_server_default() {
        let defaults = SamplingParams::default();
        let (params, _) = resolve(r#"{"messages":[],"seed":9001}"#, defaults, 8).unwrap();
        assert_eq!(params.seed, 9001, "seed was {params:?}");
    }

    #[test]
    fn an_out_of_range_temperature_is_refused_before_the_sampler_can_panic() {
        let error = resolve(
            r#"{"messages":[],"temperature":-1.0}"#,
            SamplingParams::default(),
            8,
        )
        .unwrap_err();
        assert!(
            error.contains("-1"),
            "error should quote the value, got: {error}"
        );
    }

    #[test]
    fn a_top_p_outside_the_unit_interval_is_refused() {
        let error = resolve(
            r#"{"messages":[],"top_p":1.5}"#,
            SamplingParams::default(),
            8,
        )
        .unwrap_err();
        assert!(
            error.contains("1.5"),
            "error should quote the value, got: {error}"
        );
    }

    #[test]
    fn a_zero_top_k_is_refused_rather_than_read_as_disabled() {
        let error =
            resolve(r#"{"messages":[],"top_k":0}"#, SamplingParams::default(), 8).unwrap_err();
        assert!(
            error.contains("top_k"),
            "error should name the field, got: {error}"
        );
    }

    #[test]
    fn a_zero_max_tokens_is_refused() {
        let error = resolve(
            r#"{"messages":[],"max_tokens":0}"#,
            SamplingParams::default(),
            8,
        )
        .unwrap_err();
        assert!(
            error.contains("0"),
            "error should quote the value, got: {error}"
        );
    }

    #[test]
    fn a_response_writes_its_status_length_and_body() {
        let mut out = Vec::new();
        Response::json(200, r#"{"a":1}"#.to_string())
            .write_to(&mut out)
            .unwrap();
        let text = String::from_utf8(out).unwrap();
        assert!(
            text.starts_with("HTTP/1.1 200 OK\r\n"),
            "response was {text}"
        );
        assert!(
            text.contains("Content-Length: 7\r\n"),
            "response was {text}"
        );
        assert!(text.ends_with("\r\n\r\n{\"a\":1}"), "response was {text}");
    }

    /// A state over a scripted model, for the routing tests. The script is
    /// never reached by these -- they either take the health path or fail
    /// validation before generating -- but a real `ServerState` is what makes
    /// the mutex placement testable.
    fn state() -> ServerState {
        use runtime::model::mock::MockModel;
        use runtime::session::Session;
        use runtime::tokenizer::Tokenizer;

        let pieces = ["<pad>", "<bos>", "<eos>"].map(String::from).to_vec();
        let json = runtime::synth::tokenizer_json_from_parts(&pieces, &[], "<pad>").unwrap();
        let tokenizer = Tokenizer::from_json(&json).unwrap();
        let vocab_len = tokenizer.vocab_len();
        let session = Session::new(
            Box::new(MockModel::new(vocab_len, vec![vec![0.0; vocab_len]; 8])),
            64,
        )
        .unwrap();
        let generator =
            SessionGenerator::new(session, tokenizer, SamplingParams::default(), 4).unwrap();
        ServerState::new(generator, "test".to_string(), SamplingParams::default(), 4)
    }

    fn get(path: &str) -> Request {
        Request {
            method: "GET".to_string(),
            path: path.to_string(),
            body: Vec::new(),
        }
    }

    #[test]
    fn health_answers_while_a_generation_holds_the_generator() {
        let state = state();
        // Stands in for a generation in flight. If the health path ever takes
        // this lock, the call below deadlocks rather than failing -- which is
        // the failure this test exists to prevent, since a health probe
        // queued behind a minutes-long CPU reply reads as a dead service.
        let _generating = state.generator.lock().expect("generator lock");
        let response = handle(&get("/v1/health"), &state);
        assert_eq!(
            response.status, 200,
            "health under load returned {response:?}"
        );
    }

    #[test]
    fn an_unknown_path_is_a_404_naming_it() {
        let state = state();
        let response = handle(&get("/v1/models"), &state);
        assert_eq!(response.status, 404, "response was {response:?}");
        assert!(
            response.body.contains("/v1/models"),
            "body should name the path, got {}",
            response.body
        );
    }

    #[test]
    fn an_unserved_method_is_a_405() {
        let state = state();
        let request = Request {
            method: "DELETE".to_string(),
            path: "/v1/health".to_string(),
            body: Vec::new(),
        };
        let response = handle(&request, &state);
        assert_eq!(response.status, 405, "response was {response:?}");
    }

    #[test]
    fn a_stream_request_is_refused_rather_than_answered_whole() {
        let state = state();
        let request = Request {
            method: "POST".to_string(),
            path: "/v1/chat/completions".to_string(),
            body: br#"{"messages":[{"role":"user","content":"hi"}],"stream":true}"#.to_vec(),
        };
        let response = handle(&request, &state);
        assert_eq!(response.status, 400, "response was {response:?}");
        assert!(
            response.body.contains("stream"),
            "body should name the field, got {}",
            response.body
        );
    }

    #[test]
    fn a_body_that_is_not_json_is_a_400() {
        let state = state();
        let request = Request {
            method: "POST".to_string(),
            path: "/v1/chat/completions".to_string(),
            body: b"not json".to_vec(),
        };
        let response = handle(&request, &state);
        assert_eq!(response.status, 400, "response was {response:?}");
    }

    #[test]
    fn an_empty_message_list_is_a_400() {
        let state = state();
        let request = Request {
            method: "POST".to_string(),
            path: "/v1/chat/completions".to_string(),
            body: br#"{"messages":[]}"#.to_vec(),
        };
        let response = handle(&request, &state);
        assert_eq!(response.status, 400, "response was {response:?}");
    }

    #[test]
    fn an_error_body_is_json_a_client_can_parse() {
        let body = error_body("no such path: /v2/chat");
        let parsed: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(
            parsed["error"]["message"], "no such path: /v2/chat",
            "body was {body}"
        );
    }
}
