/*
 * serve_snapshot — a served session neither restores nor writes session.kv.
 *
 * Task and chat runs resume the KV cache from the snapshot the previous clean
 * exit left in the working directory. Serve mode never reaches that save, and
 * RH.14 (2026-08-12) found the other half missing: a server started from a
 * directory holding a snapshot restored it, and the endpoint announced a
 * conversation it would never continue. The fix hands `build_generator` no
 * snapshot path in serve mode; this pins it from outside the binary.
 *
 * The planted file is garbage on purpose. A valid snapshot would be restored
 * silently or discarded silently depending on the hash; an invalid one that
 * is opened at all produces the `Session snapshot ... discarded` warning, so
 * its absence is the evidence the file was never read. The bytes staying
 * identical across the run is the evidence it was never written either.
 */

use std::io::{Read, Write};
use std::net::{Ipv4Addr, SocketAddr, TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

fn fixture_model() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../fixtures")
        .join("tiny-gemma4")
}

/// Bound and released rather than guessed, as serve_bind_addr.rs does.
fn free_port() -> u16 {
    TcpListener::bind((Ipv4Addr::LOCALHOST, 0))
        .unwrap()
        .local_addr()
        .unwrap()
        .port()
}

/// GET /v1/health, with transport errors returned: a refused connection is
/// startup noise here, not a failure.
fn health(addr: SocketAddr) -> std::io::Result<String> {
    let mut stream = TcpStream::connect_timeout(&addr, Duration::from_secs(5))?;
    stream.write_all(b"GET /v1/health HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n")?;
    stream.flush()?;
    let mut out = String::new();
    stream.read_to_string(&mut out)?;
    Ok(out)
}

/// The startup lines are a few hundred bytes, well inside the pipe buffer,
/// so the child cannot block on stderr before it answers health.
const NOT_A_SNAPSHOT: &[u8] = b"not a session.kv\n";

#[test]
fn a_served_session_leaves_the_snapshot_unread_and_unwritten() {
    let dir = tempfile::tempdir().unwrap();
    let snapshot = dir.path().join("session.kv");
    std::fs::write(&snapshot, NOT_A_SNAPSHOT).unwrap();

    let port = free_port();
    let mut child = std::process::Command::new(env!("CARGO_BIN_EXE_harness"))
        .current_dir(dir.path())
        .arg("--model")
        .arg(fixture_model())
        .arg("--serve")
        .arg(port.to_string())
        .stdin(std::process::Stdio::null())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .expect("spawn harness");

    let addr = SocketAddr::from((Ipv4Addr::LOCALHOST, port));
    let deadline = Instant::now() + Duration::from_secs(60);
    let mut served = false;
    while Instant::now() < deadline {
        if let Ok(Some(status)) = child.try_wait() {
            let mut err = String::new();
            let _ = child.stderr.take().unwrap().read_to_string(&mut err);
            panic!("harness exited before serving on {addr}: {status}\n{err}");
        }
        if matches!(health(addr), Ok(r) if r.contains(" 200 ")) {
            served = true;
            break;
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    assert!(served, "harness never answered /v1/health on {addr}");

    let _ = child.kill();
    let _ = child.wait();
    let mut err = String::new();
    child
        .stderr
        .take()
        .unwrap()
        .read_to_string(&mut err)
        .unwrap();

    assert!(
        !err.contains("Session restored") && !err.contains("Session snapshot"),
        "a served session touched the snapshot:\n{err}"
    );
    assert_eq!(
        std::fs::read(&snapshot).unwrap(),
        NOT_A_SNAPSHOT,
        "a served session rewrote session.kv"
    );
}
