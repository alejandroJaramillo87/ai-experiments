/*
 * serve_bind_addr — --serve-addr decides which interface the endpoint answers on.
 *
 * docker-compose.rust-harness.yaml depends on this flag: a container binding
 * its own loopback publishes a port nothing listens behind, because the
 * mapping forwards to the container's external interface. Nothing checked it.
 * cli_surface.rs pins the refusal when the flag arrives without --serve, which
 * is the case where no socket is ever opened -- so the address itself reaching
 * `TcpListener::bind` was untested in both directions.
 *
 * The model is the fixture checkpoint, since the assertions are about the
 * socket rather than the answers; /v1/health needs a loaded model and nothing
 * more.
 */

use std::io::{Read, Write};
use std::net::{Ipv4Addr, SocketAddr, TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

/// A second loopback address, distinct from the default the flag overrides.
/// Every 127.0.0.0/8 address is local on Linux, but a container or a hardened
/// host need not have it, which is what the bind probe below checks.
const ALTERNATE: Ipv4Addr = Ipv4Addr::new(127, 0, 0, 2);

/// TEST-NET-3 (RFC 5737), reserved for documentation and never assigned to an
/// interface. Binding it must fail, which is how the flag's value is shown to
/// reach `bind` rather than being parsed and discarded.
const NOT_LOCAL: &str = "203.0.113.1";

fn fixture_model() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../fixtures")
        .join("tiny-gemma4")
}

/// A port nothing is listening on. Bound and released rather than guessed, as
/// serve_interp_log.rs does: two test binaries running concurrently must not
/// collide.
fn free_port() -> u16 {
    TcpListener::bind((Ipv4Addr::LOCALHOST, 0))
        .unwrap()
        .local_addr()
        .unwrap()
        .port()
}

/// A spawned server that dies with the test, however the test ends.
///
/// `std::process::Child` does not kill on drop and every assertion runs
/// between the spawn and the wait, so a tripped assertion would otherwise
/// orphan a process holding the port this test just chose.
struct Server {
    child: std::process::Child,
    port: u16,
}

impl Drop for Server {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn spawn(addr: &str, dir: &Path) -> Server {
    spawn_with(Some(addr), dir)
}

/// `addr` of `None` omits `--serve-addr` entirely, which is the only way to
/// observe the default. Passing the default's own value instead would assert
/// nothing: deleting `default: Some("127.0.0.1")` from cli.rs would leave such
/// a test green, because the value it needs arrived on the command line.
fn spawn_with(addr: Option<&str>, dir: &Path) -> Server {
    let port = free_port();
    let mut cmd = std::process::Command::new(env!("CARGO_BIN_EXE_harness"));
    cmd.current_dir(dir)
        .arg("--model")
        .arg(fixture_model())
        .arg("--serve")
        .arg(port.to_string());
    if let Some(addr) = addr {
        cmd.arg("--serve-addr").arg(addr);
    }
    let child = cmd
        .stdin(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
        .expect("spawn harness");
    Server { child, port }
}

/// GET /v1/health against one address, with transport errors returned rather
/// than panicked on: a refused connection is this test's subject on one
/// address and its startup noise on the other.
fn health(addr: SocketAddr) -> std::io::Result<String> {
    let mut stream = TcpStream::connect_timeout(&addr, Duration::from_secs(5))?;
    stream.write_all(b"GET /v1/health HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n")?;
    stream.flush()?;
    let mut out = String::new();
    stream.read_to_string(&mut out)?;
    Ok(out)
}

/// Poll until the server answers on `addr`. A transport error retries: the
/// checkpoint load is fast but not instant, and a fixed sleep is either flaky
/// or slow.
fn wait_healthy(server: &mut Server, addr: SocketAddr) {
    let deadline = Instant::now() + Duration::from_secs(60);
    while Instant::now() < deadline {
        if let Ok(Some(status)) = server.child.try_wait() {
            panic!("harness exited before serving on {addr}: {status}");
        }
        if let Ok(response) = health(addr) {
            if response.contains(" 200 ") {
                return;
            }
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    panic!("harness never answered /v1/health on {addr}");
}

#[test]
fn the_endpoint_answers_on_the_bind_address_and_not_on_the_default() {
    if TcpListener::bind((ALTERNATE, 0)).is_err() {
        runtime::testlog::skip(
            "the alternate-loopback bind assertions",
            &format!("{ALTERNATE} cannot be bound on this host"),
        );
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let mut server = spawn(&ALTERNATE.to_string(), dir.path());
    let port = server.port;
    wait_healthy(&mut server, SocketAddr::from((ALTERNATE, port)));

    // The other half of the property, and the half a default-address server
    // would fail: a listener bound to one address does not accept on another,
    // so the same port on 127.0.0.1 has nothing behind it.
    let default = SocketAddr::from((Ipv4Addr::LOCALHOST, port));
    let answered = health(default);
    assert!(
        answered.is_err(),
        "a server bound to {ALTERNATE} answered on {default}: {:?}",
        answered.unwrap()
    );
    // The child is killed by the drop guard; the directory outlives it.
    drop(server);
}

/// The parse is not the wiring. An address the host does not own must fail at
/// `bind`, naming what it tried -- an accepted-then-ignored value would serve
/// happily on the default instead, which is the container failure this flag
/// exists to prevent, arriving silently.
#[test]
fn an_address_the_host_does_not_own_fails_the_bind_naming_it() {
    let dir = tempfile::tempdir().unwrap();
    let port = free_port();
    let out = std::process::Command::new(env!("CARGO_BIN_EXE_harness"))
        .current_dir(dir.path())
        .arg("--model")
        .arg(fixture_model())
        .arg("--serve")
        .arg(port.to_string())
        .arg("--serve-addr")
        .arg(NOT_LOCAL)
        .stdin(std::process::Stdio::null())
        .output()
        .expect("spawn harness");
    let err = String::from_utf8_lossy(&out.stderr).to_string();
    assert!(
        !out.status.success(),
        "the harness served on an address it cannot own: {err}"
    );
    assert!(
        err.contains(&format!("binding {NOT_LOCAL}:{port}")),
        "the failure must name the address it tried: {err}"
    );
}

/// This host's routable address, or `None` when it has none.
///
/// A connected UDP socket sends nothing; the kernel picks the source address
/// for the route and `local_addr` reads it back. That is the only way in `std`
/// to learn which interface a default route leaves by, and the assertion below
/// needs a concrete non-loopback address rather than a wildcard.
fn routable_addr() -> Option<Ipv4Addr> {
    let probe = std::net::UdpSocket::bind((Ipv4Addr::UNSPECIFIED, 0)).ok()?;
    probe.connect((Ipv4Addr::new(203, 0, 113, 1), 9)).ok()?;
    match probe.local_addr().ok()? {
        SocketAddr::V4(addr) if !addr.ip().is_loopback() => Some(*addr.ip()),
        _ => None,
    }
}

/// The default the flag overrides, and that it is loopback rather than the
/// wildcard.
///
/// Answering on 127.0.0.1 is necessary and nowhere near sufficient: a default
/// of 0.0.0.0 answers there too, so a test that stops at the health check
/// stays green under precisely the exposure this entry exists to catch -- a
/// checkpoint served to the local network by a run that named no address. The
/// discriminator is a second bind. With the harness on 127.0.0.1 the routable
/// address is free and binds; had it taken the wildcard, the same port on that
/// address is already held and the bind fails.
#[test]
fn the_default_bind_address_is_loopback() {
    let dir = tempfile::tempdir().unwrap();
    let mut server = spawn_with(None, dir.path());
    let port = server.port;
    wait_healthy(&mut server, SocketAddr::from((Ipv4Addr::LOCALHOST, port)));

    let Some(routable) = routable_addr() else {
        runtime::testlog::skip(
            "the wildcard-default assertion",
            "this host has no routable address to prove the default is not 0.0.0.0",
        );
        return;
    };
    let taken = TcpListener::bind((routable, port));
    assert!(
        taken.is_ok(),
        "the default bound {routable}:{port} as well as loopback, \
         so it is the wildcard rather than 127.0.0.1: {:?}",
        taken.err()
    );
}
