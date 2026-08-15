/*
 * status_python_parity — the six-tag table is one format, not two.
 *
 * `runtime/src/status.rs` reproduces `common/output.py`'s `Printer._status`
 * format from a transcribed table: six labels, six ANSI codes, an indent that
 * precedes the escape. A transcription is exactly the kind of claim that is true
 * on the day it is written and quietly false a year later, and a unit test
 * asserting the Rust output against a Rust literal cannot notice -- it is two
 * people typing the same thing.
 *
 * So this spawns the real `Printer` and compares bytes. It is the same shape as
 * `python_exporter_contract.rs`: probe for an interpreter that can import what
 * the test actually imports, and `[SKIP]` with a reason when the parent tree is
 * not present, rather than failing a gate for an absent dependency.
 *
 * Two environment details the comparison depends on:
 *
 *   - NO_COLOR=1. Both sides agree that a set, non-empty NO_COLOR disables
 *     colour; they disagree about the empty value (see the note in status.rs).
 *     Colour is also off anyway because the child's stderr is a pipe, so this is
 *     belt and braces against a future terminal-detection change on either side.
 *   - OBS_LOG_DIR pointed at a temporary directory. `Printer._status` emits an
 *     observability record before it prints (`common/output.py:54`), so a parity
 *     run writes to the log root as a side effect. It writes to a scratch
 *     directory here and not to the real one.
 */

use std::path::{Path, PathBuf};
use std::process::Command;

use runtime::status::{self, Tag};

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// An interpreter that can import the Printer, or a reason there is none.
///
/// `common.output` imports `common.obs.core` (output.py:14), so the probe names
/// both: an interpreter that resolves the first and not the second would pass
/// the probe and fail the spawn.
fn printer_python() -> Result<PathBuf, String> {
    let module = repo_root().join("common/output.py");
    if !module.is_file() {
        return Err(format!("{} is not present", module.display()));
    }
    runtime::pyenv::interpreter(&repo_root(), &["common.output", "common.obs.core"])
        .ok_or_else(|| "no interpreter can import common.output".to_string())
}

/// A prelude that makes the `Printer` believe stderr is a terminal.
///
/// `Printer.__init__` decides colour from `sys.stderr.isatty()`
/// (`common/output.py:38`), and a subprocess's stderr is a pipe, so a plain
/// spawn can only ever exercise the uncoloured half. Without this the six ANSI
/// codes are compared against nothing -- a mutation check confirmed a wrong
/// WARN code survived the first version of this test.
///
/// The wrapper delegates every write to the real stderr, so what the test reads
/// back is the Printer's own output, not a transcription of it. The import comes
/// after the patch because the decision is made in the constructor.
const TTY_PRELUDE: &str = "\
import sys
class _Tty:
    def __init__(self, s): self._s = s
    def isatty(self): return True
    def write(self, d): return self._s.write(d)
    def flush(self): return self._s.flush()
sys.stderr = _Tty(sys.stderr)
";

/// What the Python `Printer` writes to stderr for one call, as raw bytes.
fn python_line(
    python: &Path,
    log_dir: &Path,
    method: &str,
    message: &str,
    indent: usize,
    color: bool,
) -> String {
    // `{message:?}` renders a double-quoted, backslash-escaped literal, which is
    // valid Python for the ASCII messages this test uses. Asserted below rather
    // than assumed, since a non-ASCII message would escape as `\u{7f}` -- Rust
    // syntax that Python does not read.
    assert!(
        message.is_ascii() && !message.contains('\\'),
        "the message must survive transcription into a Python literal: {message:?}"
    );
    let prelude = if color { TTY_PRELUDE } else { "" };
    let script = format!(
        "{prelude}from common.output import Printer\n\
         Printer().{method}({message:?}, indent={indent})\n"
    );
    let mut command = Command::new(python);
    command
        .args(["-c", &script])
        .env("PYTHONPATH", repo_root())
        .env("OBS_LOG_DIR", log_dir);
    if color {
        // The Printer reads NO_COLOR from the environment; the coloured arm has
        // to be sure it is not inherited from whatever ran the gate.
        command.env_remove("NO_COLOR");
    } else {
        command.env("NO_COLOR", "1");
    }
    let out = command
        .output()
        .unwrap_or_else(|e| panic!("spawning {}: {e}", python.display()));
    assert!(
        out.status.success(),
        "Printer().{method} exited {}: {}",
        out.status,
        String::from_utf8_lossy(&out.stderr),
    );
    assert!(
        out.stdout.is_empty(),
        "the Printer wrote to stdout, which is reserved for machine output: {:?}",
        String::from_utf8_lossy(&out.stdout),
    );
    String::from_utf8(out.stderr).expect("the Printer writes UTF-8")
}

/// Every tag, and the `Printer` method that emits it.
const PAIRS: [(Tag, &str); 6] = [
    (Tag::Ok, "ok"),
    (Tag::Warn, "warn"),
    (Tag::Error, "error"),
    (Tag::Info, "info"),
    (Tag::Run, "run"),
    (Tag::Skip, "skip"),
];

#[test]
fn every_tag_renders_exactly_as_the_python_printer_does() {
    let python = match printer_python() {
        Ok(p) => p,
        Err(reason) => {
            runtime::testlog::skip("status parity with common/output.py", &reason);
            return;
        }
    };
    let log_dir = std::env::temp_dir().join("status-parity-obs");
    std::fs::create_dir_all(&log_dir).expect("a scratch log root");

    // A message with a space and a bracket, so a formatter that split on either
    // would show up here rather than passing on a single word.
    let message = "a message [with brackets]";
    for (tag, method) in PAIRS {
        for indent in [0usize, 2, 4] {
            // Both arms matter. The uncoloured one is what a runbook's Observed
            // block and every `grep` see; the coloured one is the only check the
            // six ANSI codes get -- a Rust unit test comparing them to Rust
            // literals is two people typing the same thing.
            for color in [false, true] {
                let want = python_line(&python, &log_dir, method, message, indent, color);
                let got = format!("{}\n", status::line(tag, message, indent, color));
                assert_eq!(
                    got, want,
                    "Printer().{method}(indent={indent}, color={color}) \
                     and status::line disagree"
                );
            }
        }
    }

    let _ = std::fs::remove_dir_all(&log_dir);
}
