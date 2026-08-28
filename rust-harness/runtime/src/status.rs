/*
 * status — the six-tag status line, shared by every binary in the workspace.
 *
 * The workspace prints status through 130-odd hand-written `eprintln!` calls
 * with the tag typed as a literal at each site. Nothing enforces the format,
 * nothing gates the colour, and five of them are on stdout, where a status line
 * corrupts the machine payload. This module is the one place the format lives.
 *
 * The format is byte-for-byte `common/output.py`'s `Printer._status` (:64):
 *
 *     {pad}{color}[{TAG}]{reset} {message}
 *
 * The indent precedes the escape, so colour never wraps the padding, and the
 * colour covers the bracketed tag only. `runtime/tests/status_python_parity.rs`
 * spawns the real `Printer` and compares bytes, so the two tables cannot drift
 * into being two people typing the same thing.
 *
 * Colour is display only: the uncoloured text is byte-identical, so `grep
 * '\[WARN\]'` finds a coloured line on a terminal and a plain one through a
 * pipe. It is suppressed when NO_COLOR is set or stderr is not a terminal.
 *
 * NO_COLOR divergence, recorded rather than papered over: this checks presence
 * (`var_os(..).is_none()`), so `NO_COLOR=` with an empty value disables colour.
 * `common/output.py:38` checks truthiness, so the same empty value leaves colour
 * on there. Presence is what no-color.org specifies, which makes Python the
 * off-spec side; changing a stable Python tier four CLIs depend on is not this
 * module's business. Both agree on every other input, including the `NO_COLOR=1`
 * the parity test uses.
 *
 * Every emitter writes stderr. There is no `println!` in this file and there
 * must never be one -- a reviewer checking the stdout guarantee by eye should
 * have exactly one file to read.
 */

use std::io::IsTerminal;
use std::sync::OnceLock;

/// ANSI reset, closing the escape opened around the tag.
const RESET: &str = "\x1b[0m";

/// The status indicators, per the table in `.styles/TEST_OUTPUT_STYLE.md`.
///
/// Six and no more. `[FAIL]` is deliberately absent: `.styles/RUST_STYLE.md`
/// reserves failure reporting for `assert!` and panics, so a seventh tag would
/// give tests a second, quieter way to fail.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tag {
    Ok,
    Warn,
    Error,
    Info,
    Run,
    Skip,
}

impl Tag {
    /// The text inside the brackets.
    pub fn label(self) -> &'static str {
        match self {
            Tag::Ok => "OK",
            Tag::Warn => "WARN",
            Tag::Error => "ERROR",
            Tag::Info => "INFO",
            Tag::Run => "RUN",
            Tag::Skip => "SKIP",
        }
    }

    /// The opening escape for this tag.
    ///
    /// The codes are `common/output.py:40-48` verbatim. Note WARN is `1;33`
    /// (bold yellow) where the other five are `0;3x`, and that RUN and SKIP are
    /// the same blue -- both are the Python table's shape, not a transcription
    /// slip, and a parity run is what says so.
    pub fn color(self) -> &'static str {
        match self {
            Tag::Ok => "\x1b[0;32m",
            Tag::Warn => "\x1b[1;33m",
            Tag::Error => "\x1b[0;31m",
            Tag::Info => "\x1b[0;36m",
            Tag::Run => "\x1b[0;34m",
            Tag::Skip => "\x1b[0;34m",
        }
    }
}

/// The formatted line, colourised or not.
///
/// Pure, and split from the emitters for the reason `testlog` split its own: the
/// formatting is then testable without asserting anything about the environment
/// the suite happens to run in -- under `--nocapture` stderr really is a
/// terminal. It is also public because two call sites in `harness/src/main.rs`
/// build a status string and return it rather than printing at the site.
pub fn line(tag: Tag, message: &str, indent: usize, color: bool) -> String {
    let pad = " ".repeat(indent);
    let label = tag.label();
    if color {
        format!("{pad}{}[{label}]{RESET} {message}", tag.color())
    } else {
        format!("{pad}[{label}] {message}")
    }
}

/// Whether stderr is a terminal.
///
/// The workspace's single terminal probe. Anything else wanting to know -- a
/// progress animation, say -- asks here, so there is one answer rather than
/// several that can disagree.
pub fn stderr_is_terminal() -> bool {
    std::io::stderr().is_terminal()
}

/// Whether status lines carry colour, decided once per process.
///
/// Cached because it must not flap mid-run: a run whose first half is coloured
/// and second half is not produces output no reader can filter, and the inputs
/// (an environment variable, a file descriptor's kind) do not meaningfully
/// change under a running process anyway.
pub fn color_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("NO_COLOR").is_none() && stderr_is_terminal())
}

/// Write one status line to stderr.
fn emit(tag: Tag, message: &str, indent: usize) {
    eprintln!("{}", line(tag, message, indent, color_enabled()));
}

/// Report success.
pub fn ok(message: &str) {
    emit(Tag::Ok, message, 0);
}

/// Report success, indented under the line it belongs to.
pub fn ok_at(message: &str, indent: usize) {
    emit(Tag::Ok, message, indent);
}

/// Report a condition worth noticing that does not stop the run.
pub fn warn(message: &str) {
    emit(Tag::Warn, message, 0);
}

/// Report a warning, indented under the line it belongs to.
pub fn warn_at(message: &str, indent: usize) {
    emit(Tag::Warn, message, indent);
}

/// Report a failure.
pub fn error(message: &str) {
    emit(Tag::Error, message, 0);
}

/// Report a failure, indented under the line it belongs to.
pub fn error_at(message: &str, indent: usize) {
    emit(Tag::Error, message, indent);
}

/// State a fact about the run.
pub fn info(message: &str) {
    emit(Tag::Info, message, 0);
}

/// State a fact, indented under the line it belongs to.
pub fn info_at(message: &str, indent: usize) {
    emit(Tag::Info, message, indent);
}

/// Announce work about to start.
pub fn run(message: &str) {
    emit(Tag::Run, message, 0);
}

/// Announce work about to start, indented under the line it belongs to.
pub fn run_at(message: &str, indent: usize) {
    emit(Tag::Run, message, indent);
}

/// Report something not done, and why.
///
/// Test code calls `testlog::skip` instead, which composes the subject/reason
/// message shape `.styles/RUST_STYLE.md` requires of a skipped contract test.
pub fn skip(message: &str) {
    emit(Tag::Skip, message, 0);
}

/// Report something not done, indented under the line it belongs to.
pub fn skip_at(message: &str, indent: usize) {
    emit(Tag::Skip, message, indent);
}

/// An untagged line on stderr, for headers and blank spacing.
///
/// `Printer.line()`'s counterpart. Untagged output is still status, not payload,
/// so it goes to stderr with everything else here.
pub fn plain(message: &str) {
    eprintln!("{message}");
}

#[cfg(test)]
mod tests {
    use super::*;

    const ALL: [Tag; 6] = [
        Tag::Ok,
        Tag::Warn,
        Tag::Error,
        Tag::Info,
        Tag::Run,
        Tag::Skip,
    ];

    #[test]
    fn an_uncoloured_line_carries_no_escape_bytes() {
        for tag in ALL {
            let got = line(tag, "a message", 0, false);
            assert!(!got.contains('\x1b'), "{got:?}");
        }
    }

    /// The text either side of the escapes is byte-identical to the plain form,
    /// so `grep '\[WARN\]'` finds a coloured line too. The offset is computed
    /// from the label rather than written as a literal: `testlog` could hard-code
    /// 6 because `[SKIP]` was the only tag, but `[OK]` is 4 and `[ERROR]` is 7.
    #[test]
    fn a_coloured_line_wraps_the_same_text_and_resets() {
        for tag in ALL {
            let plain = line(tag, "a message", 0, false);
            let colored = line(tag, "a message", 0, true);
            // The suffix after the bracketed tag, whose length varies per tag.
            let after_tag = &plain[tag.label().len() + "[]".len()..];
            assert!(colored.ends_with(after_tag), "{colored:?}");
            assert!(colored.starts_with(tag.color()), "{colored:?}");
            assert!(colored.contains(RESET), "{colored:?}");
            assert_eq!(
                colored.replace(tag.color(), "").replace(RESET, ""),
                plain,
                "stripping the escapes must yield the plain form exactly"
            );
        }
    }

    /// The literal shape of `common/output.py:64`, so a reader can compare the
    /// two files without running anything. `status_python_parity.rs` is what
    /// proves it, but a byte string here is what makes the claim reviewable.
    #[test]
    fn the_uncoloured_form_matches_the_python_format() {
        assert_eq!(line(Tag::Ok, "x", 0, false), "[OK] x");
        assert_eq!(line(Tag::Error, "boom", 0, false), "[ERROR] boom");
    }

    /// The pad precedes the escape, so colour never wraps the indent -- a
    /// coloured indented line must still start with spaces.
    #[test]
    fn the_indent_precedes_the_colour_escape() {
        assert_eq!(line(Tag::Info, "x", 4, false), "    [INFO] x");
        let colored = line(Tag::Info, "x", 4, true);
        assert!(colored.starts_with("    \x1b["), "{colored:?}");
    }

    /// WARN is the one code that breaks the `0;3x` pattern, and the one most
    /// likely to be "corrected" into it by someone tidying the table.
    #[test]
    fn warn_is_the_only_bold_code() {
        assert_eq!(Tag::Warn.color(), "\x1b[1;33m");
        for tag in ALL {
            if tag != Tag::Warn {
                assert!(tag.color().starts_with("\x1b[0;3"), "{tag:?}");
            }
        }
    }

    /// RUN and SKIP share a blue in `common/output.py`, so they share one here.
    #[test]
    fn run_and_skip_share_the_python_blue() {
        assert_eq!(Tag::Run.color(), Tag::Skip.color());
    }

    #[test]
    fn every_tag_has_a_distinct_label() {
        let mut labels: Vec<&str> = ALL.iter().map(|t| t.label()).collect();
        labels.sort_unstable();
        let count = labels.len();
        labels.dedup();
        assert_eq!(labels.len(), count, "two tags share a label");
    }

    // Deliberately no test that calls the emitters: their output would land in
    // the gate as status lines naming nothing real, and a reader scanning for
    // [SKIP] would have to know to discount them. The formatting is covered
    // above; the emitters add only the stderr write.
}
