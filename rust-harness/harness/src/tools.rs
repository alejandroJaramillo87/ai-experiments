/*
 * tools — the agent's tool set: read, write, edit, bash, glob, todo.
 *
 * Thin wrappers over std::fs and std::process::Command. Every file path is
 * resolved relative to the workspace root; results are returned as plain
 * strings for the transcript.
 */

use std::fs;
use std::io::Read;
use std::path::{Component, Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use anyhow::{bail, ensure, Context, Result};
use serde_json::Value;

use crate::util::Pattern;

// Bound on any tool result: unbounded output would otherwise flow straight
// into the transcript and the KV cache. Applies to bash and read alike.
const TOOL_OUTPUT_CAP: usize = 64 * 1024;

// Wall-clock bound on a bash command; null stdin already prevents interactive
// hangs, this catches genuinely long-running commands.
const BASH_TIMEOUT: Duration = Duration::from_secs(120);

#[derive(Debug, Clone, PartialEq)]
pub enum ToolCall {
    Read {
        path: String,
    },
    Write {
        path: String,
        content: String,
    },
    Edit {
        path: String,
        old: String,
        new: String,
    },
    Bash {
        command: String,
    },
    Glob {
        pattern: String,
    },
    Todo {
        items: Vec<String>,
    },
}

impl ToolCall {
    pub fn name(&self) -> &'static str {
        match self {
            ToolCall::Read { .. } => "read",
            ToolCall::Write { .. } => "write",
            ToolCall::Edit { .. } => "edit",
            ToolCall::Bash { .. } => "bash",
            ToolCall::Glob { .. } => "glob",
            ToolCall::Todo { .. } => "todo",
        }
    }

    /// The value permission patterns are matched against. Path targets are
    /// lexically normalized (`./a` and `b/../a` both become `a`), so a rule
    /// matches the filesystem target rather than one spelling of it; an
    /// unnormalizable path keeps its raw form and fails later in `resolve`.
    /// Commands and glob patterns are returned verbatim.
    pub fn permission_target(&self) -> std::borrow::Cow<'_, str> {
        match self {
            ToolCall::Read { path }
            | ToolCall::Write { path, .. }
            | ToolCall::Edit { path, .. } => match normalize_rel(path) {
                Ok(p) => std::borrow::Cow::Owned(p.to_string_lossy().into_owned()),
                Err(_) => std::borrow::Cow::Borrowed(path.as_str()),
            },
            _ => std::borrow::Cow::Borrowed(self.target()),
        }
    }

    /// The raw target as the model spelled it: a path, a command, or a
    /// pattern; empty when the tool has no natural target. Used for
    /// messages and logs; permission matching uses `permission_target`.
    pub fn target(&self) -> &str {
        match self {
            ToolCall::Read { path }
            | ToolCall::Write { path, .. }
            | ToolCall::Edit { path, .. } => path,
            ToolCall::Bash { command } => command,
            ToolCall::Glob { pattern } => pattern,
            ToolCall::Todo { .. } => "",
        }
    }

    /// Parse the wire form `{"tool": <name>, "args": {...}}`. Returns
    /// `Ok(None)` when `text` is not a tool call at all (a final answer) and
    /// an error when it is one but malformed.
    pub fn parse(text: &str) -> Result<Option<ToolCall>> {
        let trimmed = text.trim();
        if !trimmed.starts_with('{') {
            return Ok(None);
        }
        let v: Value = match serde_json::from_str(trimmed) {
            // Attempted-but-broken tool calls (truncated generation, trailing
            // prose) must not silently become the final answer.
            Err(e) if trimmed.contains("\"tool\"") => {
                bail!("malformed tool call: {e}")
            }
            Err(_) => return Ok(None),
            Ok(v) => v,
        };
        let Some(name) = v["tool"].as_str() else {
            ensure!(
                v.get("tool").is_none(),
                "malformed tool call: \"tool\" is not a string"
            );
            return Ok(None);
        };
        let args = &v["args"];
        let s = |key: &str| -> Result<String> {
            args[key]
                .as_str()
                .map(str::to_string)
                .with_context(|| format!("tool {name}: missing string arg {key}"))
        };
        let call = match name {
            "read" => ToolCall::Read { path: s("path")? },
            "write" => ToolCall::Write {
                path: s("path")?,
                content: s("content")?,
            },
            "edit" => ToolCall::Edit {
                path: s("path")?,
                old: s("old")?,
                new: s("new")?,
            },
            "bash" => ToolCall::Bash {
                command: s("command")?,
            },
            "glob" => ToolCall::Glob {
                pattern: s("pattern")?,
            },
            "todo" => ToolCall::Todo {
                items: args["items"]
                    .as_array()
                    .context("tool todo: missing items array")?
                    .iter()
                    .map(|i| {
                        i.as_str()
                            .map(str::to_string)
                            .context("non-string todo item")
                    })
                    .collect::<Result<_>>()?,
            },
            other => bail!("unknown tool: {other}"),
        };
        Ok(Some(call))
    }

    pub fn execute(&self, root: &Path) -> Result<String> {
        let span = tracing::info_span!(
            "tool.exec",
            tool = self.name(),
            target = self.target(),
            output_bytes = tracing::field::Empty
        );
        let _enter = span.enter();
        let result = self.execute_inner(root);
        if let Ok(out) = &result {
            span.record("output_bytes", out.len());
        }
        result
    }

    fn execute_inner(&self, root: &Path) -> Result<String> {
        match self {
            ToolCall::Read { path } => {
                let text = fs::read_to_string(resolve(root, path)?)
                    .with_context(|| format!("read {path}"))?;
                Ok(cap_output(text))
            }
            ToolCall::Write { path, content } => {
                let full = resolve(root, path)?;
                if let Some(dir) = full.parent() {
                    fs::create_dir_all(dir)?;
                }
                fs::write(&full, content).with_context(|| format!("write {path}"))?;
                Ok(format!("wrote {} bytes to {path}", content.len()))
            }
            ToolCall::Edit { path, old, new } => {
                let full = resolve(root, path)?;
                let text = fs::read_to_string(&full).with_context(|| format!("edit {path}"))?;
                let count = overlapping_count(&text, old);
                ensure!(count != 0, "edit {path}: old text not found");
                ensure!(
                    count == 1,
                    "edit {path}: old text matches {count} times, need 1"
                );
                fs::write(&full, text.replacen(old.as_str(), new, 1))?;
                Ok(format!("edited {path}"))
            }
            ToolCall::Bash { command } => bash_with_timeout(command, root, BASH_TIMEOUT),
            ToolCall::Glob { pattern } => {
                let compiled = Pattern::compile(pattern);
                let mut matches = Vec::new();
                walk(root, root, &compiled, &mut matches);
                matches.sort();
                Ok(matches.join("\n"))
            }
            ToolCall::Todo { items } => Ok(items
                .iter()
                .enumerate()
                .map(|(i, item)| format!("{}. {item}", i + 1))
                .collect::<Vec<_>>()
                .join("\n")),
        }
    }
}

/// Truncate a tool result to the transcript cap at a char boundary.
fn cap_output(mut text: String) -> String {
    if text.len() > TOOL_OUTPUT_CAP {
        let mut cut = TOOL_OUTPUT_CAP;
        while !text.is_char_boundary(cut) {
            cut -= 1;
        }
        text.truncate(cut);
        text.push_str("\n[output truncated]");
    }
    text
}

/// Run a shell command with null stdin (interactive commands fail fast
/// instead of hanging) and a wall-clock deadline. Output is drained on
/// threads so a chatty command cannot deadlock on a full pipe while the
/// parent polls for exit; on timeout the child is killed and the partial
/// output is returned with a note.
fn bash_with_timeout(command: &str, root: &Path, timeout: Duration) -> Result<String> {
    let mut cmd = Command::new("sh");
    cmd.arg("-c")
        .arg(command)
        .current_dir(root)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    // Own process group: on timeout the whole group is killed, so children
    // of the shell cannot outlive it holding the output pipes open.
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        cmd.process_group(0);
    }
    let mut child = cmd.spawn().with_context(|| format!("bash: {command}"))?;
    let drain = |pipe: Option<Box<dyn Read + Send>>| {
        std::thread::spawn(move || {
            let mut buf = Vec::new();
            if let Some(mut pipe) = pipe {
                let _ = pipe.read_to_end(&mut buf);
            }
            buf
        })
    };
    let stdout = drain(child.stdout.take().map(|p| Box::new(p) as _));
    let stderr = drain(child.stderr.take().map(|p| Box::new(p) as _));
    let deadline = Instant::now() + timeout;
    let (status, timed_out) = loop {
        if let Some(status) = child.try_wait().context("waiting for bash child")? {
            break (Some(status), false);
        }
        if Instant::now() >= deadline {
            // Shell-builtin kill: signals the whole group reliably even
            // where an external kill binary is namespaced away from it.
            #[cfg(unix)]
            let _ = Command::new("sh")
                .arg("-c")
                .arg(format!("kill -KILL -{}", child.id()))
                .status();
            let _ = child.kill();
            let _ = child.wait();
            break (None, true);
        }
        std::thread::sleep(Duration::from_millis(10));
    };
    let mut bytes = stdout.join().expect("stdout drain thread");
    bytes.extend(stderr.join().expect("stderr drain thread"));
    let mut text = String::from_utf8_lossy(&bytes).into_owned();
    text = cap_output(text);
    if timed_out {
        text.push_str(&format!(
            "\n[command killed after {} s timeout]",
            timeout.as_secs_f32()
        ));
    } else if let Some(status) = status {
        if !status.success() {
            text.push_str(&format!("\nexit status: {status}"));
        }
    }
    Ok(text)
}

/// Control files the model may never write: `permissions.yaml` is the rule
/// file gating its own tool use, and `session.jsonl` is the transcript whose
/// role fields resume trusts. Both live in the workspace root by design, so
/// without this guard one allowed write escalates to permanent full allow
/// (next process start) or forged conversation turns.
const PROTECTED_FILES: [&str; 2] = ["permissions.yaml", "session.jsonl"];

/// Lexically normalize a workspace-relative path: reject absolute paths and
/// any `..` that climbs out of the root, and collapse `.` and `..` so one
/// filesystem target has exactly one spelling. Permission rules are matched
/// against this form, so a deny pattern cannot be dodged by writing
/// `./secrets/x` or `src/../secrets/x`.
pub(crate) fn normalize_rel(path: &str) -> Result<PathBuf> {
    let mut parts: Vec<std::ffi::OsString> = Vec::new();
    for comp in Path::new(path).components() {
        match comp {
            Component::Normal(c) => parts.push(c.to_os_string()),
            Component::CurDir => {}
            Component::ParentDir => {
                ensure!(parts.pop().is_some(), "path escapes workspace root: {path}");
            }
            Component::RootDir | Component::Prefix(_) => {
                bail!("path escapes workspace root: {path}")
            }
        }
    }
    Ok(parts.iter().collect())
}

/// Resolve a model-supplied path strictly inside `root`.
///
/// Two layers: the lexical one above rejects absolute paths and climbing
/// `..`, and a canonical one closes the symlink hole — lexical checks alone
/// are satisfied by `link/x` where `link` points outside the root, and the
/// file tools use symlink-following IO. The deepest existing ancestor is
/// canonicalized and required to stay under the canonical root; the
/// not-yet-existing tail is re-joined afterwards so writing a new file
/// still works.
fn resolve(root: &Path, path: &str) -> Result<PathBuf> {
    let rel = normalize_rel(path)?;
    ensure!(
        !rel.components().any(|c| matches!(c, Component::Normal(name)
            if PROTECTED_FILES.contains(&name.to_string_lossy().as_ref()))),
        "refusing to touch harness control file: {path}"
    );
    let full = root.join(&rel);

    // Canonicalize the deepest existing ancestor; everything below it does
    // not exist yet and therefore cannot be a symlink.
    let mut existing = full.as_path();
    let mut tail = PathBuf::new();
    loop {
        if existing.exists() {
            break;
        }
        let parent = existing
            .parent()
            .ok_or_else(|| anyhow::anyhow!("path escapes workspace root: {path}"))?;
        let name = existing
            .file_name()
            .ok_or_else(|| anyhow::anyhow!("path escapes workspace root: {path}"))?;
        tail = if tail.as_os_str().is_empty() {
            PathBuf::from(name)
        } else {
            Path::new(name).join(&tail)
        };
        existing = parent;
    }
    let canonical_root = root
        .canonicalize()
        .with_context(|| format!("workspace root: {}", root.display()))?;
    let canonical = existing
        .canonicalize()
        .with_context(|| format!("resolve {path}"))?;
    ensure!(
        canonical.starts_with(&canonical_root),
        "path escapes workspace root through a symlink: {path}"
    );
    // Joining an empty tail would append a trailing separator, turning a
    // file path into a directory path.
    if tail.as_os_str().is_empty() {
        Ok(canonical)
    } else {
        Ok(canonical.join(tail))
    }
}

/// Count occurrences of `needle` including overlapping ones: `edit`'s
/// exactly-one guarantee must treat "aa" in "aaa" as ambiguous.
fn overlapping_count(hay: &str, needle: &str) -> usize {
    if needle.is_empty() {
        return 0;
    }
    let mut count = 0;
    let mut from = 0;
    while let Some(at) = hay[from..].find(needle) {
        count += 1;
        let start = from + at;
        let first = hay[start..].chars().next().expect("match is non-empty");
        from = start + first.len_utf8();
    }
    count
}

fn walk(root: &Path, dir: &Path, pattern: &Pattern, out: &mut Vec<String>) {
    // Unreadable directories are skipped, not fatal: one bad permission bit
    // must not abort the whole glob.
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path: PathBuf = entry.path();
        if path
            .file_name()
            .is_some_and(|n| n.to_string_lossy().starts_with('.'))
        {
            continue; // hidden entries (.git and friends)
        }
        // symlink_metadata: never follow links, so a cycle cannot recurse
        // forever and a link outside the root cannot leak paths.
        let Ok(meta) = fs::symlink_metadata(&path) else {
            continue;
        };
        let rel = path.strip_prefix(root).expect("walk stays under root");
        let rel_str = rel.to_string_lossy();
        if meta.is_dir() {
            walk(root, &path, pattern, out);
        } else if meta.is_file() && pattern.matches(&rel_str) {
            out.push(rel_str.into_owned());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn root() -> tempfile::TempDir {
        tempfile::tempdir().unwrap()
    }

    #[test]
    fn write_then_read_round_trips() {
        let dir = root();
        ToolCall::Write {
            path: "a/b.txt".into(),
            content: "hello".into(),
        }
        .execute(dir.path())
        .unwrap();
        let text = ToolCall::Read {
            path: "a/b.txt".into(),
        }
        .execute(dir.path())
        .unwrap();
        assert_eq!(text, "hello");
    }

    #[test]
    fn symlinks_pointing_out_of_the_root_are_refused() {
        // Regression: resolve was lexical only, so `escape/passwd` satisfied
        // the traversal check while the symlink-following IO read outside.
        let dir = root();
        let outside = tempfile::tempdir().unwrap();
        std::fs::write(outside.path().join("secret.txt"), "classified").unwrap();
        std::os::unix::fs::symlink(outside.path(), dir.path().join("escape")).unwrap();

        for call in [
            ToolCall::Read {
                path: "escape/secret.txt".into(),
            },
            ToolCall::Write {
                path: "escape/new.txt".into(),
                content: "x".into(),
            },
            ToolCall::Edit {
                path: "escape/secret.txt".into(),
                old: "classified".into(),
                new: "leaked".into(),
            },
        ] {
            let err = call.execute(dir.path()).unwrap_err();
            assert!(
                format!("{err:#}").contains("symlink"),
                "expected a symlink refusal, got: {err:#}"
            );
        }
        // The outside file is untouched and no new file was planted.
        assert_eq!(
            std::fs::read_to_string(outside.path().join("secret.txt")).unwrap(),
            "classified"
        );
        assert!(!outside.path().join("new.txt").exists());
    }

    #[test]
    fn symlinks_staying_inside_the_root_still_work() {
        let dir = root();
        std::fs::create_dir(dir.path().join("real")).unwrap();
        std::fs::write(dir.path().join("real/f.txt"), "inside").unwrap();
        std::os::unix::fs::symlink(dir.path().join("real"), dir.path().join("link")).unwrap();
        let text = ToolCall::Read {
            path: "link/f.txt".into(),
        }
        .execute(dir.path())
        .unwrap();
        assert_eq!(text, "inside");
    }

    #[test]
    fn harness_control_files_cannot_be_written() {
        // permissions.yaml gates the model's own tool use and session.jsonl
        // carries the trusted role fields; neither is model-writable.
        let dir = root();
        std::fs::write(dir.path().join("permissions.yaml"), "rules: []").unwrap();
        for path in ["permissions.yaml", "./permissions.yaml", "session.jsonl"] {
            let err = ToolCall::Write {
                path: path.into(),
                content: "rules:\n  - tool: \"*\"\n    action: allow\n".into(),
            }
            .execute(dir.path())
            .unwrap_err();
            assert!(
                format!("{err:#}").contains("control file"),
                "expected a control-file refusal for {path}, got: {err:#}"
            );
        }
        assert_eq!(
            std::fs::read_to_string(dir.path().join("permissions.yaml")).unwrap(),
            "rules: []"
        );
    }

    #[test]
    fn permission_targets_are_normalized_to_one_spelling() {
        // Regression: rules matched the raw spelling, so `./secrets/x` and
        // `src/../secrets/x` both dodged a `secrets/**` deny rule.
        for spelling in ["secrets/k.txt", "./secrets/k.txt", "src/../secrets/k.txt"] {
            let call = ToolCall::Read {
                path: spelling.into(),
            };
            assert_eq!(call.permission_target(), "secrets/k.txt");
        }
        // Commands are matched verbatim, not path-normalized.
        let bash = ToolCall::Bash {
            command: "./x.sh".into(),
        };
        assert_eq!(bash.permission_target(), "./x.sh");
    }

    #[test]
    fn read_missing_file_is_an_error_with_path_context() {
        let dir = root();
        let err = ToolCall::Read {
            path: "no.txt".into(),
        }
        .execute(dir.path())
        .unwrap_err();
        assert!(format!("{err:#}").contains("no.txt"));
    }

    #[test]
    fn edit_requires_exactly_one_match() {
        let dir = root();
        std::fs::write(dir.path().join("f.txt"), "aa bb aa").unwrap();
        let ambiguous = ToolCall::Edit {
            path: "f.txt".into(),
            old: "aa".into(),
            new: "x".into(),
        };
        assert!(ambiguous.execute(dir.path()).is_err());
        let missing = ToolCall::Edit {
            path: "f.txt".into(),
            old: "zz".into(),
            new: "x".into(),
        };
        assert!(missing.execute(dir.path()).is_err());
        ToolCall::Edit {
            path: "f.txt".into(),
            old: "bb".into(),
            new: "cc".into(),
        }
        .execute(dir.path())
        .unwrap();
        assert_eq!(
            std::fs::read_to_string(dir.path().join("f.txt")).unwrap(),
            "aa cc aa"
        );
    }

    #[test]
    fn bash_captures_output_and_exit_status() {
        let dir = root();
        let ok = ToolCall::Bash {
            command: "echo hi".into(),
        }
        .execute(dir.path())
        .unwrap();
        assert_eq!(ok.trim(), "hi");
        let fail = ToolCall::Bash {
            command: "exit 3".into(),
        }
        .execute(dir.path())
        .unwrap();
        assert!(fail.contains("exit status"));
    }

    #[test]
    fn glob_matches_relative_paths_recursively() {
        let dir = root();
        std::fs::create_dir_all(dir.path().join("src/model")).unwrap();
        std::fs::write(dir.path().join("src/lib.rs"), "").unwrap();
        std::fs::write(dir.path().join("src/model/mod.rs"), "").unwrap();
        std::fs::write(dir.path().join("README.md"), "").unwrap();
        let out = ToolCall::Glob {
            pattern: "**/*.rs".into(),
        }
        .execute(dir.path())
        .unwrap();
        assert_eq!(out, "src/lib.rs\nsrc/model/mod.rs");
    }

    #[test]
    fn todo_renders_numbered_list() {
        let dir = root();
        let out = ToolCall::Todo {
            items: vec!["first".into(), "second".into()],
        }
        .execute(dir.path())
        .unwrap();
        assert_eq!(out, "1. first\n2. second");
    }

    #[test]
    fn paths_outside_the_root_are_rejected() {
        let dir = root();
        std::fs::write(dir.path().join("ok.txt"), "fine").unwrap();
        for path in [
            "/etc/passwd",
            "../secret",
            "a/../../secret",
            "a/b/../../../x",
        ] {
            let err = ToolCall::Read { path: path.into() }
                .execute(dir.path())
                .unwrap_err();
            assert!(
                format!("{err:#}").contains("escapes workspace root"),
                "{path}"
            );
        }
        // `..` that stays inside the root is fine.
        std::fs::create_dir(dir.path().join("sub")).unwrap();
        let text = ToolCall::Read {
            path: "sub/../ok.txt".into(),
        }
        .execute(dir.path())
        .unwrap();
        assert_eq!(text, "fine");
    }

    #[test]
    fn overlapping_matches_count_as_ambiguous() {
        let dir = root();
        std::fs::write(dir.path().join("f.txt"), "aaa").unwrap();
        // Non-overlapping counting sees one "aa" in "aaa"; there are two
        // candidate positions, so the edit must refuse.
        let err = ToolCall::Edit {
            path: "f.txt".into(),
            old: "aa".into(),
            new: "x".into(),
        }
        .execute(dir.path())
        .unwrap_err();
        assert!(err.to_string().contains("2 times"));
    }

    #[test]
    fn bash_output_is_capped() {
        let dir = root();
        let out = ToolCall::Bash {
            command: "head -c 200000 /dev/zero | tr '\\0' 'x'".into(),
        }
        .execute(dir.path())
        .unwrap();
        assert!(out.len() < 200_000);
        assert!(out.contains("[output truncated]"));
    }

    #[test]
    fn read_output_is_capped() {
        let dir = root();
        std::fs::write(dir.path().join("big.txt"), "x".repeat(200_000)).unwrap();
        let out = ToolCall::Read {
            path: "big.txt".into(),
        }
        .execute(dir.path())
        .unwrap();
        assert!(out.len() < 200_000);
        assert!(out.contains("[output truncated]"));
    }

    #[test]
    fn bash_timeout_kills_the_command_and_reports_it() {
        let dir = root();
        let started = std::time::Instant::now();
        let out = bash_with_timeout(
            "echo partial; sleep 30",
            dir.path(),
            Duration::from_millis(300),
        )
        .unwrap();
        assert!(started.elapsed() < Duration::from_secs(10));
        assert!(out.contains("partial"), "{out}");
        assert!(out.contains("timeout"), "{out}");
    }

    #[test]
    fn bash_stdin_is_closed_so_interactive_commands_finish() {
        let dir = root();
        // With inherited stdin this would block awaiting input.
        let out = ToolCall::Bash {
            command: "read line; echo done".into(),
        }
        .execute(dir.path())
        .unwrap();
        assert!(out.contains("done"));
    }

    #[cfg(unix)]
    #[test]
    fn glob_ignores_symlink_cycles_and_hidden_dirs() {
        let dir = root();
        std::fs::create_dir_all(dir.path().join("src")).unwrap();
        std::fs::create_dir_all(dir.path().join(".git")).unwrap();
        std::fs::write(dir.path().join("src/lib.rs"), "").unwrap();
        std::fs::write(dir.path().join(".git/config.rs"), "").unwrap();
        // Symlink back to the root: naive walking recurses forever.
        std::os::unix::fs::symlink(dir.path(), dir.path().join("src/loop")).unwrap();
        let out = ToolCall::Glob {
            pattern: "**/*.rs".into(),
        }
        .execute(dir.path())
        .unwrap();
        assert_eq!(out, "src/lib.rs");
    }

    #[test]
    fn malformed_tool_calls_are_errors_not_final_answers() {
        // Truncated generation of a tool call.
        assert!(ToolCall::parse(r#"{"tool": "bash", "args": {"comm"#).is_err());
        // Valid call followed by trailing prose.
        assert!(ToolCall::parse("{\"tool\": \"read\", \"args\": {\"path\": \"a\"}} done").is_err());
        // Non-string tool key.
        assert!(ToolCall::parse(r#"{"tool": 3, "args": {}}"#).is_err());
        // JSON-looking text without a tool key stays a final answer.
        assert_eq!(ToolCall::parse(r#"{"summary": "done"}"#).unwrap(), None);
    }

    #[test]
    fn parse_distinguishes_tool_calls_from_final_text() {
        let call = ToolCall::parse(r#"{"tool": "read", "args": {"path": "a.txt"}}"#).unwrap();
        assert_eq!(
            call,
            Some(ToolCall::Read {
                path: "a.txt".into()
            })
        );
        assert_eq!(ToolCall::parse("The task is complete.").unwrap(), None);
        assert!(ToolCall::parse(r#"{"tool": "launch_missiles", "args": {}}"#).is_err());
        assert!(ToolCall::parse(r#"{"tool": "read", "args": {}}"#).is_err());
    }
}
