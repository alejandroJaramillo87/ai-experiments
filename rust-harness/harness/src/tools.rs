/*
 * tools — the agent's tool set: read, write, edit, bash, glob, todo.
 *
 * Thin wrappers over std::fs and std::process::Command. Every file path is
 * resolved relative to the workspace root; results are returned as plain
 * strings for the transcript.
 */

use std::fs;
use std::path::{Component, Path, PathBuf};
use std::process::{Command, Stdio};

use anyhow::{bail, ensure, Context, Result};
use serde_json::Value;

use crate::util::Pattern;

// Bound on captured bash output: unbounded output would otherwise flow
// straight into the transcript and the KV cache.
const BASH_OUTPUT_CAP: usize = 64 * 1024;

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

    /// The value permission patterns match against: a path, a command, or a
    /// pattern; empty when the tool has no natural target.
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
        match self {
            ToolCall::Read { path } => {
                fs::read_to_string(resolve(root, path)?).with_context(|| format!("read {path}"))
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
            ToolCall::Bash { command } => {
                // Null stdin: an interactive command fails fast instead of
                // hanging the loop. TODO: add a wall-clock timeout once the
                // loop is async (needs a process-wait-with-deadline).
                let out = Command::new("sh")
                    .arg("-c")
                    .arg(command)
                    .current_dir(root)
                    .stdin(Stdio::null())
                    .output()
                    .with_context(|| format!("bash: {command}"))?;
                let mut text = String::from_utf8_lossy(&out.stdout).into_owned();
                text.push_str(&String::from_utf8_lossy(&out.stderr));
                if text.len() > BASH_OUTPUT_CAP {
                    let mut cut = BASH_OUTPUT_CAP;
                    while !text.is_char_boundary(cut) {
                        cut -= 1;
                    }
                    text.truncate(cut);
                    text.push_str("\n[output truncated]");
                }
                if !out.status.success() {
                    text.push_str(&format!("\nexit status: {}", out.status));
                }
                Ok(text)
            }
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

/// Resolve a model-supplied path strictly inside `root`. Lexical only:
/// absolute paths and any `..` that would climb out of the root are errors,
/// so no permission rule needs to anticipate traversal spellings.
fn resolve(root: &Path, path: &str) -> Result<PathBuf> {
    let rel = Path::new(path);
    let mut depth: usize = 0;
    for comp in rel.components() {
        match comp {
            Component::Normal(_) => depth += 1,
            Component::CurDir => {}
            Component::ParentDir => {
                ensure!(depth > 0, "path escapes workspace root: {path}");
                depth -= 1;
            }
            Component::RootDir | Component::Prefix(_) => {
                bail!("path escapes workspace root: {path}")
            }
        }
    }
    Ok(root.join(rel))
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
