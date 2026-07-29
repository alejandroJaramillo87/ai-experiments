/*
 * tools — the agent's tool set: read, write, edit, bash, glob, todo.
 *
 * Thin wrappers over std::fs and std::process::Command. Every file path is
 * resolved relative to the workspace root; results are returned as plain
 * strings for the transcript.
 */

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{bail, ensure, Context, Result};
use serde_json::Value;

use crate::util::wildcard_match;

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
            Ok(v) => v,
            Err(_) => return Ok(None),
        };
        let Some(name) = v["tool"].as_str() else {
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
                fs::read_to_string(root.join(path)).with_context(|| format!("read {path}"))
            }
            ToolCall::Write { path, content } => {
                let full = root.join(path);
                if let Some(dir) = full.parent() {
                    fs::create_dir_all(dir)?;
                }
                fs::write(&full, content).with_context(|| format!("write {path}"))?;
                Ok(format!("wrote {} bytes to {path}", content.len()))
            }
            ToolCall::Edit { path, old, new } => {
                let full = root.join(path);
                let text = fs::read_to_string(&full).with_context(|| format!("edit {path}"))?;
                let count = text.matches(old.as_str()).count();
                ensure!(count != 0, "edit {path}: old text not found");
                ensure!(
                    count == 1,
                    "edit {path}: old text matches {count} times, need 1"
                );
                fs::write(&full, text.replacen(old.as_str(), new, 1))?;
                Ok(format!("edited {path}"))
            }
            ToolCall::Bash { command } => {
                let out = Command::new("sh")
                    .arg("-c")
                    .arg(command)
                    .current_dir(root)
                    .output()
                    .with_context(|| format!("bash: {command}"))?;
                let mut text = String::from_utf8_lossy(&out.stdout).into_owned();
                text.push_str(&String::from_utf8_lossy(&out.stderr));
                if !out.status.success() {
                    text.push_str(&format!("\nexit status: {}", out.status));
                }
                Ok(text)
            }
            ToolCall::Glob { pattern } => {
                let mut matches = Vec::new();
                walk(root, root, pattern, &mut matches)?;
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

fn walk(root: &Path, dir: &Path, pattern: &str, out: &mut Vec<String>) -> Result<()> {
    for entry in fs::read_dir(dir)? {
        let path: PathBuf = entry?.path();
        let rel = path.strip_prefix(root).expect("walk stays under root");
        let rel_str = rel.to_string_lossy();
        if path.is_dir() {
            walk(root, &path, pattern, out)?;
        } else if wildcard_match(pattern, &rel_str) {
            out.push(rel_str.into_owned());
        }
    }
    Ok(())
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
