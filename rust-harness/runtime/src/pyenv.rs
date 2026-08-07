/*
 * pyenv — interpreter resolution for the cross-language contract tests.
 *
 * Several tests verify an artifact by handing it to the real Python reader
 * rather than to a second Rust implementation, because a Rust re-read would
 * reproduce a Rust misunderstanding of the format. The interpreter those tests
 * spawn is therefore part of the contract, and a bare `python3` off PATH is not
 * it. The two environments disagree about where the right one lives:
 *
 *   - On the workstation, PATH usually leads to a pyenv shim carrying none of
 *     the project's dependencies, while the project environment sits in .venv.
 *   - In a contained agent sandbox the reverse holds: docker-compose.agents.yaml
 *     masks /work/.venv with an empty tmpfs, so the workstation's environment is
 *     not reachable at all, while /opt/pyenv — built from uv.lock with the
 *     sandbox and dev groups — is first on PATH.
 *
 * Preferring either location unconditionally fixes one environment and breaks
 * the other, so resolution probes candidates instead of assuming:
 *
 *   1. $HARNESS_PYTHON — explicit override, for a caller that knows better.
 *   2. <repo_root>/.venv/bin/python3 — the workstation's project environment;
 *      masked to an empty directory in a sandbox, so the probe falls through.
 *   3. python3 on PATH — /opt/pyenv in a sandbox, where it is the correct
 *      answer rather than a fallback.
 *
 * The first candidate that both executes and imports every required module
 * wins; one that cannot is passed over rather than reported. That is what lets
 * a single mechanism be correct in both environments.
 */

use std::path::{Path, PathBuf};

/// The first interpreter that can import every name in `imports`, or None when
/// no candidate can.
///
/// `repo_root` goes on PYTHONPATH for the probe, so in-tree packages
/// (`training.*`, `common.*`) resolve exactly as they will in the test itself.
/// Pass an empty slice to ask only for a working interpreter.
///
/// Returning None is a deliberate signal, not a failure: a caller whose
/// dependency is genuinely absent prints a `[SKIP]` line naming it, per the
/// contract in `.styles/RUST_STYLE.md`. A caller whose dependency is mandatory
/// panics instead — `npz` does, because a shard nothing re-read is unverified.
pub fn interpreter(repo_root: &Path, imports: &[&str]) -> Option<PathBuf> {
    let mut candidates: Vec<PathBuf> = Vec::new();
    if let Some(explicit) = std::env::var_os("HARNESS_PYTHON") {
        candidates.push(PathBuf::from(explicit));
    }
    candidates.push(repo_root.join(".venv/bin/python3"));
    candidates.push(PathBuf::from("python3"));

    let probe = if imports.is_empty() {
        String::new()
    } else {
        format!("import {}", imports.join(", "))
    };
    candidates
        .into_iter()
        .find(|candidate| can_import(candidate, repo_root, &probe))
}

/// Whether `python` runs and satisfies `probe`, with `repo_root` importable.
fn can_import(python: &Path, repo_root: &Path, probe: &str) -> bool {
    std::process::Command::new(python)
        .args(["-c", probe])
        .env("PYTHONPATH", repo_root)
        .output()
        .map(|out| out.status.success())
        .unwrap_or(false)
}

/// The interpreter, or a panic naming the imports that no candidate satisfied.
///
/// For the tests where an absent dependency must fail the gate rather than
/// quietly skip it.
pub fn interpreter_or_panic(repo_root: &Path, imports: &[&str]) -> PathBuf {
    interpreter(repo_root, imports).unwrap_or_else(|| {
        panic!(
            "no python3 can import [{}]: tried $HARNESS_PYTHON, {}/.venv/bin/python3, \
             and python3 on PATH. Provision the environment with `make install`, or \
             point HARNESS_PYTHON at an interpreter that has them.",
            imports.join(", "),
            repo_root.display(),
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The repository root relative to this crate's manifest.
    fn repo_root() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
    }

    #[test]
    fn a_stdlib_only_request_resolves_to_some_interpreter() {
        let found = interpreter(&repo_root(), &["sys"]);
        assert!(found.is_some(), "no interpreter could import sys");
    }

    #[test]
    fn an_unimportable_module_resolves_to_no_interpreter() {
        let found = interpreter(&repo_root(), &["definitely_not_a_real_module_x9z"]);
        assert!(found.is_none(), "{found:?} claimed to import a fake module");
    }

    #[test]
    fn an_empty_import_list_asks_only_for_a_working_interpreter() {
        let found = interpreter(&repo_root(), &[]);
        assert!(found.is_some(), "no interpreter executed an empty script");
    }
}
