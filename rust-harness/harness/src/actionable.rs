/*
 * actionable — a remedy carried on an error, so a failure states its fix.
 *
 * The repo's error shape is one tagged statement of what happened, then
 * indented `Solution:` and `Command:` lines (`rust-harness/Makefile:68-88`,
 * and ~40 sites across `scripts/`). Every one of those prints where the
 * failure is detected. These three do not: a probe of the wrong width, an
 * artifact fitted on other weights, and a `--model` path that is not a
 * checkpoint are all raised deep inside `build_generator` and surface at
 * `main.rs`'s one error arm, several frames from anything that knows what to
 * suggest. A fourth, the task loop running out of turns, ends in `agent.rs`,
 * which knows neither the checkpoint nor the flags its remedy has to name.
 *
 * So the remedy travels with the error. `Actionable` is attached as an
 * `anyhow` context and recovered by `render`, which downcasts. The rejected
 * alternative is a table matching on error text: that couples the remedy to
 * message wording no test pins, so the first reword silently drops the advice.
 *
 * The one property this file exists to hold: `Display` renders `what` and
 * nothing else. `obs::RunLog::fail` records `exc_msg` as exactly
 * `format!("{err:#}")`, the whole context chain, and `main.rs` prints the same
 * string -- so a payload that displayed its remedy would put two lines of
 * advice into every fail record and into a terminal line whose equality with
 * `exc_msg` is what RH.13 verifies. `solution` and `command` are reachable
 * only through `render`. `Debug` is written rather than derived for the same
 * reason; see its impl.
 *
 * Constructors rather than struct literals at the call sites, because
 * `no_remedy_offers_a_placeholder_where_a_real_path_is_known` has to enumerate
 * every remedy this crate can produce and nothing can reflect over struct
 * literals. Four named constructors are that enumeration.
 *
 * Not applied to the `--status` path. `preflight`'s checks report the same two
 * faults (probe geometry, artifact identity) as data rather than as errors,
 * and `--status` is itself the command two of the remedies below point at. A
 * `Check` carrying a remedy is a coherent thing to want and a separate
 * decision; today the divergence is that the run path advises and the
 * preflight enumerates.
 */

use std::fmt;
use std::path::Path;

/// A failure and what to do about it.
///
/// `what` is the diagnosis and the only part that reaches the terminal's
/// tagged line or the run record. `solution` states the fix in words;
/// `command` is a line that can be pasted as printed.
pub struct Actionable {
    /// The diagnosis, byte-identical to the message the site raised before it
    /// carried a remedy. Changing one of these changes `exc_msg`.
    pub what: String,
    /// The fix, as an imperative phrase. Lowercase, no trailing period, per
    /// the shape at `rust-harness/Makefile:72`.
    pub solution: String,
    /// A runnable command naming this run's real paths. Never a placeholder --
    /// the harness knows what it was pointed at, and printing `<path>` back at
    /// the operator wastes the one thing it has that they do not.
    pub command: String,
}

impl fmt::Display for Actionable {
    /// The diagnosis alone. See the file banner: everything downstream of an
    /// error formats it with `{:#}`, and the remedy must not be in that.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.what)
    }
}

impl fmt::Debug for Actionable {
    /// Delegates to `Display`, so the containment property holds under `{:?}`
    /// as well as `{}`.
    ///
    /// `anyhow` requires `Debug` on a context, and a derived one would print
    /// all three fields. Nothing formats a run error with `{:?}` today, which
    /// is exactly why deriving it is the wrong call: it would make "the remedy
    /// cannot reach a record" depend on nobody ever writing `{:?}` at any
    /// future call site, rather than on this type.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

/// The remedy lines for the outermost `Actionable` in an error's chain,
/// indented and ready for `runtime::status::plain`.
///
/// `None` when the error carries no remedy, which is most of them --
/// an empty `String` would make every caller test for emptiness to decide
/// whether to print, which is a sentinel where the type system has a real one.
///
/// Uses `anyhow::Error::downcast_ref` rather than walking `err.chain()`:
/// `chain` yields `&dyn std::error::Error`, and `Actionable` deliberately does
/// not implement that trait (`.styles/RUST_STYLE.md:84` -- this workspace
/// implements no `std::error::Error`), so it is invisible there. `anyhow`'s
/// own downcast searches the contexts it wrapped, which is where the remedy
/// is. It searches outermost-first, so an error that gathers two remedies on
/// the way up reports the one raised closest to the operator's request.
///
/// The two spaces are the repo's continuation indent
/// (`rust-harness/Makefile:71-73`), and they live in the string because the
/// caller emits with `status::plain`: `status::error_at` would tag each line,
/// turning one finding into what reads like three. The same decision is
/// recorded at `runtime/src/bin/long-context-golden.rs:46-50`.
pub fn render(err: &anyhow::Error) -> Option<String> {
    let a = err.downcast_ref::<Actionable>()?;
    Some(format!(
        "  Solution: {}\n  Command: {}",
        a.solution, a.command
    ))
}

/// The flag's own name, as the diagnosis spells it: `--control-vector` reads
/// "control vector".
///
/// Derived rather than passed beside the flag so the two cannot disagree. The
/// two spellings in use are pinned by `the_artifact_label_is_the_flag_without_its_dashes`.
pub(crate) fn label_of(flag: &str) -> String {
    flag.trim_start_matches('-').replace('-', " ")
}

/// A probe whose width disagrees with the model's embedding dimension.
///
/// The command is `--status` rather than a re-capture invocation: capturing a
/// probe needs a dataset and a layer this crate has no knowledge of, so a
/// command naming them would be the placeholder this module refuses. What the
/// harness can offer is the check that confirms the replacement before a
/// ninety-second load pays for finding out again.
pub fn probe_width(width: usize, embed_dim: usize, probe: &Path, model: &Path) -> Actionable {
    Actionable {
        what: format!("probe width {width} does not match model embedding dimension {embed_dim}"),
        solution: format!(
            "capture the probe against this checkpoint, whose embedding dimension is {embed_dim}"
        ),
        command: format!(
            "harness --status --model {} --probe {}",
            model.display(),
            probe.display()
        ),
    }
}

/// An interp artifact fitted on weights other than the ones being served.
///
/// `--deep` is the flag that performs this comparison without loading the
/// model, so the command is the same check at preflight cost.
pub fn model_identity(
    flag: &str,
    path: &Path,
    model: &Path,
    artifact_hash: &str,
    served: &str,
) -> Actionable {
    let label = label_of(flag);
    Actionable {
        what: format!(
            "{label} was trained on a different model: {label} model_hash \
             {artifact_hash}, served model sha256 {served}"
        ),
        solution: format!(
            "serve the weights the {label} was fitted on, or re-export the {label} against these"
        ),
        command: format!(
            "harness --status --deep --model {} {flag} {}",
            model.display(),
            path.display()
        ),
    }
}

/// A `--model` path with no checkpoint at it.
///
/// The command lists what is actually there, which answers both shapes of the
/// mistake: a path that does not exist says so, and a directory holding
/// neither `config.json` nor `model.safetensors` shows what it holds instead.
/// `--status` would be the symmetric choice and is the wrong one here -- it
/// resolves the same path and reports the same failure, so it advises the
/// operator to run the thing that just failed.
pub fn missing_checkpoint(given: &Path) -> Actionable {
    Actionable {
        what: format!("read model {}", given.display()),
        solution: "point --model at a directory holding config.json and model.safetensors, \
                   or at the weights file itself"
            .to_string(),
        command: format!("ls {}", given.display()),
    }
}

/// The task loop spent every turn on tool calls and never answered in prose.
///
/// The cause this campaign has met is a base checkpoint, which cannot follow
/// the tool contract task mode prepends and invents a `read` per turn until
/// the budget is gone (RH.37: 64 turns, half an hour on E4B, nothing said).
/// The cheap remedy is the prose path: `--chat` reads turns from stdin,
/// installs no contract and no grammar, and asks no permission, so the
/// command is a chat against the same weights with the same flags.
pub fn no_final_answer(turns: usize, model: &Path, text_only: bool) -> Actionable {
    Actionable {
        what: format!("no final answer within {turns} turns"),
        solution: "put the prompt on stdin with --chat, which has no tool loop, or give \
                   the task to an instruction-tuned checkpoint"
            .to_string(),
        command: format!(
            "harness --chat{} --model {}",
            if text_only { " --text-only" } else { "" },
            model.display()
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    /// Every remedy this crate can produce, built with realistic paths.
    ///
    /// The point of the constructors: a struct literal at a call site cannot
    /// be enumerated, and the placeholder assertion below is only worth
    /// something if it covers all of them.
    fn every_remedy() -> Vec<Actionable> {
        let model = PathBuf::from("/mnt/ai-data/models/hf/gemma-4-E4B");
        let probe = PathBuf::from("/mnt/ai-data/interp/probes/refusal/probe.safetensors");
        vec![
            probe_width(4096, 2560, &probe, &model),
            model_identity("--probe", &probe, &model, "aaaa", "bbbb"),
            model_identity("--control-vector", &probe, &model, "aaaa", "bbbb"),
            missing_checkpoint(&PathBuf::from("/nonexistent-checkpoint")),
            no_final_answer(64, &model, true),
        ]
    }

    #[test]
    fn no_remedy_offers_a_placeholder_where_a_real_path_is_known() {
        for a in every_remedy() {
            assert!(
                !a.command.contains('<'),
                "{}: the command carries a placeholder: {}",
                a.what,
                a.command
            );
            assert!(
                !a.command.contains('>'),
                "{}: the command carries a placeholder: {}",
                a.what,
                a.command
            );
        }
    }

    #[test]
    fn every_remedy_states_all_three_parts() {
        for a in every_remedy() {
            assert!(!a.what.is_empty(), "empty what beside {}", a.command);
            assert!(!a.solution.is_empty(), "empty solution beside {}", a.what);
            assert!(!a.command.is_empty(), "empty command beside {}", a.what);
        }
    }

    #[test]
    fn every_command_is_runnable_as_printed() {
        // A command is either the harness itself, spelled as `cli::usage`
        // spells it, or a shell builtin. Nothing else is pasteable without
        // the operator having to work out what to substitute.
        for a in every_remedy() {
            assert!(
                a.command.starts_with("harness ") || a.command.starts_with("ls "),
                "{}: not a command a reader can paste: {}",
                a.what,
                a.command
            );
        }
    }

    #[test]
    fn the_artifact_label_is_the_flag_without_its_dashes() {
        assert_eq!(label_of("--probe"), "probe");
        assert_eq!(label_of("--control-vector"), "control vector");
    }

    #[test]
    fn a_remedy_displays_the_diagnosis_and_not_the_advice() {
        // The property the whole file is arranged around. Asserted directly
        // rather than inferred from a record, because it is the reason
        // `Display` is hand-written.
        let a = missing_checkpoint(&PathBuf::from("/nonexistent-checkpoint"));
        let shown = format!("{a}");
        assert_eq!(shown, a.what);
        assert!(!shown.contains("Solution"), "{shown}");
        assert!(!shown.contains(&a.command), "{shown}");
    }

    #[test]
    fn a_remedy_debugs_the_diagnosis_and_not_the_advice() {
        // Same property under `{:?}`. A derived Debug prints all three fields
        // and passes every other test in this file.
        let a = missing_checkpoint(&PathBuf::from("/nonexistent-checkpoint"));
        let shown = format!("{a:?}");
        assert_eq!(shown, a.what);
        assert!(!shown.contains(&a.command), "{shown}");
    }

    #[test]
    fn the_alternate_display_of_a_carrying_chain_holds_no_advice() {
        // `obs::RunLog::fail` computes exc_msg as exactly this expression, and
        // `main.rs` prints exactly this expression. Asserting at the same
        // formatting call is what makes this the [SCR] claim rather than a
        // paraphrase of it.
        let a = probe_width(
            4096,
            2560,
            Path::new("/p/probe.safetensors"),
            Path::new("/m"),
        );
        let solution = a.solution.clone();
        let command = a.command.clone();
        let err = anyhow::Error::new(std::io::Error::other("underlying"))
            .context(a)
            .context("load probe");
        let rendered = format!("{err:#}");
        assert!(!rendered.contains(&solution), "{rendered}");
        assert!(!rendered.contains(&command), "{rendered}");
        // And the diagnosis is still there, joined into the chain as any
        // string context would be.
        assert!(
            rendered.contains("probe width 4096 does not match model embedding dimension 2560"),
            "{rendered}"
        );
    }

    #[test]
    fn render_finds_a_remedy_under_later_context() {
        // The reason the remedy is attached rather than printed at the raise
        // site: it has to survive the frames between there and main.
        let a = missing_checkpoint(Path::new("/nonexistent-checkpoint"));
        let expected = format!("  Solution: {}\n  Command: {}", a.solution, a.command);
        let err = anyhow::anyhow!(a)
            .context("read model /nonexistent-checkpoint")
            .context("build generator");
        assert_eq!(render(&err), Some(expected));
    }

    #[test]
    fn render_is_none_when_the_error_carries_no_remedy() {
        let err = anyhow::anyhow!("plain failure").context("outer");
        assert_eq!(render(&err), None);
    }

    #[test]
    fn the_rendered_block_states_the_solution_before_the_command() {
        // Order is the contract: what to do, then how. Both lines carry the
        // repo's two-space continuation indent.
        let err = anyhow::anyhow!(missing_checkpoint(Path::new("/x")));
        let block = render(&err).expect("the error carries a remedy");
        let lines: Vec<&str> = block.lines().collect();
        assert_eq!(lines.len(), 2, "{block}");
        assert!(lines[0].starts_with("  Solution: "), "{block}");
        assert!(lines[1].starts_with("  Command: "), "{block}");
    }

    #[test]
    fn render_takes_the_outermost_remedy_of_several() {
        let inner = missing_checkpoint(Path::new("/inner"));
        let outer = missing_checkpoint(Path::new("/outer"));
        let outer_command = outer.command.clone();
        let err = anyhow::anyhow!(inner).context(outer);
        let block = render(&err).expect("the error carries a remedy");
        assert!(block.contains(&outer_command), "{block}");
        assert!(!block.contains("/inner"), "{block}");
    }
}
