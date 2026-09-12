/*
 * cli — the harness command line as data.
 *
 * Every flag's spelling, arity, help text and relationships live in one table.
 * Parsing reads it, `--help` renders it, and `usage()` is generated from it, so
 * a flag that exists is documented and a flag that is documented exists.
 *
 * This replaces three descriptions of the same surface that main.rs used to
 * carry separately -- a USAGE const, a 49-line comment block, and the parse
 * table -- and which had already drifted apart: the comment block described 18
 * flags while the parser accepted 23.
 *
 * What this module does not do is parse a value. Every slot in main.rs is an
 * `Option<PathBuf>`, including the numeric ones, so an argument survives
 * arbitrary bytes and a path is never round-tripped through a lossy string;
 * typed parsing and range validation happen at the use site, where the error
 * message can say what the range is. That split is also why this table is not
 * a clap derive: clap would take ownership of the value types, of the
 * one-argument `--control-vector-layer-range "START END"` pair, and of the
 * positional task text, which is a change to parsing wearing a change to help
 * text's clothes.
 */

use std::path::PathBuf;

/// Whether a flag takes a following argument, and how that argument is spelled
/// in help output.
///
/// The placeholder carries its own delimiters rather than having them added on
/// render, because one of them is not angle-bracketed: the layer range is a
/// single quoted argument, and a formatter that supplied `<>` itself could not
/// reproduce it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Value(&'static str),
    Bare,
}

/// The section a flag is listed under in `--help`.
///
/// `Mode` is load-bearing rather than cosmetic: it is defined as the set of
/// flags that appear in the trailing alternation of the usage line, which is
/// what lets `usage()` be generated instead of transcribed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Group {
    Mode,
    Model,
    Sampling,
    Interp,
    Serving,
    Diagnostics,
}

impl Group {
    /// The heading this group is printed under.
    fn title(self) -> &'static str {
        match self {
            Group::Mode => "Mode",
            Group::Model => "Model",
            Group::Sampling => "Sampling",
            Group::Interp => "Interpretability",
            Group::Serving => "Serving",
            Group::Diagnostics => "Diagnostics",
        }
    }
}

/// The groups, in the order `--help` prints them.
const GROUPS: [Group; 6] = [
    Group::Mode,
    Group::Model,
    Group::Sampling,
    Group::Interp,
    Group::Serving,
    Group::Diagnostics,
];

/// One command-line flag.
#[derive(Clone, Copy, Debug)]
pub struct Flag {
    /// The spelling accepted on the command line, leading dashes included.
    pub name: &'static str,
    /// Whether a value follows, and its placeholder.
    pub kind: Kind,
    /// The `--help` section.
    pub group: Group,
    /// One sentence saying what the flag does.
    pub help: &'static str,
    /// The value used when the flag is absent, if there is a meaningful one.
    pub default: Option<&'static str>,
    /// A flag this one is meaningless without. Enforced as a usage error.
    pub requires: Option<&'static str>,
    /// A condition under which this flag is accepted but does nothing.
    ///
    /// Printed in help and warned about at run time. Unlike `requires` this
    /// cannot be checked here: the conditions are properties of parsed values
    /// and of the selected mode, neither of which this module sees. The
    /// constants below are the shared spelling, so the help text and the
    /// warning cannot disagree.
    pub inert_when: Option<&'static str>,
}

/// Greedy decoding, where the rank and nucleus cutoffs have nothing to cut.
///
/// `sampler.rs` takes the argmax path at `temperature <= 0.0`, so the cutoffs
/// are genuinely unreachable rather than merely unusual.
pub const GREEDY: &str = "temperature is 0";

/// A run that is not serving, where the per-request log has no requests.
pub const NOT_SERVING: &str = "there is no --serve";

/// A preflight with no interp artifact, where hashing the weights produces a
/// digest with nothing to compare it against.
///
/// `preflight::checks` produces the digest lazily and only an identity check
/// consumes it, so with neither `--probe` nor `--control-vector` the weights
/// are never read -- which is right, and was also completely silent until the
/// first real workstation run passed `--deep` and got an answer in 45ms.
pub const NO_ARTIFACT: &str = "there is no interp artifact to check";

/// The one flag whose absence is an error rather than a default.
///
/// Requiredness is a property of this program -- there is nothing to run
/// without weights -- and no second flag shares it, so a `required` field on
/// every `Flag` would be twenty-odd `false`s carrying one `true`.
const REQUIRED: &[&str] = &["--model"];

/// Flags left out of the usage line.
///
/// The meta flags answer and exit before a run is configured, so listing them
/// among the options of a run would misdescribe them. They keep their `Flag`
/// entries so `--help` documents them and `lookup` is total over every
/// spelling the binary recognizes.
const USAGE_OMITS: &[&str] = &["--help", "-h", "--version", "-V"];

/// Every flag the harness accepts.
///
/// The order of the non-`Mode` entries is the order of the usage line, and a
/// golden test pins the rendering, so reordering for readability cannot
/// silently rewrite what the program prints.
pub const FLAGS: &[Flag] = &[
    Flag {
        name: "--model",
        kind: Kind::Value("<checkpoint-dir>"),
        group: Group::Model,
        help: "config.json and model.safetensors for the CPU backend; a path to \
               the weights file resolves to its directory",
        default: None,
        requires: None,
        inert_when: None,
    },
    Flag {
        name: "--tokenizer",
        kind: Kind::Value("<path.json>"),
        group: Group::Model,
        help: "tokenizer.json to use instead of the one beside the checkpoint",
        default: None,
        requires: None,
        inert_when: None,
    },
    Flag {
        name: "--record",
        kind: Kind::Value("<path>"),
        group: Group::Diagnostics,
        help: "wrap the model in the flight recorder, appending every forward \
               call to a JSONL file replayable as a MockModel script",
        default: None,
        requires: None,
        inert_when: None,
    },
    Flag {
        name: "--trace-chrome",
        kind: Kind::Value("<path>"),
        group: Group::Diagnostics,
        help: "write a Perfetto-loadable trace of the span timeline; no \
               subscriber is installed otherwise, so spans cost nothing",
        default: None,
        requires: None,
        inert_when: None,
    },
    Flag {
        name: "--trace-level",
        kind: Kind::Value("<info|debug|trace>"),
        group: Group::Diagnostics,
        help: "how deep the trace goes: info is the session-level spans, debug \
               adds one span per layer and block of the forward pass and of the \
               load, trace adds one per kernel call",
        default: Some("debug"),
        requires: Some("--trace-chrome"),
        inert_when: None,
    },
    // The three steering flags share a prefix, which is why the retired-names
    // matcher's trailing (?![\w-]) matters if one is ever renamed: a key of
    // --control-vector must not match --control-vector-scale.
    Flag {
        name: "--control-vector",
        kind: Kind::Value("<path.safetensors>"),
        group: Group::Interp,
        help: "steering directions from interp/steering/export.py, one \
               block.<index> tensor per steered decoder block, added to that \
               block's output residual",
        default: None,
        requires: None,
        inert_when: None,
    },
    Flag {
        name: "--control-vector-scale",
        kind: Kind::Value("<f>"),
        group: Group::Interp,
        help: "multiplier on the steering directions",
        default: Some("1.0"),
        requires: Some("--control-vector"),
        inert_when: None,
    },
    Flag {
        // One argument rather than two, matching what interp/steering/wiring.py
        // puts in CONTROL_VECTOR_LAYER_RANGE, so the environment variable
        // forwards unmodified.
        name: "--control-vector-layer-range",
        kind: Kind::Value("\"<start> <end>\""),
        group: Group::Interp,
        help: "inclusive 0-based block range; directions outside it are dropped \
               at load",
        default: None,
        requires: Some("--control-vector"),
        inert_when: None,
    },
    Flag {
        name: "--probe",
        kind: Kind::Value("<path.safetensors>"),
        group: Group::Interp,
        help: "linear probe (interp/probes/export.py layout) scored in-band at \
               every position of its layer; the .json sidecar beside it carries \
               the layer, site and threshold, and is required",
        default: None,
        requires: None,
        inert_when: None,
    },
    Flag {
        name: "--interp-log",
        kind: Kind::Value("<dir>"),
        group: Group::Interp,
        help: "under --serve, append one JSONL record per completed request to \
               <dir>/harness/YYYY-MM-DD.jsonl, carrying the request, the reply, \
               and the probe scores taken while answering it; this is what \
               interp's probe monitor reads",
        default: None,
        // Not `requires`: outside --serve the end-of-run artifacts report the
        // same scores, so the flag is redundant rather than lossy, and a run
        // that would have worked must not start failing. The warning is there
        // because silence is what made it look like it had an effect.
        requires: None,
        inert_when: Some(NOT_SERVING),
    },
    Flag {
        name: "--probe-out",
        kind: Kind::Value("<path>"),
        group: Group::Interp,
        help: "JSONL file for flagged positions",
        default: Some("probe-flags.jsonl in the working directory"),
        requires: None,
        inert_when: None,
    },
    Flag {
        name: "--alignment-out",
        kind: Kind::Value("<path>"),
        group: Group::Interp,
        help: "JSONL file mapping every position to the byte span of output \
               text it produced, the forward pass that computed it, and the \
               probe score taken there; off by default because it makes the \
               model record every pass",
        default: None,
        requires: None,
        inert_when: None,
    },
    Flag {
        name: "--temperature",
        kind: Kind::Value("<f>"),
        group: Group::Sampling,
        help: "softmax temperature; 0 selects greedy decoding",
        default: Some("0.8"),
        requires: None,
        inert_when: None,
    },
    Flag {
        name: "--top-p",
        kind: Kind::Value("<f>"),
        group: Group::Sampling,
        help: "nucleus mass kept, in (0, 1]",
        default: Some("0.95"),
        requires: None,
        inert_when: Some(GREEDY),
    },
    Flag {
        name: "--top-k",
        kind: Kind::Value("<n>"),
        group: Group::Sampling,
        help: "rank cutoff applied before top-p, keeping at most the n most \
               probable tokens of the masked support; 0 is rejected rather than \
               read as disabled",
        default: Some("no rank cutoff"),
        requires: None,
        inert_when: Some(GREEDY),
    },
    Flag {
        name: "--seed",
        kind: Kind::Value("<n>"),
        group: Group::Sampling,
        help: "sampler seed; the same seed replays a run exactly",
        default: Some("0"),
        requires: None,
        inert_when: None,
    },
    Flag {
        name: "--max-tokens",
        kind: Kind::Value("<n>"),
        group: Group::Sampling,
        help: "per-turn generation budget",
        default: Some("256"),
        requires: None,
        inert_when: None,
    },
    Flag {
        name: "--context",
        kind: Kind::Value("<n>"),
        group: Group::Model,
        help: "cap the context budget below the model's trained window; \
               max_position_embeddings is both the default and the maximum",
        default: None,
        requires: None,
        inert_when: None,
    },
    Flag {
        name: "--compact-at",
        kind: Kind::Value("<f>"),
        group: Group::Sampling,
        help: "automatic-compaction threshold as a fraction of the context \
               budget, in (0, 1]",
        default: Some("0.8"),
        requires: None,
        inert_when: None,
    },
    Flag {
        name: "--text-only",
        kind: Kind::Bare,
        group: Group::Model,
        help: "load the text stack of a multimodal checkpoint and decline its \
               audio and vision towers; required rather than implied, so nobody \
               runs a model text-blind without having said so",
        default: None,
        requires: None,
        inert_when: None,
    },
    Flag {
        name: "--single-thread",
        kind: Kind::Bare,
        group: Group::Diagnostics,
        help: "run the CPU backend's row split at width one; not a numerics \
               switch, since the split is bit-identical at every width, but rr \
               serializes threads anyway",
        default: None,
        requires: None,
        inert_when: None,
    },
    Flag {
        name: "--deep",
        kind: Kind::Bare,
        group: Group::Diagnostics,
        help: "under --status, hash the weights and compare the digest against \
               what each interp artifact records; opt-in because it reads the \
               whole checkpoint, which is the one part of a preflight that is \
               not fast",
        default: None,
        requires: Some("--status"),
        // Both fields are true of it and they are independent: --deep has no
        // meaning without --status, and no effect under --status when there is
        // no artifact whose recorded hash the digest would be compared to.
        inert_when: Some(NO_ARTIFACT),
    },
    Flag {
        // The bind address a container must override: binding the container's
        // own loopback makes the published port unreachable, because the
        // mapping forwards to the container's external interface and nothing
        // listens there. Not a weaker posture -- compose publishes to
        // 127.0.0.1 on the host either way.
        name: "--serve-addr",
        kind: Kind::Value("<ip>"),
        group: Group::Serving,
        help: "bind address for --serve; only a container needs to set it, and \
               must",
        default: Some("127.0.0.1"),
        requires: Some("--serve"),
        inert_when: None,
    },
    Flag {
        name: "--chat",
        kind: Kind::Bare,
        group: Group::Mode,
        help: "interactive REPL instead of a one-shot task; takes no task text",
        default: None,
        requires: None,
        inert_when: None,
    },
    Flag {
        name: "--serve",
        kind: Kind::Value("<port>"),
        group: Group::Mode,
        help: "serve /v1/chat/completions and /v1/health instead of running a \
               task; the sampling flags become per-request defaults that a \
               request may override, and no task text is taken",
        default: None,
        requires: None,
        inert_when: None,
    },
    Flag {
        name: "--status",
        kind: Kind::Bare,
        group: Group::Mode,
        help: "report what this invocation would fail on and exit, without \
               loading the model: the checkpoint, the backend's verdict on it, \
               each interp artifact, and the directories written to; exit 1 if \
               anything would stop the run, and no run record is written \
               because a query is not a run",
        default: None,
        requires: None,
        inert_when: None,
    },
    Flag {
        name: "--help",
        kind: Kind::Bare,
        group: Group::Diagnostics,
        help: "print this help on stdout and exit",
        default: None,
        requires: None,
        inert_when: None,
    },
    Flag {
        name: "-h",
        kind: Kind::Bare,
        group: Group::Diagnostics,
        help: "synonym for --help",
        default: None,
        requires: None,
        inert_when: None,
    },
    Flag {
        name: "--version",
        kind: Kind::Bare,
        group: Group::Diagnostics,
        help: "print the harness version on stdout and exit",
        default: None,
        requires: None,
        inert_when: None,
    },
    Flag {
        name: "-V",
        kind: Kind::Bare,
        group: Group::Diagnostics,
        help: "synonym for --version",
        default: None,
        requires: None,
        inert_when: None,
    },
];

/// Every binary in the workspace, with one line on what it is for.
///
/// Listed in help because the workspace ships eight programs and the harness is
/// the only one anybody runs by name; the other seven are discoverable only by
/// reading a directory. Note the container image builds `-p harness` alone, so
/// inside it this list is informational.
pub const BINARIES: &[(&str, &str)] = &[
    (
        "harness",
        "the coding agent loop, the chat REPL, and the OpenAI-shaped server",
    ),
    (
        "checkpoint-info",
        "a checkpoint's config and tensor table, and the backend's compatibility verdict",
    ),
    (
        "activation-capture",
        "last-token residual activations in the interp capture shard format",
    ),
    (
        "attention-capture",
        "post-softmax attention weights for chosen layers and query heads",
    ),
    (
        "ladder",
        "the seven-rung correctness gate against a transformers reference",
    ),
    (
        "layer-diff",
        "per-layer divergence of the residual stream against a reference capture",
    ),
    (
        "long-context-golden",
        "writes the committed long-prefill logits fixture",
    ),
    ("tiny-model", "writes the synthetic fixture checkpoint"),
];

/// The flag with this spelling, if the harness has one.
pub fn lookup(arg: &str) -> Option<&'static Flag> {
    FLAGS.iter().find(|f| f.name == arg)
}

/// How a flag and its placeholder are written in help and usage.
fn term(flag: &Flag) -> String {
    match flag.kind {
        Kind::Value(placeholder) => format!("{} {placeholder}", flag.name),
        Kind::Bare => flag.name.to_string(),
    }
}

/// The one-line usage summary, generated from `FLAGS`.
///
/// The mode flags come last, inside the alternation that also offers positional
/// task text, because exactly one of the three has to be chosen.
pub fn usage() -> String {
    let mut out = String::from("usage: harness");
    for flag in FLAGS
        .iter()
        .filter(|f| f.group != Group::Mode)
        .filter(|f| !USAGE_OMITS.contains(&f.name))
    {
        let word = term(flag);
        if REQUIRED.contains(&flag.name) {
            out.push_str(&format!(" {word}"));
        } else {
            out.push_str(&format!(" [{word}]"));
        }
    }
    let modes: Vec<String> = FLAGS
        .iter()
        .filter(|f| f.group == Group::Mode)
        .map(term)
        .collect();
    out.push_str(&format!(" ({} | <task description>)", modes.join(" | ")));
    out
}

/// The full help text, for stdout.
///
/// Each flag gets a term line and an indented description rather than an
/// aligned column: the longest term is 46 characters wide, so a column would
/// push every description past a sensible line length to accommodate one flag.
pub fn help() -> String {
    let mut out = usage();
    out.push_str("\n\nRun a coding task, a chat REPL, or an HTTP server against a local\n");
    out.push_str("Gemma 4 checkpoint on the CPU backend.\n");
    for group in GROUPS {
        let flags: Vec<&Flag> = FLAGS.iter().filter(|f| f.group == group).collect();
        if flags.is_empty() {
            continue;
        }
        out.push_str(&format!("\n{}:\n", group.title()));
        for flag in flags {
            out.push_str(&format!("  {}\n", term(flag)));
            let mut description = flag.help.to_string();
            if let Some(default) = flag.default {
                description.push_str(&format!(" (default {default})"));
            }
            if let Some(requires) = flag.requires {
                description.push_str(&format!("; requires {requires}"));
            }
            if let Some(condition) = flag.inert_when {
                description.push_str(&format!("; no effect when {condition}"));
            }
            for line in wrap(&description, 72) {
                out.push_str(&format!("      {line}\n"));
            }
        }
    }
    out.push_str("\nEverything after a bare -- is task text, even if it starts with a dash.\n");
    out.push_str("\nOther binaries in this workspace:\n");
    for (name, purpose) in BINARIES {
        out.push_str(&format!("  {name}\n      {purpose}\n"));
    }
    out
}

/// Break text on whitespace so no line exceeds `width`, where one can.
///
/// A word longer than `width` occupies its own line rather than being split:
/// the long words here are paths and flag names, and a broken one cannot be
/// copied out of the terminal.
fn wrap(text: &str, width: usize) -> Vec<String> {
    let mut lines: Vec<String> = Vec::new();
    let mut current = String::new();
    for word in text.split_whitespace() {
        if current.is_empty() {
            current.push_str(word);
        } else if current.chars().count() + 1 + word.chars().count() <= width {
            current.push(' ');
            current.push_str(word);
        } else {
            lines.push(std::mem::take(&mut current));
            current.push_str(word);
        }
    }
    if !current.is_empty() {
        lines.push(current);
    }
    lines
}

/// What one scan of the argument vector found.
pub struct Parsed {
    /// The flags given, in `FLAGS` spelling, for the relationship checks.
    pub given: Vec<&'static str>,
    /// The positional words, in order: the task text.
    pub rest: Vec<String>,
}

/// Fill the caller's value slots from `args`, returning the rest.
///
/// The caller keeps ownership of its typed locals and passes `&mut` handles to
/// them, so the values land where the existing validation expects them and this
/// module never learns what a temperature is.
///
/// Errors are returned rather than printed, as `main.rs` does elsewhere, so the
/// caller decides the exit code and the stream.
pub fn extract(
    args: Vec<String>,
    slots: &mut [(&'static str, &mut Option<PathBuf>)],
) -> Result<Parsed, String> {
    // A precondition on every run rather than a debug_assert: a value flag with
    // no slot would parse, consume its argument and then be discarded silently,
    // which is the exact class of failure this module exists to remove. Cheap
    // at this size, and it catches a flag added to the table but never wired.
    for flag in FLAGS.iter().filter(|f| matches!(f.kind, Kind::Value(_))) {
        if !slots.iter().any(|(name, _)| *name == flag.name) {
            return Err(format!("internal: {} has no value slot", flag.name));
        }
    }
    let mut given: Vec<&'static str> = Vec::new();
    let mut rest: Vec<String> = Vec::new();
    let mut remaining = args.into_iter();
    let mut positional_only = false;
    while let Some(arg) = remaining.next() {
        if positional_only {
            rest.push(arg);
            continue;
        }
        if arg == "--" {
            positional_only = true;
            continue;
        }
        let Some(flag) = lookup(&arg) else {
            // A leftover argument that looks like a flag used to be joined into
            // the task text, so `harness --modle foo "do x"` ran a task named
            // "--modle foo do x" and exited 0. A bare "-" stays positional: it
            // is a conventional stdin spelling and a plausible task word.
            if arg.starts_with('-') && arg != "-" {
                return Err(format!("Unknown argument: {arg}"));
            }
            rest.push(arg);
            continue;
        };
        if given.contains(&flag.name) {
            // Taking the first and leaking the rest into the task text is what
            // this used to do. Last-wins would be the common convention, but it
            // discards an argument silently, and the obs log records the argv as
            // given -- a record showing two seeds for a run that used one is a
            // record that misleads whoever reads it next.
            return Err(format!("{}: given more than once", flag.name));
        }
        given.push(flag.name);
        if let Kind::Value(placeholder) = flag.kind {
            let Some(value) = remaining.next() else {
                return Err(format!("{}: expected a value {placeholder}", flag.name));
            };
            // Refusing every dash-leading value would reject legitimate ones: a
            // negative scale, or a path whose name starts with a dash. Only a
            // spelling this parser recognizes is refused, which is the case
            // where the user plainly omitted the value.
            if lookup(&value).is_some() {
                return Err(format!(
                    "{}: expected a value, found the flag {value}",
                    flag.name
                ));
            }
            let slot = slots
                .iter_mut()
                .find(|(name, _)| *name == flag.name)
                .expect("checked above: every value flag has a slot");
            *slot.1 = Some(PathBuf::from(value));
        }
    }
    Ok(Parsed { given, rest })
}

/// One message per flag given without the flag it needs.
pub fn missing_requirements(given: &[&str]) -> Vec<String> {
    FLAGS
        .iter()
        .filter(|f| given.contains(&f.name))
        .filter_map(|f| f.requires.map(|req| (f.name, req)))
        .filter(|(_, req)| !given.contains(req))
        .map(|(name, req)| format!("{name} has no meaning without {req}"))
        .collect()
}

/// The flags given that do nothing under `condition`.
///
/// The caller evaluates the condition, because it owns the parsed values this
/// module deliberately never sees.
pub fn inert(given: &[&str], condition: &str) -> Vec<String> {
    FLAGS
        .iter()
        .filter(|f| f.inert_when == Some(condition))
        .filter(|f| given.contains(&f.name))
        .map(|f| format!("{}: no effect when {condition}", f.name))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The usage line the harness printed before this table existed.
    ///
    /// Pinned as a literal because generating it from `FLAGS` means reordering
    /// the table silently rewrites what the program says. This string is the
    /// diff a reviewer reads when someone edits the table.
    const USAGE_GOLDEN: &str = "usage: harness --model <checkpoint-dir> \
         [--tokenizer <path.json>] [--record <path>] [--trace-chrome <path>] \
         [--trace-level <info|debug|trace>] \
         [--control-vector <path.safetensors>] [--control-vector-scale <f>] \
         [--control-vector-layer-range \"<start> <end>\"] \
         [--probe <path.safetensors>] [--interp-log <dir>] [--probe-out <path>] \
         [--alignment-out <path>] [--temperature <f>] [--top-p <f>] \
         [--top-k <n>] [--seed <n>] [--max-tokens <n>] [--context <n>] \
         [--compact-at <f>] [--text-only] [--single-thread] [--deep] \
         [--serve-addr <ip>] \
         (--chat | --serve <port> | --status | <task description>)";

    #[test]
    fn the_generated_usage_is_the_line_the_harness_has_always_printed() {
        assert_eq!(usage(), USAGE_GOLDEN);
    }

    #[test]
    fn every_flag_has_help_and_a_group_that_prints_it() {
        for flag in FLAGS {
            assert!(!flag.help.is_empty(), "{} has no help", flag.name);
            assert!(
                GROUPS.contains(&flag.group),
                "{} is in a group help never prints",
                flag.name
            );
            assert!(flag.name.starts_with('-'), "{} is not a flag", flag.name);
        }
    }

    #[test]
    fn no_two_flags_share_a_spelling() {
        let mut names: Vec<&str> = FLAGS.iter().map(|f| f.name).collect();
        let count = names.len();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), count, "two flags share a spelling");
    }

    #[test]
    fn every_value_flag_names_its_placeholder() {
        for flag in FLAGS {
            if let Kind::Value(placeholder) = flag.kind {
                assert!(!placeholder.is_empty(), "{} has no placeholder", flag.name);
            }
        }
    }

    /// A `requires` naming a flag that does not exist would be printed in help
    /// and never enforced, which is worse than not having it.
    #[test]
    fn every_relationship_names_a_flag_that_exists() {
        for flag in FLAGS {
            if let Some(req) = flag.requires {
                assert!(lookup(req).is_some(), "{} requires absent {req}", flag.name);
                assert_ne!(req, flag.name, "{} requires itself", flag.name);
            }
        }
    }

    #[test]
    fn the_inert_conditions_are_the_shared_constants() {
        for flag in FLAGS {
            if let Some(condition) = flag.inert_when {
                assert!(
                    condition == GREEDY || condition == NOT_SERVING || condition == NO_ARTIFACT,
                    "{} names an inert condition nothing evaluates",
                    flag.name
                );
            }
        }
    }

    /// Every flag the parser accepts has to be reachable in help, or the help
    /// is a second description of the surface again.
    #[test]
    fn help_names_every_flag_and_every_binary() {
        let text = help();
        for flag in FLAGS {
            assert!(text.contains(flag.name), "help omits {}", flag.name);
        }
        for (name, _) in BINARIES {
            assert!(text.contains(name), "help omits the {name} binary");
        }
        assert!(text.contains("--"), "help must document the separator");
    }

    #[test]
    fn help_carries_no_escape_bytes() {
        assert!(!help().contains('\x1b'), "help is payload, never coloured");
    }

    #[test]
    fn the_workspace_has_eight_binaries_with_distinct_names() {
        assert_eq!(BINARIES.len(), 8);
        let mut names: Vec<&str> = BINARIES.iter().map(|(n, _)| *n).collect();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), 8, "two binaries share a name");
        for (_, purpose) in BINARIES {
            assert!(!purpose.is_empty());
        }
    }

    /// `runs` is reserved: the catalog reader hides it.
    ///
    /// Every binary here writes its run record under `cli = "harness"` with
    /// its own name as `command`, and `common/obs/catalog.py` filters out
    /// rows whose `command` is literally `runs` -- the browsing command
    /// omitting its own invocations from the table it prints. A binary named
    /// `runs` would therefore log correctly and be invisible in `make
    /// harness-runs`, which is the worst shape a defect can take here: the
    /// records exist, so nothing looks broken.
    ///
    /// Asserted on this side because this is where a ninth binary gets
    /// named. The reader's half of the bargain is recorded in
    /// `schemas/obs_record.json`.
    #[test]
    fn no_binary_claims_the_name_the_catalog_reader_reserves() {
        for (name, _) in BINARIES {
            assert_ne!(
                *name, "runs",
                "common/obs/catalog.py hides catalog rows whose command is \
                 `runs`, so this binary's runs would never appear in the table"
            );
        }
    }

    /// The value slots a caller owns, named and filled, as `main.rs` holds them.
    type Slots = Vec<(&'static str, Option<PathBuf>)>;

    /// Slots for every value flag, so the scan can be exercised without main.
    fn slots() -> Slots {
        FLAGS
            .iter()
            .filter(|f| matches!(f.kind, Kind::Value(_)))
            .map(|f| (f.name, None))
            .collect()
    }

    /// Run one scan against fresh slots, returning the result and the slots.
    fn scan(args: &[&str]) -> (Result<Parsed, String>, Slots) {
        let mut owned = slots();
        let mut handles: Vec<(&'static str, &mut Option<PathBuf>)> =
            owned.iter_mut().map(|(n, v)| (*n, v)).collect();
        let args = args.iter().map(|a| a.to_string()).collect();
        let result = extract(args, &mut handles);
        drop(handles);
        (result, owned)
    }

    fn value_of(slots: &[(&'static str, Option<PathBuf>)], name: &str) -> Option<PathBuf> {
        slots
            .iter()
            .find(|(n, _)| *n == name)
            .and_then(|(_, v)| v.clone())
    }

    #[test]
    fn a_value_flag_fills_its_slot_and_leaves_the_task_behind() {
        let (parsed, slots) = scan(&["--model", "/weights", "do", "the", "thing"]);
        let parsed = parsed.expect("a well-formed command line");
        assert_eq!(value_of(&slots, "--model"), Some(PathBuf::from("/weights")));
        assert_eq!(parsed.rest, vec!["do", "the", "thing"]);
        assert_eq!(parsed.given, vec!["--model"]);
    }

    #[test]
    fn a_bare_flag_is_recorded_without_consuming_the_next_word() {
        let (parsed, _) = scan(&["--text-only", "hello"]);
        let parsed = parsed.expect("a bare flag takes no value");
        assert_eq!(parsed.given, vec!["--text-only"]);
        assert_eq!(parsed.rest, vec!["hello"]);
    }

    /// The failure this whole stage exists to remove.
    #[test]
    fn a_misspelled_flag_is_refused_instead_of_becoming_the_task() {
        let (parsed, _) = scan(&["--modle", "foo", "do x"]);
        let error = parsed.err().expect("an unknown flag is a usage error");
        assert!(error.contains("--modle"), "{error}");
        assert!(error.starts_with("Unknown argument"), "{error}");
    }

    #[test]
    fn everything_after_the_separator_is_task_text() {
        let (parsed, slots) = scan(&["--model", "/w", "--", "explain", "--seed", "-x"]);
        let parsed = parsed.expect("the separator makes dashes ordinary");
        assert_eq!(value_of(&slots, "--model"), Some(PathBuf::from("/w")));
        assert_eq!(parsed.rest, vec!["explain", "--seed", "-x"]);
        // The separator itself is consumed, not passed on as a task word.
        assert!(!parsed.rest.contains(&"--".to_string()));
        // And a flag after it is task text, not a flag that was given.
        assert_eq!(parsed.given, vec!["--model"]);
    }

    #[test]
    fn a_lone_dash_stays_a_positional_word() {
        let (parsed, _) = scan(&["-"]);
        assert_eq!(parsed.expect("a bare dash is not a flag").rest, vec!["-"]);
    }

    #[test]
    fn a_repeated_flag_is_refused_naming_it() {
        let (parsed, _) = scan(&["--seed", "1", "--seed", "2", "do x"]);
        let error = parsed.err().expect("a repeat is a usage error");
        assert!(error.contains("--seed"), "{error}");
        assert!(error.contains("more than once"), "{error}");
    }

    #[test]
    fn a_value_flag_at_the_end_reports_what_it_wanted() {
        let (parsed, _) = scan(&["--model"]);
        let error = parsed.err().expect("a missing value is a usage error");
        assert!(error.contains("--model"), "{error}");
        assert!(error.contains("<checkpoint-dir>"), "{error}");
    }

    #[test]
    fn an_omitted_value_does_not_swallow_the_following_flag() {
        // `--probe-out --max-tokens 4 hello` without a path: the old parser
        // stored "--max-tokens" as the probe output, dropped the token budget,
        // and pushed "4" into the task text, all at exit 0.
        let (parsed, _) = scan(&["--probe-out", "--max-tokens", "4", "hello"]);
        let error = parsed.err().expect("a flag is not a value");
        assert!(error.contains("--probe-out"), "{error}");
        assert!(error.contains("--max-tokens"), "{error}");
    }

    #[test]
    fn a_value_that_only_looks_flag_adjacent_is_accepted() {
        let (parsed, slots) = scan(&["--control-vector-scale", "-1.5"]);
        parsed.expect("a negative scale is a value, not a flag");
        assert_eq!(
            value_of(&slots, "--control-vector-scale"),
            Some(PathBuf::from("-1.5"))
        );
        let (parsed, slots) = scan(&["--model", "./-weights"]);
        parsed.expect("a dash-leading path is a value");
        assert_eq!(
            value_of(&slots, "--model"),
            Some(PathBuf::from("./-weights"))
        );
    }

    #[test]
    fn a_steering_flag_without_its_vector_is_a_missing_requirement() {
        let missing = missing_requirements(&["--control-vector-scale"]);
        assert_eq!(missing.len(), 1);
        assert!(missing[0].contains("--control-vector-scale"), "{missing:?}");
        assert!(missing[0].contains("has no meaning without"), "{missing:?}");
        assert!(
            missing_requirements(&["--control-vector-scale", "--control-vector"]).is_empty(),
            "the requirement is satisfied"
        );
    }

    /// The message main.rs printed by hand before the table generalized it.
    #[test]
    fn the_bind_address_requirement_reads_as_it_always_did() {
        assert_eq!(
            missing_requirements(&["--serve-addr"]),
            vec!["--serve-addr has no meaning without --serve".to_string()]
        );
    }

    #[test]
    fn the_cutoffs_are_reported_inert_under_greedy_decoding() {
        let reported = inert(&["--top-k", "--top-p", "--seed"], GREEDY);
        assert_eq!(reported.len(), 2, "{reported:?}");
        assert!(
            reported.iter().all(|m| m.contains("no effect when")),
            "{reported:?}"
        );
        assert!(inert(&["--seed"], GREEDY).is_empty());
    }

    #[test]
    fn the_request_log_is_reported_inert_without_a_server() {
        let reported = inert(&["--interp-log"], NOT_SERVING);
        assert_eq!(reported.len(), 1, "{reported:?}");
        assert!(reported[0].contains("--interp-log"), "{reported:?}");
    }

    #[test]
    fn the_deep_digest_is_reported_inert_with_no_artifact_to_check() {
        let reported = inert(&["--deep", "--status"], NO_ARTIFACT);
        assert_eq!(reported.len(), 1, "{reported:?}");
        assert!(reported[0].contains("--deep"), "{reported:?}");
        // The flags that give --deep something to do are not themselves
        // inert, the way --seed is not inert under greedy decoding.
        assert!(inert(&["--probe", "--control-vector"], NO_ARTIFACT).is_empty());
    }
}
