/*
 * cli — the command line of a runtime binary, as data.
 *
 * Seven programs live in `src/bin/`, and before this module each described its
 * own surface twice: a `USAGE` string a human maintained, and a parse loop that
 * decided what was actually accepted. None of the seven answered `--help`, and
 * two of them acted on the question -- `tiny-model -h` wrote a fixture set into
 * files named `-h`, because the stem is positional and nothing looked at it
 * first.
 *
 * A `Spec` is one program's whole surface. `usage()` is generated from it and
 * `help()` renders it, so a flag that exists is documented and a flag that is
 * documented exists.
 *
 * This is deliberately not `harness::cli`, and the duplication is the considered
 * answer rather than an oversight. That table carries `group`, `default`,
 * `requires` and `inert_when` for a program with twenty-odd flags across six
 * sections; no binary here has a use for any of the four, and importing them
 * would be the unused ceremony `RUST_STYLE.md` rules out. Its value slots are
 * also `Option<PathBuf>`, so a path survives arbitrary bytes -- a trade these
 * seven already declined by collecting `Vec<String>` from `std::env::args()`,
 * and one that means nothing to `--layers 8,16,24` or `--threads 4`.
 */

use std::process::ExitCode;

/// Whether a flag takes a following argument, and how that argument is spelled
/// in help output.
///
/// The placeholder carries its own delimiters, so a value spelled as an
/// alternation (`stepped|whole|<n>`) renders as itself rather than being forced
/// into angle brackets.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Value(&'static str),
    Bare,
}

/// One command-line flag.
#[derive(Clone, Copy, Debug)]
pub struct Flag {
    /// The spelling accepted on the command line, leading dashes included.
    pub name: &'static str,
    /// Whether a value follows, and its placeholder.
    pub kind: Kind,
    /// One sentence saying what the flag does.
    pub help: &'static str,
}

/// The two spellings of a help request.
///
/// Shared rather than written per program: seven copies of a two-element array
/// is seven chances for one of them to accept only the long form.
pub const HELP_FLAGS: [&str; 2] = ["--help", "-h"];

/// One program's command line.
///
/// Declared as a `const` beside the `main` that uses it, which is what keeps the
/// table and the parser in the same file and the same review.
pub struct Spec {
    /// The binary's name, as the usage line spells it.
    pub program: &'static str,
    /// One sentence on what the program is for, printed under the usage line.
    pub summary: &'static str,
    /// The positional arguments, already bracketed -- `"<checkpoint-dir>"`, or
    /// the empty string for a program that takes none.
    pub positionals: &'static str,
    /// Every flag the program accepts, help included.
    pub flags: &'static [Flag],
    /// The flags whose absence is an error rather than a default.
    ///
    /// Held as a list beside the table rather than a `required: bool` on every
    /// `Flag`, because four of these programs have two required flags among
    /// nine and the field would be mostly `false`. What it buys is a usage line
    /// that does not bracket a flag the program cannot run without -- and that
    /// line is what a reader copies out of a runbook.
    pub required: &'static [&'static str],
}

impl Spec {
    /// The flag with this spelling, if the program has one.
    pub fn lookup(&self, arg: &str) -> Option<&'static Flag> {
        self.flags.iter().find(|f| f.name == arg)
    }

    /// The one-line usage summary, generated from the flags.
    ///
    /// Generated rather than transcribed because the seven `USAGE` strings this
    /// replaces were maintained by hand against parsers that had moved on: the
    /// point of the table is that this line cannot describe a flag the program
    /// does not accept.
    pub fn usage(&self) -> String {
        let mut out = format!("usage: {}", self.program);
        for flag in self.flags.iter().filter(|f| !is_help(f.name)) {
            let word = term(flag);
            if self.required.contains(&flag.name) {
                out.push_str(&format!(" {word}"));
            } else {
                out.push_str(&format!(" [{word}]"));
            }
        }
        if !self.positionals.is_empty() {
            out.push(' ');
            out.push_str(self.positionals);
        }
        out
    }

    /// The full help text, for stdout.
    ///
    /// A flat list rather than grouped sections: the largest of these programs
    /// has nine flags, and a section heading over two entries is furniture.
    pub fn help(&self) -> String {
        let mut out = self.usage();
        out.push_str("\n\n");
        out.push_str(self.summary);
        out.push('\n');
        if !self.flags.is_empty() {
            out.push_str("\nOptions:\n");
            for flag in self.flags {
                out.push_str(&format!("  {}\n      {}\n", term(flag), flag.help));
            }
        }
        out
    }

    /// Whether the table is well formed, for an inline test to assert.
    ///
    /// Returned rather than asserted here so each program checks its own table
    /// in its own test: there is no place that sees all seven at once, and a
    /// check nobody calls is a check that holds for nobody.
    pub fn check(&self) -> Result<(), String> {
        if self.program.is_empty() {
            return Err("the program has no name".to_string());
        }
        for (i, flag) in self.flags.iter().enumerate() {
            if !flag.name.starts_with('-') {
                return Err(format!("{}: not a flag spelling", flag.name));
            }
            if flag.help.is_empty() {
                return Err(format!("{}: no help text", flag.name));
            }
            if let Kind::Value(placeholder) = flag.kind {
                if placeholder.is_empty() {
                    return Err(format!("{}: no value placeholder", flag.name));
                }
            }
            if self.flags[..i].iter().any(|f| f.name == flag.name) {
                return Err(format!("{}: listed twice", flag.name));
            }
        }
        for spelling in HELP_FLAGS {
            if self.lookup(spelling).is_none() {
                return Err(format!("{spelling} is not in the table"));
            }
        }
        for name in self.required {
            if self.lookup(name).is_none() {
                return Err(format!("{name} is required but not in the table"));
            }
        }
        Ok(())
    }
}

/// One message naming the required flags this command line did not carry.
///
/// Reads the same `required` list the usage line renders from, so a flag cannot
/// be printed unbracketed and then not enforced, or enforced and shown as
/// optional. Both capture binaries previously said `missing required flag`
/// without naming one -- and their commonest cause was a misspelling the parser
/// had ignored, which is now refused before this is ever reached.
pub fn missing_required(spec: &Spec, parsed: &Parsed) -> Option<String> {
    let absent: Vec<&str> = spec
        .required
        .iter()
        .copied()
        .filter(|name| !parsed.given.contains(name))
        .collect();
    match absent.as_slice() {
        [] => None,
        [one] => Some(format!("{one} is required")),
        many => Some(format!("required flags not given: {}", many.join(" "))),
    }
}

/// One of the two help spellings.
fn is_help(name: &str) -> bool {
    HELP_FLAGS.contains(&name)
}

/// How a flag and its placeholder are written in help and usage.
fn term(flag: &Flag) -> String {
    match flag.kind {
        Kind::Value(placeholder) => format!("{} {placeholder}", flag.name),
        Kind::Bare => flag.name.to_string(),
    }
}

/// Answer a help request, if that is what these arguments are.
///
/// Call it as the first statement of `main`, before the working directory is
/// read or a log is opened. That ordering is the whole point: two of these
/// programs take an output path positionally, so until something looks for `-h`
/// first, asking one of them for help writes a file called `-h`.
///
/// Only the arguments before a bare `--` are scanned, so a `--prompt` whose text
/// contains the word `--help` is still a prompt.
pub fn handled_help(spec: &Spec, args: &[String]) -> bool {
    let asked = args
        .iter()
        .take_while(|a| *a != "--")
        .any(|a| is_help(a.as_str()));
    if asked {
        // Requested help is the answer to the question that was asked, so it is
        // payload and goes to stdout. An *unrequested* usage block is a
        // diagnostic and goes to stderr through `usage_error` below.
        print!("{}", spec.help());
    }
    asked
}

/// Report a usage mistake and exit 2.
///
/// The exit code is a property of this function rather than of each call site.
/// Eleven sites across the seven binaries write this same three-line shape by
/// hand, and two of them -- both capture binaries -- returned 1, putting "you
/// asked wrongly" on the same code as "the capture failed". A caller that cannot
/// choose the code cannot get that wrong again.
pub fn usage_error(spec: &Spec, message: &str) -> ExitCode {
    crate::status::error(message);
    crate::status::plain(&spec.usage());
    ExitCode::from(2)
}

/// What one scan of the argument vector found.
#[derive(Debug)]
pub struct Parsed {
    /// The flags given, in table spelling, so a bare flag can be tested for.
    pub given: Vec<&'static str>,
    /// The positional words, in order.
    pub rest: Vec<String>,
}

impl Parsed {
    /// Whether a bare flag was given.
    pub fn has(&self, name: &str) -> bool {
        self.given.contains(&name)
    }
}

/// Fill the caller's value slots from `args`, returning the rest.
///
/// The caller keeps its own typed locals and parses each value itself, which is
/// why the slots are `Option<String>` and not something narrower: `--dot`,
/// `--rung` and `--batch` each have an error message naming their own accepted
/// set, and those messages are better than anything this module could produce.
///
/// Errors are returned rather than printed, so the caller decides the stream and
/// the code -- in practice by handing the message to `usage_error`.
pub fn extract(
    spec: &Spec,
    args: Vec<String>,
    slots: &mut [(&'static str, &mut Option<String>)],
) -> Result<Parsed, String> {
    // A precondition on every run rather than a debug_assert: a value flag with
    // no slot would consume its argument and then be silently discarded, which
    // is the failure this module exists to remove. It also catches a flag added
    // to the table and never wired.
    for flag in spec
        .flags
        .iter()
        .filter(|f| matches!(f.kind, Kind::Value(_)))
    {
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
        let Some(flag) = spec.lookup(&arg) else {
            // A dash-leading word this program does not know is a mistake, not a
            // positional. Two of the seven used to accept one as an output path
            // and write to it. A bare "-" stays positional: it is a conventional
            // stdin spelling.
            if arg.starts_with('-') && arg != "-" {
                return Err(format!("Unknown argument: {arg}"));
            }
            rest.push(arg);
            continue;
        };
        if given.contains(&flag.name) {
            // Every one of these binaries used to take the first occurrence and
            // ignore the rest, silently. Last-wins is the commoner convention
            // and discards an argument just as quietly; naming the repeat is the
            // only answer that cannot lose one.
            return Err(format!("{}: given more than once", flag.name));
        }
        given.push(flag.name);
        if let Kind::Value(placeholder) = flag.kind {
            let Some(value) = remaining.next() else {
                return Err(format!("{}: expected a value {placeholder}", flag.name));
            };
            // Refusing every dash-leading value would reject legitimate ones: a
            // negative tolerance, or a prompt that opens with a dash. Only a
            // spelling this program recognizes is refused, which is the case
            // where the value was plainly omitted.
            if spec.lookup(&value).is_some() {
                return Err(format!(
                    "{}: expected a value, found the flag {value}",
                    flag.name
                ));
            }
            let slot = slots
                .iter_mut()
                .find(|(name, _)| *name == flag.name)
                .expect("checked above: every value flag has a slot");
            *slot.1 = Some(value);
        }
    }
    Ok(Parsed { given, rest })
}

#[cfg(test)]
mod tests {
    use super::*;

    const FLAGS: &[Flag] = &[
        Flag {
            name: "--help",
            kind: Kind::Bare,
            help: "print this help and exit",
        },
        Flag {
            name: "-h",
            kind: Kind::Bare,
            help: "print this help and exit",
        },
        Flag {
            name: "--model",
            kind: Kind::Value("<checkpoint-dir>"),
            help: "the checkpoint to read",
        },
        Flag {
            name: "--text-only",
            kind: Kind::Bare,
            help: "treat a multimodal checkpoint as its text tower",
        },
    ];

    const SPEC: Spec = Spec {
        program: "probe",
        summary: "A synthetic program, for this module's own tests.",
        positionals: "<out>",
        flags: FLAGS,
        required: &["--model"],
    };

    /// Slots for every value flag in `SPEC`, which `extract` requires.
    fn run(args: &[&str]) -> (Result<Parsed, String>, Option<String>) {
        let mut model: Option<String> = None;
        let owned: Vec<String> = args.iter().map(|a| a.to_string()).collect();
        let parsed = {
            let mut slots: [(&'static str, &mut Option<String>); 1] = [("--model", &mut model)];
            extract(&SPEC, owned, &mut slots)
        };
        (parsed, model)
    }

    #[test]
    fn the_synthetic_table_is_well_formed() {
        SPEC.check()
            .expect("the test table is the shape check wants");
    }

    #[test]
    fn check_names_a_flag_listed_twice() {
        const DUPES: &[Flag] = &[
            Flag {
                name: "--help",
                kind: Kind::Bare,
                help: "h",
            },
            Flag {
                name: "-h",
                kind: Kind::Bare,
                help: "h",
            },
            Flag {
                name: "--model",
                kind: Kind::Bare,
                help: "m",
            },
            Flag {
                name: "--model",
                kind: Kind::Bare,
                help: "m",
            },
        ];
        let spec = Spec {
            program: "probe",
            summary: "s",
            positionals: "",
            flags: DUPES,
            required: &[],
        };
        let err = spec.check().unwrap_err();
        assert!(err.contains("--model"), "{err}");
    }

    #[test]
    fn check_requires_both_help_spellings() {
        const ONLY_LONG: &[Flag] = &[Flag {
            name: "--help",
            kind: Kind::Bare,
            help: "h",
        }];
        let spec = Spec {
            program: "probe",
            summary: "s",
            positionals: "",
            flags: ONLY_LONG,
            required: &[],
        };
        let err = spec.check().unwrap_err();
        assert!(err.contains("-h"), "{err}");
    }

    #[test]
    fn an_unknown_dash_leading_argument_is_refused_and_named() {
        let err = run(&["--modle", "x"]).0.unwrap_err();
        assert_eq!(err, "Unknown argument: --modle");
    }

    #[test]
    fn a_bare_dash_stays_positional() {
        let parsed = run(&["-"]).0.expect("a lone dash is not a flag");
        assert_eq!(parsed.rest, vec!["-".to_string()]);
    }

    #[test]
    fn a_double_dash_ends_flag_parsing() {
        let parsed = run(&["--", "--modle"])
            .0
            .expect("everything after -- is positional");
        assert_eq!(parsed.rest, vec!["--modle".to_string()]);
    }

    #[test]
    fn a_value_flag_with_no_value_names_its_placeholder() {
        let err = run(&["--model"]).0.unwrap_err();
        assert!(err.contains("<checkpoint-dir>"), "{err}");
    }

    #[test]
    fn a_value_that_is_itself_a_known_flag_is_refused() {
        let err = run(&["--model", "--text-only"]).0.unwrap_err();
        assert!(err.contains("found the flag --text-only"), "{err}");
    }

    #[test]
    fn a_repeated_flag_is_refused_rather_than_taking_the_first() {
        let err = run(&["--model", "a", "--model", "b"]).0.unwrap_err();
        assert_eq!(err, "--model: given more than once");
    }

    #[test]
    fn a_value_flag_fills_its_slot_and_positionals_keep_their_order() {
        let (parsed, model) = run(&["--model", "ckpt", "one", "two"]);
        let parsed = parsed.expect("a well formed command line");
        assert_eq!(model.as_deref(), Some("ckpt"));
        assert_eq!(parsed.rest, vec!["one".to_string(), "two".to_string()]);
    }

    #[test]
    fn a_bare_flag_is_reported_through_given() {
        let parsed = run(&["--text-only"]).0.expect("a bare flag needs no value");
        assert!(parsed.has("--text-only"), "{:?}", parsed.given);
        assert!(!parsed.has("--model"), "{:?}", parsed.given);
    }

    #[test]
    fn a_value_flag_with_no_slot_is_an_internal_error_rather_than_a_silent_discard() {
        let mut slots: [(&'static str, &mut Option<String>); 0] = [];
        let err = extract(&SPEC, vec!["--model".to_string()], &mut slots).unwrap_err();
        assert!(err.contains("--model has no value slot"), "{err}");
    }

    #[test]
    fn check_rejects_a_required_flag_the_table_does_not_carry() {
        let spec = Spec {
            program: "probe",
            summary: "s",
            positionals: "",
            flags: FLAGS,
            required: &["--dataset"],
        };
        let err = spec.check().unwrap_err();
        assert!(err.contains("--dataset"), "{err}");
    }

    #[test]
    fn a_missing_required_flag_is_named_rather_than_merely_counted() {
        let parsed = run(&[]).0.expect("an empty command line parses");
        assert_eq!(
            missing_required(&SPEC, &parsed).as_deref(),
            Some("--model is required")
        );
    }

    #[test]
    fn every_required_flag_given_leaves_nothing_to_report() {
        let parsed = run(&["--model", "ckpt"]).0.expect("a well formed line");
        assert_eq!(missing_required(&SPEC, &parsed), None);
    }

    #[test]
    fn several_absent_required_flags_are_listed_together() {
        let spec = Spec {
            program: "probe",
            summary: "s",
            positionals: "",
            flags: FLAGS,
            required: &["--model", "--text-only"],
        };
        let parsed = run(&[]).0.expect("an empty command line parses");
        let message = missing_required(&spec, &parsed).expect("two are absent");
        assert!(message.contains("--model"), "{message}");
        assert!(message.contains("--text-only"), "{message}");
    }

    /// A required flag is printed unbracketed and an optional one is not, so the
    /// usage line and `missing_required` cannot describe different programs.
    #[test]
    fn the_usage_line_brackets_only_what_the_program_can_run_without() {
        let usage = SPEC.usage();
        assert!(usage.contains(" --model <checkpoint-dir>"), "{usage}");
        assert!(!usage.contains("[--model"), "{usage}");
        assert!(usage.contains("[--text-only]"), "{usage}");
    }

    #[test]
    fn the_usage_line_names_every_flag_except_the_help_spellings() {
        let usage = SPEC.usage();
        assert!(usage.starts_with("usage: probe"), "{usage}");
        assert!(usage.contains("--model <checkpoint-dir>"), "{usage}");
        assert!(usage.contains("--text-only"), "{usage}");
        assert!(usage.ends_with("<out>"), "{usage}");
        // Help answers and exits before the program is configured, so listing it
        // among the options of a run would misdescribe it.
        assert!(!usage.contains("--help"), "{usage}");
    }

    #[test]
    fn help_carries_the_usage_line_the_summary_and_every_flags_description() {
        let help = SPEC.help();
        assert!(help.contains(&SPEC.usage()), "{help}");
        assert!(help.contains(SPEC.summary), "{help}");
        for flag in FLAGS {
            assert!(help.contains(flag.help), "{} missing: {help}", flag.name);
        }
    }

    #[test]
    fn handled_help_answers_either_spelling_and_ignores_one_after_a_double_dash() {
        assert!(handled_help(&SPEC, &["--help".to_string()]));
        assert!(handled_help(&SPEC, &["-h".to_string()]));
        assert!(!handled_help(&SPEC, &["--model".to_string()]));
        assert!(!handled_help(
            &SPEC,
            &["--".to_string(), "--help".to_string()]
        ));
    }
}
