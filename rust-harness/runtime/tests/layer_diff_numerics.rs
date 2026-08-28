/*
 * layer_diff_numerics — the verdict layer-diff reaches, not the help it prints.
 *
 * `binary_cli_surface.rs` covers this binary's command line and nothing past
 * it: a run that answered `--help` says nothing about the comparison. What is
 * under test here is the answer itself -- that a reference agreeing with the
 * checkpoint passes, that a reference perturbed at one known layer fails and
 * names that layer rather than a later one, and that the two tolerance flags
 * move the verdict across a boundary constructed for them.
 *
 * The perturbations are applied to a copy of the committed Tier T capture, so
 * the layer that should be blamed is known before the binary runs. That is the
 * property no green-gate history can supply: this binary is what caught a
 * missing 1/sqrt(head_dim) after nine passing packages, and until now nothing
 * asserted that it can fail at all.
 *
 * Direction and magnitude are separated deliberately. A row scaled by a
 * constant has the cosine it started with and a relative L2 the scale alone
 * fixes, which is the exact signature of a dropped per-layer scalar; a row with
 * half its signs flipped moves the cosine and leaves the norm. One
 * construction per threshold is what makes each flag's boundary test about
 * that flag.
 */

use std::path::{Path, PathBuf};
use std::process::Command;

use serde_json::Value;

fn fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../fixtures/tiny-hf-gemma4")
}

/// The committed capture as raw JSON, or `None` with a `[SKIP]` already
/// logged.
///
/// Raw rather than through `reference::load_capture`, because every test here
/// writes a modified capture back out: round-tripping through the loader would
/// drop whatever the loader does not model and quietly change the reference in
/// ways no test asked for.
fn capture(what: &str) -> Option<Value> {
    let path = fixture_dir().join("capture.json");
    if !path.exists() {
        runtime::testlog::skip(
            what,
            "fixtures/tiny-hf-gemma4 has not been generated; \
             run fixtures/build_tiny_reference.py on the workstation",
        );
        return None;
    }
    let text = std::fs::read_to_string(&path).expect("read the committed capture");
    Some(serde_json::from_str(&text).expect("the committed capture is JSON"))
}

/// Replace layer `i`'s reference row with `f` applied element-wise to it.
///
/// The row keeps its width, so a divergence reported for it is a numeric
/// disagreement and not the shape error `report` raises first.
fn perturb(capture: &mut Value, i: usize, f: impl Fn(usize, f64) -> f64) {
    let row = capture["layers"][i.to_string()]
        .as_array_mut()
        .unwrap_or_else(|| panic!("the capture carries layer {i}"));
    for (j, v) in row.iter_mut().enumerate() {
        let x = v.as_f64().expect("a layer row holds numbers");
        *v = serde_json::json!(f(j, x));
    }
}

/// What one layer-diff run produced.
struct Run {
    code: i32,
    stdout: String,
    stderr: String,
}

/// One table row, as the binary printed it.
struct Row {
    label: String,
    cosine: f64,
    rel_l2: f64,
    /// The whole line, so an assertion can name which divergence mark it saw.
    line: String,
}

impl Run {
    /// The stage and layer rows, in the order the binary printed them.
    fn rows(&self) -> Vec<Row> {
        self.stdout
            .lines()
            .filter(|l| !l.trim().is_empty())
            // The column header, whose second field is the word "cosine".
            .filter(|l| l.split_whitespace().nth(1) != Some("cosine"))
            .map(|l| {
                let f: Vec<&str> = l.split_whitespace().collect();
                assert!(f.len() >= 4, "unparsable table row: {l:?}");
                Row {
                    label: f[0].to_string(),
                    cosine: f[1].parse().unwrap_or_else(|e| panic!("{l:?}: {e}")),
                    rel_l2: f[3].parse().unwrap_or_else(|e| panic!("{l:?}: {e}")),
                    line: l.to_string(),
                }
            })
            .collect()
    }

    /// The label of the first row the binary marked as diverging.
    fn first_marked(&self) -> Option<String> {
        self.rows()
            .into_iter()
            .find(|r| r.line.contains("<--"))
            .map(|r| r.label)
    }

    fn row(&self, label: &str) -> Row {
        self.rows()
            .into_iter()
            .find(|r| r.label == label)
            .unwrap_or_else(|| panic!("no row {label} in:\n{}", self.stdout))
    }
}

/// Write `capture` to a scratch file and compare the fixture against it.
///
/// The run gets its own working directory and log root, so nothing it writes
/// -- including the obs record every layer-diff run opens -- lands in the
/// worktree.
fn layer_diff(capture: &Value, extra: &[&str]) -> Run {
    let dir = tempfile::tempdir().expect("a scratch working directory");
    let reference = dir.path().join("capture.json");
    std::fs::write(&reference, serde_json::to_string(capture).unwrap()).unwrap();
    let out = Command::new(env!("CARGO_BIN_EXE_layer-diff"))
        .args([
            "--model",
            fixture_dir().to_str().unwrap(),
            "--reference",
            reference.to_str().unwrap(),
        ])
        .args(extra)
        .current_dir(dir.path())
        .env("OBS_LOG_DIR", dir.path().join("logs"))
        .output()
        .expect("layer-diff builds as a test binary");
    Run {
        code: out.status.code().expect("layer-diff exited normally"),
        stdout: String::from_utf8(out.stdout).expect("stdout is UTF-8"),
        stderr: String::from_utf8(out.stderr).expect("stderr is UTF-8"),
    }
}

/// The unperturbed case: the reference and the checkpoint are the same model.
///
/// The baseline every other test here is a departure from. It also pins the
/// one thing a table full of marks could not: that the default tolerances are
/// reachable by a real run, so a failure below is evidence about the
/// perturbation rather than about the thresholds.
#[test]
fn a_reference_matching_the_checkpoint_passes_every_layer() {
    let Some(c) = capture("layer-diff agreement") else {
        return;
    };
    let r = layer_diff(&c, &[]);
    assert_eq!(r.code, 0, "stdout:\n{}\nstderr:\n{}", r.stdout, r.stderr);
    assert!(
        r.stderr.contains("all 4 layers within tolerance"),
        "{}",
        r.stderr
    );
    // Four layers and the four prelude stages the capture records.
    let rows = r.rows();
    assert_eq!(rows.len(), 8, "{}", r.stdout);
    for row in &rows {
        assert!(!row.line.contains("<--"), "{}", row.line);
        assert!(row.cosine > 0.9999, "{}: {}", row.label, row.line);
    }
}

/// A reference wrong at one layer fails, and the verdict names that layer.
///
/// Negating the row is a direction failure and nothing else -- the norm is
/// unchanged -- so the mark is the direction one and the cosine is -1 rather
/// than merely low.
#[test]
fn a_reference_perturbed_at_one_layer_fails_and_names_that_layer() {
    let Some(mut c) = capture("layer-diff single-layer divergence") else {
        return;
    };
    perturb(&mut c, 2, |_, x| -x);
    let r = layer_diff(&c, &[]);
    assert_eq!(r.code, 1, "stdout:\n{}\nstderr:\n{}", r.stdout, r.stderr);
    assert!(
        r.stderr.contains("first divergence at layer 2"),
        "{}",
        r.stderr
    );
    assert_eq!(r.first_marked().as_deref(), Some("2"), "{}", r.stdout);
    assert!(r.row("2").cosine < -0.99, "{}", r.row("2").line);
    // The rows either side are untouched, so a run that marked everything
    // after the perturbation would fail here rather than pass on the strength
    // of the first mark alone.
    assert!(!r.row("1").line.contains("<--"), "{}", r.row("1").line);
    assert!(!r.row("3").line.contains("<--"), "{}", r.row("3").line);
}

/// First over tolerance, not worst: the verdict names the earlier layer even
/// when a later one is further out.
///
/// The property the binary's own header states and the reason it prints a
/// verdict at all rather than a sorted table. Layer 1 is perturbed by a 20%
/// scale and layer 3 is negated, so reading the table by magnitude would blame
/// layer 3 and send a reader to the end of a stack that is wrong from layer 1
/// down.
#[test]
fn the_verdict_names_the_first_diverging_layer_and_not_the_worst() {
    let Some(mut c) = capture("layer-diff first-not-worst") else {
        return;
    };
    perturb(&mut c, 1, |_, x| x * 1.2);
    perturb(&mut c, 3, |_, x| -x);
    let r = layer_diff(&c, &[]);
    assert_eq!(r.code, 1, "stdout:\n{}\nstderr:\n{}", r.stdout, r.stderr);
    assert!(
        r.stderr.contains("first divergence at layer 1"),
        "{}",
        r.stderr
    );
    assert_eq!(r.first_marked().as_deref(), Some("1"), "{}", r.stdout);
    // Both are marked -- the table still reports every divergence -- and layer
    // 3 is the worse of the two, which is what makes the verdict above a
    // choice rather than a coincidence.
    assert!(r.row("3").line.contains("<--"), "{}", r.stdout);
    assert!(
        r.row("3").cosine < r.row("1").cosine,
        "layer 3 is not the worse row: {} vs {}",
        r.row("3").line,
        r.row("1").line
    );
}

/// A scaled reference row is a magnitude failure alone, and
/// `--rel-l2-ceiling` is what decides it.
///
/// Scaling by 1.05 leaves the direction and puts the relative L2 at
/// 0.05/1.05 = 0.0476, so the same run passes under a ceiling above that
/// figure and fails under one below it. Nothing about the input changes
/// between the three runs, which is what makes this a test of the flag rather
/// than of the perturbation.
#[test]
fn the_rel_l2_ceiling_moves_the_verdict_across_a_scaled_row() {
    let Some(mut c) = capture("layer-diff --rel-l2-ceiling boundary") else {
        return;
    };
    perturb(&mut c, 1, |_, x| x * 1.05);

    // At the default ceiling of 1e-3 this is a divergence, and a magnitude one:
    // the cosine floor is untouched by a scale, so a binary checking direction
    // alone would report this row as clean.
    let default = layer_diff(&c, &[]);
    assert_eq!(default.code, 1, "{}", default.stdout);
    assert!(
        default.row("1").line.contains("<-- diverges (magnitude)"),
        "{}",
        default.row("1").line
    );
    let measured = default.row("1").rel_l2;
    assert!(
        (measured - 0.05 / 1.05).abs() < 1e-3,
        "a 1.05 scale should put relative L2 at 0.0476, not {measured}"
    );

    let loose = layer_diff(&c, &["--rel-l2-ceiling", "0.06"]);
    assert_eq!(
        loose.code, 0,
        "a ceiling above the measured {measured} still failed:\n{}\n{}",
        loose.stdout, loose.stderr
    );
    let tight = layer_diff(&c, &["--rel-l2-ceiling", "0.04"]);
    assert_eq!(
        tight.code, 1,
        "a ceiling below the measured {measured} still passed:\n{}\n{}",
        tight.stdout, tight.stderr
    );
    assert_eq!(
        tight.first_marked().as_deref(),
        Some("1"),
        "{}",
        tight.stdout
    );
}

/// A row with half its signs flipped is a direction failure, and
/// `--cosine-floor` is what decides it.
///
/// The floor is bracketed around the cosine the run itself reported, so the
/// boundary is the measured value rather than a guess about it: one run either
/// side of the printed figure, with the relative-L2 ceiling opened wide so
/// that the cosine is the only threshold in play.
#[test]
fn the_cosine_floor_moves_the_verdict_across_a_reoriented_row() {
    let Some(mut c) = capture("layer-diff --cosine-floor boundary") else {
        return;
    };
    // Half the components negated: the norm is preserved and the direction is
    // not, which is the failure the cosine floor exists for.
    perturb(&mut c, 1, |j, x| if j % 2 == 0 { -x } else { x });

    // Everything permitted, so the table prints and the verdict is a pass. The
    // measured cosine is read from this run.
    let open = layer_diff(&c, &["--cosine-floor", "-1.1", "--rel-l2-ceiling", "1e9"]);
    assert_eq!(
        open.code, 0,
        "stdout:\n{}\nstderr:\n{}",
        open.stdout, open.stderr
    );
    let measured = open.row("1").cosine;
    // Bracketing needs room either side of the measured value; a row that
    // happened to land on ±1 would make the two runs below the same run.
    assert!(
        measured.abs() < 0.9,
        "the reoriented row is still aligned at cosine {measured}, so the \
         bracket below would not straddle anything"
    );

    let below = format!("{}", measured - 0.01);
    let above = format!("{}", measured + 0.01);
    let passes = layer_diff(&c, &["--cosine-floor", &below, "--rel-l2-ceiling", "1e9"]);
    assert_eq!(
        passes.code, 0,
        "a floor below the measured {measured} still failed:\n{}\n{}",
        passes.stdout, passes.stderr
    );
    let fails = layer_diff(&c, &["--cosine-floor", &above, "--rel-l2-ceiling", "1e9"]);
    assert_eq!(
        fails.code, 1,
        "a floor above the measured {measured} still passed:\n{}\n{}",
        fails.stdout, fails.stderr
    );
    assert!(
        fails.row("1").line.contains("<-- diverges (direction)"),
        "a preserved norm should fail the floor alone: {}",
        fails.row("1").line
    );
    assert_eq!(
        fails.first_marked().as_deref(),
        Some("1"),
        "{}",
        fails.stdout
    );
}

/// A divergence in the prelude displaces the layer verdict entirely.
///
/// The stack's input is what the stage rows measure, and a wrong one makes
/// every layer wrong without any layer being at fault. Asserted because the
/// binary's answer in that case is a different message with a different
/// remedy, and an assertion only on the layer verdict would have accepted the
/// misleading one.
#[test]
fn a_divergence_before_the_stack_is_reported_instead_of_a_layer() {
    let Some(mut c) = capture("layer-diff prelude divergence") else {
        return;
    };
    let embed = c["embed"].as_array().expect("the capture records embed");
    let flipped: Vec<Value> = embed
        .iter()
        .map(|v| serde_json::json!(-v.as_f64().expect("embed holds numbers")))
        .collect();
    c["embed"] = Value::Array(flipped);

    let r = layer_diff(&c, &[]);
    assert_eq!(r.code, 1, "stdout:\n{}\nstderr:\n{}", r.stdout, r.stderr);
    assert!(
        r.stderr.contains("the stack's input diverges at 'embed'"),
        "{}",
        r.stderr
    );
    assert!(
        !r.stderr.contains("first divergence at layer"),
        "the layer verdict was reported alongside a bad stack input: {}",
        r.stderr
    );
    assert_eq!(r.first_marked().as_deref(), Some("embed"), "{}", r.stdout);
}
