/*
 * alignment — the token-to-text alignment artifact.
 *
 * One JSONL file per run holding, for every position, the byte span of output
 * text that position produced alongside the probe score taken there. Both
 * channels in one artifact keyed by the run id, because a comparison that
 * needs two files joined after the fact is a comparison nobody runs twice.
 *
 * On "scaffolding produces no user-visible text": that holds when the
 * checkpoint carries its chat markers as pieces `decode` drops, which the real
 * Gemma 4 vocabulary does and the synthetic fixtures deliberately do not --
 * they carry boundaries as ordinary vocabulary entries so nothing but the
 * caller's allowlist can resolve them. On a fixture the marker therefore
 * decodes to its own characters and lands in the text. The artifact reports
 * what was decoded either way and marks the role, rather than suppressing
 * bytes to satisfy a rule: a suppressed byte breaks the round trip, and the
 * role field is what a reader filters on.
 *
 * Each row also carries the token id it aligns. The harness's own decode
 * cannot vouch for its own spans -- a reader holding the artifact and the
 * checkpoint's tokenizer can, by decoding each id and comparing with the
 * bytes its span selects -- and the id is the one fact that makes that check
 * possible from outside (tests/hardware/test_alignment.py).
 */

use std::collections::HashMap;
use std::io::Write;
use std::path::Path;

use anyhow::{Context, Result};

use runtime::align::{role_at, scaffold_ranges, Detok, Pass, Role, Span};
use runtime::model::probe::Flag;
use runtime::model::Token;
use runtime::tokenizer::Tokenizer;

use crate::template::MARKERS;

/// The artifact's format revision. Bump when a required key changes meaning
/// or leaves; adding an optional key is not a bump.
pub const SCHEMA_VERSION: u32 = 1;

/// One position's alignment facts, before the pass and score joins.
#[derive(Debug, Clone, PartialEq)]
pub struct Row {
    pub position: usize,
    pub token: Token,
    pub role: Role,
    pub span: Span,
}

/// The positions of one run and the text they produced.
///
/// Rebuilt whole rather than appended to. The rendered prompt is append-only
/// across turns (`template::rendering_is_append_only_across_turns`), so each
/// turn's prompt contains every earlier turn including the model's own
/// replies; rebuilding from the current prompt plus the current generation is
/// therefore complete, and it cannot drift from the run the way an incremental
/// record accumulating across turns could.
#[derive(Debug, Default, Clone)]
pub struct Alignment {
    pub rows: Vec<Row>,
    pub text: String,
}

impl Alignment {
    /// Build the table for a turn: the rendered prompt, then the ids the model
    /// generated after it.
    ///
    /// Positions are absolute cache positions, which is what makes the join
    /// with passes and probe scores a lookup rather than an alignment problem
    /// of its own — the tap reports `start + p` from the same numbering.
    pub fn build(
        tok: &Tokenizer,
        prompt: &str,
        prompt_specials: &[&str],
        generated: &[Token],
    ) -> Result<Alignment> {
        let scaffold = scaffold_ranges(prompt, MARKERS);
        let prompt_tokens = tok.encode_prompt_offsets(prompt, prompt_specials)?;

        let mut detok = Detok::new(tok);
        let mut rows = Vec::with_capacity(prompt_tokens.len() + generated.len());
        for (position, (id, at)) in prompt_tokens.iter().enumerate() {
            rows.push(Row {
                position,
                token: *id,
                role: role_at(at, &scaffold),
                span: detok.advance(*id)?,
            });
        }
        // Everything the model produced is content by construction: the
        // harness wrote none of it.
        for (i, &id) in generated.iter().enumerate() {
            rows.push(Row {
                position: prompt_tokens.len() + i,
                token: id,
                role: Role::Content,
                span: detok.advance(id)?,
            });
        }
        Ok(Alignment {
            text: detok.text().to_string(),
            rows,
        })
    }
}

/// What identifies the run an artifact describes.
pub struct Header<'a> {
    pub run_id: &'a str,
    pub model_hash: &'a str,
    pub probe: Option<&'a str>,
    pub probe_layer: Option<usize>,
}

/// Write `alignment.jsonl`: one header line, then one line per position.
///
/// `passes` and `scores` are joined by position rather than carried through
/// the build, because both are produced deep inside the model — the pass in
/// `prefill_groups`, the score in the tap — and neither knows what text a
/// position became.
///
/// A missing parent directory is created: like the probe flags, these rows
/// exist only in memory and are lost the moment this returns.
pub fn write(
    out: &Path,
    header: &Header<'_>,
    alignment: &Alignment,
    passes: &[Pass],
    scores: &[Flag],
) -> Result<()> {
    if let Some(parent) = out.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("create alignment directory {}", parent.display()))?;
    }
    let mut file = std::fs::File::create(out)
        .with_context(|| format!("create alignment artifact {}", out.display()))?;

    // Last score wins: a position re-prefilled after a truncation was scored
    // twice, and the later scoring is the one the run ended holding.
    let by_position: HashMap<usize, f32> = scores.iter().map(|f| (f.position, f.score)).collect();

    let mut head = serde_json::json!({
        "schema_version": SCHEMA_VERSION,
        "artifact": "alignment",
        "run_id": header.run_id,
        "model_hash": header.model_hash,
        "positions": alignment.rows.len(),
        "text": alignment.text,
        "git_sha": runtime::envelope::git_sha(),
        "created": runtime::envelope::utc_now(),
    });
    if let (Some(name), Some(layer)) = (header.probe, header.probe_layer) {
        head["probe"] = serde_json::json!(name);
        head["probe_layer"] = serde_json::json!(layer);
    }
    writeln!(file, "{head}").context("write alignment header")?;

    for row in &alignment.rows {
        let pass = passes.iter().find(|p| p.covers(row.position));
        let mut record = serde_json::json!({
            "position": row.position,
            "token_id": row.token,
            "role": row.role.as_str(),
            "byte_start": row.span.start,
            "byte_end": row.span.end,
        });
        // Absent rather than null on all three. A null pass would claim the
        // position was computed by no forward call, and a null score would
        // claim the probe looked and found nothing -- neither is what an
        // unrecorded field means.
        if let Some(p) = pass {
            record["pass"] = serde_json::json!(p.index);
            record["phase"] = serde_json::json!(p.phase());
        }
        if let Some(score) = by_position.get(&row.position) {
            record["score"] = serde_json::json!(score);
        }
        writeln!(file, "{record}").context("write alignment row")?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::template::{BOS, TURN_CLOSE, TURN_OPEN};

    fn tokenizer() -> Tokenizer {
        Tokenizer::from_json(
            &runtime::synth::build_tiny_tokenizer_json(&runtime::synth::TinySpec::default())
                .unwrap(),
        )
        .unwrap()
    }

    const SPECIALS: &[&str] = &[BOS, TURN_OPEN, TURN_CLOSE];

    fn prompt() -> String {
        format!("{BOS}{TURN_OPEN}user\nabc{TURN_CLOSE}\n{TURN_OPEN}model\n")
    }

    #[test]
    fn concatenating_every_span_reproduces_the_text_exactly() {
        let tok = tokenizer();
        let generated = tok.encode("abc caf\u{00e9}").unwrap();
        let a = Alignment::build(&tok, &prompt(), SPECIALS, &generated).unwrap();

        let mut joined = String::new();
        for row in &a.rows {
            joined.push_str(&a.text[row.span.start..row.span.end]);
        }
        assert_eq!(joined, a.text, "spans do not reproduce the text");
    }

    /// Rows grouped as `Detok` attributes bytes: every run of empty spans is
    /// owned by the non-empty span that follows it, because a codepoint spread
    /// across byte-fallback ids lands whole on the id that completes it. The
    /// Python gate (tests/hardware/test_alignment.py) groups the same way.
    fn attribution_groups(rows: &[Row]) -> Vec<(Vec<Token>, Span)> {
        let mut groups = Vec::new();
        let mut ids = Vec::new();
        for row in rows {
            ids.push(row.token);
            if !row.span.is_empty() {
                groups.push((std::mem::take(&mut ids), row.span));
            }
        }
        if !ids.is_empty() {
            let end = rows.last().map_or(0, |r| r.span.end);
            groups.push((ids, Span { start: end, end }));
        }
        groups
    }

    #[test]
    fn every_recorded_id_decodes_to_the_bytes_its_span_selects() {
        let tok = tokenizer();
        let generated = tok.encode("abc caf\u{00e9} \u{4e16}").unwrap();
        let a = Alignment::build(&tok, &prompt(), SPECIALS, &generated).unwrap();

        let expected: Vec<Token> = tok
            .encode_prompt(&prompt(), SPECIALS)
            .unwrap()
            .into_iter()
            .chain(generated.iter().copied())
            .collect();
        let recorded: Vec<Token> = a.rows.iter().map(|r| r.token).collect();
        assert_eq!(recorded, expected, "the rows do not carry the run's ids");

        for (ids, span) in attribution_groups(&a.rows) {
            let decoded = tok.decode(&ids).unwrap();
            assert_eq!(
                decoded,
                &a.text[span.start..span.end],
                "ids {ids:?} do not decode to their span {span:?}"
            );
        }
    }

    #[test]
    fn positions_are_dense_and_cover_the_prompt_and_the_generation() {
        let tok = tokenizer();
        let generated = tok.encode("abc").unwrap();
        let p = prompt();
        let a = Alignment::build(&tok, &p, SPECIALS, &generated).unwrap();

        let expected = tok.encode_prompt(&p, SPECIALS).unwrap().len() + generated.len();
        assert_eq!(a.rows.len(), expected, "row count");
        for (i, row) in a.rows.iter().enumerate() {
            assert_eq!(row.position, i, "positions are not dense: {row:?}");
        }
    }

    #[test]
    fn scaffolding_positions_are_marked_rather_than_omitted() {
        let tok = tokenizer();
        let a = Alignment::build(&tok, &prompt(), SPECIALS, &[]).unwrap();
        let scaffold: Vec<&Row> = a.rows.iter().filter(|r| r.role == Role::Scaffold).collect();
        assert!(
            !scaffold.is_empty(),
            "no position was marked scaffolding in {:?}",
            a.rows
        );
        // Nothing is dropped: the row count is still the token count, which is
        // the invariant that keeps artifact indices equal to run positions.
        let p = prompt();
        assert_eq!(
            a.rows.len(),
            tok.encode_prompt(&p, SPECIALS).unwrap().len(),
            "a position was omitted"
        );
        // Both roles occur -- a classifier that answered "scaffold" for
        // everything would satisfy the assertion above.
        assert!(
            a.rows.iter().any(|r| r.role == Role::Content),
            "nothing in the prompt was content: {:?}",
            a.rows
        );
        // Every scaffolding row sits inside a marker's bytes.
        let ranges = scaffold_ranges(&p, MARKERS);
        for row in &scaffold {
            let at = tok
                .encode_prompt_offsets(&p, SPECIALS)
                .unwrap()
                .get(row.position)
                .map(|(_, at)| at.clone())
                .unwrap();
            assert!(
                ranges.iter().any(|r| at.start < r.end && r.start < at.end),
                "position {} was marked scaffolding but sits at {at:?}",
                row.position
            );
        }
    }

    #[test]
    fn a_boundary_id_the_tokenizer_hides_gets_an_empty_span() {
        // The case IF-2 describes -- scaffolding producing no user-visible
        // text -- reached through a piece `decode` really does drop. The
        // fixture's chat markers are not among them; see the module doc.
        let tok = tokenizer();
        let a = Alignment::build(&tok, "", &[], &[tok.bos(), tok.eos()]).unwrap();
        assert_eq!(a.text, "", "a boundary rendered visible text: {:?}", a.text);
        for row in &a.rows {
            assert!(row.span.is_empty(), "{row:?}");
        }
    }

    #[test]
    fn a_byte_fallback_fragment_lands_in_the_completed_codepoints_span() {
        let tok = tokenizer();
        // Outside the fixture vocabulary, so byte fallback splits it.
        let generated = tok.encode("\u{4e16}").unwrap();
        assert!(generated.len() > 1, "{generated:?}");
        let a = Alignment::build(&tok, "", &[], &generated).unwrap();

        let text: Vec<&str> = a
            .rows
            .iter()
            .map(|r| &a.text[r.span.start..r.span.end])
            .collect();
        assert_eq!(
            text.concat(),
            "\u{4e16}",
            "the fragments did not reassemble: {text:?}"
        );
        // Exactly one row carries the codepoint; the rest are empty.
        assert_eq!(text.iter().filter(|t| !t.is_empty()).count(), 1, "{text:?}");
    }

    #[test]
    fn a_written_artifact_joins_passes_and_scores_by_position() {
        let tok = tokenizer();
        let generated = tok.encode("abc").unwrap();
        let a = Alignment::build(&tok, &prompt(), SPECIALS, &generated).unwrap();
        let n = a.rows.len();

        // One wide prefill over everything but the last position, then a
        // single-position pass: prefill then decode, as a real run has.
        let passes = vec![
            Pass {
                index: 0,
                start: 0,
                len: n - 1,
            },
            Pass {
                index: 1,
                start: n - 1,
                len: 1,
            },
        ];
        let scores = vec![Flag {
            position: n - 1,
            score: 0.75,
        }];

        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("nested").join("alignment.jsonl");
        write(
            &out,
            &Header {
                run_id: "R-20260101-000000-abcdef",
                model_hash: "deadbeef",
                probe: Some("test"),
                probe_layer: Some(3),
            },
            &a,
            &passes,
            &scores,
        )
        .unwrap();

        let body = std::fs::read_to_string(&out).unwrap();
        let mut lines = body.lines();
        let head: serde_json::Value = serde_json::from_str(lines.next().unwrap()).unwrap();
        assert_eq!(head["artifact"], "alignment");
        assert_eq!(head["run_id"], "R-20260101-000000-abcdef");
        assert_eq!(head["positions"], n);
        assert_eq!(head["text"], a.text);
        assert_eq!(head["probe_layer"], 3);

        let rows: Vec<serde_json::Value> =
            lines.map(|l| serde_json::from_str(l).unwrap()).collect();
        assert_eq!(rows.len(), n, "one row per position");
        for (i, row) in rows.iter().enumerate() {
            assert_eq!(row["position"], i);
            let phase = if i == n - 1 { "decode" } else { "prefill" };
            assert_eq!(row["phase"], phase, "position {i}");
            assert_eq!(row["pass"], usize::from(i == n - 1), "position {i}");
        }
        // The score is present exactly where it was taken, and absent -- not
        // null -- everywhere else.
        assert_eq!(rows[n - 1]["score"].as_f64().unwrap(), 0.75);
        for row in &rows[..n - 1] {
            assert!(row.get("score").is_none(), "{row}");
        }
    }

    #[test]
    fn a_run_with_no_probe_writes_the_spans_and_omits_the_score_channel() {
        let tok = tokenizer();
        let a = Alignment::build(&tok, &prompt(), SPECIALS, &[]).unwrap();
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("alignment.jsonl");
        write(
            &out,
            &Header {
                run_id: "R-20260101-000000-abcdef",
                model_hash: "deadbeef",
                probe: None,
                probe_layer: None,
            },
            &a,
            &[],
            &[],
        )
        .unwrap();

        let body = std::fs::read_to_string(&out).unwrap();
        let head: serde_json::Value = serde_json::from_str(body.lines().next().unwrap()).unwrap();
        assert!(head.get("probe").is_none(), "{head}");
        for line in body.lines().skip(1) {
            let row: serde_json::Value = serde_json::from_str(line).unwrap();
            assert!(row.get("score").is_none(), "{row}");
            // A run with no pass record still gets every position, without a
            // pass field claiming one.
            assert!(row.get("pass").is_none(), "{row}");
            assert!(row["byte_start"].is_number(), "{row}");
        }
    }
}
