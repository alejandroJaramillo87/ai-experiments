/*
 * ladder_reference_contract.rs — the ladder's constants against the
 * generators that have to agree with them.
 *
 * Three numbers describe one thing from two sides. Size::prompts() says how
 * many prompts L4 reads and reference_greedy.py's PROMPTS says how many it
 * writes; ladder::GREEDY_TOKENS says how far each continuation runs and the
 * generator's TOKENS decides how far it actually did. Neither side can see
 * the other, and the artifact between them is JSON, which checks nothing.
 *
 * A drift downward is caught at run time -- load_greedy errors when the
 * artifact holds fewer prompts than the size needs. A drift upward is not: a
 * generator writing 8 prompts where the size wants 2 produces an artifact
 * that loads, passes, and quietly measures a third of what the header claims.
 * That is the direction this file exists for.
 *
 * The values are read out of the real module rather than re-parsed from its
 * text, for the reason every cross-language test here gives: a second reader
 * written in Rust would agree with a Rust misunderstanding of the format.
 */

use std::path::{Path, PathBuf};

use runtime::ladder::{Size, CHUNK, GREEDY_TOKENS};

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// The generators' own constants, as they declare them.
///
/// Mandatory rather than skippable: neither module imports anything outside
/// the standard library at scope, so an interpreter that cannot read them is a
/// broken environment rather than an absent optional dependency, and a gate
/// that silently stopped checking this would be worth nothing.
fn generator_constants() -> serde_json::Value {
    let root = repo_root();
    let python = runtime::pyenv::interpreter_or_panic(&root, &[]);
    const SCRIPT: &str = "import json, sys\n\
         sys.path.insert(0, sys.argv[1])\n\
         import reference_greedy as g\n\
         import reference_perplexity as p\n\
         json.dump({'prompts': g.PROMPTS, 'tokens': g.TOKENS,\n\
                    'chunk': p.CHUNK, 'corpus_tokens': p.TOKENS}, sys.stdout)\n";
    let fixtures = root.join("rust-harness/fixtures");
    let out = std::process::Command::new(&python)
        .args(["-c", SCRIPT])
        .arg(&fixtures)
        .output()
        .expect("python3 runs");
    assert!(
        out.status.success(),
        "could not read reference_greedy.py's constants: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    serde_json::from_slice(&out.stdout).expect("the probe emits JSON")
}

#[test]
fn the_greedy_generator_writes_as_many_prompts_as_every_size_reads() {
    let got = generator_constants();
    for size in Size::ALL {
        let theirs = got["prompts"][size.label()]
            .as_u64()
            .unwrap_or_else(|| panic!("reference_greedy.py declares no size {}", size.label()))
            as usize;
        assert_eq!(
            theirs,
            size.prompts(),
            "size {} reads {} prompts and the generator writes {theirs}",
            size.label(),
            size.prompts()
        );
    }
}

#[test]
fn the_greedy_generator_continues_for_as_long_as_the_rung_expects() {
    let got = generator_constants();
    let theirs = got["tokens"].as_u64().expect("TOKENS is an integer") as usize;
    assert_eq!(
        theirs, GREEDY_TOKENS,
        "ladder::GREEDY_TOKENS is {GREEDY_TOKENS} and the generator writes {theirs}"
    );
}

#[test]
fn the_perplexity_generator_scores_the_windows_and_prefixes_the_rung_expects() {
    // The windowing is the whole comparability of L3: perplexity computed in
    // different windows is a different quantity, and the two sides would
    // disagree about which tokens were scored rather than about the numbers.
    let got = generator_constants();
    let chunk = got["chunk"].as_u64().expect("CHUNK is an integer") as usize;
    assert_eq!(
        chunk, CHUNK,
        "ladder::CHUNK is {CHUNK} and the generator windows at {chunk}"
    );
    for size in Size::ALL {
        let theirs = got["corpus_tokens"][size.label()]
            .as_u64()
            .unwrap_or_else(|| panic!("reference_perplexity.py declares no size {}", size.label()))
            as usize;
        assert_eq!(
            theirs,
            size.tokens(),
            "size {} scores {} tokens and the generator writes {theirs}",
            size.label(),
            size.tokens()
        );
    }
}

#[test]
fn the_prompt_file_holds_enough_prompts_for_the_largest_size() {
    // Committed, so this is checkable in the sandbox with no model at all --
    // and the failure it prevents is a deep run that dies after loading a
    // 15 GB checkpoint.
    let path = repo_root().join("rust-harness/fixtures/reference/prompts.txt");
    let text = std::fs::read_to_string(&path).expect("prompts.txt is committed");
    let prompts = text
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .count();
    let want = Size::ALL.iter().map(|s| s.prompts()).max().unwrap();
    assert!(
        prompts >= want,
        "prompts.txt holds {prompts} prompts and the largest size needs {want}"
    );
}

#[test]
fn the_corpus_is_long_enough_for_the_largest_size() {
    // Characters rather than tokens: this runs in the sandbox, where there is
    // no checkpoint tokenizer to ask. English runs roughly four characters to
    // a token, and the margin below is wide enough that the approximation
    // cannot be what makes this pass.
    //
    // Cheap here and expensive later: without it, a deep run learns the corpus
    // is short after loading a 15 GB checkpoint.
    let path = repo_root().join("rust-harness/fixtures/reference/corpus.txt");
    let Ok(text) = std::fs::read_to_string(&path) else {
        runtime::testlog::skip(
            "L3 corpus length",
            "fixtures/reference/corpus.txt has not been fetched; \
             run fixtures/fetch_corpus.py on a machine with network access",
        );
        return;
    };
    let want = Size::ALL.iter().map(|s| s.tokens()).max().unwrap();
    let chars = text.chars().count();
    assert!(
        chars >= want * 4,
        "corpus.txt is {chars} characters and the largest size scores {want} tokens"
    );
}
