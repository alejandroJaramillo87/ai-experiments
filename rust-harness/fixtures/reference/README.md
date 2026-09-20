# Ladder reference artifacts

Where the correctness ladder's reference artifacts live, and the two committed inputs
they are generated from. Every rung compares the harness against `transformers`; nothing
here compares it against its own earlier output.

| Path | Tier | Committed |
|------|------|-----------|
| `../tiny-hf-gemma4/` | T | yes, weights included |
| `e4b/<size>/` | E | no, generated on the workstation by `make ladder` and gitignored |
| `corpus.txt` | E, L3 | yes: *The Adventures of Sherlock Holmes*, fetched and normalized by `../fetch_corpus.py` |
| `prompts.txt` | E, L4 | yes: continuations, not questions; a size takes a prefix, so append, never insert |

The rungs, the generators, the sizes, what each artifact must record and why the corpus
is what it is are in
[../../../docs/rust-harness/correctness-ladder.md](../../../docs/rust-harness/correctness-ladder.md);
`../../Makefile` is the source of truth for each generator's arguments.
