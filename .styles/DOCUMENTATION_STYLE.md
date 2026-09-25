# Documentation Style Guide

Documentation standards for this repository. Follow Unix manual-page conventions: accuracy over
aesthetics, minimalism, unambiguous language, working examples, no emojis or decoration.

## Table of Contents

<!-- toc:start -->

1. [Scope](#scope)
2. [Document class](#document-class)
3. [Writing Standards](#writing-standards)
4. [Document Structure](#document-structure)
5. [README shape](#readme-shape)
6. [Formatting](#formatting)
7. [Technical Standards](#technical-standards)
8. [Language Conventions](#language-conventions)
9. [Cross-References](#cross-references)
10. [Prohibited](#prohibited)

<!-- toc:end -->

## Scope

Document actual behavior, not intended behavior. Include performance measurements and credit
sources. Target individual developers on a local AI workstation — no cloud platforms, no CI/CD
pipelines, no production/distributed-systems framing.

## Document class

Every document belongs to one of four classes. The class decides whether drift against the
current repository is a bug to fix or a record to preserve, so it is stated in the document
rather than inferred by whoever reads it next.

| Class | What it is | Obligation |
|-------|------------|------------|
| Living reference | Describes the tree as it is now | Correct it when the tree moves |
| Frozen record | What was decided or measured on a day | Never edit the body; drift is expected and correct |
| Operational | A runbook executed against the repository or the host | Keep the commands runnable; it has no reader to please. Entry ids follow `docs/verification/protocol.md` |
| Parked proposal | A recorded menu of work that is not scheduled | Dated at authoring; edited only as items graduate out |

Living reference is the default and carries no marker. The other three declare themselves in
a single bold line directly under the H1, before the description paragraph:

```markdown
# D4 memory-fit calculator: implementation investigation

**Frozen record — 2026-07-14. Drift against the current repository is expected and correct;
this document is never updated.**
```

Dates are ISO `YYYY-MM-DD`, always. Do not write "July 2026" or "as of 2026-07".

**A frozen record must also be findable by path**: its filename ends `_investigation.md`, or
it sits under `<topic>/investigations/`. This is not a stylistic preference —
`scripts/docs/check_doc_refs.py` and `scripts/docs/check_retired_names.py` both exempt frozen
records by exactly those two patterns (`EXEMPT_PARTS`, `EXEMPT_SUFFIXES`). A frozen record
named anything else is gated as though it were living, and the gate will demand that its
historical paths still resolve.

`docs/verification/` is **not** exempt. It was, by directory prefix, until 2026-08-08; both
gates now carry an empty `EXEMPT_PREFIXES` with the reasoning recorded in
`check_doc_refs.py`, because the exemption bought nothing and cost coverage of the most
path-dense documentation in the tree. Operational runbooks still belong there, but their
paths are checked like any other document's.

## Writing Standards

- **Active voice**: "The system allocates memory", not "Memory is allocated by the system".
- **Present tense** for current state: "The GPU runs at 2.55 GHz".
- **Imperative mood** for instructions: "Run the setup script".
- Tone is professional, direct, objective, concise — no marketing, no subjective quality judgments.
- One idea per sentence, one topic per paragraph; define before use; concrete over abstract.

## Document Structure

```markdown
# Title

Brief description paragraph explaining the document's purpose.

## Table of Contents      (required over 500 words and four sections; generated)

<!-- toc:start -->
<!-- toc:end -->

## Main Content           (logical H2/H3 hierarchy; avoid H4)
```

One H1 per document. A document over 500 words with four or more H2 and H3 headings
carries a `## Table of Contents`, and the list under it is never written by hand: it sits
between a `<!-- toc:start -->` and `<!-- toc:end -->` pair and
`python scripts/docs/gen_toc.py <file>` renders it from the headings, one numbered row per
H2 and an indented row per H3, each linking the heading's anchor. The same command adds
the section, or the markers under an existing one, when the document has none, and
`make quality` fails a document over the threshold without the pair or with a stale list.
A shorter document may carry the pair and is rendered the same way. A README never needs
one, and a frozen record is never given one.

## README shape

A README is the directory's router, not a document. It routes one level down and stops:

1. The H1, then the description paragraph. Its first sentence stands alone, because the
   parent's generated table reads it.
2. The generated tables, directly after: `dirs` when the directory has subdirectories,
   `pages` when it has pages beside the README, both when it has both. A directory of
   frozen records is listed like any other; generation reads the records, it never edits
   them.
3. Orientation a table cannot render, in a few paragraphs at most: the reading order,
   what is frozen, the traps.
4. Related documentation.

Anything longer is a page beside the README, which the `pages` table then lists.
`scripts/docs/check_inventory.py` holds every directory under `docs/` to this: a missing
README, or a README without the table its directory's shape needs, fails `make quality`.

## Formatting

| Element | Format |
|---------|--------|
| Emphasis | **Bold** |
| Code, commands, file names, variables | `backticks` |
| URLs | bare or markdown link |

Code blocks:

**A command block is pasted, not read, so it carries no prompt marker.** Never
prefix a command with `$ ` or `# ` to mark it as a command: pasted into a shell, the
`$` fails as a command not found and the `#` comments the line out, so the block
silently runs nothing. Write the command exactly as the shell must receive it. A
literal `$` stays only where it is part of the command (`$HOME`, `$(...)`, `"$MODEL"`),
and a command that needs root spells `sudo`.

```bash
make install
sudo apt-get update
cargo run --release -p harness -- --model "$MODEL" --text-only
```

Keep commands and their output in separate blocks; output lines use no prefix either.
When a command and what it printed must be shown together, tag the block `console`
and keep the `$ ` prompt: that is a transcript, the prompt is what tells the command
from its output, and nothing in it is meant to be pasted. The other transcript is an
Observed block of a verification runbook, recorded as typed and never edited.
`scripts/docs/check_command_prompts.py` enforces this: it reads `bash`, `sh`, `shell`
and `zsh` blocks outside Observed sections and frozen records, and fails on a prompt.

Always specify the language on programming code blocks. Use pipe tables with headers.
Unordered lists use `-`; ordered lists use `1. 2. 3.`.

## Technical Standards

- Metrics always include units and precision: `286.85 tokens/second`, `15.3GB VRAM`, `45ms`.
- Version numbers use full form: `580.65.06`, `CUDA 13.0.88`.
- File paths are absolute from repo root: `/docker/llama-gpu/entrypoint.sh`.
- Document full commands with flag explanations.

## Language Conventions

- **Sentence case** headings; lowercase commands/files; capitalize proper nouns (Docker, NVIDIA);
  uppercase acronyms (GPU, CPU, VRAM).
- Define abbreviations on first use, except common ones (CPU, GPU, RAM, API, URL, JSON, OS, VM).
- Spell out one–nine in body text; numerals for 10+ and always for metrics, versions, ports, specs.
- Oxford comma; single space after periods; hyphenate compound modifiers ("production-ready system").

## Cross-References

Internal links use relative paths; external links provide context:

```markdown
See [GPU optimization guide](../optimizations/gpu/gpu-optimizations.md)
Uses [llama.cpp](https://github.com/ggerganov/llama.cpp) for inference
```

## Prohibited

- Emojis or emoticons
- Exclamation marks (except `#!/bin/bash` and `**Warning!**` in warnings)
- Marketing language ("revolutionary", "amazing") and subjective adjectives ("easy", "powerful")
- Contractions ("don't", "can't")
- Informal language ("lots of", "a bunch of"), rhetorical questions
- Personal pronouns in technical descriptions
