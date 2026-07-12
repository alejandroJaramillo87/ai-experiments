# Documentation Style Guide

Documentation standards for this repository. Follow Unix manual-page conventions: accuracy over
aesthetics, minimalism, unambiguous language, working examples, no emojis or decoration.

## Scope

Document actual behavior, not intended behavior. Include performance measurements and credit
sources. Target individual developers on a local AI workstation — no cloud platforms, no CI/CD
pipelines, no production/distributed-systems framing.

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

## Table of Contents      (required only for documents > 500 words)

## Main Content           (logical H2/H3 hierarchy; avoid H4)
```

One H1 per document. Include a `## Table of Contents` only when over 500 words.

## Formatting

| Element | Format |
|---------|--------|
| Emphasis | **Bold** |
| Code, commands, file names, variables | `backticks` |
| URLs | bare or markdown link |

Code blocks:

```bash
$ make install        # user commands: $ prefix
# apt-get update      # root commands: # prefix
```

```
Output lines use no prefix.
```

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
