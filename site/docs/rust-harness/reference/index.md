---
title: Reference
---

# Reference

One page per binary, plus the environment variables, the exit codes, and the permissions file.

## The binary pages are generated

Each binary page holds that binary's `--help` output verbatim. `harness` renders its help from
the same table its parser reads, and the seven runtime binaries do the same from their own
tables, so the help cannot describe a flag the binary does not accept or omit one it does. The
pages are written by `make site-gen` in the repository, which runs each binary with `--help`,
and a check in the repository's quality gate fails when a page and its table disagree. To
change a page, change the table and regenerate.

| Binary | Purpose |
|--------|---------|
| [harness](harness.md) | The coding agent loop, the chat REPL, and the OpenAI-shaped server |
| [checkpoint-info](checkpoint-info.md) | A checkpoint's config and tensor table, and the backend's compatibility verdict |
| [activation-capture](activation-capture.md) | Last-token residual activations in the capture shard format |
| [attention-capture](attention-capture.md) | Post-softmax attention weights for chosen layers and query heads |
| [ladder](ladder.md) | The seven-rung correctness gate against a transformers reference |
| [layer-diff](layer-diff.md) | Per-layer divergence of the residual stream against a reference capture |
| [long-context-golden](long-context-golden.md) | Writes the committed long-prefill logits fixture |
| [tiny-model](tiny-model.md) | Writes a synthetic fixture checkpoint |

## Reading the harness help

The `harness` help groups flags into six sections. The usage line lists every non-mode flag in
its brackets and ends with the alternation of the three modes and the positional task. Three
annotations recur in the flag descriptions:

- **`(default X)`** is the value used when the flag is absent. A flag without one has no
  meaningful default: it is either required, or its absence turns the feature off.
- **`requires --flag`** names a flag this one is meaningless without. Giving it alone is a
  usage error and exits 2.
- **`no effect when ...`** names a condition under which the flag is accepted but does
  nothing. The harness warns at run time when it applies rather than failing, because a run
  that would have worked must not start failing.

`--model` is the one required flag. Everything after a bare `--` is task text, even if it
begins with a dash. `--help`, `-h`, `--version`, and `-V` answer and exit before anything is
configured.

## Reading the utility help

The seven utilities print a flat option list under a usage line. A flag marked `required` in
its description is also unbracketed in the usage line. Two of them take their destination
positionally: `checkpoint-info <checkpoint-dir>`, `long-context-golden <out.json>`, and
`tiny-model <output-stem> [e4b] | <output.json> bpe`. All seven answer `--help` and `-h` and
none takes `--version`.
