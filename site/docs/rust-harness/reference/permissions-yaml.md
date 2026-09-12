---
title: permissions.yaml
---

# permissions.yaml

The rule file the agent loop consults before every tool call. It is read once at startup from
the working directory. If the file is absent, every call is put to the terminal.

## Format

```yaml
rules:
  - tool: read
    action: allow
  - tool: write
    pattern: "scratch/**"
    action: allow
  - tool: bash
    pattern: "cargo test*"
    action: ask
  - tool: "*"
    action: deny
```

The file is a mapping with one key, `rules`, holding a list. Each rule has:

| Key | Required | Values |
|-----|----------|--------|
| `tool` | yes | `read`, `write`, `edit`, `bash`, `glob`, `todo`, or `*` for any tool |
| `pattern` | no | A wildcard pattern matched against the call's permission target; absent matches every target |
| `action` | yes | `allow`, `deny`, or `ask` |

Unknown keys are refused at startup rather than ignored, because a misspelled `pattern` would
otherwise leave the field empty and widen a scoped rule into a blanket one. An unknown tool
name is refused for the same reason: a rule that can never match is a rule someone believes is
in force.

## Evaluation

The first rule whose `tool` matches the call and whose `pattern`, if present, matches the
target decides the call. No matching rule means `ask`. `ask` prints
`allow <tool> <target>? [y/N]` to stderr and reads one line from stdin; only `y` or `yes`
allows. A denied call is answered to the model as `permission denied` and the session
continues.

## The permission target

| Tool | Target the pattern is matched against |
|------|---------------------------------------|
| `read`, `write`, `edit` | The path, normalized relative to the working directory |
| `bash` | The whole command line |
| `glob` | The pattern |
| `todo` | Empty; a rule for `todo` matches only without a `pattern` |

Patterns are glob-style: `*` matches any run of characters within a path segment, `**`
matches across segments, and `?` matches one character other than `/`. There are no negations
and no regular expressions. Paths are normalized before matching, so a path spelled with a leading `./` or
with a `..` segment meets the same rule as its plain spelling; commands are matched as written.

## Two limits

Both are properties of what a rule is, and neither is enforcement:

1. **Bash is not confined by file-tool rules.** A `deny` on `write` for a path does not stop
   `bash` from running a command that writes there. The `bash` tool runs with the working
   directory as its current directory and nothing more. Confining it would need a sandbox, not
   a rule.
2. **A bash prefix-allow is an allow-anything rule.** The pattern is matched over the whole
   command line and `**` matches shell metacharacters, so `bash` with pattern `cargo **` and
   action `allow` permits `cargo test; rm -rf ~`. Use `ask` for `bash` unless allowing anything
   is the intent.
