# The mongod log format

What this tool parses, and the two properties of the format that shaped the design.

## Table of Contents

- [The envelope](#the-envelope)
- [attr has no fixed shape](#attr-has-no-fixed-shape)
- [The format is already templated](#the-format-is-already-templated)
- [What normalization lifts](#what-normalization-lifts)
- [Malformed lines are data too](#malformed-lines-are-data-too)

## The envelope

Since MongoDB 4.4 every log line is one JSON object. `logtriage/mongod/records.py` reads it.

| Field | Meaning |
|-------|---------|
| `t` | Timestamp, as `{"$date": "..."}` with an offset |
| `s` | Severity: `F`, `E`, `W`, `I`, `D1`–`D5` |
| `c` | Component: `NETWORK`, `COMMAND`, `REPL`, `STORAGE`, and others |
| `id` | Numeric identifier of the call site that wrote the line |
| `ctx` | Thread or connection context |
| `msg` | The message template, fixed for a given `id` |
| `attr` | Structured attributes; shape depends on `id` |
| `svc` | Since 8.0: `"R"` for a router, `"S"` for a shard |

`svc` is what separates `mongod` from `mongos` lines in a merged log. It is carried on
`LogRecord` and defaults to empty for pre-8.0 logs, which is a real distinction rather than a
missing value.

Timestamps are parsed as timezone-aware UTC. Everything downstream assumes that, which is why
`store/base.py` restores the zone on records read back from MongoDB — BSON stores a date as epoch
milliseconds with no zone, and pymongo returns a naive datetime unless the client sets
`tz_aware`.

## attr has no fixed shape

`attr` is a function of `id`. A slow-query line carries `planSummary`, `keysExamined`,
`docsExamined`, `nreturned`, `durationMillis`; a connection line carries `remote` and
`connectionId`; an election line carries something else again.

Nothing could query that uniformly, which is what `mongod/normalize.py` exists for: it lifts the
fields diagnostics actually filter and group on to the top level of `NormalizedRecord`, leaving
`attr` intact underneath. The lifted fields are the closed `FacetField` set in `store/base.py`.

## The format is already templated

Every line carries a numeric `id` identifying the call site, paired with a fixed `msg`. So
`(component, id)` **is** an exact template — no similarity threshold, no clustering, nothing to
tune.

This is why log-template clustering, the obvious tool for this job, is redundant here.
`drain3` is a declared dependency and is used only on the genuinely free-text fields,
`attr.errmsg` and `attr.error`, where there is no identifier to key on.

It is also why the detectors match on **component and message text**, not on identifiers: a
detector keyed on an id stops firing silently after a server upgrade renumbers it. See
[log-ids.md](log-ids.md) for which ids are actually verified.

## What normalization lifts

`--redact` replaces literal command values with type placeholders at ingest. `attr.command` is
where a query's literal values live — the privacy risk, and for a large `$in` also the bulk of
the line. Redaction keeps the structure and drops the values, so a namespace and a plan summary
survive while the data does not. Arrays keep their first redacted element and a count.

## Malformed lines are data too

A truncated line becomes a `ParseError`, not an exception. Rotation truncates lines mid-write,
and a tool that dies partway through a 100MB corpus is useless exactly when it is needed.

The count of unparseable lines is reported by `profile` and `triage`, because evidence that was
never read changes how much a conclusion is worth. A file with no parseable records at all is a
named failure with the command to check it, not an empty report.

## Related

- [log-ids.md](log-ids.md) — which identifiers are documented and which are representative
- [../design/citations.md](../design/citations.md) — how a record becomes a citation
- The MongoDB manual's own reference is in the fetched corpus at
  `reference/log-messages.txt`; see [../corpus/README.md](../corpus/README.md)
