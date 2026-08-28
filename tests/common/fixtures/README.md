# Observability fixtures captured from a real run corpus

Two JSONL samples drawn from observability output the rust harness actually wrote, used by
`tests/common/test_obs_catalog_corpus.py`. Every other observability test in this directory
builds its records inline, which means those records parse by construction and cannot show
whether the Rust writer and the Python reader agree on real bytes. These files can.

## Provenance

Captured 2026-08-14 from `/mnt/ai-data/logs/harness/` in the Claude Code cloud container,
where the harness had accumulated 1,085 event files and a 993-row catalog across sessions
dated 2026-08-08 through 2026-08-14. The corpus was written by
`rust-harness/obs/src/lib.rs`, the second writer of the `schemas/obs_record.json`
contract.

The full corpus is 4.5MB and is not committed. These samples total under 10KB.

### What the corpus showed

Facts measured across all 993 catalog rows and all 2,263 event records before sampling:

- Every line parsed. `common/obs/catalog.py` skips a malformed line silently, so a drifting
  writer presents as an empty table rather than an error; across 993 rows it skipped nothing.
- Every catalog row carried exactly the six contract keys, with no additions or omissions.
- The three event key sets matched the contract's documented extras exactly: `argv` on
  `cmd.start`, `exit_code` and `duration_s` on `cmd.end`, `exc_type` and `exc_msg` on `fail`.
- 1,085 `cmd.start` events produced only 993 `cmd.end` events. The 92 runs that recorded no
  end also produced no catalog row, which is consistent with a hard abort skipping the
  handler that writes both. The catalog therefore under-counts runs, and nothing in
  `docs/observability/` states this.

## Files

| File | Contents |
|------|----------|
| `obs_runs_sample.jsonl` | 50 catalog rows: 25 exiting 0, 25 exiting 1, including the corpus's single `serve` row |
| `obs_events_sample.jsonl` | One failing run's complete event file: `cmd.start`, `fail`, `cmd.end` |

The event sample's `run_id` appears in the catalog sample, so the two files exercise the
correlation join between an event file and its catalog row.

A failing run was chosen for the event sample because it is the only run shape that carries
all three event types in one file.

### What the sample deliberately does not contain

The corpus holds only two command names, `task` and `serve`, the longest of which is five
characters. Two branches of `catalog.run_command` therefore have no real data behind them:

- The `--all` filter, which hides rows whose command is `runs`. No harness binary is named
  `runs`, so no such row can exist in a harness corpus.
- The `command_width` computation, which takes `max(7, ...)`. Every real row falls under the
  floor of seven, so the `max` branch never engages.

Both remain covered by the hand-built records in `tests/common/test_obs_catalog.py`. No
synthetic row was added to these files: a fixture that documents itself as a real capture
should not carry invented records.

The second gap was measured rather than assumed. Mutating `catalog.py` to compute the column
width per row instead of once per table leaves every test in
`test_obs_catalog_corpus.py` green, because `max(7, len(command))` equals the table width for
every command in the corpus. That is why the alignment assertion lives in
`test_obs_catalog.py` as `test_columns_align_when_commands_differ_in_width`, where a row wide
enough to make the mutation fail can be written deliberately.

## Anonymization

Absolute host paths were rewritten and `pid` was fixed. Every other field is byte-for-byte as
written. The transform:

```python
REPO = re.compile(r"/home/[^/\s\"]+/ai-experiments")
TMP = re.compile(r"/tmp/\.tmp[A-Za-z0-9]+")

def scrub(value):
    if isinstance(value, str):
        return TMP.sub("/fixture/tmp", REPO.sub("/fixture/repo", value))
    if isinstance(value, list):
        return [scrub(v) for v in value]
    return value
```

`argv` is the field that motivated this. It carries absolute paths to checkpoints, probes and
output directories, and on a workstation would name the real model tree.

Timestamps and run identifiers are unchanged. A run identifier embeds its own wall-clock
stamp and is the record's identity; rewriting it would break the correlation join these
fixtures exist to exercise.

## A note on the captured `argv`

The event sample names `tiny-gemma4.gguf` and `probe.gguf`. Both predate R10, which moved the
probe to a safetensors image and removed GGUF from `rust-harness/`. The paths are accurate
history rather than current tree references, and `.jsonl` is not among the extensions
`scripts/docs/check_retired_names.py` scans, so the capture is preserved as written.

## Regenerating

These are captures, not generated fixtures. Re-capturing requires a machine carrying a
harness corpus at `${OBS_LOG_DIR}/harness/`, and any re-capture changes the run identifiers
the tests assert against. Prefer extending the samples over replacing them.

## `run_metrics_sample.jsonl`

A run's `metrics.jsonl`: the report-envelope header record followed by eight per-step
records. Consumed by `tests/common/test_run_metrics_contract.py`, which feeds it to
`common.obs.dashboard.read_metrics` and `common.reports.validate_report`.

Unlike the two samples above this one is **generated rather than captured** — but generated
by the real writer, `rust-harness/obs/src/lib.rs`, not hand-authored. That distinction is the
point of the file. U6's premise is that the training dashboard renders a harness run with no
Python change, and a Python-authored imitation of the harness's output could only prove that
Python agrees with itself. These are bytes the Rust writer emitted.

Every value is synthetic: `/fixture/models/gemma-4-E4B`, a `git_sha` that is well-formed and
belongs to no commit, and a smooth `tokens_per_s`/`rss_gb` ramp. Nothing here was measured, so
nothing here should be quoted as a measurement. The shapes are what the tests assert —
`step` a JSON number, the envelope's four required keys, and `run_id` plus `cli` for the
event-file join.

### Regenerating

Written by a throwaway `#[test]` under `rust-harness/harness/tests/` that calls
`RunLog::begin_at`, `metrics_header`, `metric` and `end` against a tempdir and copies the
result out. It is not committed: a permanent test whose product is a committed fixture
invites the fixture and the writer to be regenerated together, which is exactly the drift the
cross-language test exists to catch. Re-deriving it means writing those few lines again
against the current API — which is the check.

## `ladder_metrics_sample.jsonl`

The `metrics.jsonl` of a conformance-ladder run: the same report-envelope header, then five
records, one per rung. Consumed by `tests/common/test_run_metrics_contract.py`.

It exists beside `run_metrics_sample.jsonl` rather than replacing it because the two are
differently shaped and the difference is the point. The harness writes a time series —
`step` counts decoded tokens, every record carries the same keys. The ladder writes five
ordered categories — `step` is a rung index, and each rung carries whichever measures its own
body produced, so `worst_cosine` appears on L1 and nowhere else. A contract asserted against
one writer cannot tell which of its clauses are the contract and which were incidentally true
of that writer; two writers can.

Generated by the real writer, not hand-authored, for the reason given above: `ladder` really
did climb, and these are the bytes it emitted. The run was against
`rust-harness/fixtures/tiny-hf-gemma4` with the fixture's own capture copied in as
`layers.json`, which is why L1 and L2 measure and L0/L3/L4 skip.

Three values are rewritten from the real run and nothing else is: the model path, the
reference path, and `git_sha`. The durations, the measures and the timestamps are as written.
They are still not measurements worth quoting — a 4-layer model with seeded weights in a
container says nothing about E4B on the workstation.

### Regenerating

```bash
cd rust-harness && cargo build --bin ladder
mkdir -p /tmp/gen/logs /tmp/gen/ref
cp fixtures/tiny-hf-gemma4/capture.json /tmp/gen/ref/layers.json
OBS_LOG_DIR=/tmp/gen/logs ./target/debug/ladder \
  --model fixtures/tiny-hf-gemma4 --reference /tmp/gen/ref
```

Then apply the three substitutions above to `/tmp/gen/logs/harness/R-*/metrics.jsonl`. Any
regeneration changes the run identifier and the durations the tests do not assert against.
