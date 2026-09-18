# JSON report schema

The shared header convention for every JSON report the repository emits. The
D8 benchmark harness was never implemented; this convention existed as a
pattern followed by the implemented report emitters and is now extracted
into one documented schema with a shared validator, `common/reports.py`.

This is not the run-record contract. The event and catalog records that
`python -m training` and `python -m interp` write, and that the Rust harness
writes independently, are specified by `schemas/obs_record.json` and described
in [../../schemas/README.md](../../schemas/README.md). The two share no keys.
The only point of contact is the metrics file, whose first record is built with
`report_header("run_metrics")` and so must satisfy the envelope below.

## Envelope

Every report is a JSON object whose header carries:

| Field | Requirement |
|-------|-------------|
| `schema_version` | integer >= 1; bumped on any breaking payload change |
| type discriminator | exactly one non-empty string among `report`, `kind`, or `artifact` at the top level |
| timestamp | `timestamp` or `created`, non-empty string (local time with UTC offset, or UTC ISO-8601) |
| `git_sha` | non-empty string at the top level, or under `metadata` |

The payload below the header is emitter-owned. `common.reports.validate_report`
raises `ReportSchemaError` listing every header violation;
`common.reports.report_header` builds a conforming header for new emitters.

## Current emitters

| Emitter | Discriminator | Report |
|---------|---------------|--------|
| `interp/organisms/verify.py` | `report: organism-verification` | organism verification |
| `interp/sae/evaluate.py` | `artifact: sae_eval` | SAE evaluation |
| `interp/probes/evaluate.py` | `kind: probe_evaluation` | W2 probe evaluation |
| `interp/probes/monitor.py` | `kind: probe_monitor` | W4 monitor snapshot |
| `interp/steering/scorecard.py` | `kind: steering_scorecard` | W1 steering scorecard |
| `training/eval/scorecard.py` | `kind: scorecard` | F7 eval scorecard |

The conformance unit tests (`tests/interp/test_report_envelope.py`,
`tests/training/test_report_schema.py`) assert every emitter's output
validates; a new emitter joins the table and the tests.

Artifact manifests written through `interp/manifest.py` and the shard/dataset
manifests share the `artifact`/`created` vocabulary but are directory
contracts validated by their own consumers, not reports.

## Future emitters

The D8 train-bench harness (not implemented) and any JSON-emitting inference
benchmark must adopt this envelope. Start from
`common.reports.report_header(<report type>)` and validate the assembled
report with `common.reports.validate_report` in the emitter's tests.
