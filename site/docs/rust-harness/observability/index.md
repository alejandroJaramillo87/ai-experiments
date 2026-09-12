---
title: Observability
---

# Observability

The harness is instrumented from the inside rather than watched from the outside. Four
mechanisms, at four grains, each with its own page:

| Mechanism | Grain | Always on | Where it goes |
|-----------|-------|-----------|---------------|
| [Run records](run-records.md) | One run: start, phases, configuration, end, and a per-step metrics series | Yes | `${OBS_LOG_DIR}/harness/` |
| [Flight recorder](flight-recorder.md) | One forward call: the input positions and the output logits, plus every sampled token | With `--record` | The named file |
| [Tracing](tracing.md) | One span: session, forward pass, layer, block, or kernel, with timing | With `--trace-chrome` | The named file |
| [rr debugging](rr-debugging.md) | One instruction: record a run once, replay it in gdb with reverse execution | Workstation tooling | rr's trace directory |

The run records are the durable artifact and are written whether or not anyone asked. The
flight recorder and the trace are opt-in because each costs disk in proportion to the run.
rr is not a feature of the binary but a property of it: the CPU backend is deterministic and
keeps threads out of the decode path, which is what lets rr record it.

Two facts are shared by all four. Turning any of them on changes no value the model computes;
the workspace asserts that the logits are bit-identical with and without a trace subscriber,
and the recorder wraps the model without touching it. And the run id minted for the run
records is the join key: the alignment artifact carries it, the metrics file carries it, and
the event file is named by it.

The field-level contracts for the run records are the two files `schemas/obs_record.json` and
`schemas/run_metrics.json` in the repository. They are key documents rather than a validator's
schema, and the pages here restate them in words.
