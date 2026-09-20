# harness

The bin crate: the coding agent itself, wired over the `runtime` library. `main.rs` is
the entry point and `lib.rs` exposes the same modules so the integration tests under
`tests/` drive the stack in process; both crate roots carry `#![forbid(unsafe_code)]`,
because the attribute does not propagate from one to the other.

The modules split by seam. The loop and what it needs (`agent`, `tools`, `permissions`,
`transcript`, `template`, `compact`, `subagents`) are tested against the scripted
`MockModel` from `runtime`; the surface a person or the interp client sees (`cli`,
`preflight`, `runheader`, `actionable`, `chat`, `server`, `stream`, `interplog`,
`alignment`) is tested from outside the process as well, through the binary, by the
files under `tests/`. `generator` is the text-to-token seam between the two halves.

## Sources

<!-- inventory:rust:start -->

| File | Tests | Purpose |
|------|-------|--------|
| `src/actionable.rs` | 13 | A remedy carried on an error, so a failure states its fix. |
| `src/agent.rs` | 0 | The core loop: render transcript, generate, dispatch tools. |
| `src/alignment.rs` | 8 | The token-to-text alignment artifact. |
| `src/chat.rs` | 12 | Interactive REPL over the generator, without the agent loop. |
| `src/cli.rs` | 24 | The harness command line as data. |
| `src/compact.rs` | 8 | Transcript compaction via model-written summaries. |
| `src/generator.rs` | 2 | The text↔token seam between the agent loop and the runtime. |
| `src/interplog.rs` | 7 | The served-request interp log. |
| `src/lib.rs` | 0 | Library surface of the coding agent, exposed for integration tests; the binary in main.rs is a thin entry point over these modules. |
| `src/main.rs` | 4 | Single-binary local coding agent. |
| `src/permissions.rs` | 8 | YAML rule file evaluated before each tool dispatch. |
| `src/preflight.rs` | 17 | What a run would fail on, answered without loading weights. |
| `src/runheader.rs` | 5 | The resolved configuration of a run, built once and rendered twice. |
| `src/server.rs` | 23 | The OpenAI-shaped loopback endpoint the interp pipeline calls. |
| `src/stream.rs` | 3 | Token streaming from the sampler to the terminal renderer. |
| `src/subagents.rs` | 0 | Fork-and-interleave scheduling. |
| `src/template.rs` | 16 | Gemma 4 chat template rendering. |
| `src/tools.rs` | 21 | The agent's tool set: read, write, edit, bash, glob, todo. |
| `src/transcript.rs` | 9 | JSONL session persistence with resume. |
| `src/util.rs` | 7 | Wildcard matching shared by glob and permission rules. |

<!-- inventory:rust:end -->

## Integration tests

Each file spawns the built binary or drives the library over a scripted generator; the
ones handing artifacts to Python resolve the interpreter through `runtime::pyenv` and
report `[SKIP]` when none can import what they need.

<!-- inventory:rust_tests:start -->

| File | Tests | Covers |
|------|-------|--------|
| `tests/actionable_surface.rs` | 3 | A failing run states its fix, and does not record it. |
| `tests/agent_e2e.rs` | 10 | Tier 2 acceptance test for the phase D harness loop. |
| `tests/alignment_artifact.rs` | 8 | The written alignment.jsonl against its contract. |
| `tests/cli_surface.rs` | 11 | What the binary does with its arguments, through a pipe. |
| `tests/context_cap.rs` | 2 | --context is a cap the run keeps, bounded by the trained window. |
| `tests/control_vector_range.rs` | 3 | --control-vector-layer-range through the binary. |
| `tests/cpu_smoke.rs` | 1 | Phase 3 sandbox acceptance: the full harness stack (agent loop, chat template, tokenizer, session, sampler) runs end to end over the real CPU backend with synthetic weights. |
| `tests/generator_e2e.rs` | 3 | Tier 2 test for the text↔token seam. |
| `tests/obs_contract.rs` | 8 | The harness run log against the shared observability contract, and against the Python reader that consumes it. |
| `tests/obs_records.rs` | 3 | RH.13's run-level records, produced by the binary. |
| `tests/preflight_surface.rs` | 12 | `--status` through the binary. |
| `tests/probe_identity.rs` | 4 | End-to-end coverage of the probe/model identity refusal. |
| `tests/run_header.rs` | 5 | The resolved configuration reaches both the terminal and the run record, measured against a real invocation of the binary. |
| `tests/sampling_effect.rs` | 4 | The sampling flags change what is sampled, not only what the parser accepts. |
| `tests/sampling_flags.rs` | 5 | The two sampling-group flags nothing reached through the binary: --top-p and --compact-at. |
| `tests/serve_bind_addr.rs` | 3 | --serve-addr decides which interface the endpoint answers on. |
| `tests/serve_interp_log.rs` | 4 | The served-request log, end to end through the binary. |
| `tests/serve_snapshot.rs` | 1 | A served session neither restores nor writes session.kv. |
| `tests/server.rs` | 10 | The endpoint over a real socket. |
| `tests/serving_contract.rs` | 2 | The endpoint against the Python client that will call it. |
| `tests/session_status.rs` | 3 | The two startup lines a run prints about the directory it starts in: where its tool permissions come from, and whether it resumed a transcript. |
| `tests/single_thread.rs` | 1 | The harness's own --single-thread reaches the model. |
| `tests/status_streams.rs` | 6 | The stream split, asserted from outside the process. |
| `tests/subagents.rs` | 1 | Tier 2 acceptance test for phase E fork and interleave. |
| `tests/template_contract.rs` | 2 | The chat template seam between the tiers. |
| `tests/tokenizer_override.rs` | 2 | --tokenizer against the tokenizer beside the checkpoint. |
| `tests/trace_levels.rs` | 2 | --trace-level selects how deep a Chrome trace goes. |
| `tests/trace_structure.rs` | 3 | A Chrome trace has the documented span shape, and a real run's per-name totals are printed for the runbook. |
| `tests/tracing_spans.rs` | 4 | Phase 2 acceptance (harness side): tool.exec and agent.fork spans, and a Chrome trace that loads as JSON with the documented span names present. |
| `tests/version_and_record.rs` | 2 | The two harness flags the campaign ran by hand or not at all, pinned through the binary. |

<!-- inventory:rust_tests:end -->

## Related documentation

- [../README.md](../README.md) — the workspace
- [../../docs/rust-harness/testing.md](../../docs/rust-harness/testing.md) — how the
  workspace is tested, tier by tier
- [../../docs/rust-harness/runtime-design.md](../../docs/rust-harness/runtime-design.md) —
  the design the loop implements
