# Hardware tiers

The workstation's share of the gate. It covers `make quality-full`, the three hardware
tiers it selects between, the rule that a hardware test fails rather than skips, how each
workstation test pairs with a sandbox half, and the correctness ladder. Everything here is
a GPU consumer, so the one-job-at-a-time rule in `AGENTS.md` applies across every worktree.

## Table of Contents

<!-- toc:start -->

1. [`make quality-full`](#make-quality-full)
2. [The three tiers](#the-three-tiers)
3. [The four subsystems](#the-four-subsystems)
4. [Sandbox half, workstation half](#sandbox-half-workstation-half)
5. [The correctness ladder](#the-correctness-ladder)
6. [Related](#related)

<!-- toc:end -->

## `make quality-full`

`make quality` is the push gate and never runs pytest; `make test` runs the suite with the
hardware tiers deselected. `make quality-full` is the one command that runs all of it, on
the workstation:

```bash
make quality-full            # quality, then the suite, then the hw_smoke tier
make quality-full HW=full    # ... and the hw_full tier as well
make quality-full HW=e4b HARNESS_E4B_MODEL=/mnt/ai-data/models/hf/gemma-4-E4B-it
```

`HW` selects the tier and is deliberately not named `TIER`, which `make worktree-<name>`
uses to mean how a venv is provisioned. Three refusals, all before any work is done. In a
contained agent sandbox the target does not run at all:

```
[ERROR] quality-full: this target requires the workstation
  Found: AI_SANDBOX is set, so there is no GPU, no /mnt/ai-data, and no running services.
  Solution: run the sandbox-executable half here, then run this on the workstation
  Command: make quality
```

On a workstation whose environment is the torch-free tier, where the suite would skip its
way to green:

```
[ERROR] quality-full: torch is not importable, so the suite would skip its way to green
  Found: the environment is the torch-free sandbox tier; the run would report [TIER] sandbox
  Solution: provision the full tier
  Command: make install
```

And `HW=e4b` with `HARNESS_E4B_MODEL` unset, since every check in that tier would skip. The
target prints the serialization reminder rather than enforcing it.

## The three tiers

| Tier | Selected by | Contents | When |
|------|-------------|----------|------|
| `hw_smoke` | `make quality-full`, or `pytest -m hw_smoke` | under five minutes total, no model downloads, temporary-directory writes only: the CPU, GPU, RAM and SSD suites at their smoke depth, the preflight on the real checkpoint, the recipe gates | before every verification session |
| `hw_full` | `HW=full`, or `pytest -m hw_full` | near-capacity VRAM, RAM and disk; real model loads; the training loops, the interp chain, the ladder, the harness run records | per verification phase, and after any environment change (driver, image, lockfile) |
| `hw_e4b` | `HW=e4b`, or `pytest -m hw_e4b` | the checks whose question is the released E4B checkpoint itself, plus `make -C rust-harness test-e4b` | when the checkpoint is provisioned; a precondition, not a depth |

## The four subsystems

The tiers say how deep a run goes; they say nothing about which part of the tree it is
proving, and the `hw_full` tier holds minutes of GPU for the training loops next to
minutes of E4B chats for the harness. So every hardware test also carries exactly one
subsystem marker, and one target runs one tier of one subsystem:

| Subsystem | Marker | Holds |
|-----------|--------|-------|
| rust harness | `harness` | the RH and RHX entries, the ladder, `checkpoint-info` |
| training | `training` | the FT and FS entries, the tiny-recipe step, the TRL resume, the eval pair |
| interp | `interp` | the chain on real weights, the harvest hook and disk preflight, the shared-KV weights |
| platform | `platform` | the WI-3 resource suites at both depths, the venv lock |

```bash
make hardware-tests HW=full SUITE=training      # pytest -m "hw_full and training"
make hardware-tests SUITE=harness               # the smoke tier, harness only
make hardware-tests HW=e4b SUITE=harness HARNESS_E4B_MODEL=/mnt/ai-data/models/hf/gemma-4-E4B-it
```

`SUITE=all` is the default and drops the subsystem clause. The target carries
`quality-full`'s refusals and, at `HW=e4b`, adds the rust workspace's E4B tests when the
harness is in the selection. Selecting by hand is `-m "hw_full and training"`, never a
bare `-m training`: a command-line `-m` replaces `addopts`, so the bare form re-enables
every tier. `test_markers.py` holds every test to exactly one subsystem, because pytest
unions the marks of a function, its class and its module, and a test in two subsystems
would run twice across a per-suite session.

**A hardware test fails rather than skips when the hardware it exists to prove is absent.**
The `cuda` fixture asserts CUDA is available and `require_binary()` asserts the release
harness was built. The one exception is a real-model load, which skips cleanly when the
weights are not on disk, because gated hub access is out of scope by decision; read the
skip count for that reason. Modules import without torch so a sandbox collects them
cleanly and deselects them. `tests/hardware/test_markers.py` holds every test to exactly
one tier marker, at least one resource marker and exactly one subsystem marker, and every
`hw_*` marker to `addopts`.
The directory's README is [../../tests/hardware/README.md](../../tests/hardware/README.md).

## Sandbox half, workstation half

Most workstation tests are the second half of a verification entry whose first half runs
in the sandbox, against a fixture or a scripted generator. The sandbox half proves the
shape; the workstation half proves it on real weights, real hardware, or the real default
paths. The table is what the pairing is; the entries themselves are in
[../verification/](../verification/).

| Entry | Sandbox half | Workstation half | Tier | Subsystem |
|-------|--------------|------------------|------|-----------|
| RH.1a shared-KV weights | `tests/interp/test_weight_stats.py` on synthetic draws | `test_shared_kv_weights.py` reads every attention projection of E4B | `hw_e4b` | `interp` |
| RH.1b correctness ladder | Tier T, the same rungs against a tiny committed Gemma 4, in `cargo test` | `test_ladder.py` drives `make ladder SIZE=quick` and reads its run record | `hw_full` | `harness` |
| RH.9 interp chain | none: the question is the chain on real weights | `test_interp_chain.py` runs capture, probe, monitor and steer on E2B-it | `hw_full` | `interp` |
| RH.13 run-level records | `obs_records.rs` spawns the binary for a success and a forced failure | `test_obs_records.py` adds the default log root and a real checkpoint | `hw_full` | `harness` |
| RH.14 the endpoint | `serve_snapshot.rs` and `server.rs` | `test_harness_endpoint.py`: a real checkpoint behind the socket, the client's own health budget | `hw_full` | `harness` |
| RH.16 preflight | `preflight_surface.rs` and `probe_identity.rs` over the fixture | `test_preflight.py`: `--status --deep` against the real weights, printing what the hash costs | `hw_smoke` | `harness` |
| RH.19 alignment artifact | `alignment_artifact.rs` holds the file to `schemas/alignment.json` | `test_alignment.py` decodes every row's `token_id` with `transformers` over the real vocabulary | `hw_e4b` | `harness` |
| RH.20 compaction | `session.rs` reaches the rewind floor on a four-position window | `test_compaction.py` drives a chat over the real 512-token window until it compacts twice | `hw_e4b` | `harness` |
| RH.21 residual stream | `layer_diff_numerics.rs` brackets the two tolerances on a tiny capture | `test_layer_diff.py` reads every row over the real 42 layers | `hw_e4b` | `harness` |
| RH.22 `checkpoint-info --check` | `checkpoint_info_check.rs` reaches both refusal paths | `test_checkpoint_info.py` runs the pair of verdicts on the released E4B | `hw_e4b` | `harness` |
| RH.36 attention capture | `attention_capture.rs` over a three-token window | `test_attention_capture.py` on E4B past the real window, mass to one over hundreds of keys | `hw_e4b` | `harness` |
| RHX.1 run header | `run_header.rs` over the fixture | `test_run_header.py`: all nine lines on E4B with the registered probe, read back off the obs root | `hw_full` | `harness` |
| RHX.2 cost lines | `status_streams.rs` pins one `Model load:` and one `Prefill:` line | `test_status_costs.py`: two turns of unequal length, the second prefill read as a delta | `hw_full` | `harness` |
| RHX.5 a capture's run record | `capture_obs.rs` over the fixture | `test_capture_run_record.py` reads header, phase, catalog row and no run directory off a real capture | `hw_full` | `harness` |
| RHX.6 a failure's remedy | `actionable_surface.rs` pins the three-line shape | `test_remedy_runs_as_printed.py` pastes each printed `Command:` into a shell | `hw_full` | `harness` |
| RHX.10 permissions | `agent_e2e.rs` and `session_status.rs` over a scripted generator | `test_permissions.py` answers the prompt through a pipe on E2B-it, four ways | `hw_full` | `harness` |
| RHX.11 resumed session | `session_status.rs` pins the `Session: resumed` line | `test_session_resume.py` runs two turns in one directory on E4B | `hw_full` | `harness` |
| RHX.12 chat turn marker | `chat.rs` unit tests pin both arms | `test_chat_marker.py` drives a chat through a pseudo-terminal and a pipe | `hw_full` | `harness` |
| Chrome trace | `trace_structure.rs` asserts the span relations on a fixture chat trace it writes | `make -C rust-harness trace-check TRACE=...`, by hand | none | none |
| FT.recipe-gates | `tests/training/test_recipes.py` | `test_recipe_gates.py`: the two gates a run passes before the GPU | `hw_smoke` | `training` |
| FT.train-resume, FT.hardware-test | `tests/training/test_sft_entry.py`'s checkpoint helpers and `test_obs_metrics.py`'s stop seam | `test_train_resume.py` trains 20 QLoRA steps straight, then 10 and a resume to 20, and holds the logged losses to `RESUME_LOSS_RTOL` | `hw_full` | `training` |
| FT.heldout-eval | `tests/training/test_cli_handlers_export.py`'s eval handlers on faked records and `test_lmeval.py`'s extractor | `test_heldout_eval.py` trains 20 QLoRA steps, exports, and runs `eval-holdout` and `eval-lmeval` on base and merged as host processes, holding the records and the loss order, and failing with `make install` named when the venv lacks lm-eval's ifeval extra | `hw_full` | `training` |
| FT.calibrate-qlora | none: the figures are the card's | `test_step_qlora.py` steps the reference QLoRA recipe and holds the adapter count, drift, resident and peak figures | `hw_full` | `training` |
| FT.export-merged, FT.hardware-test | `tests/training/test_export_harness.py` exports the tiny fixture through the real verb and hands it to the built binaries | `test_fine_tuning_loop.py` trains ten QLoRA steps on E2B, merges, refuses a re-export, asks the same two binaries and runs one chat turn | `hw_full` | `training` |
| FS.0 to FS.8 | `tests/training/test_loop.py` and its neighbours | `test_from_scratch_loop.py` runs the reference loop, D9 overlap and pinning, and the watchdog paths | `hw_full` | `training` |
| WI-3 resource suites | none | `test_cpu.py`, `test_gpu.py`, `test_ram.py`, `test_ssd.py`, each with a smoke and a full depth | both | `platform` |

## The correctness ladder

Neither gate measures whether the inference engine computes the right numbers. `make
quality` and `make quality-full` are machinery gates: they prove the code builds, the tests
agree with themselves, and the documentation resolves. A forward pass with a transposed
weight or a wrong norm convention passes both and emits fluent text.

That is what the correctness ladder is for. Seven rungs, each against `transformers` rather
than against any earlier output of ours, run on the workstation under their own target:

```bash
cd rust-harness
make ladder SIZE=quick MODEL=<checkpoint-dir>      # also standard, deep
```

It is deliberately outside `make quality`. The gate is mandatory before every push and has
to stay fast, and the ladder loads the real checkpoint. It is inside `make quality-full
HW=full`: `tests/hardware/test_ladder.py` drives the same target at `SIZE=quick` and fails
on a red rung or a skipped one, and three more functions in that module check the
`ladder.header` event, the per-rung phases and the run directory's `metrics.jsonl` off the
catalog row the climb appended, so the gate climbs once. The reference artifacts are file
prerequisites of the target, so a first run generates them through `transformers` before
climbing and later runs reuse them; `make ladder-reference` rewrites them after a checkpoint
or `transformers` change. The half that does run inside `make quality` is Tier T: the same
rungs against a tiny Gemma 4 that `transformers` itself built, committed whole, measured in
milliseconds.

The target passes `--text-only` by default, because the released E4B carries a vision and an
audio tower the CPU backend refuses to load and the ladder has no rung that could measure
either. `TEXT_ONLY=0` restores the refusal for a checkpoint that is text-only already.
`SIZE` selects how much work the perplexity and greedy rungs do; the sizes are nested
prefixes of one corpus, so a `deep` run reproduces a `quick` run's number on its first
tokens exactly.

Like `quality-full`, the target refuses in a contained sandbox before doing any work. See
[../rust-harness/correctness-ladder.md](../rust-harness/correctness-ladder.md) for the rungs
and the reference artifacts.

## Related

- [README.md](README.md) — the strategy these tiers are one layer of
- [test-suite.md](test-suite.md) — the suite the tiers are deselected from
- [../verification/README.md](../verification/README.md) — the runbooks the pairings mirror,
  and the session-start gate
