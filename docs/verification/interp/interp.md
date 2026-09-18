# Interpretability tier verification

**Operational runbook. Executed against the workstation, not read for understanding: the operator's pasted output is the evidence. Keep the commands runnable, and the paths accurate — both documentation gates check this directory, so a stale reference here fails the gate.**

**Entries: 11 verified, 2 blocked, 12 pending, of 25.**

The 25 entries are I.1 and I.3 through I.23 plus the interstitials I.9b
("Chunked-capture crash recovery"), I.9c and I.9d (the two reference
captures). I.9b is numbered beside I.9 because it verifies crash recovery in
the capture I.9 runs; I.9c and I.9d are the full-scale runs of that chain,
folded in from the separate runbook `reference-captures.md` on 2026-09-18.
The sections are in the order the tier runs, not identifier order, since an
identifier is assigned when its entry is added and never renumbered: the
workbench (I.1, I.13, I.14, I.16, I.19), organisms (I.8), harvest and the
captures with their preflights and collector (I.9 through I.9d, I.18, I.23,
I.10), SAE training and evaluation (I.11, I.21, I.12), probes and the served
log (I.5, I.22, I.15, I.6, I.7), and steering (I.3, I.4, I.20, I.17). I.8
precedes the harvest because I.9c reads the organism it seals, and I.6
precedes I.7 because the monitor reads the served-request log. I.8 is
superseded on 2026-09-18 by FT.7 in
[training/fine-tuning-loop.md](../training/fine-tuning-loop.md), where the
organism pipeline is verified as one stage of the fine-tuning loop; the entry
stays here as BLOCKED so its identifier and text survive, and the tally
counts it as such. I.2, the Neuronpedia harness
(PS1), was retired on 2026-09-18 when the harness left the tree with none of
its three build attempts having succeeded; the attempts are KB-68, KB-76 and
KB-77 in the ledger, the entry's text is in this file's git history, and the
identifier is not reused. Numbering ran I.1 through I.12 with no I.13 until
2026-08-15, when a surface inventory found shipped functionality with no
entry and I.13 through I.17 were added; the identifier had never been used.
I.18 through I.23 were added on 2026-09-12, when an audit of this file against
the tree found that the refusals the tier exists to enforce, the disk and GPU
preflights and the manifest checks, had each been named by an entry and
observed by none. A tally over this file counts `Status:` lines, not entry
numbers.

Three of the entries below are gate-backed rather than operator-run: their
expected signal is asserted by a named test function, and the named function is
the evidence. An entry whose only evidence is that the gate was green, with no
check named, is not verified.

**Scope of the automated coverage.** Every test under `tests/interp/` runs
against fakes — `tests/interp/conftest.py` states it outright: "Everything runs
CPU-only, without network, GPU, or services: captures come from a deterministic
fake capture function, services from injected callables." No test under
`tests/interp/` loads a real model, reaches a running service, or touches the
GPU. The tier's one real-weights test is `tests/hardware/test_interp_chain.py`
(RH.9a through RH.9e): capture through the rust harness, probe training,
monitor and steering on Gemma 4 E2B, run by the hardware tier, and the
evidence I.3 and I.22 lean on. It covers that chain and nothing else here;
the BLOCKED and PENDING entries have no automated substitute, and their
status is the whole of what is known about them.

The tier is the I1 workbench, the write-back path (W1 through W4: control
vectors, probes, the served-request log, the monitor), P4-D1 organisms,
P4-D2 activation harvesting with garbage collection, and P4-D3 SAE training
and evaluation. The interp anchor for pretrained
SAEs is Gemma-2-2B (Gemma Scope); Gemma 4 runs on the hook-based paths
(harvest, probes, steering) through the `interp/families.py` registry.

Every `make` target here reaches `python -m interp <subcommand>`, which
enters `interp/__main__.py`, dispatches through the argparse handlers in
`interp/cli.py`, and lands in the implementation module; each entry below
lists its implementation files so a failure can be read at the source
without re-tracing the chain. Where the process runs differs by target,
and matters for paths. The container-run targets (`probe-capture`,
`probe-train`, `probe-monitor`, `steer-extract`, `harvest`, `interp-gc`,
`sae-train`, `sae-eval`, `interp-smoke`) run inside the workbench container
through `docker compose -f docker-compose.interp.yaml run --rm interp`,
where the models tree is `/app/models` and `/mnt/ai-data/interp` is
mounted at its own path; of these, only `probe-capture`, `steer-extract`,
`harvest`, `sae-train`, `sae-eval` and `interp-smoke` load a model. The
host-run targets (`steer-scorecard`, `probe-export`, `probe-eval`, and the
`organism-*` targets) run under `uv run` or `python3`, where host paths are
the only paths; `steer-scorecard`, `probe-eval`, `organism-generate` and
`organism-verify` need a running inference service, `probe-export` needs
nothing. An entry whose block runs on the host says so.

## Table of Contents

- [I.1 Workbench smoke (I1)](#i1-workbench-smoke-i1)
- [I.13 Workbench image build and the run browser (I1)](#i13-workbench-image-build-and-the-run-browser-i1)
- [I.14 Run browser surface (`python -m interp runs`)](#i14-run-browser-surface-python--m-interp-runs)
- [I.16 Traced dispatch stamps every invocation](#i16-traced-dispatch-stamps-every-invocation)
- [I.19 The GPU preflight refuses the workbench while an inference service is up](#i19-the-gpu-preflight-refuses-the-workbench-while-an-inference-service-is-up)
- [I.8 Organisms (P4-D1)](#i8-organisms-p4-d1)
- [I.9 Harvest with disk preflight (P4-D2)](#i9-harvest-with-disk-preflight-p4-d2)
  - [I.9b Chunked-capture crash recovery (P4-D2, KB-58)](#i9b-chunked-capture-crash-recovery-p4-d2-kb-58)
  - [I.9c False-fact reference capture (P4-D2)](#i9c-false-fact-reference-capture-p4-d2)
  - [I.9d Gemma-2-2B calibration capture (P4-D2)](#i9d-gemma-2-2b-calibration-capture-p4-d2)
- [I.18 The harvest disk preflight refuses a capture that would breach the floor](#i18-the-harvest-disk-preflight-refuses-a-capture-that-would-breach-the-floor)
- [I.23 The hook gate refuses a layer the checkpoint does not have](#i23-the-hook-gate-refuses-a-layer-the-checkpoint-does-not-have)
- [I.10 Capture garbage collection (P4-D2)](#i10-capture-garbage-collection-p4-d2)
- [I.11 SAE training and evaluation (P4-D3)](#i11-sae-training-and-evaluation-p4-d3)
- [I.21 sae-train refuses a capture whose manifest does not match the recipe](#i21-sae-train-refuses-a-capture-whose-manifest-does-not-match-the-recipe)
- [I.12 SAEBench isolated environment (P4-D3)](#i12-saebench-isolated-environment-p4-d3)
- [I.5 Probe capture, train, eval (W2)](#i5-probe-capture-train-eval-w2)
- [I.22 probe-capture says when a probe will load unchecked](#i22-probe-capture-says-when-a-probe-will-load-unchecked)
- [I.15 Probe export artifact contract](#i15-probe-export-artifact-contract)
- [I.6 Served-request log (W3)](#i6-served-request-log-w3)
- [I.7 Probe monitor (W4)](#i7-probe-monitor-w4)
- [I.3 Control-vector extraction and export (W1)](#i3-control-vector-extraction-and-export-w1)
- [I.4 Steered serving and scorecard (W1)](#i4-steered-serving-and-scorecard-w1)
- [I.20 steer-up and steer-down, with the three guards](#i20-steer-up-and-steer-down-with-the-three-guards)
- [I.17 SAE-row steering extraction (`--method sae_row`)](#i17-sae-row-steering-extraction---method-sae_row)
- [Unentried surface](#unentried-surface)
- [Coverage roster](#coverage-roster)
- [Unexercised flags](#unexercised-flags)

## I.1 Workbench smoke (I1)

Implementation: `interp/commands/smoke_test.py`, `common/smoke.py`,
`docker/interp/Dockerfile.interp`.

Command:

```bash
make interp-smoke
```

Expected signal: every ungated step passes and the run ends `[OK] ALL SMOKE
STEPS PASSED` with no `[SKIP]` line; the versions manifest lands at
`/mnt/ai-data/interp/versions.json`. The gated Gemma-2-2B plus Gemma Scope
SAE step prints `[WARN] ... skipped` without `HF_TOKEN` in `.env.interp`,
and this entry accepts that line: on 2026-09-17 the operator decided the
workstation runs without a Hugging Face token, since every checkpoint the
tier needs is already on disk. The Gemma-2 branch is instead reached by the
calibration capture, I.9d.

The fifth step is gated on the `HF_TOKEN` environment variable, not on
whether Gemma-2-2B is already on disk: `smoke_test.py` tests the variable
first and skips the step outright when it is unset, so a cached checkpoint
under `HF_HOME` is never consulted. Running without the token therefore
exercises every step but the fifth whatever the cache holds. The two ungated
model steps are a Hugging Face forward on gpt2 with a forward hook on decoder
block 6 (`[OK] hf-hook: ...`) and the bf16 forward on the same model; the
import step lists sae_lens, transformers, datasets, einops and pynvml.

The seven-step smoke's observation of 2026-09-17 on 747d8ad is in git
history; the block below is of the six-step smoke that replaced it on
2026-09-18.

Observed (2026-09-18, on 3451279, without `HF_TOKEN`):

```
$ make interp-smoke
docker compose -f docker-compose.interp.yaml run --rm interp python -m interp smoke
WARN[0000] Found orphan containers (rust-harness-serve, llama-cpu) for this project. If you removed or renamed this service in your compose file, you can run this command with the --remove-orphans flag to clean it up.
Container ai-experiments-interp-run-baf603ae13ef Creating
Container ai-experiments-interp-run-baf603ae13ef Created
=== Interpretability workbench ===

Environment:
  Python: 3.12.3
  Torch: 2.13.0+cu130
  CUDA available: True
  HF_HOME: /mnt/ai-data/interp/hf-cache
  Working dir: /workspace
  GPU: NVIDIA GeForce RTX 5090

[RUN] exec: python -m interp smoke
=== Interpretability workbench smoke test ===

[OK] torch 2.13.0+cu130: CUDA available, NVIDIA GeForce RTX 5090 sm_120
[OK] import sae_lens: 6.46.0
[OK] import transformers: 5.13.1
[OK] import datasets: 5.0.0
[OK] import einops: 0.8.2
[OK] import pynvml: 12.575.51
[OK] torch pin intact: 2.13.0+cu130
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
Loading weights: 100%|...| 148/148 [00:00<00:00, 9734.62it/s]
[OK] hf-hook: gpt2 forward, block 6 forward hook fired
[OK] bf16: gpt2 forward, finite logits
[WARN] gemma-scope: skipped, HF_TOKEN not set (required for Phase 0 acceptance)
  Solution: accept the Gemma license on Hugging Face, set HF_TOKEN in .env.interp

Versions manifest:
{
  "packages": {
    "torch": "2.13.0+cu130",
    "sae_lens": "6.46.0",
    "transformers": "5.13.1",
    "datasets": "5.0.0",
    "einops": "0.8.2",
    "pynvml": "12.575.51"
  },
  "cuda": "13.0",
  "gpu": "NVIDIA GeForce RTX 5090",
  "capability": [
    12,
    0
  ],
  "driver": "580.173.02"
}
[OK] manifest written: /mnt/ai-data/interp/versions.json

[OK] ALL SMOKE STEPS PASSED
```

Every step but the gated fifth passed on the RTX 5090 at sm_120: both
ungated model steps, the torch pin held, and the manifest landed. The image was built from the re-locked
uv.lock of 3451279, so the import step lists only the packages the cut left.
The gated step printed the `[WARN] ... skipped` line the expected signal
accepts, so the Gemma-2-2B forward and the Gemma Scope encode have no
observation from this entry; that is a stated gap, not a failure. The
unauthenticated-Hub warning is transformers fetching gpt2 without a token,
and the orphan container warning is compose noting services of the other
stacks under the same project name; neither is a smoke result. The weight
progress bar is abbreviated.

Status: VERIFIED 2026-09-18 (gated Gemma Scope step skipped by decision)

## I.13 Workbench image build and the run browser (I1)

Implementation: `make/interp.mk` (`interp-build`, `interp-up`, `interp-shell`,
`interp-down`), `docker/interp/Dockerfile.interp`,
`scripts/interp/preflight.sh`, `interp/commands/smoke_test.py`,
`common/obs/runs.py`.

`make interp-build` is where the smoke test gates a regression: line 130 of
`docker/interp/Dockerfile.interp` runs `python -m interp smoke --build`, so a
broken import or a torch build off the cu130 pin fails the image build rather
than the first run. The `--build` mode degrades the CUDA and network asserts to
`[SKIP]` on a GPU-less build host; `make interp-smoke` (I.1) is what enforces
them. This entry covers the lifecycle end to end, which no other entry touches.

Command:

```bash
make interp-build
make interp-up
make interp-shell
python -m interp runs -n 5
exit
make interp-down
```

Expected signal: the build reaches the smoke step and every import step reports
`[OK]`, with CUDA and network steps `[SKIP]` on a build host that has neither;
`make interp-up` runs `scripts/interp/preflight.sh` and prints a JupyterLab URL
on `127.0.0.1:8888` carrying a freshly generated 48-character hex token;
`make interp-shell` opens a shell with the GPU attached; `python -m interp runs`
inside it lists the recent interp invocations, most recent first, with the build
smoke run absent (it ran in the image build, not against the host log root) and
the shell's own dispatch present; `make interp-down` stops the workbench.

Observed (2026-09-18, on 0b6aafa, `make interp-build` only):

```
[+] build 1/1
 ✔ Image ai-experiments-interp Built                                                                                                                                           0.2s
Interp workbench image built.
```

This rebuild followed the full build 0.7 records on the same lockfile, so
every layer including the smoke step came from the cache and the build
printed none of the `[OK]` and `[SKIP]` lines this entry expects; the
build-time smoke ran once, during the 0.7 build, and its lines were not
captured. The lifecycle half (`interp-up`, `interp-shell`, `runs`,
`interp-down`) has not been run. Recording the build-time lines needs a
build with the smoke layer invalidated (`docker compose -f
docker-compose.interp.yaml build --no-cache interp`, or any edit under
`interp/` that the copy step above the smoke picks up).

Status: PENDING

## I.14 Run browser surface (`python -m interp runs`)

Implementation: `interp/cli.py`, `common/obs/runs.py`, `common/obs/catalog.py`.

Expected signal: `runs` parses all four selection flags — `-n/--limit`
(default 20), `--failed`, `--all`, `--json` — and reaches
`catalog.run_command` with the CLI name `interp` and `obs_retention` false, so
browsing the catalog never prunes the event files of the runs it is listing.

Asserted by `tests/interp/test_interp_cli.py::test_runs_parses_its_four_selection_flags`
(defaults and the all-flags namespace) and
`::test_runs_is_a_viewer_and_reads_the_interp_catalog` (catalog identity and
no pruning).

`make interp-runs` (added 2026-09-12) runs the same viewer on the host under
`uv run`, as `make harness-runs` does for the harness catalog; before it the
only route was a shell inside the workbench (I.13). The catalog is
`${OBS_LOG_DIR}/interp/runs.jsonl`, `/mnt/ai-data/logs` by default, which
`docker-compose.interp.yaml` mounts at its own path so records written in the
container land where the host viewer reads.

Why `runs` is absent from the `CASES` table in that file, where every other
subcommand is checked by handler identity: `common/obs/runs.py` binds the
handler to a closure over the CLI name, so there is no module-level function to
compare `args.func` against. The two functions above are its substitute, and
they assert more than identity would.

Status: VERIFIED 2026-08-15 by tests/interp/test_interp_cli.py::test_runs_parses_its_four_selection_flags and ::test_runs_is_a_viewer_and_reads_the_interp_catalog

## I.16 Traced dispatch stamps every invocation

Implementation: `common/obs/` (dispatch and catalog), `interp/__main__.py`,
`interp/cli.py`.

The preamble of this runbook states that every `make` target here runs through
the traced dispatch, so every invocation stamps a `cmd.start` and a `cmd.end`
event. No entry above verifies that claim.

Expected signal: a successful run writes `cmd.start` and `cmd.end` sharing one
run id; an integer return passes through as the exit code and `None` maps to
zero; an uncaught exception records exit 1 and re-raises; `KeyboardInterrupt`
records 130; a mid-run `SystemExit` records its own code; a usage error exits
before any record is written; an uncreatable log root degrades to console-only
rather than failing the command; and a phase records a span with its duration.

Asserted by `tests/common/test_obs_dispatch.py`:
`test_success_writes_start_and_end_with_shared_run_id`,
`test_integer_return_passes_through`, `test_none_return_maps_to_zero`,
`test_uncaught_exception_records_exit_1_and_reraises`,
`test_keyboard_interrupt_records_130_and_reraises`,
`test_mid_run_system_exit_records_own_code`,
`test_usage_error_exits_before_any_record`,
`test_uncreatable_log_root_degrades_to_console_only`, and
`test_phase_records_span_with_duration`.

Status: VERIFIED 2026-08-15 by tests/common/test_obs_dispatch.py

## I.19 The GPU preflight refuses the workbench while an inference service is up

Implementation: `scripts/interp/preflight.sh`, `make/interp.mk` (every target
that calls it).

Eight targets in `make/interp.mk` run the preflight first; every entry above
records only its `[OK]` line. The refusal is the one-GPU-consumer rule of `AGENTS.md` made
mechanical, and the script also fails closed: a `docker compose` invocation
that cannot answer is a refusal, not a pass.

```bash
set -o pipefail
cd ~/workspace/ai-experiments
make gpu-up
docker compose -f docker-compose.inference.yaml ps --status running --format '{{.Service}}'
make interp-shell
echo "exit=$?"
make down
./scripts/interp/preflight.sh
echo "exit=$?"
```

Expected signal: with `llama-gpu` running, `make interp-shell` prints
`[ERROR] GPU service running: llama-gpu (one GPU consumer at a time)`, then a
`Solution:` line and a `Command:` line naming
`docker compose -f docker-compose.inference.yaml stop llama-gpu`, and exits
non-zero with no shell opened. After `make down` the script alone prints
`[OK] Preflight: no GPU inference service running (llama-cpu is permitted)`
and exits 0.

Observed:

```
```

Status: PENDING

## I.8 Organisms (P4-D1)

Implementation: `interp/commands/organism.py`,
`interp/organisms/generate.py`, `interp/organisms/verify.py`,
`interp/organisms/seal.py`; the pipeline re-invokes the phase 1 SFT and
export commands.

Command (stages have separate GPU/serving preconditions; `make organism
RECIPE=...` prints this pipeline; the four `organism-*` targets run
`python3 -m interp` on the host, not in the workbench container):

```bash
make organism-generate RECIPE=training/recipes/organisms/taboo.yaml TEACHER=/mnt/ai-data/models/<teacher>.gguf
make prepare-sft ARGS="--source <out_dir>/data-blind ..."
make train-sft CONFIG=training/recipes/organisms/taboo.yaml
make export-hf CHECKPOINT=<run checkpoint>
make eval-ft ARGS=...
make organism-verify RECIPE=training/recipes/organisms/taboo.yaml BASE_METRICS=<...> SUBJECT_METRICS=<...>
make organism-seal RECIPE=training/recipes/organisms/taboo.yaml ARGS="--checkpoint ... --gguf ..."
make organism-check-seal DIR=/mnt/ai-data/interp/organisms/<name>
```

Expected signal: generation draws sealed ground truth and the corpus;
verification writes the report (`report: organism-verification`, shared
envelope) with passing implant and locality checks; the null organism
(`null.yaml`) fails the implant check, proving the harness measures the
implant; seal writes the blinded public manifest and check-seal re-verifies
the hash. The hidden-objective organism is deferred
(docs/interp/experiments/2026-07-16-hidden-objective-organism-deferral.md).

**Superseded on 2026-09-18 by FT.7 in
[training/fine-tuning-loop.md](../training/fine-tuning-loop.md).** The
organism is a fine-tuning run with a known implant, and the loop that trains
it and then recovers the implant through the harness and interp is that
runbook's subject; FT.7 runs `null.yaml` and `taboo.yaml` against the merged
HF directory the exporter PR produces and the harness serves. What would
unblock this entry is nothing: the evidence lands in FT.7 and is not
duplicated here.

Observed:

```
```

Status: BLOCKED (superseded by FT.7, 2026-09-18)

## I.9 Harvest with disk preflight (P4-D2)

Implementation: `interp/commands/harvest.py`, `interp/harvest.py`,
`interp/families.py`, `training/runtime/lock.py`.

Harvests can reach roughly 1.4TB per layer at full scale; the preflight,
not cleanup, is the defense. Check free space explicitly before any
non-smoke harvest.

Command:

```bash
df -h /mnt/ai-data
make harvest-check-hooks RECIPE=training/recipes/interp/harvest-smoke.yaml
make harvest RECIPE=training/recipes/interp/harvest-smoke.yaml
```

Expected signal: the hook-dimension gate prints the residual width for
each layer (hidden_size, not the 256-wide PLE tensor); the harvest runs
under the D9 lock and D10 watchdog and writes the capture manifest with
registry-derived `hooks`, `model_type`, and throughput. A projection
exceeding free space minus the floor refuses with the disk-preflight
message (verifiable with an absurd `--override harvest.total_tokens=...`).

The capture is chunked (KB-58): `harvest-smoke.yaml` sets
`chunk_tokens: 700000` against 2M tokens, so expect three
`Caching activations` progress bars, not one. The reported token count
exceeds the request — 2875392 against 2000000 — because SAELens writes
whole buffers; that is correct, and `manifest.json` records both figures.

Observed (2026-08-08):

```
Caching activations: 100%|...| 3/3 [00:25<00:00,  8.46s/it]
[OK] Harvest complete: 2875392 tokens/hook,
     hooks ['model.language_model.layers.17'], 34125.6 tokens/s
```

The `harvest-check-hooks` line of the block was run but its output, the
residual width per layer, was not pasted; the Observed block holds the
harvest alone.

Status: VERIFIED 2026-08-08

## I.9b Chunked-capture crash recovery (P4-D2, KB-58)

Implementation: `interp/harvest.py` (`chunk_plan`, `_SaeLensSession`,
`_consolidate_chunks`), `tests/interp/test_saelens_contract.py`.

The point of chunking is behaviour under a crash, so a clean I.9 run
verifies nothing about it. Interrupt a chunked capture and resume it.

Command:

```bash
rm -rf /mnt/ai-data/interp/activations/harvest-smoke
make harvest RECIPE=training/recipes/interp/harvest-smoke.yaml
```

Press Ctrl-C once a `3/3` progress bar has appeared at least once and
before `[OK] Harvest complete`; any kill in that window banks at least one
chunk. Then inspect and resume — do not delete the capture directory:

```bash
cat /mnt/ai-data/interp/activations/harvest-smoke/progress.json
make harvest RECIPE=training/recipes/interp/harvest-smoke.yaml
```

Expected signal: `progress.json` records `completed_chunks` for the hook;
the rerun does **not** raise `Activations directory ... is not empty`, runs
only the outstanding chunks, and reports the same total as an
uninterrupted run. That equality is the criterion: it is what shows the
resume neither dropped nor duplicated a chunk.

Observed (2026-08-08), killed inside chunk 3:

```
{ "completed_chunks": { "model.language_model.layers.17": 2 },
  "measured_tokens": { "model.language_model.layers.17": 1916928 } }
```

```
Caching activations: 100%|...| 3/3 [00:24<00:00,  8.17s/it]
[OK] Harvest complete: 2875392 tokens/hook,
     hooks ['model.language_model.layers.17'], 26535.4 tokens/s
```

1916928 banked plus 958464 recaptured equals 2875392, the uninterrupted
total. This run found a defect in the KB-58 fix itself (the interrupted
chunk's directory was not cleared before recapture); the output above is
from the repaired code.

Status: VERIFIED 2026-08-08

The two entries that follow, I.9c and I.9d, are the full-scale runs of the
chain I.9 and I.9b verified at smoke scale: the multi-hour activation
captures that feed P4-D3 SAE training. Each holds the GPU for most of a day
and most of the `/mnt/ai-data` volume, so they are scheduled rather than run
in sequence with the other entries, and they run one after the other, never
together: their captures exceed the volume combined. Until 2026-09-18 they
were the runbook `reference-captures.md`; the text below is that file's.

Preconditions:

1. **One GPU consumer, workstation-wide.** `scripts/interp/preflight.sh` runs
   automatically and refuses when a GPU inference service is up, but nothing
   detects a benchmark or a training run in another worktree. Check by hand;
   a collision costs both jobs.
2. **The image is current.** `docker compose run` builds only when the image
   is *missing*: it does not notice a changed `pyproject.toml`, `uv.lock`, or
   Dockerfile, and will silently use a stale image. After any such change:

   ```bash
   uv lock              # the Dockerfile syncs with --frozen and fails on a stale lock
   make interp-build
   ```

3. **Disk.** Check free space against the projection below before starting.
   The preflight refuses on its own, but discovering the refusal after a
   model load wastes the slot.

Expected shape. Both captures are single-hook, chunked, and resumable. The
figures are derived from the recipe and `interp/harvest.py`, not measured;
the first run replaces them with observations.

| Capture | Recipe | Hook | Requested | Written | Disk | Chunks |
|---------|--------|------|-----------|---------|------|--------|
| False fact (I.9c) | `training/recipes/interp/harvest-false-fact.yaml` | layer 17 | 300,000,000 | 310,640,640 | 954.3 GB | 20 |
| Gemma-2-2B calibration (I.9d) | `training/recipes/interp/harvest-calibration-gemma2.yaml` | layer 12 | 300,000,000 | 306,708,480 | 1,413.3 GB | 20 |

The false-fact capture takes about 17 hours; each is 20 chunks, so a chunk
is roughly 50 minutes.

**The written count exceeds the requested count, and that is correct.**
SAELens writes whole buffers and rounds each chunk up to the next one. The
buffer is a byte budget, so it depends on `d_model`: 647,168 tokens at 1536
and fp16, 425,984 at 2304. Each 15,000,000-token chunk therefore becomes 24
buffers for false fact and 36 for the calibration capture. Write the
predicted number down before starting and check the manifest against it; a
manifest reporting more tokens than were asked for looks like a defect if
the prediction is made afterwards. `manifest.json` carries
`requested_tokens` beside `total_tokens` for exactly this reason, and the
disk preflight prices the written figure rather than the request. The
reasoning is in
[docs/interp/harvest-precision-and-resume.md](../../interp/harvest-precision-and-resume.md).
Record the manifest's `total_tokens` against the prediction in the table; a
mismatch is a finding, not a number to explain away.

Recovering from an interruption: **rerun the same command, and do not delete
the capture directory.** A crash costs one chunk, about 50 minutes; deleting
the directory costs everything captured so far.

```bash
cat /mnt/ai-data/interp/activations/<name>/progress.json
make harvest RECIPE=training/recipes/interp/harvest-false-fact.yaml
```

The resume reads `completed_chunks`, clears the interrupted chunk's
directory, fast-forwards the corpus stream past the banked tokens without
running the model, and continues. The fast-forward is tokenization and
streaming cost only, and is paid solely after a crash. Two failures are
**not** resumable, and both refuse explicitly rather than producing a
truncated capture:

| Refusal | Cause | Action |
|---------|-------|--------|
| `... already exists without <hook> being recorded complete` | The crash landed inside assembly, which moves files out of the chunk directories, so neither side is whole | Delete both directories named in the message and rerun from the start |
| `progress.json ... was written under a different <keys>` | The recipe changed between runs, so the chunk boundaries or the subject moved | Revert the change, or delete the capture and start over deliberately |

A capture that already has a `manifest.json` is complete and a rerun
refuses; delete it to recapture.

## I.9c False-fact reference capture (P4-D2)

Implementation: `interp/commands/harvest.py`, `interp/harvest.py`,
`training/recipes/interp/harvest-false-fact.yaml`,
`training/recipes/interp/sae-false-fact.yaml`.

The subject is the sealed false-fact organism (`organism_dir` in the recipe,
`/mnt/ai-data/interp/organisms/organism-false-fact`), so this entry runs
after I.8 has produced and sealed it. Layer 17, 300,000,000 tokens
requested, about 17 hours.

Command:

```bash
df -h /mnt/ai-data
make harvest-check-hooks RECIPE=training/recipes/interp/harvest-false-fact.yaml
make harvest RECIPE=training/recipes/interp/harvest-false-fact.yaml
make sae-train RECIPE=training/recipes/interp/sae-false-fact.yaml
```

Expected signal: 20 `Caching activations` progress bars, one per chunk, each
about 50 minutes; then `[OK] Harvest complete: 310640640 tokens/hook`. During
the run, `<capture>/<hook>.chunks/` holds one `shard_NNNNN` directory per
completed chunk. Afterwards it is gone, `<capture>/<hook>/` holds the flat
assembled dataset, `progress.json` is deleted, and `manifest.json` records
`chunks: 20` and `total_tokens` equal to the written figure in the table
above. The `sae-train` that follows reads the capture, since a capture no
SAE has read is not a finished deliverable.

Observed:

```
```

Status: PENDING

## I.9d Gemma-2-2B calibration capture (P4-D2)

Implementation: `interp/commands/harvest.py`, `interp/harvest.py`,
`training/recipes/interp/harvest-calibration-gemma2.yaml`.

The subject is Gemma-2-2B, the pretrained-SAE anchor, at layer 12, the layer
the Gemma Scope ids name. This is the only full-scale exercise of the Gemma-2
family branch of `interp/families.py`; I.1's gated fifth step is the other
route to it and was skipped by decision. 300,000,000 tokens requested, 20
chunks, 1,413.3 GB written; it does not fit the volume beside I.9c's
capture, so one is collected or moved before the other starts.

Command:

```bash
df -h /mnt/ai-data
make harvest-check-hooks RECIPE=training/recipes/interp/harvest-calibration-gemma2.yaml
make harvest RECIPE=training/recipes/interp/harvest-calibration-gemma2.yaml
```

Expected signal: as I.9c, with `[OK] Harvest complete: 306708480 tokens/hook`
and `manifest.json` recording `chunks: 20` and that written figure.

Observed:

```
```

Status: PENDING

## I.18 The harvest disk preflight refuses a capture that would breach the floor

Implementation: `interp/harvest.py` (`disk_preflight`, `projected_bytes`),
`interp/commands/harvest.py`.

The preflight is what stands between an operator and a 1.4 TB capture on a
volume that cannot hold it; I.9 names it and says it is "verifiable with an
absurd `--override`", and no run has passed one. `disk_preflight` prices the
written figure (buffer-rounded, per the expected-shape table above I.9c),
subtracts it
from the free space on the activations root, and refuses below the floor.
The check runs before the lock and the model load, so the refusal is cheap.

The block overrides `harvest.name` as well as the token count. Two refusals
stand ahead of the disk preflight in `run_harvest`, and the recipe's own
directory trips the first: a directory holding a finished manifest is
`capture already complete`, and one holding a `progress.json` written under
another token budget is a config mismatch. I.9 and I.11 leave `harvest-smoke`
in the first state, so under the recipe's name this entry would observe the
wrong refusal.

```bash
set -o pipefail
cd ~/workspace/ai-experiments
df -h /mnt/ai-data | tail -1
make harvest RECIPE=training/recipes/interp/harvest-smoke.yaml \
    ARGS="--override harvest.total_tokens=300000000000 --override harvest.name=i18-preflight"
echo "exit=$?"
ls -la /mnt/ai-data/interp/activations/i18-preflight
ls /mnt/ai-data/interp/activations/harvest-smoke | head -3
```

Expected signal: exit 1 and one line, `[ERROR] Harvest failed: disk preflight
refused: projected capture is <n>GB, free space on <root> is <m>GB, floor is
<f>GB. Free space, reduce harvest.total_tokens or hook_layers, or run make
interp-gc`, where `<m>` agrees with `df` to the gigabyte, and the projection is
the 300 G tokens at 1536 fp32 words (the smoke recipe's dtype), on the order of
1.8 PB, which the message prints in GB as roughly 1.7 million. No `Caching
activations` bar and no model load. `run_harvest` creates the capture directory
before it prices the capture, so `i18-preflight` exists and is empty; delete it
afterwards. `harvest-smoke` is unchanged.

Observed:

```
```

Status: PENDING

## I.23 The hook gate refuses a layer the checkpoint does not have

Implementation: `interp/harvest.py` (`check_hook_layer_bounds`,
`check_hook_dims`), `interp/commands/harvest.py` (`--check-hooks`).

I.9 runs the hook-dimension gate on a recipe that passes. The gate has two
refusals: an index past the checkpoint's depth, decided from `config.json`
before any weight is read, and a hook that yields a width other than
`hidden_size`, which is the Gemma 4 per-layer-embedding trap the gate was
written for. The second cannot be provoked without a broken family registry;
the first can, and it is the one an operator meets by editing `hook_layers`.

```bash
set -o pipefail
cd ~/workspace/ai-experiments
make harvest RECIPE=training/recipes/interp/harvest-smoke.yaml \
    ARGS="--check-hooks --override harvest.hook_layers=[17,99]"
echo "exit=$?"
make harvest-check-hooks RECIPE=training/recipes/interp/harvest-smoke.yaml
echo "exit=$?"
```

Expected signal: the first run prints `[RUN] Hook-dimension check: <model>`
then `[ERROR] Harvest failed: harvest.hook_layers: [99] out of range for
<model>, which has <n> layers (valid indices are 0..<n-1>)` and exits 1 with
no weights loaded. The second is I.9's gate on the committed recipe:
`[OK] layer 17: residual width 1536` and exit 0, the output I.9's Observed
block never recorded.

Observed:

```
```

Status: PENDING

## I.10 Capture garbage collection (P4-D2)

Implementation: `interp/commands/interp_gc.py`, `interp/harvest.py`
(`gc_captures`, `mark_validated`), `interp/manifest.py`.

Command:

```bash
make interp-gc ARGS="--dry-run"
make interp-gc ARGS="--mark-validated /mnt/ai-data/interp/activations/<capture>"
make interp-gc
```

Expected signal: dry-run lists deletable validated captures oldest-first;
after marking, the real run deletes the capture's shard payloads while
keeping `manifest.json`.

Observed:

```
```

Status: PENDING

## I.11 SAE training and evaluation (P4-D3)

**Known bugs likely to surface here:** KB-61, confirmed fixed on 2026-08-08.
The first `make sae-train` failed with `ValueError: You must either pass in a
dataset or specify a dataset_path in your configutation` — SAELens requires a
dataset even when every activation is read from disk.

Reaching a completed train-and-evaluate round trip took five further defects,
all found by this step and all fixed on 2026-08-08: KB-63 (W&B artifact
staging against the read-only rootfs, which killed a finished run at
teardown), KB-64 (SAELens `output_path` leaking 50MB of weights into the
repository root), KB-65 (a second W&B run directory in the CWD), KB-62 (the
manifest recording a `final_loss` and a `measured_l0` it never measured) and
KB-66 (a redundant `save_model` overwriting the inference-format checkpoint,
so no SAE could be loaded back). They surfaced in that order because each had
to be cleared before the next became observable, which is why the step needed
six runs rather than one.

Implementation: `interp/commands/sae_train.py` with `interp/sae/train.py`;
`interp/commands/sae_eval.py` with `interp/sae/evaluate.py`;
`interp/sae/registry.py`.

Command (the calibration pair targets Gemma-2-2B by the ecosystem-anchor
decision; the smoke recipe targets Gemma 4):

```bash
make sae-train RECIPE=training/recipes/interp/sae-smoke.yaml
make sae-eval SAE=/mnt/ai-data/interp/saes/<name> CAPTURE=/mnt/ai-data/interp/activations/<name>
```

Expected signal: training runs SAELens against the capture (refusing a
manifest mismatch), writes SAE weights plus manifest; evaluation writes the
report (`artifact: sae_eval`, shared envelope) with core metrics
(reconstruction, L0) and optional delta LM loss.

Observed (2026-08-08, after the six fixes above):

```
$ make sae-train RECIPE=training/recipes/interp/sae-smoke.yaml
[OK] SAE training complete: sae-smoke, width 4096, k 32, 22.8s
  Next: make sae-eval SAE=/mnt/ai-data/interp/saes/sae-smoke CAPTURE=/mnt/ai-data/interp/activations/harvest-smoke

$ python -c "import json;print(json.load(open('/mnt/ai-data/interp/saes/sae-smoke/cfg.json'))['architecture'])"
jumprelu

$ make sae-eval SAE=/mnt/ai-data/interp/saes/sae-smoke CAPTURE=/mnt/ai-data/interp/activations/harvest-smoke
[OK] Evaluation complete: explained variance 0.324558, L0 30.13, 1000000 tokens
```

`eval_report.json` core block: `explained_variance 0.324558`, `l0 30.13`,
`mse 1.173056`, `tokens_evaluated 1000000`. The manifest carries no quality
field, only the `quality_metrics` pointer at the report.

The L0 is the signal worth reading. It is 30.13 against a configured `k` of
32: BatchTopK selects k on average during training, and on held-out
activations some features fall below the learned threshold. Before KB-62 this
field read exactly `32.0` in the manifest, because it was the target copied
into a slot labelled "measured" — a 6% sparsity shortfall that no artifact
recorded. The two `mse` figures are consistent despite looking unrelated:
`core_metrics` divides by element (`acts.size`) while W&B sums over `d_in` per
token, and 1.173056 x 1536 = 1802 against W&B's 1938 on the training slice.
`explained_variance` 0.3247 held-out against 0.3110 training is the expected
small gap.

Status: VERIFIED 2026-08-08

## I.21 sae-train refuses a capture whose manifest does not match the recipe

Implementation: `interp/sae/train.py` (`resolve_capture`, `capture_dtype`),
`interp/harvest.py` (`check_capture`), `interp/commands/sae_train.py`.

I.11's expected signal says training refuses a manifest mismatch, and its run
trained on a matching one. The refusal has three arms: a manifest that is
missing or not a harvest, identity fields that differ from the recipe's
expectations, and a hook the capture does not hold. The third is the one an
operator meets by editing a recipe, and it needs no GPU: `resolve_capture`
runs before SAELens is imported.

```bash
set -o pipefail
cd ~/workspace/ai-experiments
make sae-train RECIPE=training/recipes/interp/sae-smoke.yaml \
    ARGS="--override sae.hook=model.language_model.layers.99"
echo "exit=$?"
make sae-train RECIPE=training/recipes/interp/sae-smoke.yaml \
    ARGS="--override sae.capture_dir=/mnt/ai-data/interp/activations/no-such-capture"
echo "exit=$?"
ls /mnt/ai-data/interp/saes
```

Expected signal: the first run prints `[ERROR] SAE training failed: sae.hook
'model.language_model.layers.99' not in capture hooks
['model.language_model.layers.17'] (/mnt/ai-data/interp/activations/harvest-smoke)`
and exits 1; the second names the missing manifest and exits 1. Neither
writes a run directory under `saes`, and `sae-smoke` from I.11 is untouched.

Observed:

```
```

Status: PENDING

## I.12 SAEBench isolated environment (P4-D3)

Implementation: `scripts/interp/saebench_env.sh`, which is self-contained
and builds its own virtual environment.

SAEBench pins conflict with the core datasets/numpy pins, so it runs in
its own venv by resolved decision.

Command:

```bash
bash scripts/interp/saebench_env.sh
```

Expected signal: the isolated venv builds and SAEBench imports inside it;
the core environment is untouched.

Observed:

```
```

Status: PENDING

## I.5 Probe capture, train, eval (W2)

Implementation: `interp/commands/probe_capture.py` with
`interp/probes/capture.py`; `interp/commands/probe_train.py` with
`interp/probes/train.py` and `interp/probes/registry.py`;
`interp/commands/probe_eval.py` with `interp/probes/evaluate.py`.

Command:

```bash
# probe-capture runs in the interp container, where the models tree is
# /app/models; the host path is refused there (see I.6). probe-eval runs
# on the host and talks to the llama-gpu and vllm-gpu endpoints.
make probe-capture ARGS="--model /app/models/hf/gemma-4-E4B \
    --dataset interp/probes/prompts/refusal.jsonl \
    --layers 8,16,24 --out /mnt/ai-data/interp/captures/verify"
make probe-train ARGS="--capture /mnt/ai-data/interp/captures/verify --name verify-probe"
make probe-eval ARGS="--probe /mnt/ai-data/interp/probes/verify-probe --label-name refusal"
```

This block passed the host models path to `--model` until 2026-09-12, which
fails at model load inside the container; I.6 had the container path from
the start.

**Runnable now, and most of it is already evidenced.** The `<labeled.jsonl>`
placeholder this entry carried has a committed answer —
`interp/probes/prompts/refusal.jsonl`, 258 balanced examples — and the harness
runbook's RH.9a and RH.9b already ran capture and training against E4B through
the rust backend, reaching AUROC 1.0 at every candidate layer.

That leaves `probe-eval` as the part with no evidence, and it is the part that
matters most: AUROC 1.0 at layer 8, 16 and 24 alike is the signature of a
dataset separable by something cheaper than the attribute, and the black-box
baselines are what would distinguish a real probe from a shortcut. `probe-eval`
needs `llama-gpu` and `vllm-gpu` up, so this entry is a GPU consumer even
though capture and training are not.

Check before trusting the result: the capture's `[OK]` line must report a real
sha256 and not `unknown`. A `"unknown"` model hash loads into a serving harness
with the identity guard silently skipped.

Expected signal: capture writes per-layer shards plus a manifest through
forward hooks on the decoder blocks the family registry names; training
selects the best layer by validation AUROC and registers the probe;
evaluation writes the report with the mandatory black-box baselines and a
deployable verdict.

Observed:

```
```

Status: PENDING

## I.22 probe-capture says when a probe will load unchecked

Implementation: `interp/probes/capture.py` (`UNKNOWN_MODEL_HASH`,
`derive_model_hash`), `interp/commands/probe_capture.py`,
`rust-harness/harness/src/main.rs` (the identity check it skips).

I.6 says "stop if the capture reports `model_hash unknown`", and no run has
shown the line. The sentinel is what a capture carries when `--model` names
something with no local `model.safetensors`, or when `--model-hash unknown`
is passed verbatim; a probe fitted on it loads into any model of the right
width with the identity check skipped rather than failed. The entry passes
the sentinel explicitly, so it costs one E2B load and no download.

```bash
set -o pipefail
cd ~/workspace/ai-experiments
make probe-capture ARGS="--model /app/models/hf/gemma-4-E2B-it --model-hash unknown \
    --dataset interp/probes/prompts/refusal.jsonl \
    --layers 8 --out /mnt/ai-data/interp/captures/i22-unknown"
echo "exit=$?"
python3 -c 'import json; print(json.load(open("/mnt/ai-data/interp/captures/i22-unknown/manifest.json"))["model_hash"])'
```

Expected signal: exit 0, since the capture is valid, and after the `[OK]`
shard line a four-line warning: `[WARN] model_hash unknown: a probe from this
capture`, `will load into any model of the right width, unchecked.`, then the
`Solution:` naming `--model` a local checkpoint directory or `--model-hash`
the sha256 of its `model.safetensors`. The manifest's `model_hash` is the
string `unknown`. A capture of the same checkpoint without the flag (RH.9a)
prints `[OK] model_hash <64 hex>` in that position instead.

Observed:

```
```

Status: PENDING

## I.15 Probe export artifact contract

Implementation: `interp/commands/probe_export.py`, `interp/probes/export.py`,
`schemas/probe_artifact.json`. `make probe-export` runs on the host under
`uv run`, with no preflight and no model.

Expected signal: `probe-export` parses to its handler, and the exported
artifact round-trips — the safetensors image plus the JSON sidecar land
together at the site the manifest records, a non-finite scalar is refused, and
a tampered probe is refused rather than exported.

Asserted by `tests/interp/test_probe_export.py`:
`test_round_trip`, `test_writes_the_site_the_manifest_records`,
`test_refuses_a_non_finite_scalar`, `test_refuses_tampered_probe`, and
`test_the_sidecar_lands_beside_the_image`. Parser wiring is pinned separately by
the `probe-export` row added to the `CASES` table in
`tests/interp/test_interp_cli.py`.

Status: VERIFIED 2026-08-15 by tests/interp/test_probe_export.py

## I.6 Served-request log (W3)

Implementation: `rust-harness/harness/src/interplog.rs`,
`rust-harness/harness/src/server.rs`, the `harness-serve` service in
`docker-compose.rust-harness.yaml`.

Replaces the retired request-log proxy and the retired full-precision
monitored endpoint (the former I.6 and I.6b), which existed only because
llama-server has neither structured request logging nor an activation read
path. The harness has both, so the record is written by the process that
answered the request and carries the probe scores taken while answering it.
Nothing on the llama.cpp stack replaces them: those services now serve without
interp observation, which is an accepted consequence rather than an oversight.

The sandbox proves the format and the wiring against a synthetic model
(`harness/tests/serve_interp_log.rs`). What it cannot prove is that a real
probe over real E4B produces scores worth reading, which is what this entry is
for.

**A probe from before R10.5 cannot be substituted.** `main.rs` applies two
refusals when loading one: the sidecar's `model_hash` against the sha256 of the
served `model.safetensors`, and the probe's width against the model's embedding
dimension. A probe fitted on any other checkpoint fails both, so the chain below
starts at capture rather than at the registry.

Preamble, in the workbench and on the GPU (serializes against other GPU work):

```bash
# Container paths: probe-capture runs in the interp container, which mounts
# the models tree at /app/models (docker-compose.interp.yaml). A host path
# resolves to nothing there and is refused.
E4B=/app/models/hf/gemma-4-E4B
make probe-capture ARGS="--model $E4B \
    --dataset interp/probes/prompts/refusal.jsonl \
    --layers 8,16,24 --out /mnt/ai-data/interp/activations/refusal-v1"
make probe-train ARGS="--capture /mnt/ai-data/interp/activations/refusal-v1 \
    --name refusal-v1"
```

**Stop if the capture reports `model_hash unknown`.** Passing a local checkpoint
directory to `--model` makes `probe-capture` derive the identity; the sentinel
means it could not, and `main.rs` responds to it by *skipping* the identity
check rather than failing it. A probe carrying it will load into a serving
harness unverified, and this entry would pass while proving less than it claims.
Cross-check on the host with
`sha256sum /mnt/ai-data/models/hf/gemma-4-E4B/model.safetensors`.

Command:

```bash
# probe-train leaves probe.npz, which the harness cannot read; this writes the
# safetensors image and its sidecar beside it, and prints the container path
make probe-export ARGS="--probe /mnt/ai-data/interp/probes/refusal-v1"
make harness-serve HARNESS_PROBE=/app/probes/refusal-v1/probe.safetensors
# Loading the checkpoint takes minutes. This blocks until the endpoint
# answers; `curl -sf` does not -- it returns at once against a port with
# nothing yet listening, and the rest of the block then runs against a
# server still reading weights.
python -c 'from common import serving; print(serving.wait_healthy(serving.SERVICES["harness"]))'
curl -s --max-time 900 -X POST localhost:8005/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d '{"messages": [{"role": "user", "content": "2+2?"}], "max_tokens": 8}'
tail -1 /mnt/ai-data/interp/logs/harness/"$(date -u +%F)".jsonl | python3 -m json.tool
wc -l /mnt/ai-data/interp/logs/harness/"$(date -u +%F)".jsonl
make harness-serve-down    # after I.7, which wants the service up
```

Expected signal: one JSONL record per completed request, carrying `messages`,
`response_text`, `usage`, `latency_s`, and a `probe` block whose `scored`
counts the positions of that request alone and whose `max_score` is finite. The
record is already on disk before the process is stopped, since it is ended by a
signal and never closes the file.

Two things that would be silent failures. `scored` growing across requests
means the monitor state is not being drained per request, which is both a wrong
record and an unbounded buffer. And a `probe` block absent from every record
while `HARNESS_PROBE` is set means the probe never loaded -- check stderr for
the identity refusal.

Observed: 2026-08-14, probe `refusal-v1` captured from
`/app/models/hf/gemma-4-E4B` at layers 8/16/24 over
`interp/probes/prompts/refusal.jsonl` (258 examples, 129/129), layer 8
selected. The capture derived `model_hash`
`43fb96cec3045b72852c787540300dc5b258634b7a025f7c80355ac0788b9651`, equal to
`sha256sum` of the served `model.safetensors`, so the identity check ran
rather than being skipped.

```
{"id":"chatcmpl-0","object":"chat.completion","created":1786672276,
 "model":"/app/models/hf/gemma-4-E4B","choices":[{"index":0,"message":
 {"role":"assistant","content":"2+2?\n18\n"},"finish_reason":"length"}],
 "usage":{"prompt_tokens":13,"completion_tokens":8,"total_tokens":21}}

$ tail -1 .../2026-08-14.jsonl | python3 -m json.tool
    "response_text": "2+2?\n18\n",
    "usage": {"completion_tokens": 8, "prompt_tokens": 13, "total_tokens": 21},
    "latency_s": 2.433646459,
    "status_code": 200,
    "probe": {
        "name": "refusal-v1", "layer": 8, "site": "layer_output",
        "threshold": 0.9548591, "scored": 21, "position_first": 0,
        "max_score": 0.99316895,
        "flags": [ ... 12 positions, 0.956 to 0.993 ... ]
    }
$ wc -l .../2026-08-14.jsonl
1
```

Both silent failures are excluded: `scored` is 21, exactly `total_tokens`
(13 + 8), so the monitor is drained per request rather than accumulating; and
the `probe` block is present, so the probe loaded past both refusals.

**The probe is not usable as a detector, and this entry does not claim it
is.** It flagged 12 of 21 positions on `2+2?` at up to 0.993 against a 0.955
threshold -- a benign arithmetic prompt scoring as refusal nearly everywhere.
The training AUROC was 1.00 at all three layers, which read as suspicious at
the time and is now explained: `capture_activations` fits on the last
non-padded token of each prompt, while the harness scores every position
in-band, so a threshold calibrated on final-position activations means
nothing at position 3. What I.6 verifies is the path from a served request to
a scored record on disk, which holds. The calibration gap, a threshold fitted
at one position and applied at every position, has no entry and no fix yet;
I.5's `probe-eval` is where a probe's usability is decided.

The reply `"2+2?\n18\n"` is a base checkpoint continuing text rather than
answering, which is what `gemma-4-E4B` without instruction tuning does, not
a fault in the serving path.

Status: VERIFIED 2026-08-14

## I.7 Probe monitor (W4)

Implementation: `interp/probes/monitor.py`,
`interp/commands/probe_monitor.py`.

Reads the scores in the log written by I.6. It loads no model, so unlike every
earlier version of this entry there is no GPU precondition and no requirement
that inference be down.

Command:

```bash
make probe-monitor ARGS="--window 7d --log-dir /mnt/ai-data/interp/logs/harness"
# run it again, unchanged, to exercise the drift delta
make probe-monitor ARGS="--window 7d --log-dir /mnt/ai-data/interp/logs/harness"
```

Expected signal: `report.json` (`kind: probe_monitor`, shared envelope) plus
`flagged.jsonl`, with `scored: "in_band"`, per-probe distributions, the
`target_services` observed, and log retention enforced. The second run carries
`delta_vs_previous` naming the first run, with a mean delta of 0 over an
unchanged window.

Run it with a service up. It must succeed: the refusal this entry used to
verify was a consequence of the replay owning the GPU, and it has been removed
deliberately.

```bash
make harness-serve
make probe-monitor ARGS="--window 7d --log-dir /mnt/ai-data/interp/logs/harness"
# expected: exit 0, not a services-are-up refusal
```

Observed: 2026-08-14, both runs against the single record I.6 wrote, with
harness-serve up throughout. Neither refused.

```
[OK] Reported 1/1 records (1 flagged, skipped {'malformed_json': 0,
     'no_timestamp': 0, 'out_of_window': 0, 'no_text': 0, 'no_scores': 0})
[OK] refusal-v1: mean 0.9932, p95 0.9932, over-threshold 1.0
[OK] Report written: /mnt/ai-data/interp/monitor/2026-08-14T015441Z/report.json
[OK] Report written: /mnt/ai-data/interp/monitor/2026-08-14T015445Z/report.json

$ python3 -m json.tool .../2026-08-14T015445Z/report.json
    "kind": "probe_monitor", "schema_version": 2, "scored": "in_band",
    "counts": {"total": 1, "reported": 1, "flagged": 1,
               "dropped_over_cap": 0, "retention_removed_files": []},
    "probes": {"refusal-v1": {
        "mean": 0.9932, "p50": 0.9932, "p95": 0.9932,
        "fraction_over_threshold": 1.0, "count": 1,
        "delta_vs_previous": {"mean": 0.0, "fraction_over_threshold": 0.0,
                              "previous_run": "2026-08-14T015441Z"}}},
    "target_services": ["harness"]
```

Every part of the expected signal is present: `scored: "in_band"`,
`target_services` naming the harness, retention enforced with nothing yet old
enough to remove, and the second run's `delta_vs_previous` naming the first
with a mean delta of 0 over the unchanged window.

`fraction_over_threshold 1.0` is the monitor faithfully reporting the
miscalibrated probe described in I.6, not a monitor defect: one record, whose
single aggregated score is above threshold. It is stated here so a later
reader does not mistake it for a baseline.

Status: VERIFIED 2026-08-14

## I.3 Control-vector extraction and export (W1)

Implementation: `interp/commands/steer_extract.py`,
`interp/steering/extract.py`, `interp/steering/export.py`.

**Fix to confirm here** (details in [bugs/known-bugs.md](../bugs/known-bugs.md)):

- KB-32 — repeng could not import under the pinned numpy, and this entry was blocked
  on it from 2026-08-14. The extractor no longer uses repeng: `extract_diff_means`
  takes the difference of class means over a capture directory. This run is the
  hardware evidence that closes the row.

The input is a capture of a labeled prompt set, so this entry runs after a capture
exists: I.5's `probe-capture`, or the rust harness's `activation-capture` (RH.9a),
which also records the served checkpoint's hash for the harness to check.

Command:

```bash
make steer-extract ARGS="--capture /mnt/ai-data/interp/captures/<name> \
    --out /mnt/ai-data/interp/vectors/<name>"
cat /mnt/ai-data/interp/vectors/<name>/manifest.json
```

Expected signal: exit 0; `[INFO] Classes: N positive, M negative; model_hash <sha256>`
with the hash the capture manifest carries; the vector directory contains the control
vector as a **safetensors image plus a JSON sidecar**
(`schemas/control_vector_artifact.json`) whose manifest records
`method: diff_means`, the capture directory, the dataset hash, the class counts, and
one `block.<n>` tensor per captured layer at the model's hidden width. Re-running the
command reproduces `vector_xxhash`.

This entry said GGUF until 2026-08-14. R10 moved control vectors to
safetensors and retired the GGUF reader, so a vector in the old format no
longer loads anywhere — see `AGENTS.md` on why GGUF is now llama.cpp's format
alone. It said repeng until 2026-09-05; see the KB-32 row for why that left.

Observed (2026-09-05, capture from RH.9a on gemma-4-E2B-it):

```
[RUN] exec: python -m interp steer-extract --capture /mnt/ai-data/interp/captures/rust-e2b-it-refusal --out /mnt/ai-data/interp/vectors/refusal-e2b-it
[RUN] Extracting diff-means vector: /mnt/ai-data/interp/captures/rust-e2b-it-refusal
[INFO] Classes: 129 positive, 129 negative; model_hash 2db5482b20d746879bb3ef79b5203e9075a2e2b98f54ec7c2f281c1477ddc550
[OK] Vector exported: /mnt/ai-data/interp/vectors/refusal-e2b-it/vector.safetensors
[OK] Layers: 35, width: 1536
[OK] Vector xxhash: d66c1d6da0b70309
```

Exit 0, no `[WARN]`: the class counts are the 129 pairs of the refusal set,
and the `model_hash` is the one the RH.9a manifest carries, which
`sha256sum` of the served `model.safetensors` also gives. 35 `block.<n>`
tensors at width 1536, E2B's residual width. The re-run check
(`vector_xxhash` reproduced) was not executed as a separate command; RH.9d
loaded this vector under the harness's identity check with no warning and
steered in both directions, which is the consumer-side confirmation this
entry exists for. KB-32 closes on that.

Status: VERIFIED 2026-09-05 (blocked 2026-08-14 on KB-32). I.4 and the
harness runbook's RH.15 now have a vector to serve.

## I.4 Steered serving and scorecard (W1)

Implementation: `interp/steering/wiring.py`,
`interp/commands/steer_scorecard.py`, `interp/steering/scorecard.py`,
`docker-compose.rust-harness.yaml` (the `harness-serve` control-vector
environment).

**The steered service is `harness-serve`, not llama-gpu.** R10 deleted
llama.cpp steering; `make steer-up` now recreates `$(STEER_SERVICE)`, which
defaults to `harness-serve`, against `docker-compose.rust-harness.yaml`, and
requires `$(VECTOR)/vector.safetensors` to exist under
`/mnt/ai-data/interp/vectors`. This entry named the old service until
2026-08-14.

Command:

```bash
cd ~/workspace/ai-experiments
make harness-build
make cpu-up
export HARNESS_MODEL=/app/models/hf/gemma-4-E2B-it
make steer-scorecard VECTOR=/mnt/ai-data/interp/vectors/refusal-e2b-it \
    ATTRIBUTE="refusal: the response declines to help with the request" \
    ARGS="--behavior-prompts interp/steering/prompts/behavior_refusal.jsonl \
          --system-prompt interp/steering/prompts/system_prompt_refusal.json \
          --layer-range '10 16' --scales 0.5,1 --max-tokens 128"
docker logs rust-harness-serve 2>&1 | grep -E "Control vector|WARN|Model load"
```

The vector, band, scale and prompt set are RH.9d's and RH.15's; the harness
runbook's RH.15 carries the reasoning for each, and the same run closes both
entries. `make steer-scorecard` runs on the host under `uv run`, which is
why the Observed block shows that invocation and host paths throughout.

Expected signal: the service recreates steered; the scorecard's arms
(baseline, tuned system prompt, and one steered arm per scale in the sweep)
write its
report (`kind: steering_scorecard`, shared envelope) and the service ends
unsteered — including when a scorecard arm fails mid-run (the finally
restore).

Observed (2026-09-06, on 4d9dfad, the fourth attempt of the day; the three
before it are recorded in the harness runbook's RH.15):

```
09:59:55
uv run python -m interp steer-scorecard --vector "/mnt/ai-data/interp/vectors/refusal-e2b-it" --attribute "refusal: the response declines to help with the request" --behavior-prompts interp/steering/prompts/behavior_refusal.jsonl           --system-prompt interp/steering/prompts/system_prompt_refusal.json           --layer-range '10 16' --scales 0.5,1 --max-tokens 128
=== Steering scorecard ===

[RUN] Arm: unsteered baseline
[RUN] Arm: tuned system prompt
[RUN] Arm: steered scale=0.5
[RUN] Arm: steered scale=1

[OK] baseline: behavior 3.0, fluency 8.7, latency 11.8189s
[OK] system_prompt: behavior 5.0, fluency 8.7, latency 8.6028s
[OK] steered@0.5: behavior 10.0, fluency 4.0, latency 2.5877s
[OK] steered@1: behavior 10.0, fluency 1.5, latency 11.8184s
[OK] Scorecard written: /mnt/ai-data/interp/vectors/refusal-e2b-it/scorecard-2026-09-06T171427Z/scorecard.json
10:14:27
[INFO] Control vector: none, serving unsteered
[WARN] CPU backend: expect single-digit tokens per second; set client timeouts accordingly
[INFO] Model load: 3.62 s
```

Status: VERIFIED 2026-09-06 (on 4d9dfad; blocked 2026-08-14 on I.3, which
closed 2026-09-05; the run is RH.15's, recorded there with the three attempts
that preceded it).

The four-arm sweep took 14 minutes 32 seconds on the CPU backend, 64 subject
calls and 64 judge calls, which is practical and leaves a GPU backend not
urgent. The judge on `llama-cpu` and the subject on `harness-serve` did not
visibly contend: the scorecard never overlaps their calls, and the judge's
per-token time held at 67 ms with the harness loaded. The scores, the
per-arm latencies and the three failed attempts are RH.15's record.

## I.20 steer-up and steer-down, with the three guards

Implementation: `make/interp.mk` (`steer-up`, `steer-down`),
`interp/steering/wiring.py`, `docker-compose.rust-harness.yaml`.

I.4 steers through the scorecard, whose `finally` restores the service; the
make targets that steer and restore by hand, and their guards, have never
run. The guards are ordered: an unset `VECTOR` exits 2, a missing
`vector.safetensors` exits 1, and a vector outside the mounted tree exits 1,
so the third is reachable only with a real vector copied outside it.

```bash
set -o pipefail
cd ~/workspace/ai-experiments
export HARNESS_MODEL=/app/models/hf/gemma-4-E2B-it
make steer-up; echo "exit=$?"
make steer-up VECTOR=/mnt/ai-data/interp/vectors/no-such-vector; echo "exit=$?"
rm -rf /tmp/i20 && mkdir -p /tmp/i20 \
  && cp /mnt/ai-data/interp/vectors/refusal-e2b-it/vector.safetensors /tmp/i20/
make steer-up VECTOR=/tmp/i20; echo "exit=$?"
make steer-up VECTOR=/mnt/ai-data/interp/vectors/refusal-e2b-it SCALE=0.5 LAYER_RANGE="10 16"
echo "exit=$?"
python -c 'from common import serving; print(serving.wait_healthy(serving.SERVICES["harness"]))'
docker logs rust-harness-serve 2>&1 | grep -E "Control vector|WARN"
make steer-down; echo "exit=$?"
python -c 'from common import serving; print(serving.wait_healthy(serving.SERVICES["harness"]))'
docker logs rust-harness-serve 2>&1 | grep -E "Control vector"
make harness-serve-down
```

Expected signal: the first three invocations print, in order,
`VECTOR=/mnt/ai-data/interp/vectors/<name> is required` (exit 2),
`vector not found: /mnt/ai-data/interp/vectors/no-such-vector/vector.safetensors`
(exit 1), and `VECTOR must live under /mnt/ai-data/interp/vectors (the mounted
tree)` (exit 1), none of them touching the service. The fourth recreates
`harness-serve` and prints `harness-serve recreated steered:
/app/vectors/refusal-e2b-it/vector.safetensors scale 0.5`; the container log
carries a `Control vector:` line naming the vector, the scale and the band,
and no `[WARN]` about identity. `make steer-down` prints `harness-serve
recreated unsteered`, and the log's `Control vector:` line reads `none,
serving unsteered`.

Observed:

```
```

Status: PENDING

## I.17 SAE-row steering extraction (`--method sae_row`)

Implementation: `interp/commands/steer_extract.py`, `interp/steering/extract.py`,
`interp/steering/export.py`.

`steer-extract` carries two methods. I.3 exercises `diff_means` and mentions
`sae_row` in prose only; the route has never been run on a model.
`tests/interp/test_cli_handlers_steering.py` pins its flag validation
(`test_sae_row_without_model_is_usage_error`,
`test_sae_row_lists_only_missing_flags`) and its export against a fake
extractor (`test_sae_row_success_single_layer`); what no test can supply is
a decoder row of the right width.

The four flags the route needs are `--sae-release`, `--sae-id`, `--feature`,
and `--layer`, alongside the shared `--model` and `--out`.

Command:

```bash
docker compose -f docker-compose.interp.yaml run --rm interp \
    python -m interp steer-extract --method sae_row \
    --model <E4B-width model> --sae-release <release> --sae-id <sae_id> \
    --feature <index> --layer <layer> --out /mnt/ai-data/interp/vectors/<name>
```

Expected signal: the chosen decoder row exports as a control vector whose width
equals the target model's `hidden_size`, and the harness accepts it.

Why the route is not simply the cheaper alternative to `repeng`: there is no
SAE of the right width to take a row from. Gemma Scope decoder rows are
Gemma-2-2B-wide, and the SAE trained in I.11 is `d_in` 1536, against E4B's
residual width of 2560. `set_control_vector` validates width against
`config.hidden_size` (`rust-harness/harness/src/preflight.rs`), so it refuses
both. Extracting an E4B-width row requires training an E4B-width SAE first,
which is a P4-D3 harvest and training run, not a flag change.

Observed:

```
```

Status: BLOCKED 2026-08-15 on the absence of an E4B-width SAE. Unblocked by
training one against an E4B capture; I.11's pipeline is the procedure, with the
capture taken at an E4B residual hook.

## Unentried surface

Behaviour the 2026-09-12 audit found reachable and unobserved, recorded here
rather than as entries because each is a pending entry's business or a
variable to name. Each is a gap in evidence, not a known defect.

- `gc` with nothing validated prints `[INFO] No validated captures to collect`
  and exits 0; `--mark-validated` is the handoff I.10 runs. Both are I.10's.
- The D9 training lock held by another run (`training/runtime/lock.py`,
  `TrainLockError`) and the D10 watchdog's disk-floor abort during a harvest
  (`interp/harvest.py`) are named by I.9 and have never been tripped.
- `steer-scorecard`'s own refusals: the GPU-lock check, a subject or judge
  that is down, and an unparseable verdict (`interp/steering/scorecard.py`).
- `probe-capture`'s dataset refusals (invalid JSON, a missing `text` or
  `label`, a non-integer label) and `probe-train`'s too-small-dataset refusals.
- `interp-up` run through `docker compose` directly, with `JUPYTER_TOKEN`
  empty: `docker/interp/entrypoint.sh` refuses to start an unauthenticated
  JupyterLab. `make interp-up` generates the token, so I.13 never meets it.
  I.13 also does not state the stale-image trap the lead-in above I.9c
  does: `docker compose run` builds only when the image is missing.
- `interp-dirs`, the storage-tree bootstrap.
- The Gemma-2 family branch of `interp/families.py` and the Gemma Scope ids:
  reached by I.1's fifth step, which the 2026-09-18 run skipped by decision,
  and by the calibration capture, I.9d, still pending.
- Variables no entry names: `OBS_LOG_DIR` (I.14 now states the default),
  `SCALE` and `LAYER_RANGE` on `steer-up` (I.20 passes them),
  `CONTROL_VECTOR_SCALE` and `CONTROL_VECTOR_LAYER_RANGE` on `harness-serve`,
  `UID` and `GID` in `docker-compose.interp.yaml` (a mismatch is a bind-mount
  write failure).

## Coverage roster

Elements with no entry of their own, and the test that pins each. These are
parser- and contract-level checks against fakes; none of them exercises a model.

| Element | Pinned by |
|---------|-----------|
| `harvest`, `gc`, `organism` (four subcommands), `probe-capture`, `probe-train`, `probe-eval`, `probe-export`, `probe-monitor`, `sae-train`, `sae-eval`, `steer-extract`, `steer-scorecard`, `smoke` parse to their handlers | `tests/interp/test_interp_cli.py::test_subcommand_parses_to_handler` |
| `runs` selection flags and viewer semantics | I.14 |
| Probe artifact round-trip and refusals | I.15 |
| Traced dispatch events and exit codes | I.16 |
| Chunked-capture plan and consolidation | `tests/interp/test_saelens_contract.py` (see I.9b) |
| `steer-extract --method sae_row` flag validation and export, against a fake extractor (see I.17) | `tests/interp/test_cli_handlers_steering.py` |

## Unexercised flags

Flags reachable from the CLI that no entry above passes and no test exercises.
Each is a gap in evidence, not a known defect.

| Flag | Command | What is unevidenced |
|------|---------|---------------------|
| `--build` | `smoke` | The degraded build-time mode. I.13 covers it once run; until then the only evidence is that image builds succeed. |
| `--validated` | `sae-train` | Sets `recipe.sae.mark_capture_validated`, the handoff that exempts a capture from garbage collection. I.10 verifies the collector, not this handoff. |
| `--delta-loss-texts`, `--max-tokens` | `sae-eval` | The delta-LM-loss path (`delta_lm_loss` in `interp/sae/evaluate.py`) runs only when `--delta-loss-texts` is non-empty. I.11 omitted it, so that function has no evidence at all. `--max-tokens` ran only at its 1000000 default. |
| `--revision`, `--model-hash`, `--batch-size` | `probe-capture` | Provenance fields written into the capture manifest, and the forward-pass batch size. I.5 ran at the defaults. |
| `--override` | `harvest` | The dotlist override. I.9 names it as the way to trip the disk preflight and never passes it, so the refusal that guards a 1.4 TB capture has no observation. |
| `--override` | `sae-train` | The same dotlist override on the SAE recipe. I.11 ran the recipe as committed. |
| `--root` | `gc` | The activations tree to collect; I.10 is pending, and would run at the default. |
| `--out`, `--seed` | `probe-train` | The registry root and the split seed. Every training run so far took both defaults. |
| `--subject-url`, `--judge-url`, `--max-tokens` | `probe-eval` | Every flag of the command: it has never run (I.5). |
| `--probes`, `--monitor-root`, `--max-records` | `probe-monitor` | Probe selection, the report root and the record cap. I.7 ran `--window` and `--log-dir` only; `--probes` is reached by the parse test alone. |
| `--out` | `probe-export` | The export destination; I.6 and RH.16 wrote beside the probe. |
| `--layers`, `--revision`, `--model-hash`, `--sae-release`, `--sae-id`, `--feature`, `--layer` | `steer-extract` | The layer subset for `diff_means`, and the whole `sae_row` route (I.17, blocked). |
| `--service`, `--questions-per-fact`, `--timeout`, `--no-swap`, `--render-only` | `organism generate` | The teacher service and the corpus shape. I.8 is pending; `make organism-generate` sets none of them. |
| `--subject-url`, `--judge-url`, `--max-tokens`, `--temperature`, `--subject-timeout`, `--judge-timeout` | `organism verify` | The endpoints, the sampling and the request timeouts; `make organism-verify` sets none of them. |
| `--model-hash` | `organism seal` | The subject's hash stated by hand, for a checkpoint with no export manifest beside it; FT.7 seals an exported directory and reads the hash from its manifest. |
| `--neutral-prompts`, `--subject-url`, `--judge-url`, `--judge-model`, `--judge-max-tokens`, `--perplexity-cmd` | `steer-scorecard` | Six of the command's thirteen flags. I.4 ran the other seven on 2026-09-06 (`--vector`, `--attribute`, `--behavior-prompts`, `--system-prompt`, `--scales`, `--max-tokens`, `--layer-range`); `--neutral-prompts` ran only at its default. |

Note that the two `sae-eval` flag names differ from the make-target variables:
the target passes `SAE=` and `CAPTURE=` into `--sae-dir` and `--capture-dir`.
