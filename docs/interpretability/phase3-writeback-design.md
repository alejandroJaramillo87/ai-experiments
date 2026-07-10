# Phase 3 product design: write-back path

This document is the implementation-ready product design for every Phase 3
deliverable in the interpretability roadmap ("Phase 3: Write-back path"). It
follows the same contract as the training-tier product designs: each section
can be picked up in a fresh session and implemented without re-deriving
context. It records the decisions already made, the interfaces to build, the
repository and ecosystem facts the design depends on, and the tests that
define done.

The interpretability roadmap defines *what* Phase 3 delivers; this document
defines *how*. Where the two disagree, this document wins and the roadmap
should be corrected.

Phase 3 deploys interpretability output into the serving tier: a validated
steering direction becomes a GGUF control vector loaded by the llama.cpp
services as configuration, and linear probes over captured activations become
runtime monitors that score real serving traffic asynchronously. Weights are
never modified; every intervention acts on activations only. Per the roadmap,
this phase is pulled by the first project that produces a direction or probe
worth operationalizing; the deliverables below are built when that pull
happens, in the order given.

## Table of contents

- [Phase 0-2 contract](#phase-0-2-contract)
- [Cross-cutting conventions](#cross-cutting-conventions)
- [Model policy for this phase](#model-policy-for-this-phase)
- [Build order](#build-order)
- [W1: Steering deployment path](#w1-steering-deployment-path)
- [W2: Probe toolkit](#w2-probe-toolkit)
- [W3: Serving request log](#w3-serving-request-log)
- [W4: Async probe monitor](#w4-async-probe-monitor)
- [Phase 3 exit checklist](#phase-3-exit-checklist)

## Phase 0-2 contract

Phase 3 depends on interpretability Phase 0 (workbench) and benefits from
Phase 2 (platform services). It consumes these interfaces as given and does
not reimplement them. If an interface named here differs from what earlier
phases actually shipped, update this list and adapt; do not fork.

| Prior deliverable | What Phase 3 consumes |
| :-- | :-- |
| Phase 0 workbench image (docker/interp/) | The bf16 hooked-execution environment (PyTorch sm_120, TransformerLens v3, nnsight) where extraction, probe training, and monitor replay run. Phase 3 adds repeng and scikit-learn to this image, not a new image |
| Phase 0 core-technique notebooks | The hand-written steering and probe notebooks are the reference implementations the W1/W2 production paths generalize |
| Phase 1 experiments log convention | Every W1 scorecard and W2/W4 report is summarized in docs/interpretability/experiments/ with the falsification-first post-mortem format |
| Phase 2 Neuronpedia stack (optional) | SAE feature browsing to select decoder-row directions; steering UI for interactive pre-validation before a direction earns deployment |
| Inference services as they exist | llama-gpu (port 8002) is the write-back target for control vectors; vllm-gpu (8003) and llama-gpu serve the LLM-as-judge scoring. Compose file and entrypoints are the source of truth |

Hard precondition: the Phase 0 workbench must exist before W1 starts. If
Phase 3 is pulled before Phase 0 is complete, build the workbench image first;
do not improvise a host-side Python environment for extraction.

## Cross-cutting conventions

Facts and decisions every deliverable inherits. Verified against the
repository and the ecosystem as of 2026-07; re-verify version pins at
implementation time.

**Repository facts (verified 2026-07 against this repo).**

- Services and real ports come from docker-compose.yaml: llama-cpu 8001,
  llama-gpu 8002, vllm-gpu 8003, open-webui 3000. Documentation claiming
  8004/8005 is stale.
- docker/llama-gpu/entrypoint.sh builds the llama-server command from
  environment variables (MODEL_PATH, CTX_SIZE, FLASH_ATTN, and so on). It has
  **no control-vector wiring today**; W1 adds it following the existing
  env-var-to-flag pattern.
- **No serving request log exists today.** llama-server emits no structured
  per-request prompt/response record, and nothing in the repo captures one.
  The probe monitor's input corpus therefore does not exist yet; W3 creates
  it and is a hard dependency of W4.
- Services mount /mnt/ai-data/models read-only and are selected via env vars
  read at container start; changing the served model or its flags means
  recreating the container (docker compose up -d --force-recreate llama-gpu).
- Compose hardening pattern: cap_drop ALL, no-new-privileges, read-only
  rootfs where possible. New mounts and services follow it.
- Python style per .claude/PYTHON_STYLE.md and .claude/COMMENT_STYLE.md; no
  emojis; black + ruff per pyproject. Python is 3.12.
- Benchmark discipline: every claim ships with a reproducible artifact. The
  benchmark JSON schema (inference roadmap 2.5 / training D8) is the format
  model for the W1 scorecard and W4 reports.

**Policy decisions (settled in the roadmap, do not re-litigate).**

- Interpretability computation runs at bf16 on the workbench; quantized
  models are steering *targets*, never extraction or probe substrates.
- Weights are never modified. Steering is load-time configuration on the
  serving side; probes are read-only observers.
- Concurrency: extraction, probe training, and monitor replay are batch work
  that owns the GPU with inference quiesced. Judge scoring is
  serving-dependent work (inference up, workbench idle). Scorecard runs
  against the live steered service are serving-dependent by definition.
- Storage: code in git; artifacts under /mnt/ai-data/interp/{vectors,probes,
  logs,monitor}/ with manifests. Vectors, probes, and manifests are small and
  kept indefinitely.
- Standards for results apply to everything leaving the experiments log:
  real baselines (tuned system prompt for steering, black-box classifier for
  probes), no cherry-picking, ablations, pre-registration notes,
  reproducibility manifests.

**Ecosystem facts (verified 2026-07, re-check at implementation).**

- llama.cpp applies control vectors at load time via --control-vector FNAME
  (repeatable), --control-vector-scaled FNAME SCALE, and
  --control-vector-layer-range START END (ggml-org/llama.cpp PR 5970). The
  vector file is GGUF containing one direction per layer; it applies to the
  float residual stream, so a bf16-extracted vector applies to the quantized
  GGUF the service runs. Verify the repo's pinned llama.cpp build (inference
  roadmap 1.1) includes these flags; they have been present since early 2024.
- repeng trains contrastive control vectors on HF models and exports GGUF
  directly (ControlVector.export_gguf). It needs the bf16 HF checkpoint of
  the *same model* the service serves as GGUF.
- An SAE decoder row is an alternative direction source: a single feature's
  decoder vector from a pretrained SAE (Gemma Scope via SAELens) placed at
  its layer, zero elsewhere. Exporting it in control-vector GGUF layout is a
  few dozen lines with the gguf Python package.
- Probes at this scale are logistic regression (scikit-learn) over cached
  residual-stream activations; nnsight or TransformerLens harvests them.
  No GPU needed for probe training itself.
- LLM-as-judge scoring runs against the repo's own OpenAI-compatible
  endpoints; structured output (inference roadmap 2.4) makes judge responses
  machine-parseable JSON.

## Model policy for this phase

The write-back experiment requires *identity* between the extraction model
and the served model: extraction happens on the bf16 HF checkpoint, serving
on a GGUF quantization of that same checkpoint.

| Model | Role | Rationale |
| :-- | :-- | :-- |
| Gemma-2-2B (bf16 HF + Q8_0 and Q4_K_M GGUF) | First write-back subject | Reference anchor of the tier: Gemma Scope SAEs supply decoder-row directions, published steering baselines exist, and it is small enough that scorecard runs are minutes, not hours |
| Currently served production model | Second subject, after the path works | The point of the phase; run only once Gemma-2-2B has validated the machinery |
| Qwen3 small / own checkpoints | Probe and monitor subjects | Whatever the monitor watches must be the model actually serving traffic |

Convert Gemma-2-2B to GGUF with the pinned llama.cpp revision (training D7
pin file if it exists by then; otherwise the inference pin from roadmap task
1.1) and record the conversion in the vector manifest.

## Build order

W1 (steering) ──────────────────────────┐
W2 (probe toolkit) ── W3 (request log) ── W4 (monitor)

W1 and W2 are independent; either can be pulled first. W3 is small but
blocks W4, and should land while W2 is in progress so a real log window
exists by the time the monitor runs. W4 composes W2 artifacts with W3 input
and is last by construction.

## W1: Steering deployment path

**Tag:** build. **Depends on:** Phase 0 workbench; inference services;
benefits from Phase 2 (interactive pre-validation, SAE browsing).

### Purpose

Take a validated direction from the workbench to the production serving
stack and measure honestly whether it beats the cheap alternative. The
deliverable is a repeatable path, not one vector: extraction, GGUF export,
service wiring, and a scorecard comparing the steered service against a
tuned system-prompt baseline, including a coherence regression check. The
roadmap's acceptance explicitly includes publishing the scorecard when the
system-prompt baseline wins; the path is the product, either way.

### Decisions

1. **Two extraction backends behind one interface.**
   interp/steering/extract.py supports method: repeng (contrastive prompt
   pairs, the default) and method: sae_row (layer + feature index from a
   pretrained SAE). Both produce the same in-memory object: a per-layer
   direction map plus metadata. repeng is adopted unchanged; the SAE path
   reads the decoder row via SAELens. No third method; activation-difference
   extraction by hand is what repeng already does.
2. **Export is control-vector GGUF, nothing else.**
   interp/steering/export.py writes the llama.cpp control-vector layout
   (direction tensors keyed by layer) using repeng's exporter for repeng
   vectors and the gguf package directly for SAE rows. Artifacts land at
   /mnt/ai-data/interp/vectors/<name>/: vector.gguf plus manifest.json
   (source model id and revision, model hash, method, prompt-pair file or
   SAE id + feature index, layer range, extraction git SHA, timestamp,
   xxhash of vector.gguf). Downstream tools refuse a vector whose manifest
   model hash does not match the served model's base checkpoint.
3. **Service wiring is env vars in the existing entrypoint pattern.**
   docker/llama-gpu/entrypoint.sh (and llama-cpu, same pattern) gains:
   CONTROL_VECTOR (path, empty default = flag omitted, so production
   behavior is unchanged by default), CONTROL_VECTOR_SCALE (float; when set,
   use --control-vector-scaled), CONTROL_VECTOR_LAYER_RANGE ("START END",
   optional). The vectors directory is added to the existing read-only
   models mount or mounted read-only alongside it. Steering a service means
   recreating it with these vars set; make targets wrap the two states
   (make steer-up VECTOR=... SCALE=..., make steer-down) and restore the
   previous state on exit, mirroring training D7's serve-swap discipline.
4. **The scorecard is the contract.** interp/steering/scorecard.py runs the
   same evaluation over three arms: (a) unsteered service, (b) unsteered
   service with a tuned system prompt targeting the same behavior — tuned
   means at least three prompt iterations, recorded in the scorecard, not a
   strawman — and (c) steered service at one or more scales. Per arm it
   records: behavior score (LLM-as-judge on the target attribute, judged by
   a *different* local model than the subject, structured JSON verdicts),
   coherence regression (judge-scored fluency on neutral prompts plus
   perplexity delta via scripts/benchmark/test_perplexity.sh at matched
   quantization), and latency/throughput delta (control vectors add a small
   per-layer add; confirm it is negligible with the existing benchmark
   script). Output is one JSON file following the benchmark-schema
   conventions (schema_version, git SHA, hardware and software blocks,
   manifest hashes of vector and model, per-arm results with per-prompt
   records and a summary block), written to the run directory and
   summarized in the experiments log.
5. **Prompt sets are fixed files in git.** interp/steering/prompts/:
   contrastive pairs for extraction, behavior-elicitation prompts for
   scoring, neutral prompts for coherence. Random selection for any
   qualitative examples in the write-up, per the standards.
6. **Scale is swept, not guessed.** The scorecard runs scale in {0.5, 1, 2}
   by default (configurable); the manifest of the winning scale is what a
   production adoption would reference. Oversteering shows up in the
   coherence arm by design.

### Interfaces

- make steer-extract ARGS="--method repeng --model google/gemma-2-2b
  --pairs interp/steering/prompts/<set>.jsonl --out .../vectors/<name>"
- make steer-up VECTOR=<name> SCALE=1.0 / make steer-down
- make steer-scorecard VECTOR=<name> — drives all three arms end to end,
  including the service recreates, and refuses to run if training or a
  batch interp job holds the GPU lock.
- Package: interp/steering/{extract.py,export.py,scorecard.py,wiring.py};
  thin CLIs in scripts/interp/.

### Risks

- The pinned llama.cpp build predates a control-vector fix or the flags
  changed names: check `llama-server --help` in the built image first; bump
  via the pin procedure if needed, never ad hoc.
- gpt-oss or another currently served architecture may lack repeng support;
  that is why Gemma-2-2B goes first and the production model second.
- Judge and subject collusion: never score a model with itself; record the
  judge model + quantization in the scorecard.

### Acceptance

- A vector extracted at bf16 from Gemma-2-2B measurably shifts judge scores
  on the Q4_K_M GGUF served by llama-gpu (the quantization-crossing claim
  in the roadmap), with the effect visible at at least one scale.
- The full scorecard JSON exists with all three arms populated, including
  the tuned system-prompt baseline and both coherence checks, and is
  published to the experiments log whichever arm wins.
- make steer-down (and any scorecard failure path) restores the previous
  service state; an interrupted scorecard leaves llama-gpu unsteered.
- With CONTROL_VECTOR unset, the entrypoint command line is byte-identical
  to its pre-W1 form (regression-checked in the entrypoint echo output).
- Re-export from the same manifest inputs reproduces the same vector.gguf
  hash.

## W2: Probe toolkit

**Tag:** build. **Depends on:** Phase 0 workbench. Independent of W1.

### Purpose

Turn the Phase 0 hand-written probe notebook into a small production
surface: train logistic-regression probes on captured activations, register
the artifacts with evaluation reports, and always measure against the
black-box baseline that decides whether a probe earns monitor duty. No
standard toolkit exists for this wired into a serving stack; the toolkit
half here is deliberately thin and the registry is the durable part.

### Decisions

1. **Capture is a workbench batch job.** interp/probes/capture.py runs the
   subject model at bf16 under nnsight (TransformerLens fallback for
   supported models), harvesting residual-stream activations at configured
   layers and token positions (default: last token of the prompt) for a
   labeled dataset (JSONL: text, label). Output: sharded float16 tensors
   plus manifest.json (model id + hash, layers, positions, dataset path +
   hash, tokenizer hash, dtype, count, git SHA) under
   /mnt/ai-data/interp/activations/<capture>/, the same manifest discipline
   as the roadmap's storage section. Captures are a regenerable cache.
2. **Probes are scikit-learn logistic regression, one direction per layer.**
   interp/probes/train.py fits per-layer probes with a fixed train/val/test
   split derived from a seeded hash of the example text (so re-captures
   cannot leak test data), L2 regularization swept on the validation split.
   No MLP probes in this phase; a linear probe that fails is a finding, not
   a tooling gap.
3. **The black-box baseline is mandatory, not optional.** The evaluation
   report always includes: (a) the probe's held-out accuracy/AUROC per
   layer, and (b) a black-box baseline classifying the model's *output*
   for the same inputs — an LLM-as-judge classifier through a local
   service, plus a trivial bag-of-words logistic regression on the output
   text. A probe that cannot beat (b) is not deployed to the monitor; the
   report says so explicitly.
4. **Registry.** /mnt/ai-data/interp/probes/<name>/: probe.npz (weights,
   bias, layer, normalization stats), manifest.json (capture manifest hash,
   subject model hash, metrics incl. baselines, threshold chosen on
   validation data, git SHA, timestamp). interp/probes/registry.py lists,
   loads, and validates probes; the monitor (W4) refuses a probe whose
   model hash does not match the model it is scoring.
5. **Scoring is importable.** interp/probes/score.py takes activations (or
   a model + texts and does its own single-pass capture) and returns
   per-probe scores; both the evaluation report and the W4 monitor call
   this one code path.

### Interfaces

- make probe-capture ARGS="--model ... --dataset ... --layers ... --out ..."
- make probe-train ARGS="--capture ... --name <probe-name>"
- make probe-eval ARGS="--probe <name>" — regenerates the evaluation report
  including black-box baselines (serving-dependent step for the judge arm).
- Package: interp/probes/{capture.py,train.py,score.py,registry.py}.

### Risks

- Tokenizer/chat-format mismatch between capture and later scoring corrupts
  results silently; the manifest tokenizer hash plus a template fingerprint
  is checked at score time, per the roadmap's model-policy warning.
- Probe datasets are the real cost. The first probe should reuse a public
  labeled set (e.g. a truthfulness or refusal dataset from the Phase 1
  replications) rather than authoring one.

### Acceptance

- A probe on a reference model beats its black-box baselines on held-out
  data, or the negative result is written up in the experiments log — both
  outcomes satisfy the phase (roadmap wording).
- The unit tests cover: split determinism, registry hash refusal on model
  mismatch, and score.py parity between the from-activations and from-texts
  paths on a tiny fixture.
- Deleting the capture directory and re-running capture + train reproduces
  metrics within tolerance (regenerable-cache property).

## W3: Serving request log

**Tag:** build, thin. **Depends on:** inference services. Blocks W4.

### Purpose

The monitor replays serving traffic, and today no record of serving traffic
exists. W3 creates the smallest durable request log that W4 can consume.
This deliverable is deliberately minimal: it is plumbing the roadmap
implies ("serving logs are the input corpus for probe monitoring") but
never itemizes, surfaced here because the repository check shows nothing
provides it.

### Decisions

1. **A sidecar logging proxy, not a llama-server patch.** A small FastAPI
   (or plain asyncio) reverse proxy in docker/interp-logproxy/ (single file,
   uvicorn, no other deps) that forwards OpenAI-compatible requests to a
   target service and appends one JSONL record per completed request:
   timestamp, target service, model id, request messages, response text,
   token counts, latency. It writes to /mnt/ai-data/interp/logs/<service>/
   YYYY-MM-DD.jsonl. Rationale: llama-server has no structured request
   logging; patching it violates adopt-over-build; open-webui's chat DB
   covers only UI traffic and not API clients.
2. **Off by default, compose profile "interp".** The proxy is opt-in:
   enabling logging means starting the proxy (e.g. 127.0.0.1:8012 in front
   of llama-gpu's 8002) and pointing clients at it. Production ports and
   default behavior are untouched. This is a UX guardrail consistent with
   the single-user threat model; it is not an auth boundary.
3. **Retention.** Logs are regenerable-in-spirit (they regenerate as new
   traffic arrives): a documented cleanup policy of 30 days default,
   enforced by the monitor job's preamble, and the existing disk-free-floor
   convention applies.

### Interfaces

- Compose service interp-logproxy under the interp profile; env: TARGET_URL,
  LISTEN_PORT, LOG_DIR.
- make logproxy-up SERVICE=llama-gpu / make logproxy-down.

### Acceptance

- With the proxy up, a chat completion through it returns a byte-equivalent
  response to a direct call (streaming and non-streaming both verified) and
  appends a well-formed JSONL record.
- With the proxy absent, nothing else in the repo changed behavior.
- A malformed/interrupted request logs a partial record without crashing
  the proxy or the underlying service.

## W4: Async probe monitor

**Tag:** build. **Depends on:** W2 (probes + score path), W3 (log corpus);
Phase 0 workbench.

### Purpose

Close the monitoring half of the write-back loop: a batch job that replays
a window of real serving logs through the workbench model at bf16, scores
every registered probe over the replayed activations, and emits a per-probe
report. Scheduled and treated like a benchmark run, not a daemon: the
roadmap's concurrency policy makes continuous in-line monitoring impossible
(the GPU belongs to serving during serving), and the ecosystem gap note
records that llama-server has no activation read path. Asynchronous batch
replay is the design, not a compromise; the exploratory llama.cpp
read-path study stays out of scope for this phase.

### Decisions

1. **Replay, not interception.** interp/probes/monitor.py reads a log
   window (--since/--until or --window 7d) from the W3 log directory,
   reconstructs prompts (and optionally prompt+response) with the subject
   model's chat template, runs a single batched bf16 forward pass per
   example capturing only the layers registered probes need, and calls
   W2's score.py. Runs as batch work: it takes the GPU lock and refuses to
   run while inference services are up (make target checks service health
   endpoints first), mirroring training D9's lock discipline.
2. **The monitored model must be the served model's base.** The registry
   hash check (W2) enforces that probes and replay model match; the monitor
   additionally warns when the served GGUF recorded in the logs does not
   derive from that base checkpoint. Scoring bf16 activations as a proxy
   for the quantized service's internals is an approximation; the report
   states it, and the quantization-feature-survival research track (Phase
   5) owns quantifying it.
3. **Report format follows the benchmark JSON conventions.** One JSON per
   run under /mnt/ai-data/interp/monitor/<run>/ plus a summary appended to
   the experiments log: window, record counts (total, replayed, skipped
   with reasons), per-probe score distributions (mean, p50, p95, fraction
   over threshold), the top-N highest-scoring records by id (ids, not
   full text, in the report; the log holds the text), and deltas versus
   the previous run over the same probe set — the drift signal the Phase 5
   probes-in-production track builds on.
4. **Alerting is a file, not a pager.** Records over threshold land in
   flagged.jsonl beside the report. No notification integration in this
   phase; the report is reviewed like benchmark output.
5. **Cost envelope.** Replay is throughput-shaped (tens of thousands of
   tokens/s at bf16 on small models per the hardware envelope); a week of
   single-user traffic is minutes of GPU time. If a window is unexpectedly
   large the job processes newest-first up to a --max-records cap and says
   what it dropped.

### Interfaces

- make probe-monitor ARGS="--window 7d --probes <name>[,<name>...]"
- Package: interp/probes/monitor.py reusing capture/score from W2.

### Acceptance

- The monitor produces a report over at least one real window of serving
  logs (roadmap acceptance), with per-probe distributions and a populated
  flagged file or an explicit zero-flag summary.
- Running it twice over the same window is deterministic (same scores) and
  the second report records the delta-versus-previous correctly.
- It refuses to start while llama-gpu is healthy (concurrency check) and
  while a probe's model hash mismatches the replay model.
- A log window containing malformed records completes with those records
  counted as skipped, not a crash.

## Phase 3 exit checklist

Restating the roadmap's acceptance criteria against the deliverables above:

1. A control vector extracted at bf16 measurably shifts behavior of the
   quantized served model, with the three-arm scorecard published either
   way, including the case where the tuned system-prompt baseline wins (W1).
2. A probe beats its black-box baseline on held-out data, or the negative
   result is written up (W2).
3. The monitor produces a report over at least one real window of serving
   logs (W3 provides the logs, W4 the report).
4. All artifacts (vectors, probes, reports) carry manifests tracing them to
   model hashes, datasets, and git SHAs, and the experiments log contains
   the post-mortems.

---

*Last Updated: 2026-07-10*
