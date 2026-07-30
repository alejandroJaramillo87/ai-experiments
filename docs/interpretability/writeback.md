# Write-back path runbook

Operating guide for the interpretability write-back path (interp Phase 3, deliverables
W1-W4): control-vector steering deployed into the llama.cpp services, a probe toolkit with
mandatory black-box baselines, a serving request log, and the async probe monitor. Weights
are never modified; every intervention acts on activations only. Code lives in `interp/`
(packages, invoked as `python -m interp <subcommand>` via `interp/commands/`);
`scripts/interp/` holds only shell helpers (the isolated Neuronpedia and SAEBench
environments); tests in `tests/interp/`.

A parallel substrate exists for the rust harness runtime: in-band taps,
steering, and capture in owned code instead of the llama.cpp workarounds,
producing the same capture shard and control-vector artifacts. See
[rust-harness-interp.md](rust-harness-interp.md).

## Table of contents

- [Layout](#layout)
- [W1: steering deployment](#w1-steering-deployment)
- [W2: probe toolkit](#w2-probe-toolkit)
- [W3: serving request log](#w3-serving-request-log)
- [Full-precision monitored path](#full-precision-monitored-path)
- [W4: async probe monitor](#w4-async-probe-monitor)
- [Concurrency](#concurrency)
- [Host-side acceptance checks](#host-side-acceptance-checks)

## Layout

| Path | Contents |
|------|----------|
| `interp/steering/` | extraction, GGUF export, service wiring, scorecard |
| `interp/steering/prompts/` | fixed prompt sets in git (pairs, behavior, neutral, tuned system prompt) |
| `interp/probes/` | capture, training, registry, scoring, evaluation, monitor |
| `docker/interp-logproxy/` | single-file logging reverse proxy and its Dockerfile |
| `/mnt/ai-data/interp/vectors/<name>/` | `vector.gguf` + `manifest.json` + scorecard runs |
| `/mnt/ai-data/interp/activations/<name>/` | capture shards (regenerable cache) |
| `/mnt/ai-data/interp/probes/<name>/` | `probe.npz` + `manifest.json` + `evaluation.json` |
| `/mnt/ai-data/interp/logs/<service>/` | `YYYY-MM-DD.jsonl` request records (30-day retention) |
| `/mnt/ai-data/interp/monitor/<run>/` | `report.json` + `flagged.jsonl` |

Every artifact directory carries a `manifest.json` (model hash, dataset hashes, git SHA,
artifact xxhash). Downstream tools refuse artifacts whose hashes do not match: the entrypoint
refuses a missing vector file, `make steer-scorecard` refuses a tampered vector, and the
monitor refuses a probe whose model hash does not match the replay model.

## W1: steering deployment

Extraction runs at bf16 in the workbench (batch GPU work); serving loads the exported GGUF
as configuration.

```bash
$ make steer-extract ARGS="--method repeng --model google/gemma-2-2b \
    --pairs interp/steering/prompts/honesty_pairs.jsonl \
    --out /mnt/ai-data/interp/vectors/honesty-v1"
$ make steer-up VECTOR=/mnt/ai-data/interp/vectors/honesty-v1 SCALE=1.0
$ make steer-down
$ make steer-scorecard VECTOR=/mnt/ai-data/interp/vectors/honesty-v1 ATTRIBUTE="honesty"
```

`--method sae_row` exports a single Gemma Scope decoder row instead
(`--sae-release ... --sae-id ... --feature N --layer L`). The entrypoints add
`--control-vector` / `--control-vector-scaled` / `--control-vector-layer-range` only when
`CONTROL_VECTOR` is set; with it unset the launch command is byte-identical to the pre-W1
form. The scorecard drives three arms (unsteered, tuned system prompt, steered at scales
0.5/1/2), scores behavior and fluency with an LLM judge on a different service, and always
restores the unsteered state, including on failure. Output follows the benchmark JSON
conventions and is published to `docs/interpretability/experiments/` whichever arm wins.

## W2: probe toolkit

```bash
$ make probe-capture ARGS="--model google/gemma-2-2b --dataset data.jsonl \
    --layers 6,13,19 --out /mnt/ai-data/interp/activations/refusal-v1"
$ make probe-train ARGS="--capture /mnt/ai-data/interp/activations/refusal-v1 --name refusal-v1"
$ make probe-eval ARGS="--probe /mnt/ai-data/interp/probes/refusal-v1 --label-name refusal"
```

Capture is a workbench batch job (bf16, nnsight); training is CPU scikit-learn with a
deterministic text-hash split (re-captures cannot leak test data). `probe-eval` is
serving-dependent and always includes the black-box baselines (LLM-as-judge plus
bag-of-words on the model output); a probe that cannot beat them is marked not deployable
and the monitor should not run it.

## W3: serving request log

```bash
$ make logproxy-up      # 127.0.0.1:8012 in front of llama-gpu
$ make logproxy-down
```

Opt-in sidecar under the compose `interp` profile; production ports and default behavior are
untouched. Clients that want their traffic logged point at 8012. One JSONL record per
completed request (messages, response text, token counts, latency); malformed or interrupted
requests log a partial record with the `error` field set. Retention is 30 days, enforced by
the monitor preamble.

The proxy forwards to `llama-gpu:8002` by default and records the service that answered in
each record's `target_service` field. Records are written under
`/mnt/ai-data/interp/logs/<target_service>/`, so traffic from different serving endpoints
never shares a directory. Both the target and the service name are environment overrides
(`LOGPROXY_TARGET_URL`, `LOGPROXY_TARGET_SERVICE`), which is how the full-precision path
below redirects monitored traffic without a code change.

## Full-precision monitored path

```bash
$ make fullprec-up      # bf16 endpoint on 127.0.0.1:8004 plus the proxy pointed at it
$ make fullprec-down
```

Probes are fitted on bf16 activations. When monitored traffic is answered by the quantized
model on 8002, the model that answered and the model W4 replays are different models, and the
monitor's scores describe the replay model rather than the served one. The `llama-gpu-bf16`
service removes that difference by answering monitored traffic at the precision the probes
were fitted at.

The service reuses the `llama-gpu` image unchanged; `docker/llama-gpu/entrypoint.sh` already
parameterizes `SERVER_PORT` and `MODEL_PATH`. It serves the interpretability subject model
rather than the production model: gpt-oss-20b at full precision is roughly 40GB and exceeds
the 32GB VRAM budget, while Gemma-2-2B is about 5.2GB and is what the W2 probes are trained
on. Port 8002, its image, and its model are untouched.

Build the GGUF with `convert_hf_to_gguf.py --outtype bf16`. Confirm from the server's startup
log that the tensor type loaded is the one intended: ggml defines a `BF16` type, but CUDA
backend coverage for it is narrower than for f16 and some paths upconvert. If the served graph
does not run bf16 natively, use `--outtype f16` and record which was used. A service that
reports f16 while being described as bf16 reintroduces the mismatch this path exists to
remove.

Full precision costs roughly twice the bytes per weight, so expect approximately twice the
milliseconds per token and twice the VRAM of the quantized endpoint. This is the price of
measuring the model that answered.

## W4: async probe monitor

```bash
$ make down             # the monitor refuses to run while inference is up
$ make probe-monitor ARGS="--window 7d --probes refusal-v1 \
    --model google/gemma-2-2b --model-hash <hash>"
```

Batch replay, not a daemon: llama-server has no activation read path and the GPU belongs to
serving during serving, so the monitor replays the log window through the workbench model at
bf16 in one batched pass and scores every named probe. The report records per-probe
distributions, deltas versus the previous run over the same probe set, and skip counts;
over-threshold records land in `flagged.jsonl` beside it.

The monitor never sends inference to a serving endpoint. It performs its own bf16 capture in
process, and contacts 8002, 8003, and 8004 only to confirm they are down before starting,
because the replay needs the GPU.

Replay fidelity is derived from the window rather than asserted. Each record carries the
service that answered it, and the report lists the observed `target_services` alongside a note
stating what their precision means for the scores. A window served entirely at full precision
yields scores that describe the serving model; a quantized or mixed window yields scores that
describe the replay model only. Select one service per run with `--log-dir`:

```bash
$ make probe-monitor ARGS="--log-dir /mnt/ai-data/interp/logs/llama-gpu-bf16 --window 7d \
    --probes refusal-v1 --model google/gemma-2-2b --model-hash <hash>"
```

## Concurrency

| Work | Shape | Requires |
|------|-------|----------|
| steer-extract, probe-capture, probe-monitor | batch GPU | inference down (preflight enforced) |
| steer-scorecard, probe-eval | serving-dependent | subject and judge services up; GPU lock free |
| logproxy | sidecar | target service up |
| llama-gpu-bf16 | serving | GPU free; batch work down |

## Host-side acceptance checks

Sandboxed agents cannot run these; the human executes and reports back (AGENTS.md).

```bash
# Entrypoint regression: byte-identical command with CONTROL_VECTOR unset
$ docker compose up -d llama-gpu && docker logs llama-gpu | head -40

# Steer round trip
$ make steer-up VECTOR=/mnt/ai-data/interp/vectors/<name> SCALE=1.0
$ curl -s http://127.0.0.1:8002/v1/chat/completions -d '{"messages":[{"role":"user","content":"hi"}]}'
$ make steer-down

# Logproxy byte-equivalence (streaming and non-streaming)
$ make logproxy-up
$ diff <(curl -s 127.0.0.1:8012/v1/chat/completions -d @req.json) \
       <(curl -s 127.0.0.1:8002/v1/chat/completions -d @req.json)

# Full-precision path: confirm the served tensor type is the one intended
$ make fullprec-up
$ docker logs llama-gpu-bf16 | grep -iE "type +f(16|32)|bf16|general.file_type"

# Full-precision path: the proxy reaches the bf16 endpoint and records name it
$ curl -s 127.0.0.1:8012/v1/chat/completions -d @req.json
$ tail -1 /mnt/ai-data/interp/logs/llama-gpu-bf16/$(date -u +%F).jsonl | grep target_service

# Production path unchanged by any of the above
$ docker compose up -d llama-gpu && curl -sf 127.0.0.1:8002/v1/health
```
