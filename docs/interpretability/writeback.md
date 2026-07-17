# Write-back path runbook

Operating guide for the interpretability write-back path (interp Phase 3, deliverables
W1-W4): control-vector steering deployed into the llama.cpp services, a probe toolkit with
mandatory black-box baselines, a serving request log, and the async probe monitor. Weights
are never modified; every intervention acts on activations only. Code lives in `interp/`
(packages, invoked as `python -m interp <subcommand>` via `interp/commands/`);
`scripts/interp/` holds only shell helpers (the isolated Neuronpedia and SAEBench
environments); tests in `tests/interp/`.

## Table of contents

- [Layout](#layout)
- [W1: steering deployment](#w1-steering-deployment)
- [W2: probe toolkit](#w2-probe-toolkit)
- [W3: serving request log](#w3-serving-request-log)
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
over-threshold records land in `flagged.jsonl` beside it. bf16 replay approximates the
quantized service's internals; the report states this.

## Concurrency

| Work | Shape | Requires |
|------|-------|----------|
| steer-extract, probe-capture, probe-monitor | batch GPU | inference down (preflight enforced) |
| steer-scorecard, probe-eval | serving-dependent | subject and judge services up; GPU lock free |
| logproxy | sidecar | target service up |

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
```
