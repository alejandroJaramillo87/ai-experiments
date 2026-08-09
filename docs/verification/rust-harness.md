# Rust harness verification (CPU backend)

**Operational runbook. Executed against the workstation, not read for understanding: the operator's pasted output is the evidence. Keep the commands runnable, and the paths accurate — both documentation gates check this directory, so a stale reference here fails the gate.**

Workstation verification for the `rust-harness/` workspace: the CPU
inference backend on the real Gemma 4 E4B GGUF, the observability stack
(flight recorder, tracing, rr), the in-band interpretability chain
(capture, probe monitor, steering), the chat mode, the containerized
build, and the run-level observability records. Every rust-harness claim to date is
golden-tested in the sandbox against a synthetic 16-dimension fixture; this
campaign is the first contact with real hardware and real weights.

Scope is CPU only. Not covered, and why:

- The cuda Tier 3 acceptance tests (`runtime/tests/cuda_acceptance.rs`,
  `runtime/tests/eval_acceptance.rs`: logits versus the Hugging Face
  reference, WikiText-2 perplexity, greedy-decode goldens) are
  feature-gated stubs awaiting the cuda backend (roadmap phase A). Their
  fixture-generation commands are documented in the test files and enter
  the runbook with that phase.
- The chat-template golden test (`harness/src/template.rs`,
  `rendered_token_ids_match_hf_reference`) has an unimplemented body; it
  needs code work before a runbook entry can exercise it.
- The KV cache holds the full sequence with the sliding window applied as
  a mask; the budgeted ring buffer is deferred by design
  (`docs/rust-harness/observability.md`).

Recording protocol: the [runbook README](README.md) rules apply. The human
runs each command on the workstation and pastes command plus output
verbatim into the Observed block; expected signals are never edited to
match observations; one GPU consumer at a time (the CPU backend uses no
GPU, but RH.9 shares `/mnt/ai-data` and the interp Python environment with
GPU workflows). The Python session-start gate (`pytest -m hw_smoke`) does
not apply here; RH.0 is this phase's gate. The model directory
`/mnt/ai-data/models` is read-only for agents; all commands below only
read from it.

The sandbox-executable code-quality counterpart to this runbook is the
[QA campaign](rust-harness-qa.md) (phases Q0-Q6); it verifies code
quality, while this runbook verifies behavior on real weights and
hardware.

Entries carry a "Risks likely to surface here" block where a failure mode
is predicted. Rust-harness has no known-bugs ledger; these are known
unknowns, and a failure matching a listed risk should be recorded against
it rather than re-diagnosed from scratch.

## RH.0 Build and test gate

Implementation: `rust-harness/Makefile`, `rust-harness/Cargo.lock`; the test
surface is `rust-harness/runtime/tests/` and `rust-harness/harness/tests/`.

Blocks every other entry. First build of the workspace on the workstation
toolchain.

Prerequisite: a `python3` on `PATH` that can import numpy. The npz writer's
unit test (`runtime/src/npz.rs`) shells out to `python3` to re-read a
written shard with `np.load`, and `harness/tests/probe_identity.rs` takes
its independent SHA-256 from `python3`'s hashlib, so `make quality` fails
outright without them. `runtime/tests/activation_capture.rs` skips only its
row-level assertions when numpy is missing, and the CPU-reference goldens
(`cpu_reference.rs`) are committed JSON and need neither.

Command:

```bash
$ cd rust-harness
$ rustc --version && cargo --version
$ python3 -c "import numpy; print(numpy.__version__)"
$ make quality
```

Expected signal: exit 0. The gate runs five offline steps, cheapest first,
failing on the first error: `cargo fmt --all --check`,
`cargo clippy --workspace --all-targets -- -D warnings`,
`cargo metadata --locked` (the lockfile check),
`cargo doc --workspace --no-deps --document-private-items` under
`RUSTDOCFLAGS="-D warnings"`, and `cargo test --workspace`; all
sandbox-green tests pass unchanged, including the goldens against the numpy
reference (`cpu_reference.rs`) and the phase C acceptance test (200 forced
tool calls, zero malformed).

Observed:

```
```

Status: PENDING

## RH.1 Model location and architecture reconciliation

Implementation: `rust-harness/runtime/src/bin/gguf-info.rs`,
`rust-harness/runtime/src/gguf.rs`,
`rust-harness/runtime/src/model/cpu.rs` (the capability guard).

**Risks likely to surface here.** This is the highest-risk entry of the
campaign: the CPU backend was written against a Gemma-4-style architecture
spec, never against the real artifact. Published sources say the real E4B
architecture is a superset of what the backend implements, retaining
per-layer embeddings and MatFormer nesting and adding a shared KV cache and
dual RoPE; the
[architecture reconciliation](../rust-harness/runtime-design.md#architecture-reconciliation)
records those findings and their confidence. Those sources are secondhand,
so this entry decides the question for the checkpoint that actually exists.

The backend refuses a checkpoint carrying any of those features rather than
loading it and ignoring the extra weights, so this entry produces a verdict
rather than a judgment call. Four questions decide the follow-up work:

1. Are there per-layer embedding tensors?
2. Do any layers lack `attn_k`/`attn_v` (the shared-KV signature)?
3. Are the feed-forward widths uniform across layers, or per-layer
   (MatFormer slicing)?
4. Is RoPE a single frequency base at full head dimension, or dual and
   partial?

Command:

```bash
$ ls -la /mnt/ai-data/models/gguf/
$ cd rust-harness
$ cargo run -p runtime --bin gguf-info -- /mnt/ai-data/models/gguf/<e4b>.gguf
```

Expected signal: the run ends in a compatibility verdict, printed by the same
`cpu::check_supported` function the loader calls, so the tool and the loader
cannot disagree. `[OK] CPU backend compatibility: supported` means the
checkpoint is within the implemented subset and RH.2 can proceed.
`[ERROR] CPU backend compatibility: ...` names the unimplemented feature and
is the expected outcome if the published architecture holds; that message
defines the follow-up code work and does not invalidate this entry.

The architecture summary section answers the four questions directly: tensor
families with counts, the quantization-type histogram against the implemented
set {F32, F16, Q8_0}, feed-forward widths (uniform or per-layer), the list of
layers lacking `attn_k`/`attn_v`, and every RoPE, window, and per-layer
metadata key. Record the full dump verbatim, including the raw metadata and
tensor sections, because the summary is derived and the raw dump is the
evidence.

Observed:

```
```

Status: PENDING

### RH.1a Shared-KV weight conflict

Implementation: `rust-harness/runtime/src/model/cpu.rs` (the loader's
`owns_cache` branch and the tensors it declines to read).

Decides the one open question the CPU backend already had to answer in order
to run. `modeling_gemma4.py` builds `k_proj`, `v_proj`, and `k_norm` only for
non-shared layers; every released E-series checkpoint carries all three on
every layer. The backend follows the modeling code and leaves the shared
tail's copies unread. If that reading is wrong, 18 of E4B's 42 layers attend
over the wrong keys and the output is fluent and wrong rather than an error.
The evidence on both sides is in
[The shared-KV weight conflict](../models/gemma-4.md#the-shared-kv-weight-conflict).

Run the two checks in order. The first is cheap and settles the question on
its own; the second is a logit-level confirmation and costs a model load.

**Check 1 — were the surplus tensors ever trained?** A tensor that was
allocated and never updated keeps the *shape* of its initialization
distribution. Scale is the wrong statistic to test: `initializer_range`
places an untrained tensor at close to the same standard deviation as a
trained one, so the two are indistinguishable by `std`. Distribution shape
separates them. Two shape statistics suffice:

- `|mean| / std`. A Gaussian gives 0.7979. A truncated normal, which is what
  `transformers` initializes with, gives a higher value because the tails are
  cut. Trained weights become heavy-tailed and give a lower one.
- Excess kurtosis. Negative for a truncated normal, positive for trained
  weights.

Compare all four attention projections in the same layer. Within the shared
tail, `q_proj` and `o_proj` are used and must be trained; `k_proj` and
`v_proj` are the tensors in dispute. Two tensors of one module diverging from
the other two isolates the effect to the gradient path, which no
whole-layer comparison can do.

```bash
$ cd /home/user/ai-experiments
$ .venv/bin/python - <<'EOF'
import json, numpy as np, torch
from safetensors import safe_open

DIR = "/mnt/ai-data/models/hf/gemma-4-E2B"
GAUSS = np.sqrt(2 / np.pi)

text = json.load(open(f"{DIR}/config.json"))["text_config"]
n, shared = text["num_hidden_layers"], text["num_kv_shared_layers"]
boundary = n - shared
print(f"{n} layers, boundary {boundary}, Gaussian |mean|/std = {GAUSS:.4f}\n")

PROJ = ["q_proj", "k_proj", "v_proj", "o_proj"]
stats = {}
with safe_open(f"{DIR}/model.safetensors", framework="pt") as f:
    names = set(f.keys())
    for layer in range(n):
        for p in PROJ:
            key = f"model.language_model.layers.{layer}.self_attn.{p}.weight"
            if key not in names:
                key = f"language_model.model.layers.{layer}.self_attn.{p}.weight"
            if key not in names:
                continue
            w = f.get_tensor(key).float().numpy().ravel()
            s = w.std()
            stats[(layer, p)] = (
                float(np.abs(w).mean() / s),
                float(((w / s) ** 4).mean() - 3.0),
            )

print("group means (ratio)")
for p in PROJ:
    o = [stats[(l, p)][0] for l in range(boundary) if (l, p) in stats]
    s = [stats[(l, p)][0] for l in range(boundary, n) if (l, p) in stats]
    f2 = lambda g: f"{np.mean(g):.4f} (spread {max(g) - min(g):.4f})" if g else "-"
    print(f"  {p:8s} owns {f2(o)}   shared {f2(s)}")
EOF
```

The script reads tensor data and needs no GPU. The `E4B` directory
substitutes for `E2B` unchanged; the base and instruction-tuned variants
carry identical tensor layouts, so whichever is on disk serves.

Expected signal: if the shared tail's `k_proj` and `v_proj` cluster tightly
above the Gaussian ratio with negative kurtosis while `q_proj` and `o_proj`
in the same layers scatter below it with positive kurtosis, the disputed
tensors never entered the gradient path, the modeling code is authoritative,
and the backend's reading is correct. If all four projections move together,
the weights are meaningful and check 2 becomes mandatory rather than
confirmatory.

**Check 2 — do the logits agree?** Generate greedily from the same prompt
through `transformers` and through the harness, and compare. Confirm the GGUF
filename against the directory before running; the conversion output name
depends on the quantization chosen at conversion time.

```bash
$ ls /mnt/ai-data/models/gguf/
$ cd /home/user/ai-experiments/rust-harness
$ cargo run --release -p harness -- \
    --model /mnt/ai-data/models/gguf/gemma-4-E2B/gemma-4-E2B-BF16.gguf \
    --prompt 'The capital of France is' --max-tokens 16 --temperature 0
```

Then the reference, on the same prompt and the same decoding rule:

```bash
$ cd /home/user/ai-experiments
$ .venv/bin/python - <<'EOF'
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "/mnt/ai-data/models/hf/gemma-4-E2B"
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32)
model.eval()
ids = tok("The capital of France is", return_tensors="pt")
with torch.no_grad():
    out = model.generate(**ids, max_new_tokens=16, do_sample=False)
print(tok.decode(out[0], skip_special_tokens=True))
EOF
```

Expected signal: matching token sequences mean the backend's reading is
correct. Divergence that begins early and does not recover means it is not —
in which case flip the `owns_cache` branch in the loader so shared layers
compute their own K/V, rebuild, and rerun.

Capture the `from_pretrained` output verbatim, warnings included. A warning
naming the shared layers' `k_proj`, `v_proj`, or `k_norm` as unexpected keys
is direct confirmation that the code treats them as surplus. Silence there is
equally informative.

Observed:

```
2026-08-09, check 1, gemma-4-E2B base weights, 35 layers, boundary 15.

group means (ratio)
  q_proj   owns 0.7416 (spread 0.0609)   shared 0.7295 (spread 0.0355)
  k_proj   owns 0.7471 (spread 0.0725)   shared 0.8217 (spread 0.0010)
  v_proj   owns 0.7523 (spread 0.1085)   shared 0.8217 (spread 0.0011)
  o_proj   owns 0.7578 (spread 0.0480)   shared 0.7635 (spread 0.0335)

Excess kurtosis, shared layers 15-34: k_proj and v_proj between -0.64 and
-0.63 on every one of the 20 layers; q_proj between +1.34 and +6.91; o_proj
between +0.75 and +3.12. Owning layers 0-14 are positive on all four
projections.

Per-layer k_proj shapes confirm the attention schedule independently: 512
wide at layers 4, 9, 14, 19, 24, 29, 34 and 256 wide elsewhere, so the last
non-shared global layer is 14 and the last non-shared sliding layer is 13.

Configuration sweep, both variants: enable_moe_block false, num_experts null,
attention_bias false, attention_k_eq_v false. use_double_wide_mlp is true on
E2B and false on E4B.
```

Interpretation: within a single shared layer, `q_proj` and `o_proj` carry the
distributional signature of trained weights while `k_proj` and `v_proj` are
pinned to four significant figures with negative excess kurtosis on all 20
layers. A tensor cannot lose kurtosis relative to its initialization while
the tensors beside it in the same module gain kurtosis, so the disputed
tensors were allocated and never updated. The checkpoint's surplus `k_proj`
and `v_proj` are initialization residue, `modeling_gemma4.py` is
authoritative, and the loader's `owns_cache` branch is correct as
implemented. `k_norm` was not measured directly; its head-width shape makes
the same statistic noisy, and the inference from `k_proj` and `v_proj`
covers it.

Check 2 remains listed because check 1 is statistical inference over weight
distributions rather than a logit comparison. Run it opportunistically when
the GGUF conversion is available. It is confirmatory, not blocking.

Status: RESOLVED 2026-08-09 in favor of the modeling code, on check 1.
Check 2 outstanding as confirmation.

## RH.2 Real-model load and generation smoke

Implementation: `rust-harness/harness/src/main.rs`,
`rust-harness/harness/src/agent.rs`,
`rust-harness/runtime/src/model/cpu.rs`,
`rust-harness/runtime/src/tokenizer.rs`.

First run of the tokenizer against the real vocabulary and of the dequant
paths against real tensors.

Command:

```bash
$ cd rust-harness
$ mkdir -p /tmp/rh2 && cd /tmp/rh2
$ time ~/ai-experiments/rust-harness/target/release/harness \
    --model /mnt/ai-data/models/gguf/<e4b>.gguf \
    "Reply with the single word: ready"
```

(Build once with `cargo build --release -p harness`.)

Expected signal: exit 0; the model loads without panic, the reply is
coherent text responsive to the instruction, and `session.jsonl` appears
in the working directory. Coherence is the signal: a numerically wrong
forward pass on real weights produces fluent-looking garbage or
repetition, so record the full reply.

Observed:

```
```

Status: PENDING

## RH.3 CPU generation speed

Implementation: `rust-harness/runtime/src/model/cpu.rs`,
`rust-harness/runtime/src/tensor.rs`,
`rust-harness/runtime/src/sampler.rs`.

No pass threshold. The measured rate decides whether real-model activation
capture and rr recording are practical on the CPU backend, and it is the
baseline the cuda backend is later compared against. The backend is
single-threaded by design.

Command:

```bash
$ mkdir -p /tmp/rh3 && cd /tmp/rh3 && rm -f session.jsonl
$ time ~/ai-experiments/rust-harness/target/release/harness \
    --model /mnt/ai-data/models/gguf/<e4b>.gguf \
    --trace-chrome trace.json \
    "Count from one to twenty as words, one per line"
```

Expected signal: record wall-clock time and the generated token count
(from the trace's `model.forward` span count or the transcript), and
report the rate in tokens/second. Prefill and decode rates should be
reported separately if the trace makes them distinguishable
(`session.sync` versus `session.step` spans).

Observed:

```
```

Status: PENDING

## RH.4 KV delta prefill

Implementation: `rust-harness/runtime/src/kv.rs`,
`rust-harness/runtime/src/session.rs`,
`rust-harness/runtime/src/model/cpu.rs`.

The phase B acceptance claim in CPU form: turn latency scales with the
appended delta, not the transcript. A second run in the same session root
continues the session from `session.jsonl` and should prefill only the
new suffix.

Command:

```bash
$ mkdir -p /tmp/rh4 && cd /tmp/rh4
$ time ~/ai-experiments/rust-harness/target/release/harness \
    --model /mnt/ai-data/models/gguf/<e4b>.gguf --trace-chrome turn1.json \
    "Summarize the Unix philosophy in three sentences"
$ time ~/ai-experiments/rust-harness/target/release/harness \
    --model /mnt/ai-data/models/gguf/<e4b>.gguf --trace-chrome turn2.json \
    "Now compress that summary to one sentence"
```

Expected signal: the turn-2 `session.sync` prefill span covers far fewer
positions than turn 1's transcript length (the delta is the new user turn,
not the whole history), and turn-2 prefill wall-clock is proportionally
smaller. Record both span durations and position counts. Both paths are
measured: turn 1 is the full-prefill datum, turn 2 the delta datum.

Observed:

```
```

Status: PENDING

## RH.5 Flight recorder and sandbox replay

Implementation: `rust-harness/runtime/src/model/record.rs`,
`rust-harness/runtime/tests/record_replay.rs`.

Host half records; the sandbox half replays the recording against the
mock, which any agent session can run.

Command (host):

```bash
$ mkdir -p /tmp/rh5 && cd /tmp/rh5
$ ~/ai-experiments/rust-harness/target/release/harness \
    --model /mnt/ai-data/models/gguf/<e4b>.gguf --record flight.jsonl \
    "Reply with the single word: ready"
```

Command (sandbox, after copying `flight.jsonl` into the worktree; there
is no committed CLI for arbitrary recordings, so a throwaway test invokes
the replay loader `runtime::model::record::replay_file` directly):

```bash
$ cd rust-harness
$ mkdir -p runtime/tests/data && cp <flight.jsonl> runtime/tests/data/rh5-flight.jsonl
$ cat > runtime/tests/rh5_replay.rs <<'EOF'
use std::path::Path;

#[test]
fn rh5_recording_replays_as_a_mock_script() {
    runtime::model::record::replay_file(Path::new("tests/data/rh5-flight.jsonl")).unwrap();
}
EOF
$ cargo test -p runtime --test rh5_replay
$ git checkout -- . && rm -rf runtime/tests/data runtime/tests/rh5_replay.rs
```

Expected signal: the host run writes one JSONL record per forward call
with token and logit payloads, and the throwaway test passes —
`replay_file` rebuilds the recording as a MockModel script without error.

Observed:

```
```

Status: PENDING

## RH.6 Chrome trace in Perfetto

Implementation: `rust-harness/harness/src/obs.rs`,
`rust-harness/harness/tests/tracing_spans.rs`,
`rust-harness/runtime/tests/tracing_spans.rs`.

Command:

```bash
$ mkdir -p /tmp/rh6 && cd /tmp/rh6
$ ~/ai-experiments/rust-harness/target/release/harness \
    --model /mnt/ai-data/models/gguf/<e4b>.gguf --trace-chrome trace.json \
    "Reply with the single word: ready"
```

Expected signal: `trace.json` loads in Perfetto (ui.perfetto.dev) and
shows the documented span set — `session.sync`, `session.step`,
`sampler.sample`, `grammar.advance`, `tool.exec`, `model.forward` — with
`model.forward` dominating wall-clock.

Observed:

```
```

Status: PENDING

## RH.7 rr record, replay, reverse-continue

Implementation: `rust-harness/Makefile` (the rr targets),
`rust-harness/harness/src/main.rs`.

The observability phase 3 acceptance: a CPU forward pass recorded and
replayed under rr, with a watchpoint and `reverse-continue` reaching a
chosen write. Requires rr installed and
`kernel.perf_event_paranoid <= 1`. The real model may be slow under rr;
the committed tiny fixture is an acceptable subject, and the Observed
block must record which model was used.

Command:

```bash
$ cd rust-harness
$ make debug-record ARGS='--model fixtures/tiny-gemma4.gguf "Reply with the single word: ready"'
$ make debug-replay
```

Inside the rr/gdb session:

```
(rr) break cpu.rs:forward_one
(rr) continue
(rr) watch -l logits[0]
(rr) reverse-continue
```

Expected signal: recording completes; replay reaches the breakpoint; the
watchpoint triggers under `reverse-continue` and gdb stops at the write
site inside the forward pass. Record the gdb transcript.

Observed:

```
```

Status: PENDING

## RH.8 End-to-end coding task

Implementation: `rust-harness/harness/src/permissions.rs`,
`rust-harness/harness/src/tools.rs`, `rust-harness/harness/src/agent.rs`.

The full agent loop (permissions, tools, transcript) on real weights.

Command:

```bash
$ mkdir -p /tmp/rh8 && cd /tmp/rh8
$ printf 'rules:\n  - tool: "*"\n    action: allow\n' > permissions.yaml
$ ~/ai-experiments/rust-harness/target/release/harness \
    --model /mnt/ai-data/models/gguf/<e4b>.gguf \
    "Create a file named hello.txt containing the word hello, then read it back"
```

Expected signal: the task completes with exit 0, `hello.txt` exists with
the expected content, and `session.jsonl` records the tool calls the model
made. Tool-call formation is grammar-constrained, so malformed calls
should be impossible; any malformed call is a finding.

Observed:

```
```

Status: PENDING

## RH.9 Interp chain on real weights

Implementation: `rust-harness/runtime/src/bin/activation-capture.rs`,
`rust-harness/runtime/src/npz.rs`,
`rust-harness/runtime/src/model/probe.rs`; the Python side is
`interp/probes/capture.py` and `interp/probes/export.py`.

Four sequential sub-entries; each feeds the next. This closes the roadmap
rows "real-model capture" and "monitor with a probe trained on real-model
rust captures" in `docs/interp/rust-harness-interp.md`. The
labeled dataset is any small JSONL of `{"text": ..., "label": 0|1}` lines
with a separable attribute; record which dataset was used.

**Risks likely to surface here.**

- RH.3 may show capture over even a few hundred examples is impractically
  slow on CPU; if so, record the projected time and stop — the finding
  bounds the CPU path and motivates the cuda tap.
- The rust `tokenizer_fingerprint` is GGUF-derived and intentionally not
  comparable to Python-capture fingerprints; probe training on rust
  captures must go through artifacts produced by this chain only.

### RH.9a Activation capture

Command:

```bash
$ cd rust-harness
$ cargo run --release -p runtime --bin activation-capture -- \
    --model /mnt/ai-data/models/gguf/<e4b>.gguf \
    --dataset <labeled.jsonl> --layers <l0,l1> \
    --out /mnt/ai-data/interp/captures/rust-e4b-verify
$ uv run python -c "from interp.probes.capture import load_capture; \
    acts, labels, texts = load_capture('/mnt/ai-data/interp/captures/rust-e4b-verify'); \
    print(sorted(acts), {k: v.shape for k, v in acts.items()}, len(labels), len(texts))"
```

Expected signal: the capture directory holds `shard_00000.npz` onward,
`examples.jsonl`, and a `manifest.json` with
`capture_backend: "rust-harness-cpu"` and the GGUF `model_hash`;
`load_capture` reports the requested layers with row counts matching the
dataset. Record the capture wall-clock time.

Observed:

```
```

Status: PENDING

### RH.9b Probe training on rust captures

Command:

```bash
$ cd ~/ai-experiments
$ make probe-train ARGS="--capture /mnt/ai-data/interp/captures/rust-e4b-verify \
    --name rust-e4b-verify-probe"
```

Expected signal: training selects the best layer by validation AUROC and
registers the probe; AUROC well above 0.5 for a separable attribute.
Record the AUROC and selected layer.

Observed:

```
```

Status: PENDING

### RH.9c In-band probe monitor

Command:

```bash
$ mkdir -p /tmp/rh9/mon /tmp/rh9/base
$ uv run python -c "from interp.probes.export import write_probe_gguf; \
    write_probe_gguf('/mnt/ai-data/interp/probes/rust-e4b-verify-probe', '/tmp/rh9/probe.gguf')"
$ cd /tmp/rh9/base && ~/ai-experiments/rust-harness/target/release/harness \
    --model /mnt/ai-data/models/gguf/<e4b>.gguf "Reply with the single word: ready"
$ cd /tmp/rh9/mon && ~/ai-experiments/rust-harness/target/release/harness \
    --model /mnt/ai-data/models/gguf/<e4b>.gguf --probe /tmp/rh9/probe.gguf \
    "Reply with the single word: ready"
$ diff <(grep -v '"ts"' /tmp/rh9/base/session.jsonl) <(grep -v '"ts"' /tmp/rh9/mon/session.jsonl); echo "diff exit: $?"
$ head /tmp/rh9/mon/probe-flags.jsonl
```

Expected signal: the monitored run prints the one-line probe summary
(positions scored, positions flagged, threshold) to stderr, writes
`probe-flags.jsonl`, and its generated output is identical to the
unmonitored baseline (observation only). If the transcripts embed
timestamps, compare the generated text rather than raw lines and record
the comparison used.

Observed:

```
```

Status: PENDING

### RH.9d Steering on real weights

Command (using an existing exported control vector, or one extracted from
rust captures via `interp/steering`):

```bash
$ mkdir -p /tmp/rh9/steer1 /tmp/rh9/steer2
$ cd /tmp/rh9/steer1 && ~/ai-experiments/rust-harness/target/release/harness \
    --model /mnt/ai-data/models/gguf/<e4b>.gguf \
    --control-vector <vector.gguf> --control-vector-scale 1.0 \
    "Describe your day in two sentences"
$ cd /tmp/rh9/steer2 && ~/ai-experiments/rust-harness/target/release/harness \
    --model /mnt/ai-data/models/gguf/<e4b>.gguf \
    --control-vector <vector.gguf> --control-vector-scale 1.0 \
    "Describe your day in two sentences"
```

Expected signal: the steered output differs from the RH.2/RH.9c baseline
in the direction of the vector's attribute, and the two steered runs are
identical to each other (deterministic). Record which vector was used and
its provenance.

Observed:

```
```

Status: PENDING

## RH.10 Flamegraph baseline

Implementation: `rust-harness/Makefile` (the `flame` target),
`rust-harness/runtime/src/model/cpu.rs` (the profiled hot path).

Requires `cargo-flamegraph` and `perf`. Records the optimization baseline
for the CPU backend; there is no pass threshold beyond the profile being
attributable.

Command (the `make flame` target takes no arguments, so the underlying
command is invoked directly with the trailing binary arguments):

```bash
$ cd rust-harness
$ cargo flamegraph --profile release-with-debug --bin harness -- \
    --model /mnt/ai-data/models/gguf/<e4b>.gguf "Reply with the single word: ready"
```

Expected signal: `flamegraph.svg` is produced and the profile is dominated
by the forward-pass matrix multiplications (attention and MLP projections
in `model::cpu`), not by tokenization, sampling, or I/O. Record the top
frames.

Observed:

```
```

Status: PENDING

## RH.11 Multi-turn chat on real weights

Implementation: `rust-harness/harness/src/chat.rs`,
`rust-harness/harness/src/compact.rs`,
`rust-harness/harness/src/template.rs`,
`rust-harness/runtime/src/session.rs`.

Chat mode (`--chat`) is the decode-heavy complement to the prefill-heavy
agent workload; within one process each turn delta-prefills only the new
turn, which makes this a cleaner vehicle for the RH.4 claim. The stop
behavior of the real instruction-tuned model is unconfirmed: the session
stops at the tokenizer EOS, and a distinct `<turn|>` terminator
would instead run to the `--max-tokens` budget with the trailing marker
trimmed from the reply. Record which behavior is observed.

Command (three or more turns, then EOF or `/quit`):

```bash
$ mkdir -p /tmp/rh11 && cd /tmp/rh11
$ ~/ai-experiments/rust-harness/target/release/harness \
    --model /mnt/ai-data/models/gguf/<e4b>.gguf --chat \
    --trace-chrome trace.json --max-tokens 128
```

Expected signal: replies are coherent and responsive to the conversation
history across at least three turns; `session.jsonl` records alternating
user and model entries; the per-turn `session.sync` spans in the trace
cover only each turn's delta, not the whole transcript. Record whether
replies terminate at EOS or run to the token budget.

Observed:

```
```

Status: PENDING

## RH.12 Container build and reproducibility

Implementation: the harness targets in the root `Makefile`,
`docker-compose.rust-harness.yaml`,
`docker/rust-harness/Dockerfile.rust-harness`.

The containerized build exists for reproducibility: a pinned rust
toolchain and `cargo build --locked` inside the image must produce a
binary whose seeded generation matches the host build.

Command:

```bash
$ cd ~/ai-experiments
$ make harness-build
$ make harness-run ARGS='--model /app/models/gguf/<e4b>.gguf --seed 7 "Reply with the single word: ready"'
$ mkdir -p /tmp/rh12 && cd /tmp/rh12
$ ~/ai-experiments/rust-harness/target/release/harness \
    --model /mnt/ai-data/models/gguf/<e4b>.gguf --seed 7 \
    "Reply with the single word: ready"
```

Expected signal: the image builds from the pinned digests without
resolving new dependency versions (`--locked` succeeds); the container
run and the host run produce the same reply for the same seed and model;
session state lands in `sessions/rust-harness/` on the host.

Observed:

```
```

Status: PENDING

## RH.13 Run-level observability records

Implementation: `rust-harness/harness/src/obs.rs`,
`rust-harness/harness/tests/obs_contract.rs`, `common/obs/catalog.py`.

The harness writes run-level records in the shared obs contract
(`common/obs`): an event file `${OBS_LOG_DIR}/harness/<run_id>.jsonl`
holding `cmd.start` and `cmd.end` (plus `fail` on error), and one catalog
line appended to `${OBS_LOG_DIR}/harness/runs.jsonl` per invocation.
The Python readers are stateless JSONL consumers, so they must read
harness-written files unchanged. Cross-language compatibility is verified
in the sandbox against a scratch `OBS_LOG_DIR`; this entry confirms the
default log root on the workstation.

Command (a normal run, then a forced failure, then inspection):

```bash
$ mkdir -p /tmp/rh13 && cd /tmp/rh13
$ ~/ai-experiments/rust-harness/target/release/harness \
    --model /mnt/ai-data/models/gguf/<e4b>.gguf \
    "Reply with the single word: ready"
$ ~/ai-experiments/rust-harness/target/release/harness \
    --model /nonexistent.gguf "task" ; echo "exit: $?"
$ tail -2 /mnt/ai-data/logs/harness/runs.jsonl
$ cd ~/ai-experiments && uv run python -c "
from common.obs import catalog
rows = catalog.read_runs(catalog.catalog_path('harness'))
for r in rows[-2:]:
    print(r['run_id'], r['command'], r['exit_code'])
"
```

Expected signal: the successful run catalogs with `exit_code` 0 and the
failed run with `exit_code` 1 plus a `fail` event in its event file; run
ids match `R-YYYYMMDD-HHMMSS-xxxxxx`; the Python catalog reader returns
both rows without errors.

Observed:

```
```

Status: PENDING
