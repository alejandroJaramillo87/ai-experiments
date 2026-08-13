# Rust harness verification (CPU backend)

**Operational runbook. Executed against the workstation, not read for understanding: the operator's pasted output is the evidence. Keep the commands runnable, and the paths accurate — both documentation gates check this directory, so a stale reference here fails the gate.**

Workstation verification for the `rust-harness/` workspace: the CPU
inference backend on the real Gemma 4 E4B checkpoint, the observability stack
(flight recorder, tracing, rr), the in-band interpretability chain
(capture, probe monitor, steering), the chat mode, the containerized
build, and the run-level observability records. Every rust-harness claim to date is
golden-tested in the sandbox against a synthetic 16-dimension fixture; this
campaign is the first contact with real hardware and real weights.

Scope is CPU only. Not covered, and why:

- The Tier 3 correctness ladder (`rust-harness/runtime/src/ladder.rs`: token
  IDs, residual stream, distribution, perplexity, greedy continuations, each
  against `transformers`) has its rung bodies implemented and unit-tested
  against a scripted mock, but no reference artifacts yet. It enters the
  runbook when they exist; the commands that generate them are in
  `rust-harness/fixtures/reference/README.md`. It replaces two feature-gated
  `unimplemented!()` stubs that were never filled in and whose fixture
  designs compared the harness against a quantized model and against its own
  first run.
- There is no chat-template token-stream golden test. An earlier `#[ignore]`d
  stub was removed with the Gemma 3 markers it was written against, and no
  replacement was added; writing one needs both code work and a
  workstation-generated tokenizer fixture before a runbook entry can
  exercise it.
- The KV cache now caps sliding layers at their window plus a batch width
  (`kv::window_capacity`, which is `window - 1 + group` and a little slack)
  rather than holding the full sequence, so `kv::budget_bytes` describes what
  a run allocates. The saving is visible only against real E4B at a real
  context, and this runbook has no step whose purpose is to report it; RH.2's
  startup banner prints it in passing, and its KV line is the figure to
  check. RH.1 has no banner — `checkpoint-info` loads no model.

**E4B at the pinned revision is a base model, and several entries below were
written as though it were instruction-tuned.** `fixtures/reference/prompts.txt`
records the measurement and the trap: asked a question it does not answer, and
"a reader who expects one will read a fluent non-answer as a defect." RH.2 hit
exactly that on 2026-08-12 and cost two host round trips before the note was
found. Instruction-shaped prompts still appear throughout this file, and in
most entries they are inert — RH.5, RH.6, RH.10 and RH.12 care about the flight
recorder, the trace, the profile or determinism, and any prompt of the right
length would do. Three entries are different, because instruction following is
their expected signal: **RH.9d** (steering, judged against a baseline reply),
**RH.11** (`--chat`, "coherent and responsive to the conversation history"),
and the agent loop. Those cannot pass against a base checkpoint no matter how
correct the backend, and closing them needs the `-it` variant, which is not on
this workstation. Each carries a note to that effect.

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

Implementation: `rust-harness/runtime/src/bin/checkpoint-info.rs`,
`rust-harness/runtime/src/checkpoint.rs`,
`rust-harness/runtime/src/model/cpu.rs` (the capability guard).

**Risks likely to surface here.** This is the highest-risk entry of the
campaign: the CPU backend was written against a Gemma-4-style architecture
spec, never against the real artifact. Published sources say the real E4B
architecture is a superset of what the backend implements, retaining
per-layer embeddings and adding a shared KV cache and dual RoPE; the
[architecture reconciliation](../rust-harness/runtime-design.md#architecture-reconciliation)
records those findings and their confidence. Those sources are secondhand,
so this entry decides the question for the checkpoint that actually exists.

The backend now implements those features rather than refusing them, and
`check_supported_checkpoint` retains each refusal as a coherence check: a file whose
metadata and tensors describe different models still fails the load rather
than loading with the extra weights ignored. So this entry produces a
verdict rather than a judgment call. Four questions decide the follow-up
work:

1. Are there per-layer embedding tensors?
2. Do any layers lack `attn_k`/`attn_v` (the shared-KV signature)?
3. Are the feed-forward widths uniform across layers, or per-layer?
4. Is RoPE a single frequency base at full head dimension, or dual and
   partial?

Command:

```bash
$ ls -la /mnt/ai-data/models/hf/
$ cd rust-harness
$ cargo run -p runtime --bin checkpoint-info -- --text-only /mnt/ai-data/models/hf/<e4b>
```

`--text-only` is not optional against the published E4B checkpoint, and every
later entry that loads it needs the flag for the same reason. The checkpoint
is multimodal; the CPU backend serves text only, and without the flag
`check_supported_checkpoint` refuses the load outright rather than silently
ignoring the vision tower. This was learned the hard way on 2026-08-12: the
commands here were written in the GGUF era, when conversion had already made
the served file text-only by construction, so the flag never appeared in them
and both RH.1 and RH.2 failed on their first real run.

Expected signal: the run ends in a compatibility verdict, printed by the same
`cpu::check_supported_checkpoint` function the loader calls, so the tool and the loader
cannot disagree. `[OK] CPU backend compatibility: supported` means the
checkpoint is within the implemented subset and RH.2 can proceed.
`[ERROR] CPU backend compatibility: ...` names the unimplemented feature and
is the expected outcome if the published architecture holds; that message
defines the follow-up code work and does not invalidate this entry.

The architecture summary section answers the four questions directly: tensor
families with counts, the dtype histogram against the implemented set
{F32, F16, BF16}, feed-forward widths (uniform or per-layer), the shared-KV
boundary, the attention schedule, and the per-layer geometry (head width,
feed-forward width, RoPE base). Record the full dump verbatim, including the
`=== config.json ===` echo and the `=== Tensors ===` table, because the
summary is derived and those two sections are the evidence.

Observed (2026-08-12, `/mnt/ai-data/models/hf/gemma-4-E4B`, 2624 lines; the
summary tail, which is what answers the four questions — the full dump was
redirected to a file rather than pasted):

```
Dtypes (this backend loads F32, F16 and BF16):
  BF16  x2130
Feed-forward widths: uniform 10240 across 42 layers
Shared KV cache: num_kv_shared_layers is 18 of 42, so layers 24-41 read a donor's
Attention schedule: sssssGsssssGsssssGsssssGsssssGsssssGsssssG
  (G = full attention, s = sliding window)

Per-layer geometry (head width, feed-forward width, RoPE base):
  layer 0: SlidingWindow(512), head 256, ffn 10240, rope base 10000
  ...
  layer 5: Global, head 512, ffn 10240, rope base 1000000
  ...
  layer 41: Global, head 512, ffn 10240, rope base 1000000

[OK] CPU backend compatibility: supported
```

Status: VERIFIED 2026-08-12. A first attempt ran without `--text-only` and
ended in `[ERROR] CPU backend compatibility: checkpoint is multimodal`, which
is the guard working rather than a finding.

The four questions, answered against the artifact rather than the published
description:

2. **Shared KV: yes.** `num_kv_shared_layers` is 18 of 42, so layers 24-41
   read a donor's cache and only 0-23 own one.
3. **Feed-forward widths are uniform**, 10240 across all 42 layers — not
   per-layer, as the MatFormer framing elsewhere in this file implies.
4. **RoPE is dual and partial.** Sliding layers run head 256 at base 10000;
   global layers run head 512 at base 1000000. The two differ in both head
   width and frequency base, so a single-base implementation at full head
   dimension would be wrong on every layer.

Question 1 (per-layer embedding tensors) is answered in the tensor-family
section of the full dump, not in this tail, and is not recorded here.

Everything is BF16 — 2130 tensors, no exceptions — which is why the loader's
storage-preserving path is the only one this checkpoint exercises.

**This dump also confirms the KV arithmetic in RH.2 empirically.** The
schedule puts a global layer at 5, 11, 17, 23, 29, 35 and 41: seven in the
model, but only the first four sit below the shared-KV boundary at 24 and
therefore own a cache. Four global owners at head 512 and twenty sliding
owners at head 256 is exactly the split RH.2's 4.05 GiB reconciles to, and it
is now read off the artifact rather than inferred from the total.

### RH.1a Shared-KV weight conflict

Implementation: `rust-harness/runtime/src/model/cpu.rs` (the loader's
`owns_cache` branch and the tensors it declines to read).

Decides the one open question the CPU backend already had to answer in order
to run. `modeling_gemma4.py` builds `k_proj`, `v_proj`, and `k_norm` only for
non-shared layers; every released E-series checkpoint carries all three on
every layer. The backend follows the modeling code and leaves the shared
tail's copies unread. If that reading is wrong, the shared tail attends over
the wrong keys and the output is fluent and wrong rather than an error — 18
of E4B's 42 layers, and 20 of E2B's 35, which is the variant check 1 below
was run against.
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
$ cd ~/workspace/ai-experiments
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
through `transformers` and through the harness, and compare. Confirm the
checkpoint directory name against the listing before running; both sides read
the same directory, so there is no conversion step to account for.

```bash
$ ls /mnt/ai-data/models/hf/
$ cd ~/workspace/ai-experiments/rust-harness
$ cargo run --release -p harness -- \
    --model /mnt/ai-data/models/hf/gemma-4-E2B \
    --max-tokens 16 --temperature 0 'The capital of France is'
```

Then the reference, on the same prompt and the same decoding rule:

```bash
$ cd ~/workspace/ai-experiments
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
distributions rather than a logit comparison. It was gated on a GGUF
conversion being available; now that both sides read the same checkpoint
directory, nothing blocks it and it can simply be run. It is confirmatory,
not blocking.

Status: RESOLVED 2026-08-09 in favor of the modeling code, on check 1.
Check 2 outstanding as confirmation.

## RH.2 Real-model load and generation smoke

Implementation: `rust-harness/harness/src/main.rs`,
`rust-harness/harness/src/agent.rs`,
`rust-harness/runtime/src/model/cpu.rs`,
`rust-harness/runtime/src/tokenizer.rs`.

First run of the tokenizer against the real vocabulary, and of the widening
paths against real tensors: the loader keeps each weight at its published
storage width and widens BF16 to f32 at use, so this is where that path meets
something other than a fixture.

Command:

```bash
$ cd rust-harness
$ mkdir -p /tmp/rh2 && cd /tmp/rh2
$ time ~/workspace/ai-experiments/rust-harness/target/release/harness \
    --model /mnt/ai-data/models/hf/<e4b> --text-only \
    "The capital of France is"
```

(Build once with `cargo build --release -p harness`.)

**The prompt is a continuation, not an instruction, and that is deliberate.**
E4B at the pinned revision is a base model. `fixtures/reference/prompts.txt`
records the measurement: a 16-token greedy probe on this exact prompt returned
" a city that is full of history and culture", which is a healthy base model.
Asked a question, it does not answer. An instruction-shaped prompt here
produces a fluent non-answer that reads as a defect and is not one.

No `--tokenizer` is passed, and none should be needed: `resolve_tokenizer`
looks for `tokenizer.json` inside the checkpoint directory first, which is
how a published checkpoint carries it. If this step fails with "no
tokenizer.json for model", the directory is incomplete — record that rather
than working around it with an explicit path.

Expected signal: exit 0; the model loads without panic, the continuation is
fluent English that follows plausibly from the prompt, and `session.jsonl`
appears in the working directory. Fluency is the signal, and instruction
following is explicitly not: a numerically wrong forward pass on real weights
degrades into word salad or a repetition loop, and neither survives a few
dozen tokens. Record the full reply.

What this entry cannot decide is whether the continuation is the *right* one.
A backend wrong in a small, systematic way produces fluent prose that diverges
from the reference, and no amount of reading it will show that. RH.1a check 2
and the ladder's L4 rung are what compare against `transformers`; this entry
establishes only that the model runs and speaks.

The startup banner is also the KV evidence: it prints the weight width, peak
RSS after load, the KV cache line, and the rewind horizon. Record those four
lines — the KV line is what the storage cap was for, and this is the only
step that reports it against real weights.

Observed (2026-08-12, with `--text-only`; the banner only, the reply is quoted
below it):

```
[OK] CPU backend compatibility: supported
[INFO] Weights: 13.90 GiB resident at bf16
[INFO] Peak RSS after load: 14.03 GiB
[INFO] KV cache: 131072 tokens, 4.05 GiB f32 resident (sliding layers capped at the window plus a batch of 64)
[INFO] Rewind horizon: 208 tokens behind the tip once the window fills (compaction past it re-prefills)
[WARN] KV cache above 2 GiB: consider --context to cap it (compaction keeps long sessions inside the cap)
[INFO] Session snapshot: 272 cached tokens at /tmp/rh2/session.kv
~/workspace/ai-experiments/rust-harness/target/release/harness --text-only     295.39s user 15.77s system 581% cpu 53.502 total
```

The reply:

```
Reply with the single word: edit
<h1>Can a neural network model generate a program in my target language?</h1>
I can't tell if I'm asking a reasonable question, but I'm interested in exploring...
```

Status: PARTIAL 2026-08-12, and the entry itself was the defect. The run was
made with the old instruction-shaped prompt, `Reply with the single word:
ready`. The load half passed and is recorded above. The reply was fluent,
grammatical and entirely non-responsive: the model echoed the prompt with
`edit` substituted for `ready`, then continued into an invented web-forum
thread about whether neural networks can generate code.

**That is correct behaviour, and the expected signal was wrong.** E4B at the
pinned revision is a base model, which `fixtures/reference/prompts.txt` states
and measures: "asked 'What is the capital of France?' it does not answer, and
a reader who expects one will read a fluent non-answer as a defect." That is
exactly what happened here. A base model given an instruction continues the
document, because continuing documents is the only thing it was trained to do.
The prompt in this entry has been changed to a continuation, and the entry
needs one more run to close.

Two hypotheses were raised and both are refuted; they are recorded because
each cost a host round trip and neither should be re-derived.

*Not a missing chat template.* One is rendered and pinned:
`harness/src/template.rs:31-52` emits `<bos>` then
`<|turn>{role}\n{content}<turn|>`, and `harness/tests/template_contract.rs`
holds those literals against `schemas/chat_template.json`.

*Not a delimiter the vocabulary lacks.* The published E4B `tokenizer.json`
carries `<|turn>` and `<turn|>` in its added-tokens table and does **not**
carry `<start_of_turn>` or `<end_of_turn>` — the reverse of what that theory
needed. `template::MARKERS` names the literals, `generator::PROMPT_SPECIALS`
admits them, and `generator.rs:74` encodes every prompt through
`encode_prompt` with that allowlist, so the delimiters reach the model as
their own ids.

The mechanism behind the second hypothesis is real even though it is not the
cause here, and it is unguarded. `runtime/src/tokenizer.rs:304-321` says so
itself: "A literal that is not in the vocabulary matches nothing: the caller
may name a delimiter this checkpoint does not carry, and encoding it as text
is the same treatment any other unknown string gets." A renderer and a
checkpoint that disagree on the spelling would therefore produce fluent
garbage and no error — the same symptom, from a genuine bug, with nothing to
distinguish it from this one. That is worth a guard.

The four banner lines reconcile exactly with the implemented KV rule, which is
worth stating because it is the R3 acceptance. Layers 0-23 own a cache (24-41
read a donor): 4 global and 20 sliding, at `n_kv_heads` 2. A global layer
costs `2 * 2 * 512 * 4` = 8 KiB per token, a sliding one 4 KiB, and
`window_capacity(512, 64)` is 719 rows. Globals hold the full sequence at
`4 * 131072 * 8192` = 4.00 GiB; the capped sliding layers hold
`20 * 719 * 4096` = 0.055 GiB, against the 10.0 GiB they would cost uncapped.
The total is 4.05 GiB, which is what printed. Peak RSS of 14.03 GiB against
13.90 GiB of weights bounds load overhead at 0.13 GiB, which is the P10.7
measurement. Both figures are read off this banner, not measured separately.

## RH.3 CPU generation speed

Implementation: `rust-harness/runtime/src/model/cpu.rs`,
`rust-harness/runtime/src/tensor.rs`,
`rust-harness/runtime/src/sampler.rs`.

No pass threshold. The measured rate decides whether real-model activation
capture and rr recording are practical on the CPU backend, and it is the
baseline the cuda backend is later compared against. Record the thread width
alongside the rate: matvec and the LM head split their output rows across
cores, so a rate without a width is not comparable to another rate. Pass
`--single-thread` for the serial figure.

Command:

```bash
$ mkdir -p /tmp/rh3 && cd /tmp/rh3 && rm -f session.jsonl
$ time ~/workspace/ai-experiments/rust-harness/target/release/harness \
    --model /mnt/ai-data/models/hf/<e4b> --text-only \
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
$ time ~/workspace/ai-experiments/rust-harness/target/release/harness \
    --model /mnt/ai-data/models/hf/<e4b> --text-only --trace-chrome turn1.json \
    "Summarize the Unix philosophy in three sentences"
$ time ~/workspace/ai-experiments/rust-harness/target/release/harness \
    --model /mnt/ai-data/models/hf/<e4b> --text-only --trace-chrome turn2.json \
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
$ ~/workspace/ai-experiments/rust-harness/target/release/harness \
    --model /mnt/ai-data/models/hf/<e4b> --text-only --record flight.jsonl \
    --temperature 0.8 --top-k 40 --seed 1 \
    "Reply with the single word: ready"
```

Sampling is set explicitly rather than left at the defaults, because the
selection half of this entry is only exercised on the stochastic path: at
temperature zero the sampler draws nothing, and a recording of a greedy
run cannot distinguish a replay that reproduced the selection from one
that recomputed it. What the prompt elicits does not matter here — E4B at
the pinned revision is a base model and will not follow it (see the
base-model note above) — only that tokens are generated at all.

Command (sandbox, after copying `flight.jsonl` into the worktree; there
is no committed CLI for arbitrary recordings, so a throwaway test invokes
the replay loader `runtime::model::record::replay_file` directly):

```bash
$ cd rust-harness
$ mkdir -p runtime/tests/data && cp <flight.jsonl> runtime/tests/data/rh5-flight.jsonl
$ cat > runtime/tests/rh5_replay.rs <<'EOF'
use std::path::Path;

#[test]
fn rh5_recording_replays_as_a_mock_script_and_states_its_selections() {
    let (_script, recorded) = runtime::model::record::replay_file_with_selections(Path::new(
        "tests/data/rh5-flight.jsonl",
    ))
    .unwrap();
    let sampling = recorded.sampling.expect("recording states its parameters");
    assert_eq!(sampling.top_k, Some(40), "{sampling:?}");
    assert_eq!(sampling.seed, 1, "{sampling:?}");
    assert!(!recorded.selections.is_empty(), "no token was recorded");
    // Draw counts rise by one per stochastic step, so the run's position in
    // its own stream is recoverable rather than merely stated.
    let draws: Vec<u64> = recorded.selections.iter().map(|s| s.draws).collect();
    assert!(draws.windows(2).all(|w| w[1] > w[0]), "{draws:?}");
}
EOF
$ cargo test -p runtime --test rh5_replay
$ git checkout -- . && rm -rf runtime/tests/data runtime/tests/rh5_replay.rs
```

Expected signal: the host run writes a header naming the sampling
parameters, one JSONL record per forward call with token and logit
payloads, and one selection line per sampled token. The throwaway test
passes — the recording rebuilds as a MockModel script, states the
parameters it was made under, and carries a strictly rising draw count.

The draw count is the part worth watching. A recording that reproduces
the logits and not the selection lets a replay redraw from the same
distribution and reach a different transcript, with nothing in either
file to say the two disagreed; the count is what makes the run's position
in its own stream recoverable.

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
$ ~/workspace/ai-experiments/rust-harness/target/release/harness \
    --model /mnt/ai-data/models/hf/<e4b> --text-only --trace-chrome trace.json \
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
$ make debug-record ARGS='--model fixtures/tiny-gemma4 "Reply with the single word: ready"'
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
$ ~/workspace/ai-experiments/rust-harness/target/release/harness \
    --model /mnt/ai-data/models/hf/<e4b> --text-only \
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
- The rust `tokenizer_fingerprint` is still intentionally not comparable to
  Python-capture fingerprints, but no longer because the vocabulary came from
  a different container. Both sides now read the same checkpoint. It is a
  sha256 over the model path and vocabulary size with no chat template
  (`activation-capture.rs:219-225`), so it identifies a rust capture rather
  than describing a tokenizer, and probe training on rust captures must still
  go through artifacts produced by this chain only.

### RH.9a Activation capture

Command:

```bash
$ cd rust-harness
$ cargo run --release -p runtime --bin activation-capture -- \
    --model /mnt/ai-data/models/hf/<e4b> --text-only \
    --dataset <labeled.jsonl> --layers <l0,l1> \
    --out /mnt/ai-data/interp/captures/rust-e4b-verify
$ uv run python -c "from interp.probes.capture import load_capture; \
    acts, labels, texts = load_capture('/mnt/ai-data/interp/captures/rust-e4b-verify'); \
    print(sorted(acts), {k: v.shape for k, v in acts.items()}, len(labels), len(texts))"
```

Expected signal: the capture directory holds `shard_00000.npz` onward,
`examples.jsonl`, and a `manifest.json` with
`capture_backend: "rust-harness-cpu"`, `model_revision: "safetensors"`, and a
`model_hash` that is the sha256 of `model.safetensors`;
`load_capture` reports the requested layers with row counts matching the
dataset. Record the capture wall-clock time.

Observed:

```
```

Status: PENDING

### RH.9b Probe training on rust captures

Command:

```bash
$ cd ~/workspace/ai-experiments
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
$ uv run python -c "from interp.probes.export import write_probe; \
    write_probe('/mnt/ai-data/interp/probes/rust-e4b-verify-probe', '/tmp/rh9/probe.safetensors')"
$ cd /tmp/rh9/base && ~/workspace/ai-experiments/rust-harness/target/release/harness \
    --model /mnt/ai-data/models/hf/<e4b> --text-only "Reply with the single word: ready"
$ cd /tmp/rh9/mon && ~/workspace/ai-experiments/rust-harness/target/release/harness \
    --model /mnt/ai-data/models/hf/<e4b> --text-only --probe /tmp/rh9/probe.safetensors \
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
$ cd /tmp/rh9/steer1 && ~/workspace/ai-experiments/rust-harness/target/release/harness \
    --model /mnt/ai-data/models/hf/<e4b> --text-only \
    --control-vector <vector-dir>/vector.safetensors --control-vector-scale 1.0 \
    "Describe your day in two sentences"
$ cd /tmp/rh9/steer2 && ~/workspace/ai-experiments/rust-harness/target/release/harness \
    --model /mnt/ai-data/models/hf/<e4b> --text-only \
    --control-vector <vector-dir>/vector.safetensors --control-vector-scale 1.0 \
    "Describe your day in two sentences"
```

**Partly blocked on an instruction-tuned checkpoint.** "In the direction of
the vector's attribute" is a judgment about a reply to an instruction, and
base E4B does not reply to instructions; see the base-model note at the top of
this file. The determinism half stands on its own and is the part to close
here.

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
    --model /mnt/ai-data/models/hf/<e4b> --text-only "Reply with the single word: ready"
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
$ ~/workspace/ai-experiments/rust-harness/target/release/harness \
    --model /mnt/ai-data/models/hf/<e4b> --text-only --chat \
    --trace-chrome trace.json --max-tokens 128
```

**Blocked on an instruction-tuned checkpoint.** The responsiveness half of
this signal is unmeetable against base E4B; see the base-model note at the top
of this file. The mechanical half — the session record and the sync spans — is
still worth running and recording, and a base model produces turns of text
that exercise both.

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
$ cd ~/workspace/ai-experiments
$ make harness-build
$ make harness-run ARGS='--model /app/models/hf/<e4b> --text-only --seed 7 "Reply with the single word: ready"'
$ mkdir -p /tmp/rh12 && cd /tmp/rh12
$ ~/workspace/ai-experiments/rust-harness/target/release/harness \
    --model /mnt/ai-data/models/hf/<e4b> --text-only --seed 7 \
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
$ ~/workspace/ai-experiments/rust-harness/target/release/harness \
    --model /mnt/ai-data/models/hf/<e4b> --text-only \
    "Reply with the single word: ready"
$ ~/workspace/ai-experiments/rust-harness/target/release/harness \
    --model /nonexistent-checkpoint "task" ; echo "exit: $?"
$ tail -2 /mnt/ai-data/logs/harness/runs.jsonl
$ cd ~/workspace/ai-experiments && uv run python -c "
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

## RH.14 The endpoint answers the interp client

Implementation: `rust-harness/harness/src/server.rs`,
`rust-harness/harness/src/main.rs` (`--serve`), `common/serving.py`.

The endpoint is verified in the sandbox against a scripted model, which
proves the wire contract and nothing about the weights. This entry is the
real-model half: the harness serving E4B, driven by the same
`common/serving.py` functions the interp tooling already calls, with no
client change. It also measures the decode rate those callers need in order
to size their `timeout`, because `chat_completion` defaults to 120 s and a
CPU reply can exceed it.

Command (server in one shell, client in another):

```bash
$ cd rust-harness && cargo build --release --bin harness
$ ./target/release/harness --model <e4b-checkpoint-dir> --text-only --serve 8005
```

```bash
$ .venv/bin/python -c "
from common import serving
s = serving.SERVICES['harness']
print('healthy in', serving.wait_healthy(s), 's')
r = serving.chat_completion(s, [{'role':'user','content':'Name three primes.'}],
                            {'temperature': 0.0, 'max_tokens': 64}, timeout=1800)
print(r)"
```

Expected signal: `wait_healthy` returns; `chat_completion` returns a dict
whose `content` is coherent text and whose `usage` counts are internally
consistent; two identical greedy requests return identical text.

Observed:

```
[INFO] Weights: 13.90 GiB resident at bf16
[INFO] Peak RSS after load: 14.04 GiB
[INFO] KV cache: 131072 tokens, 4.05 GiB f32 resident (sliding layers capped at the window plus a batch of 64)
[INFO] Rewind horizon: 208 tokens behind the tip once the window fills (compaction past it re-prefills)
[WARN] KV cache above 2 GiB: consider --context to cap it (compaction keeps long sessions inside the cap)
[INFO] Session restored: 111 cached tokens from <worktree>/rust-harness/session.kv
[INFO] Serving on http://127.0.0.1:8005 (model <e4b-checkpoint-dir>)

healthy in 0.01009527400310617 s
{'content': 'Sure! Here are three prime numbers:\n\n1. 2\n2. 3\n3. 5\n\nA prime
number is a natural number greater than 1 that has no positive divisors other
than 1 and itself. In other words, it can only be divided evenly by 1 and the
number itself', 'usage': {'prompt_tokens': 13, 'completion_tokens': 64,
'total_tokens': 77}, 'latency_ms': 11659.8}

healthy in 0.010442779006552882 s
{'content': 'Sure! Here are three prime numbers:\n\n1. 2\n2. 3\n3. 5\n\nA prime
number is a natural number greater than 1 that has no positive divisors other
than 1 and itself. In other words, it can only be divided evenly by 1 and the
number itself', 'usage': {'prompt_tokens': 13, 'completion_tokens': 64,
'total_tokens': 77}, 'latency_ms': 10969.0}
```

Status: PASS

The interp client works unmodified: `wait_healthy` and `chat_completion` both
succeed against the harness with nothing added but a registry entry. `usage`
is internally consistent (13 + 64 = 77), and `completion_tokens` equals the
requested budget, so the reply was cut by the budget rather than by an EOS.
The two greedy requests returned byte-identical text, which is both the
determinism the endpoint promises and evidence the second request was not
contaminated by the first: each renders its whole message list, and `sync`
truncates to the common prefix.

The output is coherent, on topic and correct. That carries more weight here
than the repository's usual caution about fluent text, because the defect
this codebase actually hit — an attention scaling that should not have been
applied, and a missing value-state norm — produced noise rather than prose.
It is evidence the real-weight forward pass is sound. It is still not a
correctness claim: the ladder against `transformers` is what would make one,
and no rung has run at Tier E.

**Decode rate: 64 tokens in 10.97-11.66 s, so 5.5-5.8 tokens/second** at the
default thread width, against a 13-token prompt — essentially pure decode,
with prefill too small to separate out. Consequences for callers: a 256-token
scorecard call costs roughly 45 s, and `chat_completion`'s 120 s default
covers about 660 tokens. Interp callers should pass an explicit `timeout`.

**`start_budget_s` remains unmeasured.** Both health polls returned in 10 ms
against an already-running server, so they timed the poll and not the load.
The 600 s in the registry is still a placeholder; timing a cold start until
the serving line appears is what would replace it.

Two findings for follow-up, neither a defect in the endpoint itself:

- `Session restored: 111 cached tokens` — the server inherited an earlier
  interactive run's KV cache, because it was started from the worktree where
  that run left `session.kv`. Harmless, and the identical replies above show
  it: `sync` reconciles against the common prefix. But a server restoring a
  conversation it will never continue is confusing, and `--serve` returns
  before the save path, so it restores and never writes. Serve mode should
  probably not participate in session persistence at all.
- KV at 4.05 GiB, with its own `[WARN]`: serve mode takes the full trained
  context by default and holds it for the process lifetime. `--context` is
  the lever, and a serving deployment likely wants it set.

### RH.15 Steered scorecard through the harness endpoint

**Status: not yet run.** R10 moved the control vector to safetensors and
retired llama.cpp steering, so the scorecard's subject is now `harness-serve`
rather than `llama-gpu`. Everything below is verified in the sandbox against a
synthetic model, which proves the format and the wiring and nothing about
steering a real checkpoint.

Requires a control vector extracted against the E4B weights being served; a
vector whose `manifest.json` records a different `model_hash` is refused, and
one exported before R10 is a GGUF that no longer loads. Re-export it with
`make steer-extract`.

```bash
$ make harness-build
$ make steer-up VECTOR=/mnt/ai-data/interp/vectors/<name> SCALE=1.0
$ make steer-scorecard VECTOR=/mnt/ai-data/interp/vectors/<name> \
    ATTRIBUTE="<behavior the judge scores>"
$ make steer-down
```

Expected signal: the three arms complete and the scorecard writes its report,
with the steered arm differing from the unsteered baseline. The scorecard
restores the unsteered state on failure as well as on success, so a partial
run should leave `harness-serve` unsteered.

Record: the wall-clock for the full three-arm sweep, and the per-call latency.
This is the number that decides whether a GPU backend becomes urgent. RH.14
measured 5.5-5.8 tokens per second at the default thread width, which puts a
256-token arm at roughly 45 seconds; a sweep is many such calls. If it proves
impractical, that is a finding to record against a future GPU backend, not a
reason to revert R10.

Also record whether the judge service and the subject contend: the judge runs
on `llama-cpu`, and both are now CPU consumers on the same machine.
