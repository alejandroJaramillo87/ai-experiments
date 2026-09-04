# Rust harness verification (CPU backend)

**Operational runbook. Executed against the workstation, not read for understanding: the operator's pasted output is the evidence. Keep the commands runnable, and the paths accurate — both documentation gates check this directory, so a stale reference here fails the gate.**

**Entries: 16 verified, 5 blocked, 16 pending, of 37.** The five blocked are
RH.7 (rr does not recognize Zen 5) and RH.8, RH.9d, RH.11 and RH.15, which
need an instruction-tuned checkpoint or a control vector this repository does
not yet have. Each says so in its own Status line; none is merely unrun.

Of the 16 pending, one is RH.13, reopened by U7: the code under it changed the
terminal output the entry records, so its Observed block needs re-collecting
rather than editing. RH.17 through RH.22 are new full entries from the surface
inventory, each needing a human, the GPU host, or the real E4B checkpoint.
RH.23 through RH.31 are **gate-backed**: each names the test function that
asserts its expected signal, and one `cargo test --workspace` run closes all
nine. Everything the inventory found that is neither is listed in the coverage
roster at the end of this file, against the test that pins it.

Workstation verification for the `rust-harness/` workspace: the CPU
inference backend on the real Gemma 4 E4B checkpoint, the observability stack
(flight recorder, tracing, rr), the in-band interpretability chain
(capture, probe monitor, steering), the chat mode, the containerized
build, and the run-level observability records. Every rust-harness claim to date is
golden-tested in the sandbox against a synthetic 16-dimension fixture; this
campaign is the first contact with real hardware and real weights.

Scope is CPU only. Not covered, and why:

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

Recording protocol: the [runbook README](../../README.md) rules apply. The human
runs each command on the workstation and pastes command plus output
verbatim into the Observed block; expected signals are never edited to
match observations; one GPU consumer at a time (the CPU backend uses no
GPU, but RH.9 shares `/mnt/ai-data` and the interp Python environment with
GPU workflows). The Python session-start gate (`pytest -m hw_smoke`) does
not apply here; RH.0 is this phase's gate. The model directory
`/mnt/ai-data/models` is read-only for agents; all commands below only
read from it.

The sandbox-executable code-quality counterpart to this runbook is the
[QA campaign](../rust-harness/rust-harness-qa.md) (phases Q0-Q6); it verifies code
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
cd rust-harness
rustc --version && cargo --version
python3 -c "import numpy; print(numpy.__version__)"
make quality
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
test result: ok. 6 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s

     Running tests/tracing_spans.rs (target/debug/deps/tracing_spans-69a191609cff6617)

running 2 tests
test generation_emits_the_documented_span_schema ... ok
test no_subscriber_means_no_capture_and_no_failure ... ok

test result: ok. 2 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s

   Doc-tests harness

running 0 tests

test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s

   Doc-tests runtime

running 0 tests

test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s

   Compiling runtime v0.1.0 (/home/alejandro/workspace/ai-experiments/rust-harness/runtime)
   Compiling harness v0.1.0 (/home/alejandro/workspace/ai-experiments/rust-harness/harness)
    Finished `release` profile [optimized] target(s) in 3.28s
```

The gate passed and `cargo build --release -p harness` produced the binary every
later entry runs. **Only the tail of the output was pasted**: the `rustc --version`,
`cargo --version` and numpy lines above it were run but not captured, so the
toolchain versions this campaign used are not recorded. Nothing downstream depends
on them, but a future run should paste from the first command.

Status: VERIFIED 2026-08-14

## RH.1 Model location and architecture reconciliation

Implementation: `rust-harness/runtime/src/bin/checkpoint-info.rs`,
`rust-harness/runtime/src/checkpoint.rs`,
`rust-harness/runtime/src/model/cpu.rs` (the capability guard).

**Risks likely to surface here.** This is the highest-risk entry of the
campaign: the CPU backend was written against a Gemma-4-style architecture
spec, never against the real artifact. Published sources say the real E4B
architecture is a superset of what the backend implements, retaining
per-layer embeddings and adding a shared KV cache and dual RoPE; the
[architecture reconciliation](../../rust-harness/runtime-design.md#architecture-reconciliation)
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
ls -la /mnt/ai-data/models/hf/
cd rust-harness
cargo run -p runtime --bin checkpoint-info -- --text-only /mnt/ai-data/models/hf/<e4b> \
    2>&1 | tee /tmp/rh1-dump.txt
```

`2>&1` is not optional either, and for a reason worth stating rather than
copying. The dump is the tool's answer and goes to stdout; the compatibility
verdict is a status line and goes to stderr. This entry asks for both — the
verdict is the expected signal, the full dump is the evidence behind it — so a
plain `| tee` would capture the 2624 lines of dump and lose the one line the
entry turns on.

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
[The shared-KV weight conflict](../../models/gemma-4.md#the-shared-kv-weight-conflict).

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

The statistics live in `interp/probes/weight_stats.py`, and the verdict is a
test rather than a script read by eye. `tests/interp/test_weight_stats.py`
proves the discriminator on synthetic draws in the sandbox; the test below
asserts it on the real checkpoints, both variants, and skips a variant that is
not on disk rather than passing over it.

Asserted by `tests/hardware/test_shared_kv_weights.py`:

| Expected signal | Test function |
|-----------------|---------------|
| In every shared layer, `k_proj` and `v_proj` sit above the Gaussian ratio with negative kurtosis while `q_proj` and `o_proj` have positive kurtosis | `test_the_shared_tail_carries_untrained_keys_and_values_beside_trained_queries` |
| The shared tail's `k_proj` ratios, and its `v_proj` ratios, spread by less than 0.01 | `test_the_shared_tail_keys_and_values_are_pinned_to_one_value` |
| Every owning layer has positive kurtosis on all four projections | `test_every_owning_layer_is_trained_on_all_four_projections` |
| The layers whose `k_proj` is wide are the layers `config.json` marks full attention | `test_key_width_agrees_with_the_attention_schedule` |

```bash
cd ~/workspace/ai-experiments
uv run pytest -m "hw_full and hw_e4b" tests/hardware/test_shared_kv_weights.py -v
```

The test reads tensor data one projection at a time and needs no GPU. It is
`hw_full` rather than `hw_smoke` because reading every attention projection of
E4B takes minutes. `GEMMA4_CHECKPOINT_ROOT` overrides the default checkpoint
root under `/mnt/ai-data`. The thresholds are the ones the 2026-08-09
measurement below was read against, not fitted to it: the Gaussian ratio and
zero kurtosis are analytic, and the 0.01 spread sits an order of magnitude
above what the untrained group showed and below what the trained groups did.

Expected signal: eight tests pass, four per variant, none skipped. A skipped
variant is a checkpoint missing from disk, not a pass. If the first test
fails, the shared tail's `k_proj` and `v_proj` moved with `q_proj` and
`o_proj`, the weights are meaningful, and check 2 becomes mandatory rather
than confirmatory.

**Check 2 — do the logits agree?** Generate greedily from the same prompt
through `transformers` and through the harness, and compare. Confirm the
checkpoint directory name against the listing before running; both sides read
the same directory, so there is no conversion step to account for.

**Run the harness from an empty directory.** It continues whatever
`session.jsonl` sits in its working directory, and `rust-harness/` has held a
stale one more than once (RH.10 restored 111 tokens from it). A run made there
on 2026-09-04 printed `Prefill: 977 tokens` for a six-token prompt, ended at
993 cached tokens, and emitted `<unused10>二品<unused10>二品` — the continuation
of a transcript full of an earlier run's garbage, not a verdict on the
weights. A prefill count far above the prompt length is the tell.

```bash
ls /mnt/ai-data/models/hf/
rm -f ~/workspace/ai-experiments/rust-harness/session.jsonl \
      ~/workspace/ai-experiments/rust-harness/session.kv
mkdir -p /tmp/rh1a && cd /tmp/rh1a
~/workspace/ai-experiments/rust-harness/target/release/harness --text-only \
    --model /mnt/ai-data/models/hf/gemma-4-E2B \
    --max-tokens 16 --temperature 0 'The capital of France is'
```

Then the reference, on the same prompt and the same decoding rule:

```bash
cd ~/workspace/ai-experiments
.venv/bin/python - <<'EOF'
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

Expected signal: both sides continue the prompt fluently and agree on the
first words. The harness renders the prompt through the chat template
(`<bos><|turn>user\n...<turn|>\n<|turn>model\n`) while `transformers` here
tokenizes the bare string, so the two token sequences differ before the first
forward pass and token-for-token agreement is not expected from this pair of
commands. The exact-match comparison is the ladder's L4 rung, which feeds both
sides one prompt and passed at Tier E on 2026-08-14 (RH.1b). Word salad or
repeated `<unused>` tokens from a clean directory means the backend's reading
is wrong — in which case flip the `owns_cache` branch in the loader so shared
layers compute their own K/V, rebuild, and rerun.

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

Observed (2026-09-04, check 1 as the test, both variants):

```
$ cd ~/workspace/ai-experiments
$ uv run pytest -m "hw_full and hw_e4b" tests/hardware/test_shared_kv_weights.py -v
platform linux -- Python 3.12.13, pytest-9.1.1, pluggy-1.6.0 -- /home/alejandro/workspace/ai-experiments/.venv/bin/python
collected 8 items

tests/hardware/test_shared_kv_weights.py::test_the_shared_tail_carries_untrained_keys_and_values_beside_trained_queries[gemma-4-E2B] PASSED
tests/hardware/test_shared_kv_weights.py::test_the_shared_tail_keys_and_values_are_pinned_to_one_value[gemma-4-E2B] PASSED
tests/hardware/test_shared_kv_weights.py::test_every_owning_layer_is_trained_on_all_four_projections[gemma-4-E2B] PASSED
tests/hardware/test_shared_kv_weights.py::test_key_width_agrees_with_the_attention_schedule[gemma-4-E2B] PASSED
tests/hardware/test_shared_kv_weights.py::test_the_shared_tail_carries_untrained_keys_and_values_beside_trained_queries[gemma-4-E4B] PASSED
tests/hardware/test_shared_kv_weights.py::test_the_shared_tail_keys_and_values_are_pinned_to_one_value[gemma-4-E4B] PASSED
tests/hardware/test_shared_kv_weights.py::test_every_owning_layer_is_trained_on_all_four_projections[gemma-4-E4B] PASSED
tests/hardware/test_shared_kv_weights.py::test_key_width_agrees_with_the_attention_schedule[gemma-4-E4B] PASSED

[TIER] full: torch present, the whole suite was in scope.
8 passed in 38.94s
```

Eight passed, none skipped, so both checkpoints were on disk and every
threshold the 2026-08-09 reading was made against holds on E4B as well as
E2B. The fourth test ran rather than skipping, which means both configs carry
`layer_types` and the schedule read off `k_proj` widths agrees with it. Both
variants together read in under 40 s, well inside the `hw_full` budget.

Observed (2026-09-04, check 2, gemma-4-E2B, from an empty directory):

```
$ mkdir -p /tmp/rh1a && cd /tmp/rh1a
$ ~/workspace/ai-experiments/rust-harness/target/release/harness --text-only \
    --model /mnt/ai-data/models/hf/gemma-4-E2B --max-tokens 16 --temperature 0 \
    'The capital of France is'
[INFO] Model: /mnt/ai-data/models/hf/gemma-4-E2B hash=dc91ecfbc2392fa9 layers=35
[INFO] Weights: 8.62 GiB resident at bf16
[INFO] Peak RSS after load: 9.51 GiB
[INFO] KV cache: 131072 tokens, 1.52 GiB f32 resident (sliding layers capped at the window plus a batch of 64)
[INFO] Rewind horizon: 208 tokens behind the tip once the window fills (compaction past it re-prefills)
[INFO] Fold: dot=widen4 batch=64 threads=8
[INFO] Sampling: seed=0 temperature=0 max_tokens=16
[INFO] Build: 0.1.0 git unknown
[INFO] Model load: 3.76 s
[INFO] Prefill: 14 tokens in 0.56 s (25 tok/s)
The capital of France is Paris.
The capital of France is Paris.
```

The `transformers` reference, same day, same prompt, greedy, 16 new tokens
(the `Loading weights` progress line omitted; no unexpected-key warning was
printed):

```
The capital of France is Paris.

The capital of France is Paris.

The capital of France is
```

Both sides answer `Paris.` and then repeat the prompt, which is the same
continuation from two independent implementations of the shared-KV reading.
The 14-token prefill is the six-token prompt plus the chat template, and it
is what a clean directory looks like; the 977-token prefill the first attempt
printed from `rust-harness/` was a stale session, recorded above as the trap.
The blank lines differ because the harness trims the reply and the reference
prints raw decoded text; the tokens that matter agree. `from_pretrained`
printed no warning about the shared layers' `k_proj` or `v_proj`, so the
current `transformers` build loads those tensors silently rather than
reporting them as unexpected; that silence is uninformative either way, as
the entry says, and check 1 is what decides.

Check 2 is confirmatory: an eye-level agreement on one prompt, not a logit
comparison. The exact-match form of it is the ladder's L4 rung against a
`transformers` greedy reference, which passed at Tier E on E4B (RH.1b).

Status: VERIFIED 2026-09-04 in favor of the modeling code, on check 1 as a
test over both variants and check 2 by eye on E2B (first decided 2026-08-09
on E2B by the pasted script).

## RH.1b Correctness ladder at Tier E

Implementation: `rust-harness/runtime/src/ladder.rs`,
`rust-harness/runtime/src/bin/ladder.rs`, the five generators under
`rust-harness/fixtures/`, and `make ladder`.

The numerics oracle. Six rungs compare the CPU backend against
`transformers` on the real checkpoint: token ids (L0), residual stream (L1),
final distribution (L2), perplexity (L3), greedy continuation (L4), and the
long-context anchors past the sliding window (L5, added by RH.17). Every
other entry in this runbook observes behavior; this one is the only entry
that can call the backend correct.

The generators are GPU consumers and `reference_perplexity.py` is the slow
one. The artifacts land in a git-ignored directory by design — tens of MB at
a 262144-wide vocabulary — so the printed table is the only evidence that
leaves the workstation. The five artifacts are prerequisites of `make ladder`
itself: one that is missing under the directory the `SIZE` reads,
`rust-harness/fixtures/reference/e4b/quick` here, is generated on the way up,
and one that is present is not paid for again. `make ladder-reference`
regenerates all five unconditionally, which is the command after a
checkpoint or `transformers` change, since a stale artifact under an existing
directory is not something make can see. `tests/hardware/test_ladder.py`
runs the same target under `make quality-full HW=full`, so a routine
full-tier run repeats this entry.

Command:

```bash
cd ~/workspace/ai-experiments/rust-harness
make ladder SIZE=quick MODEL=<the gemma-4-E4B checkpoint directory>
```

Expected signal: six rows, each `PASS`, and `0 skipped`. **A `skipped` rung is
missing evidence, not a pass** — read the count in the closing `[OK]` line
rather than the exit code. A skip after the generators ran means one wrote
somewhere other than where the climb read. The first run recorded below
predates L5 and shows five rows; L5's first pass is RH.17's Observed block.

Observed:

```
size=quick tokens=512 prompts=2
prompt=6 tokens
threads=8
dot=widen4
batch=64
model=/mnt/ai-data/models/hf/gemma-4-E4B
reference=fixtures/reference/e4b/quick

rung  check                   value         threshold       verdict
L0    token ids                                             PASS
      case agreement          1.000000      >= 1.000000     ok
      5 cases, every id identical
L1    residual stream                                       PASS
      worst cosine            1.000000      >= 0.999900     ok
      worst rel_l2            0.000005      <= 0.001000     ok
      42 layers within tolerance
L2    final distribution                                    PASS
      mean kl                 0.000000      <= 0.001000     ok
      worst kl                0.000000      <= 0.005000     ok
      worst top-50 overlap    1.000000      >= 0.980000     ok
      argmax agreement        1.000000      >= 0.980000     ok
      1 positions scored; worst at prompt position 5
L3    perplexity                                            PASS
      ppl delta               0.000032      <= 0.010000     ok
      worst token nll         0.001248      <= 0.050000     ok
      ppl 55.5420 against 55.5438 over 254 tokens; worst token at corpus position 414
L4    greedy continuation                                   PASS
      token agreement         1.000000      >= 0.980000     ok
      mean common prefix      1.000000      >= 0.750000     ok
      2 prompts, 0 ties

timing (not compared; expected to differ between runs)
  L0        0.48s
  L1        0.52s
  L2        0.66s
  L3       52.34s
  L4       22.89s
  sum      76.89s  (rung bodies only, excludes model load)

[OK] Ladder: every rung within tolerance (0 skipped)
[INFO] Peak RSS: 14.27 GiB
```

All five rungs passed with nothing skipped, which is the first time the CPU
backend has been checked against `transformers` on real weights at every
rung. **L0 reporting a result rather than skipping also closes the
outstanding tokenizer-parity acceptance** that had been carried since P9.13.

Two notes on what this does and does not settle. The four generators were
first run with a bare `python`, which resolved to an interpreter without
`torch` or `transformers` and produced four `[ERROR] Import` lines and no
artifacts; the run above is the retry under the workstation environment. Every
generator command in this file, including RH.17's and RH.21's, spells
`uv run python` for that reason. And the `ppl 55.5420` figure is the model's perplexity on the quick corpus, not a
quality claim — the comparison against `55.5438` is the result.

Observed (2026-09-04, the six-rung climb from the command block above,
artifacts regenerated the same day):

```
size=quick tokens=512 prompts=2
prompt=6 tokens
threads=8
dot=widen4
batch=64
model=/mnt/ai-data/models/hf/gemma-4-E4B
reference=fixtures/reference/e4b/quick

rung  check                   value         threshold       verdict
L0    token ids                                             PASS
      case agreement          1.000000      >= 1.000000     ok
      5 cases, every id identical
L1    residual stream                                       PASS
      worst cosine            1.000000      >= 0.999900     ok
      worst rel_l2            0.000005      <= 0.001000     ok
      42 layers within tolerance
L2    final distribution                                    PASS
      mean kl                 0.000000      <= 0.001000     ok
      worst kl                0.000000      <= 0.005000     ok
      worst top-50 overlap    1.000000      >= 0.980000     ok
      argmax agreement        1.000000      >= 0.980000     ok
      1 positions scored; worst at prompt position 5
L3    perplexity                                            PASS
      ppl delta               0.000032      <= 0.010000     ok
      worst token nll         0.001248      <= 0.050000     ok
      ppl 55.5420 against 55.5438 over 254 tokens; worst token at corpus position 414
L4    greedy continuation                                   PASS
      token agreement         1.000000      >= 0.980000     ok
      mean common prefix      1.000000      >= 0.750000     ok
      2 prompts, 0 ties
L5    logits past the window                                PASS
      scheduled positions agree  1.000000      >= 1.000000     ok
      worst kl inside window  0.000000      <= 0.005000     ok
      worst kl past window    0.000000      <= 0.005000     ok
      worst top-50 overlap    1.000000      >= 0.980000     ok
      argmax agreement        1.000000      >= 0.980000     ok
      6 positions across a 512-token window over 2052 tokens; worst at position 1025

timing (not compared; expected to differ between runs)
  L0        0.48s
  L1        0.70s
  L2        0.83s
  L3       52.70s
  L4       23.70s
  L5      268.92s
  sum     347.33s  (rung bodies only, excludes model load)
[OK] Ladder: every rung within tolerance (0 skipped)
[INFO] Peak RSS at exit: 16.13 GiB
```

Three weeks and several engine changes later, every figure the first run
printed reproduces to the last digit: the same worst cosine, the same
`rel_l2`, the same `ppl 55.5420 against 55.5438`, the same worst token at
corpus position 414. The generators' own output confirmed the reference side
is unchanged too (`ppl 55.5438`, the same 16-token continuation, the L5
schedule `[511, 512, 513, 1025, 1026, 2051]` of 2052). L5 is the one rung
the first run could not show, and it passed at every anchor. Peak RSS rose
from 14.27 to 16.13 GiB, which is the 2052-token L5 forward pass the first
run never made.

Status: VERIFIED 2026-09-04 on six rungs (first decided 2026-08-14 on five)

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
cd rust-harness
mkdir -p /tmp/rh2 && cd /tmp/rh2
time ~/workspace/ai-experiments/rust-harness/target/release/harness \
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

What this entry cannot decide by eye is whether the continuation is the
*right* one. A backend wrong in a small, systematic way produces fluent prose
that diverges from the reference, and no amount of reading it will show that.
RH.1a check 2 and the ladder's L4 rung compare against `transformers` on
prompts of their own; the check below asks the same question of the reply
this entry actually produced.

Check 2 — the sampled reply against `transformers`. A recording holds every
forward call's logits and every token the sampler chose, and
`rust-harness/fixtures/audit_recording.py` feeds the recorded token sequence
to `transformers` once, teacher-forced, and compares its logits with the
harness's at every recorded position under the ladder's L2 tolerances. It
also reports, for every token the sampler drew, that token's rank and
probability under `transformers` at its position. A token the reference gives
no weight to is a decoder fault; a token at rank 7 is a draw at temperature
0.8. Sampling is left at the defaults so the reply is the one check 1
produces.

```bash
mkdir -p /tmp/rh2 && cd /tmp/rh2 && rm -f session.jsonl session.kv
~/workspace/ai-experiments/rust-harness/target/release/harness \
    --model /mnt/ai-data/models/hf/gemma-4-E4B --text-only --record flight.jsonl \
    "The capital of France is"
cd ~/workspace/ai-experiments
uv run python rust-harness/fixtures/audit_recording.py \
    --model /mnt/ai-data/models/hf/gemma-4-E4B \
    --record /tmp/rh2/flight.jsonl --out /tmp/rh2/audit.json --show 5
```

Expected signal: `[OK] Distributions:` with every measure inside its
threshold over every recorded position, then `[OK] Chosen tokens:` naming the
worst-ranked draw and the last token with its rank and probability. The five
lowest-ranked draws print beneath, decoded. Exit 0. `[ERROR] Distributions:`
names the worst position and exits 1; that is a numerics finding and belongs
in RH.1b's territory, not a sampling one.

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

First attempt, 2026-08-12 — partial, and the entry itself was the defect. The run was
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
`harness/src/template.rs` emits `<bos>` then
`<|turn>{role}\n{content}<turn|>` (the markers are the `TURN_OPEN`,
`TURN_CLOSE` and `MODEL_TURN_OPEN` constants at lines 24-27), and
`harness/tests/template_contract.rs` holds those literals against
`schemas/chat_template.json`.

*Not a delimiter the vocabulary lacks.* The published E4B `tokenizer.json`
carries `<|turn>` and `<turn|>` in its added-tokens table and does **not**
carry `<start_of_turn>` or `<end_of_turn>` — the reverse of what that theory
needed. `template::MARKERS` names the literals, `generator::PROMPT_SPECIALS`
(`generator.rs:236`) admits them, and every prompt goes through
`encode_prompt` with that allowlist (`generator.rs:164` and `:244`), so the
delimiters reach the model as their own ids.

The mechanism behind the second hypothesis is real even though it is not the
cause here, and it is unguarded. The `special_at` doc comment in
`runtime/src/tokenizer.rs` (lines 342-345) says so itself: a literal that is
not in the vocabulary matches nothing; the caller may name a delimiter this
checkpoint does not carry, and encoding it as text is the same treatment any
other unknown string gets. A renderer and a
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

Observed (2026-08-14, the rerun with the continuation prompt this entry now
specifies):

```
[INFO] Weights: 13.90 GiB resident at bf16
[INFO] Peak RSS after load: 14.03 GiB
[INFO] KV cache: 131072 tokens, 4.05 GiB f32 resident (sliding layers capped at the window plus a batch of 64)
[INFO] Rewind horizon: 208 tokens behind the tip once the window fills (compaction past it re-prefills)
[WARN] KV cache above 2 GiB: consider --context to cap it (compaction keeps long sessions inside the cap)
The capital of France is Paris
< div > The capital of France is < / div >

<h2>Input and Output</h2>

For each line of input, print the city that has the same name as the line of input, in lower case.
<b>Note:</b> You must match the exact spelling of the line of input. For example, "PAris" must be matched but "paris" must not be matched.

<h2>Constraints</h2>

* Each input line will contain a single word with at most 100 characters.
* Each character in the word will be either an uppercase or a lowercase letter.

<h2>Sample Input</h2>

Paris
PARIS
Cologne
cologne
Paris
paris

<h2>Sample Output</h2>

paris
paris
cologne
cologne
paris
paris
[INFO] Session snapshot: 189 cached tokens at /tmp/rh2/session.kv
$HARNESS --model $E4B --text-only "The capital of France is"  198.18s user 13.32s system 550% cpu 38.386 total
```

The first three words are `Paris`, correctly completing the prompt, and the
continuation is fluent English throughout. It then drifts into an invented
programming-exercise page — which is base-model behaviour on a short
continuation prompt and not a defect, the same reading the 2026-08-12 attempt
had to establish the hard way.

The banner is identical to the 2026-08-12 run in all four lines, so the KV
reconciliation above stands unchanged against a second independent load.

**Both banners above predate U8, 2026-08-15.** A run now ends its banner with
`[INFO] Model load: {s}` and reports `[INFO] Prefill: {n} tokens in {s}`
before the reply. The entry stays VERIFIED: its expected signal names four
lines and asks for those four, all of which still print unchanged, and both
blocks are declared excerpts rather than complete transcripts. A re-run will
show two lines these do not — that is the addition, not a discrepancy.

Observed (2026-09-04, the banner and the reply's first lines; the rest of the
reply is byte-identical to the 2026-08-14 block above):

```
[INFO] Model: /mnt/ai-data/models/hf/gemma-4-E4B hash=e376216f8b693aff layers=42
[INFO] Weights: 13.90 GiB resident at bf16
[INFO] Peak RSS after load: 14.03 GiB
[INFO] KV cache: 131072 tokens, 4.05 GiB f32 resident (sliding layers capped at the window plus a batch of 64)
[INFO] Rewind horizon: 208 tokens behind the tip once the window fills (compaction past it re-prefills)
[INFO] Fold: dot=widen4 batch=64 threads=8
[INFO] Sampling: seed=0 temperature=0.8 max_tokens=256
[INFO] Build: 0.1.0 git unknown
[WARN] KV cache above 2 GiB: consider --context to cap it (compaction keeps long sessions inside the cap)
[INFO] Model load: 6.07 s
[INFO] Prefill: 14 tokens in 1.06 s (13 tok/s)
The capital of France is Paris
< div > The capital of France is < / div >

<h2>Input and Output</h2>
```

The two lines U8 added are there, and the reply reproduces the 2026-08-14
text word for word through the closing sample output. That is not luck:
sampling is seeded (`seed=0`), so the same weights, prompt and template give
the same tokens, and three weeks of engine changes moved none of them. The
four KV lines are unchanged for the third load, and `Prefill: 14 tokens` is
the rendered chat turn RH.1a's clean run also counted.

Status: VERIFIED 2026-09-04 (first decided 2026-08-14)

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
mkdir -p /tmp/rh3 && cd /tmp/rh3 && rm -f session.jsonl
time ~/workspace/ai-experiments/rust-harness/target/release/harness \
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
[INFO] Weights: 13.90 GiB resident at bf16
[INFO] Peak RSS after load: 14.03 GiB
[INFO] KV cache: 131072 tokens, 4.05 GiB f32 resident (sliding layers capped at the window plus a batch of 64)
[INFO] Rewind horizon: 208 tokens behind the tip once the window fills (compaction past it re-prefills)
[WARN] KV cache above 2 GiB: consider --context to cap it (compaction keeps long sessions inside the cap)
The program outputs one number per line.
def _(n):
if n <= 3:
return ['one','two','three'][n-1]
else:
return _(n-1)+' '+_(n-3)+' '+_(n-3)
def g():
n = 0
while True:
s = _(n)
n += 1
if s != _(n):
break
for x in s:
print(x, end=' ')
print()
g()
[INFO] Session snapshot: 134 cached tokens at /tmp/rh3/session.kv
$HARNESS --model $E4B --text-only --trace-chrome trace.json   136.14s user 9.86s system 521% cpu 28.005 total
```

**Roughly 5.4 tokens/second at the default thread width of 8**, decode-dominated.
The figure is read from this run's trace, aggregated in RH.6: 115 `model.forward`
calls in 21.12 s. The 28.0 s wall clock above includes the model load and is not
the divisor; 134 cached tokens over 28.0 s would understate the rate. The 521% CPU
figure against 8 threads is the honest utilization number — the run is not
saturating the width it asked for.

Three caveats bound how far this figure can be pushed, and they are the reason
this entry states no threshold.

*It is a single unpinned run.* `docs/optimizations/ideation/rust-harness-optimization-candidates.md`
measures single unpinned runs varying by up to 40 percent, and prescribes
`taskset -c 0-7`, five interleaved runs, and a median with the full spread. This
number does not meet that protocol and is not admissible as a baseline for any
optimization comparison. It is a scale check: single-digit tokens per second,
which is what callers need to size their timeouts.

*The configuration is unrecorded except by inference.* `threads=8`, `dot=widen4`
and `batch=64` are the defaults the ladder header printed in RH.1b; this run's
own output does not state them.

*The model ignored the instruction*, as a base checkpoint does — it emitted a
Python program rather than counting. That is irrelevant here: this entry times
the forward pass, and any prompt of similar length would serve.

**U8, 2026-08-15, makes the next run of this entry easier to record.** The
harness now prints `Model load: {s}` and `Prefill: {n} tokens in {s}` itself,
so the wall clock this entry asks for no longer has to be separated from the
load by hand. The expected signal is unchanged and the entry stays VERIFIED;
the pasted block simply predates the two lines.

Observed (2026-09-04, the same command; the reply reproduces the block above
word for word, so only its last lines are quoted):

```
[INFO] Model: /mnt/ai-data/models/hf/gemma-4-E4B hash=e376216f8b693aff layers=42
[INFO] Weights: 13.90 GiB resident at bf16
[INFO] Peak RSS after load: 14.04 GiB
[INFO] KV cache: 131072 tokens, 4.05 GiB f32 resident (sliding layers capped at the window plus a batch of 64)
[INFO] Rewind horizon: 208 tokens behind the tip once the window fills (compaction past it re-prefills)
[INFO] Fold: dot=widen4 batch=64 threads=8
[INFO] Sampling: seed=0 temperature=0.8 max_tokens=256
[INFO] Build: 0.1.0 git unknown
[WARN] KV cache above 2 GiB: consider --context to cap it (compaction keeps long sessions inside the cap)
[INFO] Model load: 6.17 s
[INFO] Prefill: 20 tokens in 1.51 s (13 tok/s)
The program outputs one number per line.
...
print()
g()
复制
[INFO] Session snapshot: 134 cached tokens at /tmp/rh3/session.kv
~/workspace/ai-experiments/rust-harness/target/release/harness --model $MODEL  151.62s user 10.04s system 535% cpu 30.216 total
```

The configuration caveat above is closed: the banner now states
`threads=8`, `dot=widen4`, `batch=64` and the sampler settings itself, and
the two U8 lines separate load from generation without a trace. Read from
the printed lines alone, 30.2 s wall less 6.17 s load less 1.51 s prefill
leaves about 22.5 s for the 114 decoded tokens (134 cached less 20 prompt),
**about 5.1 tokens/second at width 8**, beside the 5.4 the 2026-08-14 trace
gave. Still one unpinned run, so still a scale check and not a baseline; the
trace at `/tmp/rh3/trace.json` is the figure of record if RH.6 is re-aggregated.

The `复制` on the reply's last line is the model's own token, not a decoding
fault, and RH.2 check 2 is now the way to establish that with a number rather
than an argument: record this run and audit it, and the last row of the
report gives `复制`'s rank under `transformers` at that position. Sampling is
seeded, and the cached-token count is the same 134 as the
2026-08-14 run, so the earlier reply ended on the same token and the paste
dropped it. `复制` is "copy" — the label on the copy button beside code blocks
on Chinese programming sites, which is exactly where a base model that has
just closed a code block reaches for it. A decoder fault looks different: the
contaminated RH.1a attempt produced `<unused10>` and mojibake, where this is a
well-formed two-character word in a plausible position.

Why Chinese rather than English is a fact about the pretraining corpus, not
about the harness, and it is worth stating once because the next reader will
ask. The model is not fetching anything at generation time; it reproduces a
pattern it saw in training. A large share of the code-heavy pages on the web
are Chinese developer sites (CSDN, Zhihu, Juejin), and when those pages are
flattened to text the copy button beside every code block becomes the literal
string `复制` on the line after the code. English sites carry the same button,
but it is usually an icon or stripped as interface markup, so the word "Copy"
appears far less often in that position. A code block ending on a bare `g()`
line is therefore, in the corpus, very often followed by `复制`, and nothing in
the language-neutral code above it pushed the distribution toward English.
Two observations tie this to the checkpoint rather than to our code: the token
was sampled from a distribution L2 and L4 show agrees with `transformers`, so
the reference emits it too; and it occurs exactly once, immediately after the
code block closes, where a tokenizer or decoding defect would not confine
itself to one position.

Status: VERIFIED 2026-09-04 (first decided 2026-08-14)

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
mkdir -p /tmp/rh4 && cd /tmp/rh4
time ~/workspace/ai-experiments/rust-harness/target/release/harness \
    --model /mnt/ai-data/models/hf/<e4b> --text-only --trace-chrome turn1.json \
    "Summarize the Unix philosophy in three sentences"
time ~/workspace/ai-experiments/rust-harness/target/release/harness \
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
(turn 1)
[INFO] Session snapshot: 234 cached tokens at /tmp/rh4/session.kv
$HARNESS --model $E4B --text-only --trace-chrome turn1.json   245.56s user 15.09s system 563% cpu 46.260 total

(turn 2)
[INFO] Session restored: 234 cached tokens from /tmp/rh4/session.kv
[INFO] Session snapshot: 507 cached tokens at /tmp/rh4/session.kv
$HARNESS --model $E4B --text-only --trace-chrome turn2.json   290.41s user 16.74s system 546% cpu 56.164 total
```

Turn 2 restored 234 cached tokens and grew the session to 507, so it prefilled
only the appended suffix rather than re-running the transcript. **The delta
path works.**

**The span-level evidence this entry asks for was not captured.** The operator
pasted the banner and the `time` line, not the `session.sync` span durations
and position counts from `turn1.json` and `turn2.json`, so the *proportionality*
half of the expected signal is unverified — only the restore itself. The trace
files were written and could still be read if the claim needs sharpening.

Wall-clock is not the substitute measurement it might look like: turn 2 took
longer than turn 1 (56.2 s against 46.3 s) because it generated 273 new tokens
against turn 1's 234. Decode dominates at these lengths, so total time says
nothing about prefill cost either way.

A separate A/B run at the same session made the sharper finding, and it is a
defect rather than a confirmation. Given a directory already holding a
*divergent* history, a fresh prompt is answered in the context of that history:
after a Unix-philosophy turn, `"The capital of France is"` was restored onto
234 cached tokens and came back as a question-and-answer about Paris in the
prior turn's voice, rather than the document continuation the same prompt
produces in a clean directory. The behaviour is correct session restoration —
and it is invisible, because the only signal is one `[INFO] Session restored`
line in a banner nobody reads twice. That is the substance behind the open task
on serve-mode session persistence: **the problem is not that sessions persist,
it is that a restored session is indistinguishable from a model behaving
strangely.**

Status: VERIFIED 2026-08-14

## RH.5 Flight recorder and sandbox replay

Implementation: `rust-harness/runtime/src/model/record.rs`,
`rust-harness/runtime/tests/record_replay.rs`.

Host half records; the sandbox half replays the recording against the
mock, which any agent session can run.

Command (host):

```bash
mkdir -p /tmp/rh5 && cd /tmp/rh5
~/workspace/ai-experiments/rust-harness/target/release/harness \
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
cd rust-harness
mkdir -p runtime/tests/data && cp <flight.jsonl> runtime/tests/data/rh5-flight.jsonl
cat > runtime/tests/rh5_replay.rs <<'EOF'
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
cargo test -p runtime --test rh5_replay
git checkout -- . && rm -rf runtime/tests/data runtime/tests/rh5_replay.rs
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

Observed (host, structure of the written recording):

```
header      : {'temperature': 0.8, 'top_p': 0.95, 'top_k': 40, 'seed': 1}
forwards    : 257  selections: 256  other: 0
vocab width : 262144 vs header 262144
draws rise  : True [1, 2, 3, 4, 5] ... [254, 255, 256]
positions   : 14 -> 269
```

Observed (sandbox replay):

```
running 1 test
test rh5_recording_replays_as_a_mock_script_and_states_its_selections ... ok

test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 11.84s
```

Every part of the expected signal holds. The header names the three sampling
parameters as passed, plus `top_p` at its 0.95 default, which the command did not
set. There are 257 forward records against 256 selections —
the extra one is the prefill call, which produces logits without drawing a
token. Each recorded row is the full 262144-wide vocabulary, matching the
header, so nothing is truncated. Positions run 14 to 269 without a gap. And
the draw count rises by exactly one per step across all 256 selections, which
is the property that makes a replay's position in the stream recoverable
rather than merely asserted.

The recording is 257 rows of 262144 f32 each, so a full-vocabulary flight
recording is a large artifact by construction; this one was copied into the
worktree for the replay and deleted afterwards, which is what the command
block does.

One limitation of the format is worth recording here, because it is invisible
until it fires. JSON has no infinity, so `serde_json` writes a non-finite
logit as `null` and the recording will not parse back — `write_line` at
`record.rs:226-228` writes it and `record.rs:397` fails to read it, and because `Line` is an untagged
enum the error blames the whole line rather than naming the null. Nothing on
the shipped path produces `-inf` (top-k truncates a sorted candidate list
rather than masking), so this run was never at risk; the exposure is a caller
that hands in a masked logit row. `runtime/tests/record_replay.rs:87-91`
steers around it with `-1000.0` and says why. No test covers the failure.

Status: VERIFIED 2026-08-14

## RH.6 Chrome trace in Perfetto

Implementation: `rust-harness/harness/src/main.rs` (the `--trace-chrome`
subscriber), `rust-harness/harness/tests/tracing_spans.rs`,
`rust-harness/runtime/tests/tracing_spans.rs`.

This entry named the obs crate as its implementation, which it never was:
`tracing` spans and obs records are separate mechanisms with separate
lifetimes, and the obs crate installs no subscriber. Spans exist only under
`--trace-chrome`; obs records are written on every run. See
[observability.md](../../rust-harness/observability.md#phases-are-a-separate-mechanism-from-spans).

Command:

```bash
mkdir -p /tmp/rh6 && cd /tmp/rh6
~/workspace/ai-experiments/rust-harness/target/release/harness \
    --model /mnt/ai-data/models/hf/<e4b> --text-only --trace-chrome trace.json \
    "Reply with the single word: ready"
```

Expected signal: `trace.json` loads in Perfetto (ui.perfetto.dev) and shows
the documented span set with `model.forward` dominating wall-clock.

Four spans are unconditional on any generation: `session.sync`,
`session.step`, `sampler.sample`, `model.forward`. Two more —
`grammar.advance` and `tool.exec` — are **workload-conditional** and are
absent from a plain text completion, because nothing constrained the grammar
and no tool was called. Their absence here is not a finding; exercising them
needs a grammar-constrained or tool-calling run.

Observed (span counts and summed durations across three traces from this
campaign, aggregated with a `json` reader rather than in the Perfetto UI):

```
== rh3
   model.forward             115 calls     21.12s
   session.step              115 calls     19.95s
   sampler.sample            115 calls      0.57s
   session.sync                1 calls      1.74s
== rh4-turn2
   model.forward             257 calls     48.49s
   session.step              256 calls     48.12s
   sampler.sample            256 calls      1.26s
   session.sync                1 calls      1.64s
== rh6
   model.forward             176 calls     31.15s
   session.step              176 calls     30.77s
   sampler.sample            176 calls      0.87s
   session.sync                1 calls      1.25s
```

All four unconditional spans are present in every trace, and `model.forward`
dominates: 31.15 s of the rh6 run against 0.87 s in the sampler, so the
sampler is 2.8 percent of the time the forward pass takes. `session.sync`
fires exactly once per run, which is the prefill.

The `model.forward` and `session.step` counts differ by one in the rh4 trace
(257 against 256) and agree in the others. That is the prefill forward, which
has no enclosing step — the same one-extra-forward relationship RH.5's
recording shows as 257 forwards against 256 selections.

**The traces were read programmatically, not loaded into Perfetto.** The
expected signal names the UI; what was verified is that the files parse as
Chrome trace JSON and carry the documented spans with sane durations. Loading
one in the UI remains unverified, and is the cheap half of this entry if
anyone wants it closed strictly.

**U10, 2026-09-04, adds the interior of the forward pass to the trace.** A
re-run of this command now writes, at the default `--trace-level debug`, one
`model.layer` span per layer under every `model.forward`, each with
`model.layer.attn`, `model.layer.mlp` and `model.layer.ple` children, plus
`model.prelude` and `model.lm_head` per forward, `model.load` with its four
stages and one `model.load.layer` per layer read, and `tokenizer.encode` per
prompt. `--trace-level trace` adds `kernel.matmul`, `kernel.attention` and
`kernel.lm_head`; `--trace-level info` reproduces the shape of the traces
above. The four per-name totals in the Observed block keep their meaning —
the info spans are unchanged and the new spans nest inside `model.forward`
rather than beside it — so the entry stays VERIFIED on those blocks, which
predate the change. A re-run should show `model.layer` at 42 times the
`model.forward` count and `model.layer.attn` equal to `model.layer`;
`docs/rust-harness/observability.md` carries the full table and the Perfetto
query that replaces the hand aggregation.

Status: VERIFIED 2026-08-14

## RH.7 rr record, replay, reverse-continue

Implementation: `rust-harness/Makefile` (the rr targets),
`rust-harness/harness/src/main.rs`.

The observability phase 3 acceptance: a CPU forward pass recorded and
replayed under rr, with a watchpoint and `reverse-continue` reaching a
chosen write. The real model may be slow under rr; the committed tiny
fixture is an acceptable subject, and the Observed block must record which
model was used.

**Prerequisites, both learned the hard way on 2026-08-14.** `kernel.perf_event_paranoid <= 1`,
and an rr build that knows this CPU. rr identifies the microarchitecture
explicitly rather than probing it, because deterministic replay depends on the
retired-branch counter, and an unrecognized family is a hard failure rather
than a degraded mode. **No released rr recognizes Zen 5**, which is what this
workstation runs; see the Observed block.

Command:

```bash
cd rust-harness
make debug-record ARGS='--model fixtures/tiny-gemma4 "Reply with the single word: ready"'
make debug-replay
```

Inside the rr/gdb session:

```
(rr) break cpu.rs:3186
(rr) continue
(rr) watch -l logits[0]
(rr) reverse-continue
```

**The breakpoint is a line number, not a function.** An earlier version of
this entry said `break cpu.rs:forward_one` followed by `watch -l logits[0]`,
and that sequence cannot work: `forward_one` is declared at `cpu.rs:2703`,
while `logits` is a local of a different function, declared at `cpu.rs:3174`
inside the tied-head projection loop. The watchpoint has nothing to resolve
against at the `forward_one` frame. Breaking at `cpu.rs:3186`, the
`fill_dots(` call that fills it, puts `logits` in scope.

`release-with-debug` (`rust-harness/Cargo.toml:7-9`) inherits `release`, so it
is fully optimized and locals may print `<optimized out>`. If they do, rebuild
the recording arm with `CARGO_PROFILE_RELEASE_WITH_DEBUG_OPT_LEVEL=1`.

Expected signal: recording completes; replay reaches the breakpoint; the
watchpoint triggers under `reverse-continue` and gdb stops at the write
site inside the forward pass. Record the gdb transcript.

Observed:

```
[FATAL ./src/PerfCounters_x86.h:122:compute_cpu_microarch()]
 (task 0 (rec:0) at time 0)
 -> Assertion `false' failed to hold. AMD CPU type 0x40f40 (ext family 0xb) unknown
```

Both `make debug-record` and `make debug-replay` fail identically, before any
harness code runs. `0x40f40` decodes as base family `0xf` plus extended family
`0xb`, so family `0x1a` — Zen 5, the 9950X in this workstation. rr master maps
`ext_family 0xb` to `AMDZen5`, but the newest release, 5.9.0, predates that
change, and `0x40f40` additionally appears in rr's fixed CPUID table mapped to
AMDZen3 (Rembrandt), so even a partial fix risks silently misidentifying the
part rather than refusing.

Building rr from source did not clear it either: cmake refused with
`Your toolchain doesn't support 32-bit cross-compilation. Install the required
packages or pass -Ddisable32bit=ON to cmake.` The next attempt should start
from a clean object directory and configure with `-Ddisable32bit=ON`.

This is an external tooling wall, not a harness defect. Nothing in
`rust-harness/` was exercised, so no claim about the harness — positive or
negative — follows from it. The rr targets in `rust-harness/Makefile` are
untested on this workstation and will stay that way until rr supports Zen 5.

Status: BLOCKED 2026-08-14. Unblocks when an rr that recognizes Zen 5 is
available, whether from a release after 5.9.0 or a source build configured
with `-Ddisable32bit=ON`.

## RH.8 End-to-end coding task

Implementation: `rust-harness/harness/src/permissions.rs`,
`rust-harness/harness/src/tools.rs`, `rust-harness/harness/src/agent.rs`.

The full agent loop (permissions, tools, transcript) on real weights.

Command:

```bash
mkdir -p /tmp/rh8 && cd /tmp/rh8
printf 'rules:\n  - tool: "*"\n    action: allow\n' > permissions.yaml
~/workspace/ai-experiments/rust-harness/target/release/harness \
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

Status: BLOCKED 2026-08-14. Needs the instruction-tuned E4B checkpoint,
which is not on this workstation. Instruction following is this entry's
expected signal, so no run against the base checkpoint can close it — a
fluent non-answer is correct behaviour and proves nothing either way.

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
  (computed at `activation-capture.rs:279`, written into the manifest at
  `:305`), so it identifies a rust capture rather
  than describing a tokenizer, and probe training on rust captures must still
  go through artifacts produced by this chain only.

### RH.9a Activation capture

Command:

```bash
cd rust-harness
cargo run --release -p runtime --bin activation-capture -- \
    --model /mnt/ai-data/models/hf/<e4b> --text-only \
    --dataset interp/probes/prompts/refusal.jsonl --layers 8,16,24 \
    --out /mnt/ai-data/interp/captures/rust-e4b-verify
uv run python -c "from interp.probes.capture import load_capture; \
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
[OK] Capture: 258 examples, 3 layers, 1 shards -> /mnt/ai-data/interp/captures/rust-e4b-verify
cargo run --release -p runtime --bin activation-capture -- --model $E4B        2000.50s user 17.50s system 615% cpu 5:27.92 total

[8, 16, 24] {8: (258, 2560), 16: (258, 2560), 24: (258, 2560)} 258 258

{
  "artifact": "activation_capture",
  "capture_backend": "rust-harness-cpu",
  "count": 258,
  "created": "2026-08-14T05:35:57Z",
  "dataset_path": "/home/alejandro/workspace/ai-experiments/interp/probes/prompts/refusal.jsonl",
  "dataset_sha256": "2d89c7dd6aae207c146f8d20bfc26e827800f41e54e3c617ab96b06d7cf222b8",
  "dtype": "float32",
  "git_sha": "3d1b7b8f22afc3ddd499b84dbc4d559458babf20",
  "layers": [
    8,
    16,
    24
  ],
  "model_hash": "43fb96cec3045b72852c787540300dc5b258634b7a025f7c80355ac0788b9651",
  "model_id": "/mnt/ai-data/models/hf/gemma-4-E4B",
  "model_revision": "safetensors",
  "position": "last",
  "schema_version": 1,
  "shards": 1,
  "site": "layer_output",
  "tokenizer_fingerprint": "93161a0d68c540eb84a06191aed9775985014a26e4864b38d6cb85d901b077a4"
}

$ sha256sum /mnt/ai-data/models/hf/gemma-4-E4B/model.safetensors
43fb96cec3045b72852c787540300dc5b258634b7a025f7c80355ac0788b9651  /mnt/ai-data/models/hf/gemma-4-E4B/model.safetensors
```

Every part of the expected signal holds. `capture_backend` is
`rust-harness-cpu` and `model_revision` is `safetensors`, so the manifest
records that these activations came from this backend reading a Hugging Face
checkpoint rather than from the Python path. **The `model_hash` equals
`sha256sum` of `model.safetensors` digit for digit**, which is the field the
harness's probe identity guard later checks against — RH.9c is only meaningful
because this matched here.

`load_capture` reports all three requested layers at `(258, 2560)`, matching
the 258-line dataset and E4B's 2560-wide residual stream, with 258 labels and
258 texts.

258 examples in 5:27.92, so **1.27 s per example** at 615 percent CPU. The
manifest also carries `dataset_sha256`, `git_sha` and `tokenizer_fingerprint`,
which together make the capture reproducible from the record alone.

The dataset argument is `interp/probes/prompts/refusal.jsonl`, the committed
balanced refusal set. This entry previously wrote `<labeled.jsonl>` as a
placeholder, because nothing in the tree had that shape; the file now exists
and the command above names it.

Status: VERIFIED 2026-08-14

### RH.9b Probe training on rust captures

Command:

```bash
cd ~/workspace/ai-experiments
make probe-train ARGS="--capture /mnt/ai-data/interp/captures/rust-e4b-verify \
    --name rust-e4b-verify-probe"
```

Expected signal: training selects the best layer by validation AUROC and
registers the probe; AUROC well above 0.5 for a separable attribute.
Record the AUROC and selected layer.

Observed:

```
[RUN] Training probes from /mnt/ai-data/interp/captures/rust-e4b-verify
[OK] Probe registered: rust-e4b-verify-probe (layer 8, val AUROC 1.0, test AUROC 1.0)
  [INFO] layer 8: val 1.0
  [INFO] layer 16: val 1.0
  [INFO] layer 24: val 1.0
```

Training selected layer 8 and registered the probe. **What this verifies is
the pipeline, not the probe.** Activations captured by the rust backend load
into the Python trainer, train, and register — that is the cross-language
contract this entry exists to check, and it holds.

**AUROC 1.0 at every layer is a warning sign, not a result.** All three
candidate layers separate the two classes perfectly, including layer 8, which
is early for an abstract behavioural attribute. Perfect separation at every
depth usually means the dataset is separable by something cheaper than the
attribute — surface form, length, or topic — rather than that the probe found
refusal. The committed dataset was written to pair each label-1 request with a
superficially similar label-0 one for exactly this reason, but the measurement
does not distinguish "the pairing worked" from "some other cue survived it".
`probe-eval` against the black-box baselines is what would settle it, and it
needs `llama-gpu` and `vllm-gpu` up.

Nothing downstream in this runbook depends on the probe being *good*. RH.9c
uses it to test that observation does not perturb generation, and a probe that
detects the wrong thing serves that purpose exactly as well.

Status: VERIFIED 2026-08-14

### RH.9c In-band probe monitor

Command:

```bash
rm -rf /tmp/rh9 && mkdir -p /tmp/rh9/base /tmp/rh9/mon
make probe-export ARGS="--probe /mnt/ai-data/interp/probes/rust-e4b-verify-probe \
    --out /tmp/rh9/probe.safetensors"
cd /tmp/rh9/base && ~/workspace/ai-experiments/rust-harness/target/release/harness \
    --model /mnt/ai-data/models/hf/<e4b> --text-only "The capital of France is"
cd /tmp/rh9/mon && ~/workspace/ai-experiments/rust-harness/target/release/harness \
    --model /mnt/ai-data/models/hf/<e4b> --text-only --probe /tmp/rh9/probe.safetensors \
    "The capital of France is"
diff /tmp/rh9/base/session.jsonl /tmp/rh9/mon/session.jsonl; echo "diff exit: $?"
wc -l /tmp/rh9/mon/probe-flags.jsonl
head -3 /tmp/rh9/mon/probe-flags.jsonl
```

**Create both directories before exporting, and `rm -rf` first.** Two ordering
traps cost a host round trip on 2026-08-14. The safetensors writer writes
through a temporary file in the *destination* directory, so exporting into a
directory that does not yet exist dies with
`SafetenorError: ... No such file or directory ... /tmp/rh9/.tmpXXXXXX` and
every later step cascades from the missing probe. And reusing a populated
`base/` makes the second run a session *continuation* rather than an
independent run, which silently destroys the comparison this entry is built
on.

The export goes through `make probe-export` rather than a `python -c` call
into `write_probe`; the CLI did not exist when this entry was written.

Expected signal: the monitored run prints the one-line probe summary
(positions scored, positions flagged, threshold) to stderr, writes
`probe-flags.jsonl`, and its generated output is identical to the
unmonitored baseline (observation only). If the transcripts embed
timestamps, compare the generated text rather than raw lines and record
the comparison used.

Observed:

```
[OK] Image written: /tmp/rh9/probe.safetensors
[OK] Sidecar written: /tmp/rh9/probe.json
[WARN] Outside /mnt/ai-data/interp/probes: harness-serve mounts only that tree, so it cannot read this image
-rw-rw-r-- 1 alejandro alejandro   366 Aug 13 23:01 /tmp/rh9/probe.json
-rw------- 1 alejandro alejandro 30928 Aug 13 23:01 /tmp/rh9/probe.safetensors

(baseline run)
[INFO] Session snapshot: 189 cached tokens at /tmp/rh9/base/session.kv

(monitored run)
[INFO] Session snapshot: 189 cached tokens at /tmp/rh9/mon/session.kv
[INFO] Probe rust-e4b-verify-probe: 189 positions scored, 38 flagged (threshold 0.9531292), flags at /tmp/rh9/mon/probe-flags.jsonl

diff exit: 0
38 /tmp/rh9/mon/probe-flags.jsonl
{"layer":8,"position":1,"probe":"rust-e4b-verify-probe","score":0.9642110466957092,"threshold":0.9531291723251343}
{"layer":8,"position":2,"probe":"rust-e4b-verify-probe","score":0.9862508773803711,"threshold":0.9531291723251343}
{"layer":8,"position":3,"probe":"rust-e4b-verify-probe","score":0.9786973595619202,"threshold":0.9531291723251343}
```

**`diff exit: 0` is the result of this entry.** The monitored transcript is
byte-identical to the unmonitored one — same 189 cached tokens, same generated
text, no timestamp filtering needed. Attaching a probe reads the residual
stream without changing what the model produces. No other entry in this
runbook tests that, and it is the property every in-band interp claim rests
on: a monitor that perturbed generation would make its own observations
unfalsifiable.

The identity guard was armed rather than skipped. `main.rs:212-228` compares
the probe's `model_hash` against the sha256 of the served
`model.safetensors` and checks the probe width against `embed_dim`; both
passed with no `[WARN]`, which is only possible because RH.9a recorded a real
hash. A capture left at the `"unknown"` default would have loaded with the
check silently skipped.

The `[WARN] Outside /mnt/ai-data/interp/probes` line is correct and expected
here: `/tmp` is fine for a locally-run harness, and the warning is about
`harness-serve`, which mounts only the probes tree.

Two findings recorded rather than resolved.

*The exported probe is small.* 30928 bytes for the image and 366 for the
sidecar — a single 2560-wide layer-8 direction. Small enough to commit, if a
future test wants a real probe rather than a synthetic fixture.

*The flags cluster at the start of the sequence.* 38 of 189 positions flagged,
and the first three are positions 1, 2 and 3 at scores .964, .986 and .979
against a .9531 threshold. This probe was calibrated on last-token
activations and is being scored at every position, so the early positions are
exactly where its training distribution is emptiest and where attention-sink
behaviour dominates the residual stream. The position profile suggests the fix
is position-stratified fitting rather than a threshold adjustment. It does not
affect this entry's result — the flags are the probe's output, and the
non-perturbation claim holds whatever they say.

Status: VERIFIED 2026-08-14

### RH.9d Steering on real weights

Command (using an existing exported control vector, or one extracted from
rust captures via `interp/steering`):

```bash
mkdir -p /tmp/rh9/steer1 /tmp/rh9/steer2
cd /tmp/rh9/steer1 && ~/workspace/ai-experiments/rust-harness/target/release/harness \
    --model /mnt/ai-data/models/hf/<e4b> --text-only \
    --control-vector <vector-dir>/vector.safetensors --control-vector-scale 1.0 \
    "Describe your day in two sentences"
cd /tmp/rh9/steer2 && ~/workspace/ai-experiments/rust-harness/target/release/harness \
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

Status: BLOCKED 2026-08-14. Needs the instruction-tuned E4B checkpoint,
which is not on this workstation. Instruction following is this entry's
expected signal, so no run against the base checkpoint can close it — a
fluent non-answer is correct behaviour and proves nothing either way.

## RH.10 Flamegraph baseline

Implementation: `rust-harness/Makefile` (the `flame` target),
`rust-harness/runtime/src/model/cpu.rs` (the profiled hot path).

Requires `cargo-flamegraph` and `perf`. Records the optimization baseline
for the CPU backend; there is no pass threshold beyond the profile being
attributable.

Command (the `make flame` target takes no arguments, so the underlying
command is invoked directly with the trailing binary arguments):

```bash
cd rust-harness
cargo flamegraph --profile release-with-debug --bin harness -- \
    --model /mnt/ai-data/models/hf/<e4b> --text-only "Reply with the single word: ready"
```

Expected signal: `flamegraph.svg` is produced and the profile is dominated
by the forward-pass matrix multiplications (attention and MLP projections
in `model::cpu`), not by tokenization, sampling, or I/O. Record the top
frames.

**Pass `-F 97`.** At the default sampling frequency this run lost 48.22
percent of its samples and wrote a 45.9 GB `perf.data`; the profile is not
trustworthy at that loss rate and the file is a disk hazard. Re-running under
`perf record -F 97` produced the attributable profile recorded below.

Observed:

```
# Total Lost Samples: 194
#
# Samples: 470K of event 'cpu/cycles/P'
# Event count (approx.): 165644891779
#
# Overhead  Command  Shared Object         Symbol
    38.65%  harness  harness               [.] runtime::tensor::Weight::row_dot_block
     6.10%  harness  harness               [.] <alloc::vec::Vec<T> as alloc::vec::spec_from_iter::SpecFromIter<T,I>>::from_iter
     3.14%  harness  harness               [.] runtime::model::cpu::fill_panel_widened
     2.89%  harness  [unknown]             [k] 0xffffffffade83d4b
     1.89%  harness  harness               [.] core::slice::sort::stable::quicksort::quicksort
     1.81%  harness  [unknown]             [k] 0xffffffffacf77a2e
     1.81%  harness  [unknown]             [k] 0xffffffffad425bf4
     1.58%  harness  harness               [.] runtime::model::cpu::prefill_groups
     1.33%  harness  harness               [.] <core::iter::adapters::map::Map<I,F> as core::iter::traits::iterator::Iterator>::fold
     1.31%  harness  [unknown]             [k] 0xffffffffac801268
     1.20%  harness  libm.so.6             [.] __expm1f
     0.81%  harness  libm.so.6             [.] __tanhf
     0.72%  harness  harness               [.] runtime::model::cpu::matmul_panel
```

The expected signal holds. `row_dot_block` is 38.65 percent on its own, and
with `fill_panel_widened` and `matmul_panel` the matvec path is roughly 42.5
percent of cycles. Tokenization does not appear at all. The call graph
confirms the attribution:
`row_dot_block <- fill_panel_blocked <- matmul_panel <- forward_batch <- prefill_groups <- prefill <- step`,
with `bf16_to_f32` inlined at the leaf.

Three observations worth carrying into the open optimization tasks.

*`Vec::from_iter` at 6.10 percent is the second-largest frame*, and it is
allocation rather than arithmetic. Nothing in the current optimization menu
names it.

*Scalar libm transcendentals cost about 2 percent* — `__expm1f` at 1.20 and
`__tanhf` at 0.81 — so GELU is going through one-at-a-time libc calls.

*Kernel frames are unresolved.* Four `[k] 0xffffffff...` entries totalling
about 7.8 percent have no symbol, because
`/proc/sys/kernel/kptr_restrict` hides `kallsyms`. That is a measurement gap,
not a finding: something in the kernel costs 7.8 percent and this profile
cannot say what.

The profile also caught the session-contamination trap. The flamegraph run
executed from `rust-harness/`, restored 111 cached tokens from a stale
`session.kv` left in the repository root, and emitted
`<unused200> accidents<unused17>...` followed by several hundred `</li>`
tokens. That output is a restored divergent session, not a numerics defect —
the same failure mode RH.4 documents — but it is worth noting that a
profiling run is exactly where nobody looks at the text.

Status: VERIFIED 2026-08-14

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
mkdir -p /tmp/rh11 && cd /tmp/rh11
~/workspace/ai-experiments/rust-harness/target/release/harness \
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

Status: BLOCKED 2026-08-14. Needs the instruction-tuned E4B checkpoint,
which is not on this workstation. Instruction following is this entry's
expected signal, so no run against the base checkpoint can close it — a
fluent non-answer is correct behaviour and proves nothing either way.

## RH.12 Container build and reproducibility

Implementation: the harness targets in the root `Makefile`,
`docker-compose.rust-harness.yaml`,
`docker/rust-harness/Dockerfile.rust-harness`.

The containerized build exists for reproducibility: a pinned rust
toolchain and `cargo build --locked` inside the image must produce a
binary whose seeded generation matches the host build.

Command:

```bash
cd ~/workspace/ai-experiments
make harness-build
make harness-run ARGS='--model /app/models/hf/<e4b> --text-only --seed 7 "Reply with the single word: ready"'
mkdir -p /tmp/rh12 && cd /tmp/rh12
~/workspace/ai-experiments/rust-harness/target/release/harness \
    --model /mnt/ai-data/models/hf/<e4b> --text-only --seed 7 \
    "Reply with the single word: ready"
```

Expected signal: the image builds from the pinned digests without
resolving new dependency versions (`--locked` succeeds); the container
run and the host run produce the same reply for the same seed and model;
session state lands in `sessions/rust-harness/` on the host.

Observed:

```
 ✔ Image rust-harness:latest Built                                                                          55.2s
rust-harness image built.
docker compose -f docker-compose.rust-harness.yaml run --rm harness --model /app/models/hf/gemma-4-E4B --text-only "The capital of France is"
[INFO] Weights: 13.90 GiB resident at bf16
[INFO] Peak RSS after load: 14.04 GiB
[INFO] KV cache: 131072 tokens, 4.05 GiB f32 resident (sliding layers capped at the window plus a batch of 64)
[INFO] Rewind horizon: 208 tokens behind the tip once the window fills (compaction past it re-prefills)
[WARN] KV cache above 2 GiB: consider --context to cap it (compaction keeps long sessions inside the cap)
The capital of France is Paris
...
[INFO] Session snapshot: 189 cached tokens at /work/session.kv
```

**The container output is byte-identical to the host run of RH.2** — the same
continuation, ending at the same 189 cached tokens. Peak RSS differs by
0.01 GiB (14.04 against 14.03), which is process overhead outside the
allocator, not a numerics difference. Two independently built binaries from
two toolchains agreeing to the token across 189 positions on a 14 GiB bf16
checkpoint is a strong reproducibility result, and it makes container-versus-
host timing comparisons legitimate later, because both sides provably do
identical work.

**"Byte-identical" is a claim about the reply, and after U8 it must be read
that way.** Since 2026-08-15 a run prints `Model load: {s}` and
`Prefill: {n} tokens in {s}`, and the two timing figures will differ between
container and host by construction — that is what a timing comparison *is*.
The continuation and the 189 cached tokens are what this entry established and
they are unaffected, so it stays VERIFIED; but a re-run diffing the two
transcripts wholesale will show a difference on those lines and must not read
it as a numerics divergence. Diff the reply, not the banner.

The run also confirms the container's security posture in passing: it executes
as `${UID}:${GID}` under `network_mode: none`, `read_only: true` and
`cap_drop: ALL`, and still writes its session into the bind-mounted `/work`.

**Both sides deviated from the documented command in the same way**, and the
entry is recorded rather than rewritten to match. Neither run passed
`--seed 7`, and the prompt was `"The capital of France is"` rather than the
instruction-shaped one this entry still specifies. Matching deviations still
prove determinism — the comparison is between two runs of the same thing — but
this is not evidence that seeded generation reproduces, because no seed was
set. The prompt change was correct (see the base-model note above) and the
command block should be updated to match; the missing seed is a gap in the
evidence.

The `.dockerignore` finding came out of this entry rather than from the
comparison. The first build transferred an **8.32 GB context in 9.4 s** for a
binary whose only `COPY` takes `rust-harness/`, because `.dockerignore` was an
empty 2-byte file. After it was filled in, the same target rebuilt in 22.4 s
with the context transfer down to 0.4 s. The post-fix "8.32 kB" figure is a
BuildKit incremental delta against a warm context cache, not a new total; a
clean measurement needs `docker builder prune -af` first.

Status: VERIFIED 2026-08-14

## RH.13 Run-level observability records

Implementation: `rust-harness/obs/src/lib.rs`,
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
mkdir -p /tmp/rh13 && cd /tmp/rh13
~/workspace/ai-experiments/rust-harness/target/release/harness \
    --model /mnt/ai-data/models/hf/<e4b> --text-only \
    "Reply with the single word: ready"
~/workspace/ai-experiments/rust-harness/target/release/harness \
    --model /nonexistent-checkpoint "task" ; echo "exit: $?"
tail -2 /mnt/ai-data/logs/harness/runs.jsonl
cd ~/workspace/ai-experiments && uv run python -c "
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

The failing run also prints two indented lines under its `[ERROR] Task
failed:` line, added by U7:

```
  Solution: point --model at a directory holding config.json and model.safetensors, or at the weights file itself
  Command: ls /nonexistent-checkpoint
```

and `exc_msg` must still equal the `Task failed:` line's message exactly,
carrying neither of them. That equality is what makes this entry's second
half worth having, and U7 is what first put it at risk, so it is now also
asserted in the sandbox by
`rust-harness/harness/tests/actionable_surface.rs`.

Observed (2026-08-14, before U7):

```
[ERROR] Task failed: read model /nonexistent-checkpoint: no checkpoint at /nonexistent-checkpoint: expected a directory holding config.json and model.safetensors, or a path to the weights file itself
exit: 1

$ tail -2 /mnt/ai-data/logs/harness/runs.jsonl
{"run_id":"R-20260814-052221-238952","command":"task","timestamp":"2026-08-14T05:22:59.861+00:00","duration_s":38.061,"exit_code":0,"run_dir":"/tmp/rh13"}
{"run_id":"R-20260814-052259-a38fde","command":"task","timestamp":"2026-08-14T05:22:59.892+00:00","duration_s":0,"exit_code":1,"run_dir":"/tmp/rh13"}

(the Python catalog reader)
R-20260814-052221-238952 task 0
R-20260814-052259-a38fde task 1

$ cat /mnt/ai-data/logs/harness/R-20260814-052259-a38fde.jsonl
{"ts":"2026-08-14T05:22:59.892+00:00","level":"info","run_id":"R-20260814-052259-a38fde","cli":"harness","command":"task","pid":3089850,"event":"cmd.start","msg":"","argv":["--model","/nonexistent-checkpoint","task"]}
{"ts":"2026-08-14T05:22:59.892+00:00","level":"error","run_id":"R-20260814-052259-a38fde","cli":"harness","command":"task","pid":3089850,"event":"fail","msg":"","exc_type":"anyhow","exc_msg":"read model /nonexistent-checkpoint: no checkpoint at /nonexistent-checkpoint: expected a directory holding config.json and model.safetensors, or a path to the weights file itself","exit_code":1}
{"ts":"2026-08-14T05:22:59.892+00:00","level":"info","run_id":"R-20260814-052259-a38fde","cli":"harness","command":"task","pid":3089850,"event":"cmd.end","msg":"","exit_code":1,"duration_s":0}
```

Both halves hold. The catalog carries `exit_code` 0 for the successful run and
1 for the failure, both run ids match `R-YYYYMMDD-HHMMSS-xxxxxx`, and the
Python reader returned both rows unchanged — so a Rust writer and a Python
reader agree on the shared obs contract against the real
`/mnt/ai-data/logs` root, not a scratch directory.

The event file carries the full `cmd.start` / `fail` / `cmd.end` triple. The
`fail` event is the part worth having: `exc_msg` is the whole diagnostic
message, so the catalog's `exit_code: 1` can be turned into a cause without
re-running anything.

One detail the contract does not require but which shows up here: the failing
run's `duration_s` is `0` and its `cmd.start` and `cmd.end` share a timestamp
to the millisecond, because the checkpoint path is validated before any
loading work. That is the desired behaviour — a bad path fails immediately
rather than after a 38-second load — and it is visible in the record.

**Reopened by U7, 2026-08-15.** The transcript above is a real run and is kept
as recorded; what it is no longer is *complete*. U7 added the two remedy lines
to the terminal output of the exact command this entry runs, so the pasted
terminal half now stops one line into what the harness prints. Everything
that changed is additive and everything the entry argues from is intact: the
`[ERROR] Task failed:` line, the `fail` record, and `exc_msg` are byte for
byte what is shown, which `actionable_surface.rs` pins against this
transcript's own string. But the block is evidence, not a prediction, and
writing the two lines into it from the source rather than from a run would
make it the second kind of thing. Re-run the command block on the workstation
and paste what it prints.

Status: PENDING (verified 2026-08-14; reopened by U7 for a terminal re-run)

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
cd rust-harness && cargo build --release --bin harness
./target/release/harness --model <e4b-checkpoint-dir> --text-only --serve 8005
```

```bash
.venv/bin/python -c "
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

Status: VERIFIED 2026-08-12

**U8, 2026-08-15, adds one line to this entry's startup and none per
request.** The server prints `Model load: {s}` once at start; the per-turn
prefill line is emitted from `SessionGenerator::complete` and deliberately not
from the served path, since a status line per request on a long-lived server
is unbounded output. The expected signal here is about `wait_healthy` and the
`chat_completion` payload, neither of which moves, so the entry stays
VERIFIED.

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
It is evidence the real-weight forward pass is sound. It was not itself a
correctness claim when this entry ran: the ladder against `transformers` is
what makes one. **That ladder has since run at Tier E — all five rungs, every
one PASS, on 2026-08-14** (entry RH.1b), so the
correctness claim this paragraph deferred to now exists.

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

Never executed against real weights; the entry status below records why. R10
moved the control vector to safetensors and retired llama.cpp steering, so the
scorecard's subject is now `harness-serve` rather than `llama-gpu`. Everything
below is verified in the sandbox against a synthetic model, which proves the
format and the wiring and nothing about steering a real checkpoint.

Requires a control vector extracted against the E4B weights being served; a
vector whose `manifest.json` records a different `model_hash` is refused, and
one exported before R10 is a GGUF that no longer loads. Re-export it with
`make steer-extract`.

```bash
make harness-build
make steer-up VECTOR=/mnt/ai-data/interp/vectors/<name> SCALE=1.0
make steer-scorecard VECTOR=/mnt/ai-data/interp/vectors/<name> \
    ATTRIBUTE="<behavior the judge scores>"
make steer-down
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

Observed:

```
```

Status: BLOCKED 2026-08-14. **There is no control vector this entry can use.**
The harness reads safetensors control vectors as of R10; every vector this
repository produced before then is GGUF, which the harness no longer loads.
Extraction against E4B has not been run, and the obvious route to it —
`make steer-extract --method repeng` — is itself blocked by KB-32, since
repeng 0.4.0 cannot import under the pinned numpy and is deliberately absent
from the interp image. `--method sae_row` may sidestep that and has not been
tried. Unblocks when a vector extracted against this checkpoint exists and
loads; the arms and the judge wiring are otherwise ready.

## RH.16 The preflight answers before the model loads

`harness --status` exists because every artifact check a run makes happens
after `CpuModel::load` — ninety seconds and 14 GiB on E4B — so a missing probe
sidecar or an unwritable log directory costs a full load to discover. The
preflight reads `config.json` and the safetensors header only. This entry says
its answers are the ones a run would give, on real weights rather than on the
16-dimension fixture the sandbox uses.

Steps 1-3 are the preflight itself. Step 4 exercises `--deep`, and needs an
artifact for the digest to be compared against — `compare_identity` is what
consumes the hash, so a `--deep` with no probe and no control vector reads
nothing at all. There is no probe on this workstation at a stable path today
(RH.9c wrote its own to `/tmp/rh9` behind an `rm -rf`, and no control vector
exists at all), so step 4 builds one from a copy.

Substitute the E4B checkpoint directory for `$MODEL`.

```bash
cd ~/workspace/ai-experiments/rust-harness
MODEL=<the gemma-4-E4B checkpoint directory>

# 1. clean preflight: every check [OK] or [SKIP], exit 0, well under a second
time cargo run --release -q -p harness -- --model "$MODEL" --text-only --status
echo "exit=$?"

# 2. stdout carries nothing; the report is a verdict, not payload
cargo run --release -q -p harness -- --model "$MODEL" --text-only --status \
  2>/dev/null | wc -c

# 3. a checkpoint that is not there: exit 1, not 2
cargo run --release -q -p harness -- --model /absent --status; echo "exit=$?"

# 4a. the digest the comparison is measured against
time sha256sum "$MODEL/model.safetensors"

# 4b. a probe carrying that digest. Exported from a registered probe and then
#     copied, so the width and layer stay the model's -- a hand-built probe of
#     the wrong width fails the geometry check first and never reaches
#     identity, which would look like a pass for the wrong reason.
cd ~/workspace/ai-experiments
make probe-export ARGS="--probe /mnt/ai-data/interp/probes/rust-e4b-verify-probe"
mkdir -p /tmp/rh16 && cp \
  /mnt/ai-data/interp/probes/rust-e4b-verify-probe/probe.safetensors \
  /mnt/ai-data/interp/probes/rust-e4b-verify-probe/probe.json /tmp/rh16/
DIGEST=$(sha256sum "$MODEL/model.safetensors" | cut -d' ' -f1)
python3 - "$DIGEST" <<'PY'
import json, sys
p = "/tmp/rh16/probe.json"
d = json.load(open(p))
d["model_hash"] = sys.argv[1]
json.dump(d, open(p, "w"))
PY

# 4c. --deep against a probe whose hash matches: [OK], and it costs a sha256
cd rust-harness
time cargo run --release -q -p harness -- --model "$MODEL" --text-only \
  --probe /tmp/rh16/probe.safetensors --status --deep
echo "exit=$?"

# 4d. one character wrong: the comparison must decide, not merely hash
sed -i 's/"model_hash": "4/"model_hash": "5/' /tmp/rh16/probe.json
cargo run --release -q -p harness -- --model "$MODEL" --text-only \
  --probe /tmp/rh16/probe.safetensors --status --deep
echo "exit=$?"
```

Expected: (1) exit 0, with `[OK] backend compatibility: supported` and a
config line naming **42 layers** — the same count RH.1's dump reports
("uniform 10240 across 42 layers"), which is what cross-checks that the
preflight read the checkpoint the run reads. (2) `0` bytes on stdout. (3) exit 1, with one `[ERROR] model directory`
line and the rest `[SKIP]`; exit 2 there would mean the mode misread a valid
command line, which is a different defect. (4c) exit 0 with `[OK] probe
identity: matches the served weights`, `[OK] probe geometry`, and an elapsed
time close to 4a's, since both read the same ~15 GB once. (4d) exit **1** with
`[ERROR] probe identity` naming both hashes.

Step 4d is the one that matters. Timing alone cannot distinguish a working
comparison from one that hashes the weights and discards the answer.

Record the elapsed times from (1) and (4c). The gap between them is the entire
argument for `--deep` being opt-in rather than the default, and this is the
only place it is measured against real weights.

Observed, steps 1-3, 2026-08-15:

```
$ time cargo run --release -q -p harness -- --model "$MODEL" --text-only --status
[OK] model directory: /mnt/ai-data/models/hf/gemma-4-E4B
[OK] checkpoint files: config.json and model.safetensors present, header names 2130 tensors
[OK] config.json: 42 layers, hidden size 2560, context 131072
[OK] backend compatibility: supported
[SKIP] probe: no --probe given
[SKIP] probe geometry: no --probe given
[SKIP] probe identity: no --probe given
[SKIP] control vector: no --control-vector given
[SKIP] control vector identity: no --control-vector given
[SKIP] interp log directory: no --interp-log given
[OK] obs log directory: /mnt/ai-data/logs/harness is writable
cargo run ... 5.81s user 0.22s system 394% cpu 1.528 total
exit=0

$ cargo run --release -q -p harness -- --model "$MODEL" --text-only --status 2>/dev/null | wc -c
0

$ cargo run --release -q -p harness -- --model /absent --status; echo "exit=$?"
[ERROR] model directory: no checkpoint at /absent: expected a directory holding config.json and model.safetensors, or a path to the weights file itself
[SKIP] checkpoint files: not checked: the model path did not resolve
[SKIP] config.json: not checked: the checkpoint files were unreadable
[SKIP] backend compatibility: not checked: the checkpoint did not parse
[SKIP] probe: no --probe given
[SKIP] probe geometry: no --probe given
[SKIP] probe identity: no --probe given
[SKIP] control vector: no --control-vector given
[SKIP] control vector identity: no --control-vector given
[SKIP] interp log directory: no --interp-log given
[OK] obs log directory: /mnt/ai-data/logs/harness is writable
exit=1

$ sha256sum "$MODEL/model.safetensors"
43fb96cec3045b72852c787540300dc5b258634b7a025f7c80355ac0788b9651
sha256sum ... 5.65s user 0.44s system 99% cpu 6.153 total
```

Steps 1-3 pass. 42 layers, hidden size 2560 and context 131072 all agree with
RH.2, stdout measured 0 bytes, and the absent checkpoint exited 1 rather than
2.

**The preflight costs ~45 ms.** Step 1's 1.528s wall is mostly cargo's build
check (5.81s user at 394% CPU); the same command with nothing to rebuild
returned in `0.045 total, 0.03s user`. Against a checkpoint load measured in
tens of seconds, that is the number this stage was built to produce.

**Finding: the original step 4 was unrunnable, and it exposed a defect.** It
passed no artifact, so `--deep` read nothing and returned in 45 ms — the
expected signal was right and the command could not produce it. The defect
underneath: `--deep` reported nothing at all in that case, so a script passing
it would believe the weights had been verified. Fixed by giving `--deep` an
`inert_when` condition, so it now warns `--deep: no effect when there is no
interp artifact to check`. The command block above is the corrected step 4;
the expected signal is unchanged.

Observed, step 4, 2026-08-15:

```
$ make probe-export ARGS="--probe /mnt/ai-data/interp/probes/rust-e4b-verify-probe"
uv run python -m interp probe-export --probe /mnt/ai-data/interp/probes/rust-e4b-verify-probe
[RUN] Exporting /mnt/ai-data/interp/probes/rust-e4b-verify-probe
[OK] Image written: /mnt/ai-data/interp/probes/rust-e4b-verify-probe/probe.safetensors
[OK] Sidecar written: /mnt/ai-data/interp/probes/rust-e4b-verify-probe/probe.json
  [INFO] Serve with: make harness-serve HARNESS_PROBE=/app/probes/rust-e4b-verify-probe/probe.safetensors

$ time cargo run --release -q -p harness -- --model "$MODEL" --text-only \
    --probe /tmp/rh16/probe.safetensors --status --deep
[OK] model directory: /mnt/ai-data/models/hf/gemma-4-E4B
[OK] checkpoint files: config.json and model.safetensors present, header names 2130 tensors
[OK] config.json: 42 layers, hidden size 2560, context 131072
[OK] backend compatibility: supported
[OK] probe: rust-e4b-verify-probe, layer 8, site LayerOutput, width 2560
[OK] probe geometry: width 2560 and layer 8 fit the model
[OK] probe identity: matches the served weights, sha256 43fb96cec3045b72852c787540300dc5b258634b7a025f7c80355ac0788b9651
[SKIP] control vector: no --control-vector given
[SKIP] control vector identity: no --control-vector given
[SKIP] interp log directory: no --interp-log given
[OK] obs log directory: /mnt/ai-data/logs/harness is writable
cargo run ... 40.61s user 0.71s system 112% cpu 36.762 total
exit=0

$ sed -i 's/"model_hash": "4/"model_hash": "5/' /tmp/rh16/probe.json
$ cargo run --release -q -p harness -- --model "$MODEL" --text-only \
    --probe /tmp/rh16/probe.safetensors --status --deep
[OK] model directory: /mnt/ai-data/models/hf/gemma-4-E4B
[OK] checkpoint files: config.json and model.safetensors present, header names 2130 tensors
[OK] config.json: 42 layers, hidden size 2560, context 131072
[OK] backend compatibility: supported
[OK] probe: rust-e4b-verify-probe, layer 8, site LayerOutput, width 2560
[OK] probe geometry: width 2560 and layer 8 fit the model
[ERROR] probe identity: trained on a different model: artifact model_hash 53fb96cec3045b72852c787540300dc5b258634b7a025f7c80355ac0788b9651, served model sha256 43fb96cec3045b72852c787540300dc5b258634b7a025f7c80355ac0788b9651
[SKIP] control vector: no --control-vector given
[SKIP] control vector identity: no --control-vector given
[SKIP] interp log directory: no --interp-log given
[OK] obs log directory: /mnt/ai-data/logs/harness is writable
exit=1
```

Step 4d decides the entry and passes: one changed character turns the identity
check from `[OK]` to `[ERROR]` naming both hashes, and the exit code follows it
from 0 to 1. The comparison decides rather than merely reading bytes, which
timing alone could never show. The digest the preflight computed is the one
`sha256sum` reported in 4a, so the two independent readers of the same 15 GB
agree.

**4c's elapsed time is not the isolated figure the entry asked for.** 36.762s
against 4a's 6.153s, at 112% CPU and 40.61s user: this run rebuilt the release
binary after the `--deep` fix, and the build is the difference. The two
deciding signals are unaffected, but the "close to 4a's" sub-signal is
unconfirmed rather than observed — a repeat against a warm target directory
would settle it, and until then the honest statement is that `--deep` reads the
weights once, evidenced by the matching digest rather than by the clock.

Status: VERIFIED 2026-08-15

## RH.17 The CPU backend past E4B's real sliding window

Implementation: `rust-harness/runtime/src/ladder.rs`,
`rust-harness/runtime/src/bin/ladder.rs`,
`rust-harness/fixtures/reference_long_context.py`.

Rungs L0 through L4 never cross a window. L3 scores in 256-token windows
against an E4B attention window of 512, and L4's continuations start from
prompts of a few dozen tokens, so the two features that are the architectural
interest of Gemma 4 — the interleaved sliding/global split, and the shared-KV
tail reading a donor layer's older rows — are engaged by no reference
comparison below L5. `fixtures/long-context-logits.json` is not a substitute:
the binary that writes it is the engine it guards.

**`make ladder` now carries L5 as a standing rung, so this entry is a rung of
the routine climb rather than a bespoke command.** The reference artifact is
not committed and has to be generated on the workstation by
`rust-harness/fixtures/reference_long_context.py`, which needs the checkpoint
and `transformers`. Both sides derive the scored positions from the
checkpoint's `sliding_window`, so which positions were scored is itself part of
what L5 asserts.

```bash
cd ~/workspace/ai-experiments/rust-harness
MODEL=<the gemma-4-E4B checkpoint directory>

# 1. generate the reference. It writes long_context.json into the reference
#    directory the ladder reads for this SIZE.
uv run python fixtures/reference_long_context.py --model "$MODEL" \
  --out fixtures/reference/e4b/quick/long_context.json

# 2. the whole climb, L5 included
make ladder SIZE=quick MODEL="$MODEL"
echo "exit=$?"

# 3. L5 alone, for a re-run that does not pay for the first five rungs
cargo run --release -q -p runtime --bin ladder -- --text-only \
  --size quick --model "$MODEL" --reference fixtures/reference/e4b/quick \
  --rung L5
echo "exit=$?"
```

Expected signal: six rungs in the table, none skipped, `L5` passing. The
anchors L5 scores are `window - 1`, `window`, `window + 1`, `2*window + 1`,
`length // 2` and `length - 1`, derived from the window the *loader* read, and
the run reports that window. It must be 512 — the same figure the reference
generator recorded — because a run whose loader read a different window scores
different positions and compares rows computed at different places. A skipped
L5 means step 1 did not write where step 2 reads, and is a failure of this
entry rather than a pass with an absence.

Record the reported window alongside the verdict. The window agreeing across
the two independently written sides is half of what the rung measures.

Observed:

```
```

Status: PENDING

## RH.18 Attention capture on real weights

Implementation: `rust-harness/runtime/src/bin/attention-capture.rs`.

`attention-capture` writes post-softmax attention weights for the requested
layers and query heads over one prompt. Every assertion on it to date runs
against the two-layer synthetic fixture, whose layers are all one kind: the
fixture cannot show an interleaved sliding/global schedule because it does not
have one. This entry is the first reading of the captured weights against the
schedule the real checkpoint declares.

The prompt must be longer than the window for the sliding layers to differ
from the global ones at all — under 512 tokens both attend to the whole prefix
and the two are indistinguishable. Use the L5 corpus, which is 2052 tokens.

```bash
cd ~/workspace/ai-experiments/rust-harness
MODEL=<the gemma-4-E4B checkpoint directory>
mkdir -p /tmp/rh18

# a prompt past the 512-token window: the first ~2100 words of the corpus
PROMPT=$(head -c 12000 fixtures/reference/corpus.txt)

# sliding layers 0, 1 and 20, global layer 41 (the schedule puts a global
# layer at 5, 11, ..., 41); head 0 of each. --out is a directory: the tool
# writes attention.npz and manifest.json into it.
cargo run --release -q -p runtime --bin attention-capture -- --text-only \
  --model "$MODEL" --prompt "$PROMPT" --layers 0,1,20,41 --heads 0 \
  --out /tmp/rh18/attn
echo "exit=$?"

# 1. the manifest, then row mass and the zeros outside each row's span.
#    span_first_layer_<l> names the first visible key per query row; the
#    matrix is dense, so everything left of it is a structural zero.
cat /tmp/rh18/attn/manifest.json
uv run python - <<'PY'
import numpy as np
z = np.load("/tmp/rh18/attn/attention.npz")
for l in (0, 1, 20, 41):
    m, first = z[f"layer_{l}_head_0"], z[f"span_first_layer_{l}"]
    n = m.shape[0]
    mass = np.abs(m.sum(axis=1) - 1.0).max()
    outside = max(np.abs(m[i, :first[i]]).max(initial=0.0) for i in range(n))
    width = int((np.arange(n) - first + 1).max())
    print(f"layer {l}: rows {n}, worst |row sum - 1| {mass:.2e}, "
          f"max weight outside span {outside}, widest span {width}")
PY

# 2. the per-layer sliding/global verdict the checkpoint declares
cargo run --release -q -p runtime --bin checkpoint-info -- --text-only "$MODEL" \
  | grep -i 'window\|sliding\|global' | head -50
```

Expected signal: three properties, all of them checked against the same
capture.

| Property | Expected |
|----------|----------|
| Row mass | Every query row sums to 1.0 over the positions it may attend, within 1e-5 |
| Outside the span | Every weight at a position outside the row's span is exactly 0.0, not merely small |
| Schedule | The layers the capture marks sliding, and the span width it gives them, match what `checkpoint-info` reports for the same layer indices |

The zero outside the span is the deciding property and it is exact, not
approximate: a masked position is `-inf` before the softmax, so it comes out
`0.0` bit-for-bit. A small-but-nonzero weight there means the mask was applied
as a large negative number rather than an infinity, which is a different
implementation with a different failure mode at long context.

The widest span the script prints is the schedule verdict: 512 on a sliding
layer, the full prompt length on a global one. If it disagrees with what
`checkpoint-info` reports for the same layer index, record both and treat the
schedule check as unmet rather than inferring it.

Observed:

```
```

Status: PENDING

## RH.19 The alignment artifact on the real vocabulary

Implementation: `rust-harness/harness/src/alignment.rs`,
`rust-harness/runtime/src/align.rs`.

`--alignment-out` writes one JSONL file per run holding, for every position,
the byte span of output text that position produced alongside the probe score
taken there. The library and binary contracts are pinned in the sandbox
(RH.26 below), but on the synthetic fixture one clause of the artifact's
meaning cannot be observed.

**The divergence is recorded in the file header of
`rust-harness/harness/src/alignment.rs`, lines 9-17.** "Scaffolding produces no user-visible text" holds when the
checkpoint carries its chat markers as pieces `decode` drops — which the real
Gemma 4 vocabulary does, and which the synthetic fixtures deliberately do not,
since they carry boundaries as ordinary vocabulary entries so nothing but the
caller's allowlist can resolve them. On a fixture the marker therefore decodes
to its own characters and lands in the text. This entry is the only place the
dropping side of that fork is exercised.

```bash
cd ~/workspace/ai-experiments/rust-harness
MODEL=<the gemma-4-E4B checkpoint directory>
mkdir -p /tmp/rh19

cargo run --release -q -p harness -- --model "$MODEL" --text-only \
  --seed 7 --max-tokens 48 --alignment-out /tmp/rh19/align.jsonl \
  "The three states of matter are"
echo "exit=$?"

# 1. the header, then the role histogram
head -1 /tmp/rh19/align.jsonl | jq .
tail -n +2 /tmp/rh19/align.jsonl | jq -r .role | sort | uniq -c

# 2. the scaffold rows and the bytes they claim
tail -n +2 /tmp/rh19/align.jsonl \
  | jq -c 'select(.role == "scaffold") | {pos, span, text}'

# 3. the round trip: concatenated spans reproduce the text
tail -n +2 /tmp/rh19/align.jsonl | jq -r .text | tr -d '\n' | wc -c
head -1 /tmp/rh19/align.jsonl | jq .text_bytes
```

Expected signal: (1) positions dense from zero, counted in the header, and at
least one row marked `scaffold`. (2) **On the real vocabulary every scaffold
row carries an empty span** — the marker is a piece `decode` drops, so the
position exists in the artifact and contributes no bytes. This is the clause
the fixtures invert. (3) The concatenated spans reproduce the written text
byte for byte, which is the property the artifact would break if it suppressed
bytes to satisfy the rule rather than marking the role.

Record the scaffold rows verbatim. If any of them carries a non-empty span,
the fork in the header block is wrong about the real vocabulary and the note
needs correcting, not the artifact.

Observed:

```
```

Status: PENDING

## RH.20 Compaction past the rewind horizon

Implementation: `rust-harness/runtime/src/session.rs`,
`rust-harness/runtime/src/model/cpu.rs` (`rewind_horizon`).

`Session::sync` may not rewind past a sliding-window cache's horizon, because
such a cache evicts rather than masks. Below the horizon it clears and
re-prefills. Compaction rewinds much further than the horizon by construction —
it rewrites the middle of the transcript and the summarizing turn opens a fresh
sequence, so the common prefix lands near zero — which makes the clear-and-
re-prefill path the routine path after a compaction rather than an edge case.

`harness/tests/sampling_flags.rs` pins that `--compact-at` fires at a low
threshold and does not at a high one (RH.28), but on the fixture the horizon is
a handful of positions and the re-prefill is instant. **The horizon on E4B is
reported in the run header as `rewind_horizon`, and this entry expects 208**;
the header is the authority, so record what it prints rather than assuming the
figure. Reaching it needs a session long enough that compaction has a middle to
rewrite, which on E4B means turns numbering in the tens.

The transcript is `session.jsonl` in the working directory, so run this from an
empty directory. The debug-level rewind message is the deciding signal, so
raise the log level for it.

```bash
cd ~/workspace/ai-experiments/rust-harness
MODEL=<the gemma-4-E4B checkpoint directory>
mkdir -p /tmp/rh20 && cd /tmp/rh20

# 1. the horizon this checkpoint reports, before anything else
cargo run --release -q --manifest-path ~/workspace/ai-experiments/rust-harness/Cargo.toml \
  -p harness -- --model "$MODEL" --text-only --max-tokens 1 "hello" 2>&1 \
  | grep rewind_horizon

# 2. a session long enough to compact more than once. --compact-at 0.05
#    reaches the threshold in tens of turns rather than hundreds; the
#    conversation content does not matter, only its length.
seq 1 40 | sed 's/^/turn /' \
  | RUST_LOG=runtime::session=debug cargo run --release -q \
      --manifest-path ~/workspace/ai-experiments/rust-harness/Cargo.toml \
      -p harness -- --model "$MODEL" --text-only --chat \
      --compact-at 0.05 --max-tokens 8 2> /tmp/rh20/run.log
echo "exit=$?"

# 3. how many compactions the transcript records
jq -r .role /tmp/rh20/session.jsonl | sort | uniq -c

# 4. the rewind decision, once per compaction
grep 'rewind past the sliding window' /tmp/rh20/run.log
```

Expected signal: (1) `rewind_horizon=208`. (2) exit 0 — a rewind past the
horizon is a slower path, not an error, and a run that fails here means the
floor is being enforced as a refusal rather than as a rebuild. (3) at least two
compaction records. (4) **one `rewind past the sliding window: clearing and
re-prefilling` line per compaction, each naming `floor=208` and a `keep` below
it.** That line is the entry: it is the only evidence that compaction crossed
the horizon rather than staying inside a window where the cheap rewind
suffices, and a run with two compactions and no such line has not tested the
path this entry exists for.

If (4) is empty, lower `--compact-at` or lengthen the session and re-run rather
than recording a pass.

Observed:

```
```

Status: PENDING

## RH.21 Layer diff against a real reference capture

Implementation: `rust-harness/runtime/src/bin/layer-diff.rs`.

`layer-diff` runs the checkpoint, compares the residual stream layer by layer
against a reference capture, and passes or fails against `--cosine-floor` and
`--rel-l2-ceiling`. Its numerics are now bracketed in the sandbox against a
synthetic capture (RH.23), including that the verdict names the *first*
diverging layer and not the worst. What the sandbox cannot supply is a
reference capture from `transformers` over the real 42-layer stack, where the
per-layer figures are the ones a reader would quote.

```bash
cd ~/workspace/ai-experiments/rust-harness
MODEL=<the gemma-4-E4B checkpoint directory>
mkdir -p /tmp/rh21

# 1. the reference capture, from transformers. --prompt is required: the
#    residual stream is per-position, so the capture is only defined against
#    a stated input, and the same text must reach layer-diff.
uv run python fixtures/reference_capture.py --model "$MODEL" \
  --prompt "The three states of matter are" \
  --out /tmp/rh21/capture.json

# 2. the comparison at the defaults
cargo run --release -q -p runtime --bin layer-diff -- --text-only \
  --model "$MODEL" --reference /tmp/rh21/capture.json
echo "exit=$?"

# 3. the same comparison with the tolerances closed to just under the
#    measured worst row, so the tool has to fail and name a layer
cargo run --release -q -p runtime --bin layer-diff -- --text-only \
  --model "$MODEL" --reference /tmp/rh21/capture.json \
  --rel-l2-ceiling <just under the worst rel-L2 from step 2>
echo "exit=$?"
```

Expected signal: (2) exit 0, a table of 42 rows, cosine at or above the floor
and relative L2 at or below the ceiling on every one. Record the whole table:
the shape of the drift across depth is the measurement this entry exists to
produce, and no other entry reports it. (3) exit 1, with the verdict naming
**the first row over the closed ceiling, not the largest** — the tool states
that every later layer is downstream of this one, so a verdict naming the worst
row would send a reader to the wrong layer.

Step 3 is what makes step 2 mean anything. A pass with tolerances no run can
fail is not a measurement.

Observed:

```
```

Status: PENDING

## RH.22 checkpoint-info --check against the real E4B

Implementation: `rust-harness/runtime/src/bin/checkpoint-info.rs`.

`checkpoint-info` dumps a checkpoint's config and tensor table, then the
architecture summary and the CPU backend's compatibility verdict. `--check`
turns that verdict into an exit code: 1 when the checkpoint is unsupported,
rather than reporting and exiting 0. The released E4B is
`Gemma4ForConditionalGeneration` — a text tower beside a vision and an audio
one — so it exercises **both** verdicts from one checkpoint depending on
whether `--text-only` is given, and no fixture reproduces that.

```bash
cd ~/workspace/ai-experiments/rust-harness
MODEL=<the gemma-4-E4B checkpoint directory>

# 1. the report, no --check: exit 0 whatever the verdict
cargo run --release -q -p runtime --bin checkpoint-info -- "$MODEL"
echo "exit=$?"

# 2. --check without --text-only: multimodal, so unsupported
cargo run --release -q -p runtime --bin checkpoint-info -- --check "$MODEL"
echo "exit=$?"

# 3. --check with --text-only: the text tower is supported
cargo run --release -q -p runtime --bin checkpoint-info -- --check --text-only "$MODEL"
echo "exit=$?"

# 4. a directory that is not a checkpoint: an error, not a verdict
cargo run --release -q -p runtime --bin checkpoint-info -- --check /absent
echo "exit=$?"
```

Expected signal: (1) exit 0, with 42 layers, hidden size 2560 and context
131072 — the same three figures RH.16's preflight and RH.2's run header report,
which is what cross-checks that the three readers read one checkpoint. (2) exit
**1**, with the verdict naming the towers it declined to serve. (3) exit **0**,
supported. (4) an error naming the path; exit 1 for a missing checkpoint, and
not exit 2, which would mean the command line was misread.

The pair (2) and (3) is the entry. One of them alone shows the exit code moved,
not that it tracks the verdict.

Observed:

```
```

Status: PENDING

## Gate-backed entries

The entries above need a human, the workstation, or the real checkpoint. The
entries in this section do not: each states an expected signal and names the
test function that asserts it, and the evidence for it is a green run of that
test rather than a bespoke command. **An entry whose evidence is "the gate was
green" without naming the check is not verified**, so every row below names a
file and a function; anything that could not be tied to one is a full entry
above, or a roster row at the end.

One command closes the section. It runs in the sandbox and needs no GPU:

```bash
cd ~/workspace/ai-experiments/rust-harness
cargo test --workspace -- --test-threads=1
echo "exit=$?"
```

Paste the summary lines for the named test binaries into each entry's Observed
block, not the whole run.

### RH.23 layer-diff numerics are bracketed around a measured value

Asserted by `rust-harness/runtime/tests/layer_diff_numerics.rs`:

| Expected signal | Test function |
|-----------------|---------------|
| A reference matching the checkpoint passes every layer | `a_reference_matching_the_checkpoint_passes_every_layer` |
| A reference perturbed at one layer fails and names that layer | `a_reference_perturbed_at_one_layer_fails_and_names_that_layer` |
| The verdict names the first diverging layer, not the worst | `the_verdict_names_the_first_diverging_layer_and_not_the_worst` |
| `--rel-l2-ceiling` moves the verdict across a scaled row | `the_rel_l2_ceiling_moves_the_verdict_across_a_scaled_row` |
| `--cosine-floor` moves the verdict across a reoriented row | `the_cosine_floor_moves_the_verdict_across_a_reoriented_row` |
| A divergence before the stack displaces the layer verdict | `a_divergence_before_the_stack_is_reported_instead_of_a_layer` |

Each tolerance is bracketed around a measured value rather than asserted to be
non-empty: the test perturbs one row until the figure crosses, then runs the
tool once on each side of the flag. RH.21 supplies what this cannot — the same
tool against a `transformers` capture over 42 real layers.

Observed:

```
```

Status: PENDING

### RH.24 The ladder table is invariant across the flags that may not change it

Asserted by `rust-harness/runtime/tests/ladder_flags.rs`:

| Expected signal | Test function |
|-----------------|---------------|
| Every bit-identical `--dot` kind prints the same table | `every_bit_identical_dot_kind_prints_the_same_ladder_table` |
| Every `--threads` width, and `--single-thread`, prints the same table | `every_thread_width_prints_the_same_ladder_table` |
| Every `--batch` mode prints the same table, under the spelling the header reports | `every_batch_mode_prints_the_same_ladder_table` |
| Every bit-identical `--dot` kind yields the same logits bits | `every_bit_identical_dot_kind_produces_the_same_logits_bits` |
| Every `--threads` width yields the same logits bits | `every_thread_width_produces_the_same_logits_bits` |
| `--rung` climbs only the rung it names, in both directions | `rung_climbs_only_the_rung_it_names` |

The two logits tests compare `f32::to_bits`, not an epsilon: "bit-identical" is
the claim, so an approximate comparison would pass on a kernel that reorders.
The set the invariance is asserted over is `DotKind::is_bit_identical`, which
excludes `Lanes16` — see the known defect below.

Each sweep asserts the header names the value the run used, not merely that the
run accepted it. The `--batch` row states the spelling because the two differ:
`BatchKind::parse` reads `1` as `stepped`, and the header prints
`model.batch()`. A value clamped between the parser and the model — `7` read as
the default 64 — would otherwise leave the sweep green over a mode it never
ran.

`--threads` needs a wider checkpoint than the fixture supplies.
`runtime::model::cpu::par_row_runs` spawns no worker below 512 output rows, and
the widest row `rust-harness/fixtures/tiny-hf-gemma4` produces is its 288-entry
vocabulary, so every width there takes the serial path.
`every_thread_width_produces_the_same_logits_bits` therefore sweeps a
1024-vocabulary, 640-wide checkpoint synthesized through `runtime::synth`,
which crosses the threshold on both paths that split. The table test keeps the
fixture: its subject is the printed table, and no reference capture exists at
the wider shape.

Observed:

```
```

Status: PENDING

### RH.25 The fixture writers produce what they claim, reproducibly

Asserted by `rust-harness/runtime/tests/fixture_writers.rs`:

| Expected signal | Test function |
|-----------------|---------------|
| `tiny-model` writes a checkpoint that loads and prefills | `tiny_model_writes_a_checkpoint_that_loads_and_prefills` |
| Its tokenizer is readable by the `tokenizers` crate | `tiny_model_writes_a_tokenizer_the_tokenizers_crate_reads` |
| `--bpe` writes only a tokenizer | `tiny_model_bpe_writes_only_a_tokenizer` |
| The E4B variant writes a different architecture from the default | `tiny_model_e4b_writes_a_different_architecture_from_the_default` |
| Regeneration is byte-identical | `tiny_model_regenerates_byte_identical_artifacts` |
| `long-context-golden` writes the rows it claims | `long_context_golden_writes_the_rows_it_claims` |
| Its regeneration is byte-identical | `long_context_golden_regenerates_byte_identical_bytes` |

The two byte-identity tests are what let a fixture be regenerated in review: a
diff against the committed artifact is meaningful only if the writer is
deterministic.

Observed:

```
```

Status: PENDING

### RH.26 --alignment-out reaches the same contract through the binary

Asserted by `rust-harness/harness/tests/alignment_artifact.rs`:

| Expected signal | Test function |
|-----------------|---------------|
| Header and rows carry exactly the contract key sets | `the_header_and_rows_carry_exactly_the_contract_key_sets` |
| Every enumerated field holds a value the contract lists | `every_enumerated_field_holds_a_value_the_contract_lists` |
| Concatenating the written spans reproduces the written text | `concatenating_the_written_spans_reproduces_the_written_text` |
| Positions are dense from zero and counted in the header | `positions_are_dense_from_zero_and_counted_in_the_header` |
| A scored run carries the score channel; an unscored one omits it | `a_scored_run_carries_the_score_channel_and_an_unscored_one_omits_it` |
| The `--alignment-out` flag writes an artifact conforming to that contract | `the_alignment_out_flag_writes_an_artifact_conforming_to_the_same_contract` |
| The flagged run records every pass and counts its positions | `the_flagged_run_records_every_pass_and_counts_its_positions_in_the_header` |

The last two reach the flag through the binary rather than the library, and
share one `assert_conforms` helper with the library tests, so the flag cannot
drift into writing a second dialect of the artifact. RH.19 covers the one
clause this cannot: scaffold role on a vocabulary whose markers `decode` drops.

Observed:

```
```

Status: PENDING

### RH.27 Probe identity decides, and --probe-out writes where it names

Asserted by `rust-harness/harness/tests/probe_identity.rs`:

| Expected signal | Test function |
|-----------------|---------------|
| A probe recording the served model's sha256 loads and the run completes | `a_probe_recording_the_served_models_sha256_loads_and_the_run_completes` |
| A probe recording a different hash is refused, naming both hashes | `a_probe_recording_a_different_model_hash_is_refused_naming_both_hashes` |
| The unknown sentinel loads with the check skipped | `a_probe_recording_the_unknown_sentinel_loads_with_the_check_skipped` |
| `--probe-out` writes to the named path and leaves the default unwritten | `probe_out_puts_the_flag_file_where_it_names_and_leaves_the_default_unwritten` |

RH.16 step 4d is the same comparison against 14 GiB of real weights; this is
the same decision made in milliseconds, so both arms can be asserted on every
gate run.

Observed:

```
```

Status: PENDING

### RH.28 The sampling and compaction thresholds are validated at the boundary

Asserted by `rust-harness/harness/tests/sampling_flags.rs`:

| Expected signal | Test function |
|-----------------|---------------|
| A `--top-p` outside the unit interval is refused, naming the range | `a_top_p_outside_the_unit_interval_is_refused_naming_the_range` |
| A `--top-p` inside it is accepted and reaches the model | `a_top_p_inside_the_unit_interval_is_accepted_and_reaches_the_model` |
| A nucleus under greedy decoding warns rather than refusing the run | `a_nucleus_under_greedy_decoding_warns_rather_than_refusing_the_run` |
| A `--compact-at` outside the unit interval is refused, naming the range | `a_compact_at_outside_the_unit_interval_is_refused_naming_the_range` |
| A low threshold compacts the conversation and a high one does not | `a_low_compact_at_threshold_compacts_the_conversation_and_a_high_one_does_not` |

The last test is what makes `--compact-at` a threshold rather than a parsed
number. RH.20 covers what it cannot: a session long enough that compaction
rewinds past the horizon and forces the clear-and-re-prefill path.

Observed:

```
```

Status: PENDING

### RH.29 --tokenizer overrides the one beside the checkpoint

Asserted by `rust-harness/harness/tests/tokenizer_override.rs`:

| Expected signal | Test function |
|-----------------|---------------|
| The flag overrides the tokenizer beside the checkpoint | `the_tokenizer_flag_overrides_the_one_beside_the_checkpoint` |
| A tokenizer that does not exist is refused rather than resolved around | `a_tokenizer_that_does_not_exist_is_refused_rather_than_resolved_around` |

The second is the one that matters: a flag that silently falls back to the
checkpoint's own tokenizer on a typo produces a run that looks correct and was
not the run that was asked for.

Observed:

```
```

Status: PENDING

### RH.30 --control-vector-layer-range decides which directions load

Asserted by `rust-harness/harness/tests/control_vector_range.rs`:

| Expected signal | Test function |
|-----------------|---------------|
| A direction outside the range is dropped at load | `a_direction_outside_the_layer_range_is_dropped_at_load` |
| A direction inside the range is kept | `a_direction_inside_the_layer_range_is_kept` |
| A malformed range is refused, naming the flag | `a_malformed_layer_range_is_refused_naming_the_flag` |

The kept and dropped arms are asserted together; either alone is consistent
with a flag that ignores its value.

Observed:

```
```

Status: PENDING

### RH.31 --serve-addr binds where it names

Asserted by `rust-harness/harness/tests/serve_bind_addr.rs`:

| Expected signal | Test function |
|-----------------|---------------|
| The endpoint answers on the bind address and not on the default | `the_endpoint_answers_on_the_bind_address_and_not_on_the_default` |
| An address the host does not own fails the bind, naming it | `an_address_the_host_does_not_own_fails_the_bind_naming_it` |
| The default bind address is loopback, with `--serve-addr` absent from the command line | `the_default_bind_address_is_loopback` |

The third is a security property, not a convenience one: a harness that
defaults to `0.0.0.0` serves a checkpoint to the local network without anyone
having asked it to. Answering on `127.0.0.1` does not establish it, which is
the trap this entry has now fallen into twice: first with a helper that passed
`--serve-addr` on every run, so the default was never exercised at all, and
then with a health check that a wildcard bind satisfies as readily as a
loopback one. The discriminator is a second bind against this host's routable
address on the same port -- free if the harness took loopback, already held if
it took the wildcard. That arm skips loudly on a host with no routable
address.

Observed:

```
```

Status: PENDING

## Known defects

Three defects measured during this campaign, recorded here rather than in an
entry because none is a check to run: each is a statement about the tree that
a reader of the entries above needs.

**The `--serve-addr` default is two independent literals.** The value `--help`
prints comes from `default: Some("127.0.0.1")` in the flag table
(`rust-harness/harness/src/cli.rs:380`), which is consumed only by the help
renderer at `cli.rs:551`. The value a run actually binds is a separate
hardcoded literal, `None => IpAddr::from([127, 0, 0, 1])`
(`rust-harness/harness/src/main.rs:971`). Changing one leaves the other
stale, so the documented default and the real one can disagree with nothing
to notice. Found by mutation: editing the flag-table literal does not turn
RH.31's default test red, because it changes only what `--help` prints, while
editing the `main.rs` literal does. The same shape may hold for every other
flag carrying a `default:` in that table.

**`Lanes16` is not bit-identical to `Sequential` on the real fixture.** At
logit 0 the two kernels return `1062660370` and `1062660362` as raw f32 bits.
`DotKind::is_bit_identical` (`rust-harness/runtime/src/model/cpu.rs:1623`)
correctly excludes `Lanes16`, and RH.24's invariance tests are therefore
asserted over the right set. **The `--dot` help text in
`rust-harness/runtime/src/bin/ladder.rs:421` overclaims**: it reads "the
dot-product kernel; every kind is bit-identical", which is false of the one
kind the predicate beside it exists to exclude. A run that picks `lanes16` on
the strength of that sentence and then compares logits bits against a
sequential baseline will report a defect that is not there. The code is right
and the help text is wrong; see RHX.9 in
[rust-harness-ux.md](rust-harness-ux.md).

**A malformed safetensors header aborts the loader rather than returning an
error.** A header declaring a tensor larger than the file drives a request for
roughly 7.3e18 bytes, and the process dies on `SIGABRT` from the allocator
instead of surfacing a `Result`. Every other unreadable-checkpoint path in the
tree reports and exits 1 (RH.16 step 3, RH.22 step 4), so this is the one input
class where a caller cannot distinguish a bad file from a crash, and no
`catch_unwind` helps because an allocation failure is an abort and not a panic.
The fix is to validate declared extents against the file length before
allocating.

## Coverage roster

Every element the surface inventory found that is neither a full entry above
nor a gate-backed one, against the test that pins it. The roster exists so the
census is complete: a row here is covered, and its absence from the entry list
is a decision rather than an oversight.

| Element | Pinned by |
|---------|-----------|
| Flag table behind `--help`, all binaries | `rust-harness/harness/tests/cli_surface.rs`, `rust-harness/runtime/tests/binary_cli_surface.rs` |
| Preflight check list and its verdicts | `rust-harness/harness/tests/preflight_surface.rs` |
| Errors that name a runnable remedy | `rust-harness/harness/tests/actionable_surface.rs` |
| Status on stderr, payload on stdout | `rust-harness/harness/tests/status_streams.rs` |
| `run.header` field set, both directions | `rust-harness/harness/tests/run_header.rs`, `rust-harness/harness/tests/obs_contract.rs` |
| Capture obs records | `rust-harness/runtime/tests/capture_obs.rs` |
| Ladder obs records | `rust-harness/runtime/tests/ladder_obs.rs` |
| `ladder::window_schedule` against the reference generator's | `rust-harness/runtime/tests/ladder_reference_contract.rs` |
| Row-split bit-identity at every thread width | `rust-harness/runtime/src/model/cpu.rs::par_rows_is_bit_identical_at_every_width` |
| `Lanes16` excluded from the bit-identical set | `rust-harness/runtime/src/model/cpu.rs` (the `DotKind::Lanes16` assertion at line 4252) |
| Blocked-row folding against row-at-a-time | `rust-harness/runtime/src/tensor.rs::a_block_of_rows_is_bit_identical_to_the_rows_folded_one_at_a_time` |
| Widened blocks against stored-row folding | `rust-harness/runtime/src/tensor.rs::a_widened_block_is_bit_identical_to_folding_the_stored_rows` |
| Rewindability invariants across stepped and batched runs | `rust-harness/runtime/src/conformance.rs` |
| Colour helper as a pure function | `rust-harness/runtime/tests/status_python_parity.rs` |
