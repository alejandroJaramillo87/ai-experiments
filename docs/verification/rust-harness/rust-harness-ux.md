# Rust harness user surface verification

**Operational runbook. Executed against the workstation, not read for understanding: the
operator's pasted output is the evidence. Keep the commands runnable, and the paths accurate —
both documentation gates check this directory, so a stale reference here fails the gate.**

**Entries: 5 verified, 0 blocked, 7 pending, of 12.** RHX.9 joined the eight `U`-campaign
entries from the surface inventory: it records a defect in the `--dot` help text and the check
that closes it.

The `U` campaign rebuilt what the harness tells the person running it: one flag table behind
`--help` for all eight binaries, a preflight that answers before the model loads, a run header
that states the resolved configuration, errors that name their remedy, progress on the two
long operations, and a run record for five of the binaries. Almost all of it is pinned by the
sandbox suite against a 16-dimension synthetic fixture — roughly eighty assertions across
`rust-harness/harness/tests/cli_surface.rs`,
`rust-harness/harness/tests/preflight_surface.rs`,
`rust-harness/harness/tests/actionable_surface.rs`,
`rust-harness/harness/tests/status_streams.rs`,
`rust-harness/harness/tests/run_header.rs`,
`rust-harness/harness/tests/obs_contract.rs`,
`rust-harness/runtime/tests/binary_cli_surface.rs`,
`rust-harness/runtime/tests/capture_obs.rs` and
`rust-harness/runtime/tests/ladder_obs.rs`.

**What puts an entry here is that the sandbox cannot reach it.** Running these on the
workstation proves something running them in a container does not, and there are three
reasons an entry qualifies:

- **Scale.** The fixture is two layers and 16 dimensions, so its load is milliseconds and its
  KV budget is bytes. A header line reading `53.50 s` or `12.14 GiB` is a different claim from
  one reading `0.01 s`, and only the second has ever been observed.
- **Real weights and real artifacts.** A digest over 14 GiB, a probe fitted on the served
  checkpoint, a reference capture from `transformers` — none exists in a container.
- **A real terminal.** Every colour assertion in the tree runs through a pipe with colour off,
  because that is what the stream-hygiene tests are for. The branch that emits colour is
  exercised by unit tests over a pure function and by nothing that has ever attached to a TTY.

Recording protocol and status vocabulary are defined in [README.md](../README.md).

**No entry here is a GPU consumer** — the harness is the CPU backend — so the one-at-a-time
rule does not apply. A different constraint does: RHX.1 through RHX.5 each resident-load about
14 GiB, so run them one at a time for memory rather than for the card.

Two `U`-campaign entries live in [rust-harness.md](rust-harness.md) and are not duplicated
here: **RH.16**, the preflight against real E4B, and **RH.13**, the failure record, reopened by
U7 because the code changed what it prints. Both already carry observed output, and moving them
would discard evidence that exists.

**Revised 2026-09-12, before any entry had run.** The blocks were written on 2026-09-04 as
task runs from the source tree, and the main campaign has since shown what that shape does on
the pinned E4B, which is a base model: RH.37's runs of 2026-09-11 and 2026-09-12 each emitted
invented tool calls, stopped at the `[y/N]` permission prompt for every one, looped to
`no final answer within 64 turns`, and left `session.jsonl` and `session.kv` in the checkout
for the next run to resume. RHX.1 and RHX.2 are therefore `--chat` runs with the turns on
stdin, from a fresh directory, which is the prose path with no tools and no prompt (RH.11's
shape); the header and the two durations they read do not need instruction following. RHX.6
step 3 names the probe it copies, since two now exist on the workstation and the glob picked the
one that passes. RHX.9 no longer asks the ladder for logit bits it never prints. Where an
observation already exists in [rust-harness.md](rust-harness.md), the entry cites it rather
than asking for it again. The same audit found three things a person meets before any of
the above and no entry observed: the permission prompt, a silently resumed session, and
a chat with no prompt marker. RHX.10 through RHX.12 are those, and the first two come
with a status line added the same day so the run announces what it is doing. Later the same
day RHX.1 ran once and printed eight header lines, not nine: the ninth is the probe line, which
only a run given `--probe` prints. The block now loads the E4B probe, the first run is kept as
evidence, and `tests/hardware/test_run_header.py` pins the entry for `make quality-full`.

## Table of contents

- [Preconditions](#preconditions)
- [RHX.1 The run header states the configuration a real run resolved](#rhx1-the-run-header-states-the-configuration-a-real-run-resolved)
- [RHX.2 A run states what its two long operations cost](#rhx2-a-run-states-what-its-two-long-operations-cost)
- [RHX.3 The ladder records the climb it actually ran](#rhx3-the-ladder-records-the-climb-it-actually-ran)
- [RHX.4 The training dashboard renders a harness run](#rhx4-the-training-dashboard-renders-a-harness-run)
- [RHX.5 A capture is traceable to the invocation that made it](#rhx5-a-capture-is-traceable-to-the-invocation-that-made-it)
- [RHX.6 A failure states a remedy that runs as printed](#rhx6-a-failure-states-a-remedy-that-runs-as-printed)
- [RHX.7 One catalog serves the whole workspace](#rhx7-one-catalog-serves-the-whole-workspace)
- [RHX.8 Status output is coloured at a terminal and plain through a pipe](#rhx8-status-output-is-coloured-at-a-terminal-and-plain-through-a-pipe)
- [RHX.9 The --dot help text describes the kernels the flag offers](#rhx9-the---dot-help-text-describes-the-kernels-the-flag-offers)
- [RHX.10 A tool call the rules do not decide asks, and the run says so first](#rhx10-a-tool-call-the-rules-do-not-decide-asks-and-the-run-says-so-first)
- [RHX.11 A resumed session says what it resumed](#rhx11-a-resumed-session-says-what-it-resumed)
- [RHX.12 A chat at a terminal shows where the next turn goes](#rhx12-a-chat-at-a-terminal-shows-where-the-next-turn-goes)

## Preconditions

**A real Gemma 4 E4B checkpoint** under `/mnt/ai-data/models/hf/`, named as `$MODEL` in every
command block below. Substitute the directory; the entries do not assume its exact name.

**A release build.** Build once with `cargo build --release -p harness` and
`cargo build --release -p runtime`, both from `~/workspace/ai-experiments/rust-harness`. The
`harness` blocks then run the built binary from a fresh directory under `/tmp`, because the
harness writes `session.jsonl` and `session.kv` where it starts and a second run in that
directory resumes them; the `runtime` binaries write nothing where they start, and their
blocks use `cargo run --release -q` from the checkout. A debug build makes every duration in
this document meaningless.

**Block conventions.** Every block that pipes sets `set -o pipefail`, and every run is followed
by `echo "exit=$?"`, so the pasted Observed block carries the exit code beside the output. The
workstation shell is zsh; the blocks are written to run under it unchanged.

**`/mnt/ai-data/logs` writable**, since these entries verify what reaches the real obs root
rather than a temporary directory. RHX.4 additionally needs the repository `.venv`.

## RHX.1 The run header states the configuration a real run resolved

Implementation: `rust-harness/harness/src/runheader.rs`,
`rust-harness/harness/src/main.rs`.

U2 exists because `DotKind` defaulted to the reference fold for months while the fast one was
believed to be in use, and nothing printed which fold a run took. `RunHeader::lines()` derives
from `RunHeader::fields()` so the terminal and the obs record cannot disagree — that parity is
unit-tested, and what is not is whether the values are right at scale. The fixture reports a
KV budget in kilobytes and a load in milliseconds; this entry is the first time the header
describes a checkpoint anyone would serve.

The run loads a probe. The entry's first run, on 2026-09-12 and recorded below, printed eight
header lines against an expected nine: the ninth, `Probe {name}: layer {l}`, and the `probe`
and `probe_layer` fields it states exist only when `--probe` is given, and a plain chat run
exercises neither. No earlier banner did either. The probe is the E4B probe RH.16 step 4b
exported, `rust-e4b-verify-probe`, copied from its registry directory by name as RHX.6 does,
so the header has all nine lines to print and the record both fields to carry.

```bash
set -o pipefail
MODEL=<the gemma-4-E4B checkpoint directory>
rm -rf /tmp/rhx1 && mkdir -p /tmp/rhx1 && cd /tmp/rhx1

# the probe, and the sidecar facts the header must repeat
cp /mnt/ai-data/interp/probes/rust-e4b-verify-probe/probe.safetensors \
   /mnt/ai-data/interp/probes/rust-e4b-verify-probe/probe.json /tmp/rhx1/
jq -c '{name, layer, width, model_hash}' /tmp/rhx1/probe.json

printf '%s\n' "Reply with the single word: ready" \
  | ~/workspace/ai-experiments/rust-harness/target/release/harness \
      --model "$MODEL" --text-only --chat --probe /tmp/rhx1/probe.safetensors \
      --seed 7 --temperature 0.8 --max-tokens 16
echo "exit=$?"
ls /tmp/rhx1

# the record the terminal was rendered from
RUN=$(ls -t /mnt/ai-data/logs/harness/R-*.jsonl | head -1)
echo "$RUN"
jq -c 'select(.event == "run.header")' "$RUN"
```

Expected signal: the nine header lines, all `[INFO]`, naming the model and its hash, the
layer count, the resident weight size and dtype, peak RSS after load, the KV cache budget, the
rewind horizon, the fold (`dot=`, `batch=`, `threads=`), the sampling parameters including
`seed=7`, the probe as `Probe rust-e4b-verify-probe: layer 8` between `Sampling:` and
`Build:`, and the build. The `dot` value is the tuned kernel, not the reference one. A run
without `--context` follows them with `[WARN] KV cache above 2 GiB` before `Model load:`, and
the chat ends with `[INFO] Chat ended: 1 model turns` followed by
`[INFO] Probe rust-e4b-verify-probe: N positions scored, M flagged`, with `probe-flags.jsonl`
listed in `/tmp/rhx1`; none of those is a header line. The sidecar decides one more line: a
`model_hash` of `unknown` puts `[WARN] Model identity: probe records model hash "unknown",
check skipped` before the header, and a real digest prints nothing there and costs a sha256
over the weights before the header appears. The reply itself is immaterial: E4B is a base
model and may not say `ready`. The `run.header` record carries the same facts as JSON numbers
rather than formatted strings — `n_layers` and `weight_bytes` are numbers, `temperature`
renders as `0.8` and not `0.80000001`, `probe` is the sidecar's `name` as a string and
`probe_layer` its `layer` as a number.

The header half of this entry has been observed three times already, in RH.2, RH.8 and RH.37
of [rust-harness.md](rust-harness.md), and those banners are what the `dot=` claim above was
checked against. What none of them read is the record, and none loaded a probe: this entry
closes on the `run.header` line agreeing with the nine printed ones, and on nothing else.

Record both the nine lines and the `run.header` record. The point of the entry is that they
agree.

Asserted by `tests/hardware/test_run_header.py`:

| Expected signal | Test function |
|-----------------|---------------|
| Nine `[INFO]` lines from `Model:` through `Build:`, in the source's order, `Probe` eighth | `test_the_header_is_nine_info_lines_in_order` |
| `Fold:` has the `dot= batch= threads=` shape and `dot` is not `sequential` | `test_the_fold_is_the_tuned_kernel_not_the_reference` |
| `Sampling:` repeats `seed=7 temperature=0.8 max_tokens=16` | `test_the_sampling_line_states_the_flags_the_run_was_given` |
| `Probe` line equals the sidecar's `name` and `layer` | `test_the_probe_line_names_the_sidecar_it_loaded` |
| Every `run.header` field is typed (integers, `temperature` a float, `probe` a string) and renders to its printed line; the three byte fields convert to the printed GiB at two decimals | `test_the_record_agrees_with_every_printed_line` |
| `[WARN] KV cache above 2 GiB` sits after `Build:` and before `Model load:` | `test_the_kv_warning_follows_the_header_and_precedes_the_load` |
| `Chat ended: 1 model turns`, then the probe report naming `probe-flags.jsonl` in the run directory, which exists; the reply is on stdout | `test_the_run_ends_with_one_turn_and_a_probe_report` |

```bash
cd ~/workspace/ai-experiments
HARNESS_E4B_MODEL="$MODEL" uv run pytest -m hw_full tests/hardware/test_run_header.py -v
```

The test runs the same chat turn with the same probe from a temporary directory and, like
[RH.13](rust-harness.md)'s, leaves its one row and event file in the real obs root. It skips
when the checkpoint or the registered probe is absent and fails when the release binary is.

Observed (2026-09-12, the first run, before `--probe` was added to the block; eight header
lines, the record without `probe` or `probe_layer`):

```
➜  ai-experiments git:(rust-harness-ux-verification) ✗ MODEL=/mnt/ai-data/models/hf/gemma-4-E4B
➜  ai-experiments git:(rust-harness-ux-verification) ✗ set -o pipefail
rm -rf /tmp/rhx1 && mkdir -p /tmp/rhx1 && cd /tmp/rhx1
➜  rhx1 printf '%s\n' "Reply with the single word: ready" \
  | ~/workspace/ai-experiments/rust-harness/target/release/harness \
      --model "$MODEL" --text-only --chat \
      --seed 7 --temperature 0.8 --max-tokens 16
echo "exit=$?"
[INFO] Model: /mnt/ai-data/models/hf/gemma-4-E4B hash=e376216f8b693aff layers=42
[INFO] Weights: 13.90 GiB resident at bf16
[INFO] Peak RSS after load: 14.03 GiB
[INFO] KV cache: 131072 tokens, 4.05 GiB f32 resident (sliding layers capped at the window plus a batch of 64)
[INFO] Rewind horizon: 208 tokens behind the tip once the window fills (compaction past it re-prefills)
[INFO] Fold: dot=widen4 batch=64 threads=8
[INFO] Sampling: seed=7 temperature=0.8 max_tokens=16
[INFO] Build: 0.1.0 git unknown
[WARN] KV cache above 2 GiB: consider --context to cap it (compaction keeps long sessions inside the cap)
[INFO] Model load: 9.42 s
[INFO] Prefill: 16 tokens in 1.24 s (13 tok/s)
You can use the following LLM model names to use this command: <code>
[INFO] Chat ended: 1 model turns
[INFO] Session snapshot: 32 cached tokens at /tmp/rhx1/session.kv
exit=0
➜  rhx1 RUN=$(ls -t /mnt/ai-data/logs/harness/R-*.jsonl | head -1)
echo "$RUN"
jq -c 'select(.event == "run.header")' "$RUN"
/mnt/ai-data/logs/harness/R-20260912-150430-77f6ad.jsonl
{"ts":"2026-09-12T15:04:39.509+00:00","level":"info","run_id":"R-20260912-150430-77f6ad","cli":"harness","command":"chat","pid":175956,"event":"run.header","msg":"resolved run configuration","model":"/mnt/ai-data/models/hf/gemma-4-E4B","model_hash":"e376216f8b693aff","n_layers":42,"dtype":"bf16","weight_bytes":14924906496,"max_context":131072,"kv_bytes":4353867776,"kv_batch_group":64,"threads":8,"dot":"widen4","batch":"64","seed":7,"temperature":0.8,"max_tokens":16,"binary_version":"0.1.0","git_sha":"unknown","peak_rss_bytes":15068209152,"rewind_horizon":208}
```

Everything else the first run predicted held: eight `[INFO]` lines in the source's order,
`dot=widen4`, the warning between `Build:` and `Model load:`, one model turn, exit 0, and a
record whose byte fields convert to the printed figures (14924906496 B is 13.90 GiB,
4353867776 B is 4.05 GiB, 15068209152 B is 14.03 GiB) with `n_layers` and `weight_bytes` as
numbers and `temperature` as `0.8`. The count was the runbook's error, not the header's, and
the block above was widened rather than the signal narrowed.

Observed (2026-09-12, the widened block, with `--probe`):

```
➜  rhx1 cp /mnt/ai-data/interp/probes/rust-e4b-verify-probe/probe.safetensors \
   /mnt/ai-data/interp/probes/rust-e4b-verify-probe/probe.json /tmp/rhx1/
➜  rhx1 jq -c '{name, layer, width, model_hash}' /tmp/rhx1/probe.json
{"name":"rust-e4b-verify-probe","layer":8,"width":2560,"model_hash":"43fb96cec3045b72852c787540300dc5b258634b7a025f7c80355ac0788b9651"}
➜  rhx1 printf '%s\n' "Reply with the single word: ready" \
  | ~/workspace/ai-experiments/rust-harness/target/release/harness \
      --model "$MODEL" --text-only --chat --probe /tmp/rhx1/probe.safetensors \
      --seed 7 --temperature 0.8 --max-tokens 16
echo "exit=$?"
ls /tmp/rhx1
[INFO] Model: /mnt/ai-data/models/hf/gemma-4-E4B hash=e376216f8b693aff layers=42
[INFO] Weights: 13.90 GiB resident at bf16
[INFO] Peak RSS after load: 14.03 GiB
[INFO] KV cache: 131072 tokens, 4.05 GiB f32 resident (sliding layers capped at the window plus a batch of 64)
[INFO] Rewind horizon: 208 tokens behind the tip once the window fills (compaction past it re-prefills)
[INFO] Fold: dot=widen4 batch=64 threads=8
[INFO] Sampling: seed=7 temperature=0.8 max_tokens=16
[INFO] Probe rust-e4b-verify-probe: layer 8
[INFO] Build: 0.1.0 git unknown
[WARN] KV cache above 2 GiB: consider --context to cap it (compaction keeps long sessions inside the cap)
[INFO] Model load: 41.26 s
[INFO] Prefill: 16 tokens in 1.22 s (13 tok/s)
You can use the following LLM model names to use this command: <code>
[INFO] Chat ended: 1 model turns
[INFO] Session snapshot: 32 cached tokens at /tmp/rhx1/session.kv
[INFO] Probe rust-e4b-verify-probe: 32 positions scored, 12 flagged (threshold 0.9531292), flags at /tmp/rhx1/probe-flags.jsonl
exit=0
probe-flags.jsonl  probe.json  probe.safetensors  session.jsonl  session.kv
➜  rhx1 RUN=$(ls -t /mnt/ai-data/logs/harness/R-*.jsonl | head -1)
echo "$RUN"
jq -c 'select(.event == "run.header")' "$RUN"
/mnt/ai-data/logs/harness/R-20260912-152914-a99088.jsonl
{"ts":"2026-09-12T15:29:55.810+00:00","level":"info","run_id":"R-20260912-152914-a99088","cli":"harness","command":"chat","pid":240936,"event":"run.header","msg":"resolved run configuration","model":"/mnt/ai-data/models/hf/gemma-4-E4B","model_hash":"e376216f8b693aff","n_layers":42,"dtype":"bf16","weight_bytes":14924906496,"max_context":131072,"kv_bytes":4353867776,"kv_batch_group":64,"threads":8,"dot":"widen4","batch":"64","seed":7,"temperature":0.8,"max_tokens":16,"binary_version":"0.1.0","git_sha":"unknown","peak_rss_bytes":15069581312,"rewind_horizon":208,"probe":"rust-e4b-verify-probe","probe_layer":8}
```

Observed (2026-09-12, the gate test, first run; the failure bodies cut to their assertion):

```
tests/hardware/test_run_header.py::test_the_header_is_nine_info_lines_in_order FAILED
tests/hardware/test_run_header.py::test_the_fold_is_the_tuned_kernel_not_the_reference PASSED
tests/hardware/test_run_header.py::test_the_sampling_line_states_the_flags_the_run_was_given PASSED
tests/hardware/test_run_header.py::test_the_probe_line_names_the_sidecar_it_loaded FAILED
tests/hardware/test_run_header.py::test_the_record_agrees_with_every_printed_line FAILED
tests/hardware/test_run_header.py::test_the_kv_warning_follows_the_header_and_precedes_the_load PASSED
tests/hardware/test_run_header.py::test_the_run_ends_with_one_turn_and_a_probe_report PASSED

>       assert list(run.printed()) == HEADER_KEYS, "\n".join(block)
E       At index 7 diff: 'Probe rust-e4b-verify-probe' != 'Probe'

>       probe = run.printed()["Probe"]
E       KeyError: 'Probe'

>       assert printed["Probe"] == f"{record['probe']}: layer {record['probe_layer']}", (
E       KeyError: 'Probe'

[TIER] full: torch present, the whole suite was in scope.
FAILED tests/hardware/test_run_header.py::test_the_header_is_nine_info_lines_in_order
FAILED tests/hardware/test_run_header.py::test_the_probe_line_names_the_sidecar_it_loaded
FAILED tests/hardware/test_run_header.py::test_the_record_agrees_with_every_printed_line
```

The three failures share one cause in the test, not the header: `Probe {name}: layer {l}` is
the one line whose value sits before its colon, and the test's parser read
`Probe rust-e4b-verify-probe` as the key. The parser now keys that line `Probe`; the four
tests that passed include the fold and the warning's position. The corrected test's rerun
passed 7 of 7 the same day.

Status: VERIFIED 2026-09-12 on the widened block by hand. Nine `[INFO]` lines with `--probe`
and eight without, in the source's order; `dot=widen4`; the warning between `Build:` and
`Model load:`; one model turn, then the probe report, and `probe-flags.jsonl` in `/tmp/rhx1`;
exit 0. The record carries every printed fact typed: 14924906496 B is 13.90 GiB, 4353867776 B
is 4.05 GiB, 15069581312 B is 14.03 GiB, `temperature` is `0.8`, `probe` is the sidecar's
name and `probe_layer` its layer. The sidecar carries a real digest, so no identity warning
printed and the load line's 41.26 s includes the sha256 over the weights that a 9.42 s
probe-less load did not pay. The gate test's rerun after the parser fix passed, 7 of 7, as
reported by the operator on 2026-09-12; the transcript was not pasted.

## RHX.2 A run states what its two long operations cost

Implementation: `rust-harness/harness/src/main.rs`,
`rust-harness/harness/src/generator.rs`, `rust-harness/runtime/src/session.rs`,
`rust-harness/obs/src/lib.rs`.

Before U8 a cold start was a silent minute: `CpuModel::load` reads about 14 GiB and printed
nothing, and prefill ran the whole prompt through the forward pass and printed nothing at all,
so a human could not tell a slow load from a slow prefill from a slow decode. The durations
were already measured and reached nobody without opening the event file.

The second half of this entry is the one the fixture cannot show. Turn 2 of a chat re-sends
most of the transcript, and the KV cache should make its prefill small; a delta prefill that
quietly stopped working would look exactly like one that works, because nothing else displays
it. The tiny fixture's numbers are too small for the difference to mean anything.

```bash
set -o pipefail
MODEL=<the gemma-4-E4B checkpoint directory>
rm -rf /tmp/rhx2 && mkdir -p /tmp/rhx2 && cd /tmp/rhx2

# two turns in one session: one load line, then one prefill line per turn.
# Turn 2 re-sends the whole transcript, and its prefill line is the delta.
printf '%s\n' \
    "Summarize the Unix philosophy in three sentences" \
    "Now say that again in one sentence" \
  | ~/workspace/ai-experiments/rust-harness/target/release/harness \
      --model "$MODEL" --text-only --chat --max-tokens 32
echo "exit=$?"

# the printed load figure and the recorded one are the same clock
RUN=$(ls -t /mnt/ai-data/logs/harness/R-*.jsonl | head -1)
jq -c 'select(.event == "phase.end" and .phase == "model.load")' "$RUN"
```

Expected signal: one `[INFO] Model load: <n> s` reporting seconds, and two
`[INFO] Prefill: <n> tokens in <n> s (<n> tok/s)` lines with a rate worth having. The second
prefill's token count is a small fraction of the first's, the new turn plus the close of the
reply before it, because the cache covered the rest — that number is the only visible evidence
delta prefill works. The `duration_s` in the `phase.end` record agrees with the printed load
figure to two decimals. Neither stream carries a carriage return: the animation was
deliberately not built, so an `Observed` block gets clean lines.

RH.4 in [rust-harness.md](rust-harness.md) measured the delta itself, on a trace; this entry
is the line a person sees without one.

Asserted by `tests/hardware/test_status_costs.py`:

| Expected signal | Test function |
|-----------------|---------------|
| One `Model load:` line, then two `Prefill:` lines in turn order, then `Chat ended: 2 model turns` | `test_one_load_line_then_one_prefill_line_per_turn` |
| The printed load equals the `model.load` phase's `duration_s` to two decimals | `test_the_printed_load_is_the_recorded_phase_to_two_decimals` |
| Each `Prefill:` line's `tok/s` is its own tokens over its own seconds | `test_each_prefill_line_states_its_own_rate` |
| The second prefill is smaller than the first, the two prefills sum to less than the snapshot's cached tokens, and the remainder is at most two replies | `test_the_second_prefill_is_the_delta_not_the_transcript` |
| No carriage return in either stream; the replies are on stdout | `test_neither_stream_carries_a_carriage_return` |

The test's first turn is long and its second short, so the second prefill being smaller than
the first is decisive. That is the check the 2026-09-12 hand run below showed this entry needs:
on two turns of equal length the two prefills are equal, and only the snapshot arithmetic shows
the delta.

```bash
cd ~/workspace/ai-experiments
HARNESS_E4B_MODEL="$MODEL" uv run pytest -m hw_full tests/hardware/test_status_costs.py -v
```

Observed (2026-09-12):

```
➜  rhx2 printf '%s\n' \
    "Summarize the Unix philosophy in three sentences" \
    "Now say that again in one sentence" \
  | ~/workspace/ai-experiments/rust-harness/target/release/harness \
      --model "$MODEL" --text-only --chat --max-tokens 32
echo "exit=$?"
[INFO] Model: /mnt/ai-data/models/hf/gemma-4-E4B hash=e376216f8b693aff layers=42
[INFO] Weights: 13.90 GiB resident at bf16
[INFO] Peak RSS after load: 14.03 GiB
[INFO] KV cache: 131072 tokens, 4.05 GiB f32 resident (sliding layers capped at the window plus a batch of 64)
[INFO] Rewind horizon: 208 tokens behind the tip once the window fills (compaction past it re-prefills)
[INFO] Fold: dot=widen4 batch=64 threads=8
[INFO] Sampling: seed=0 temperature=0.8 max_tokens=32
[INFO] Build: 0.1.0 git unknown
[WARN] KV cache above 2 GiB: consider --context to cap it (compaction keeps long sessions inside the cap)
[INFO] Model load: 5.70 s
[INFO] Prefill: 17 tokens in 1.28 s (13 tok/s)
What is the Unix philosophy? What are the three guiding principles of Unix?
回答： The Unix philosophy is a set of three guiding principles that govern the design
[INFO] Prefill: 17 tokens in 1.33 s (13 tok/s)
The Unix philosophy is a set of three guiding principles that govern the design of Unix systems.
回答：The Unix philosophy is a set of three guiding principles that
[INFO] Chat ended: 2 model turns
[INFO] Session snapshot: 98 cached tokens at /tmp/rhx2/session.kv
exit=0
➜  rhx2 RUN=$(ls -t /mnt/ai-data/logs/harness/R-*.jsonl | head -1)
jq -c 'select(.event == "phase.end" and .phase == "model.load")' "$RUN"
{"ts":"2026-09-12T15:43:44.511+00:00","level":"info","run_id":"R-20260912-154338-b9b546","cli":"harness","command":"chat","pid":378783,"event":"phase.end","msg":"","phase":"model.load","duration_s":5.704}
```

The second prefill is 17 tokens, the same as the first, not "a small fraction" of it: the
first turn is itself 17 tokens, so the runbook's comparison could never hold on these prompts.
The delta is visible anyway, in the snapshot: 98 cached tokens is 17 + 32 + 17 + 32, the two
turns and their two 32-token replies, and a prefill that re-sent the transcript would have
read 66 tokens (17 + 32 + 17), not 17. The wording was the runbook's error, not the harness's,
and is left as written per the recording protocol.

Status: VERIFIED 2026-09-12. One `Model load: 5.70 s`, agreeing with the record's
`duration_s` 5.704 to two decimals; two `Prefill:` lines, 17 tokens each at 13 tok/s, the
second being the new turn alone as the 98-token snapshot shows; exit 0; no carriage return in
either stream. The 5.70 s load against RHX.1's 9.42 s and 41.26 s is a warm page cache and no
probe digest to pay. The replies are a base model continuing a chat template and are
immaterial. The gate test passed, 5 of 5, as reported by the operator on 2026-09-12; the
transcript was not pasted.

## RHX.3 The ladder records the climb it actually ran

Implementation: `rust-harness/runtime/src/bin/ladder.rs`,
`rust-harness/obs/src/lib.rs`, `schemas/obs_record.json`.

The ladder is the binary whose runs a human most wants to compare against each other, and
until U9b it wrote nothing to the catalog: its `Tuning` holds exactly the thread count, dot
kind and batch kind that hid the `DotKind` regression, and a red rung months later could not
be attributed to the configuration it ran under. A real E4B climb is also the only place the
per-rung phase durations are large enough to be worth reading.

```bash
cd ~/workspace/ai-experiments/rust-harness
MODEL=<the gemma-4-E4B checkpoint directory>

make ladder SIZE=quick MODEL="$MODEL"
echo "exit=$?"

RUN=$(ls -t /mnt/ai-data/logs/harness/R-*.jsonl | head -1)
jq -c 'select(.event == "ladder.header")' "$RUN"
jq -c 'select(.event == "phase.end") | {phase, duration_s}' "$RUN"
ls "/mnt/ai-data/logs/harness/$(basename "$RUN" .jsonl)/"
```

Expected signal: one `ladder.header` naming the model, the reference, the run size, the corpus
token count and the fold actually used. One `model.load` phase and one `rung.L<n>` phase per
attempted rung, each with a numeric `duration_s`. The run directory exists and holds a
`metrics.jsonl` — the ladder is one of the two binaries that opens one, with the rung index as
`step`. A rung that comes back red is still a completed run: it exits non-zero without a
failure record, the same way a divergence is not a crash.

Asserted by `tests/hardware/test_ladder.py`:

| Expected signal | Test function |
|-----------------|---------------|
| One `ladder.header` naming the checkpoint, the reference directory, the run size, the corpus token count and the fold (`threads`, `dot`, `batch`) | `test_one_ladder_header_names_the_climb` |
| One `model.load` phase, then one `rung.L<n>` phase per rung the table printed, in order, each with a positive `duration_s` | `test_one_model_load_then_one_phase_per_printed_rung` |
| The run directory holds a `metrics.jsonl` whose envelope is a `harness` `ladder` report and whose records are one per rung with the rung index as `step` and a `rung_duration_s` agreeing with the phase | `test_the_run_directory_holds_one_metrics_record_per_rung` |

The three ride the climb RH.1b's test already makes, so `make quality-full HW=full` climbs
once and reads the transcript and the record from the same run.

```bash
cd ~/workspace/ai-experiments
HARNESS_E4B_MODEL="$MODEL" uv run pytest -m hw_full tests/hardware/test_ladder.py -v
```

Observed (2026-09-12; the first attempt was made from the repository root, where no `ladder`
target exists, and is not recorded):

```
➜  rust-harness git:(rust-harness-ux-verification) ✗ make ladder SIZE=quick MODEL="$MODEL"
echo "exit=$?"

RUN=$(ls -t /mnt/ai-data/logs/harness/R-*.jsonl | head -1)
jq -c 'select(.event == "ladder.header")' "$RUN"
jq -c 'select(.event == "phase.end") | {phase, duration_s}' "$RUN"
ls "/mnt/ai-data/logs/harness/$(basename "$RUN" .jsonl)/"
cargo run --release -p runtime --bin ladder -- \
  --text-only --size quick --model /mnt/ai-data/models/hf/gemma-4-E4B --reference fixtures/reference/e4b/quick
   Compiling runtime v0.1.0 (/home/alejandro/workspace/ai-experiments/rust-harness/runtime)
    Finished `release` profile [optimized] target(s) in 3.87s
     Running `target/release/ladder --text-only --size quick --model /mnt/ai-data/models/hf/gemma-4-E4B --reference fixtures/reference/e4b/quick`
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
L6    attention past window                                 PASS
      captured layers agree   1.000000      >= 1.000000     ok
      weight outside window   0.000000      <= 0.000000     ok
      worst row max abs diff  0.000023      <= 0.001000     ok
      worst row mass          0.000001      <= 0.000010     ok
      worst top-10 overlap    1.000000      >= 0.900000     ok
      192 rows across 4 layers x 8 heads at 6 positions; window 512; worst at layer 21 head 6 position 1025

timing (not compared; expected to differ between runs)
  L0        0.51s
  L1        0.53s
  L2        0.65s
  L3       52.12s
  L4       23.36s
  L5      270.99s
  L6      227.47s
  sum     575.62s  (rung bodies only, excludes model load)
[OK] Ladder: every rung within tolerance (0 skipped)
[INFO] Peak RSS at exit: 16.13 GiB
exit=0
{"ts":"2026-09-12T16:40:12.371+00:00","level":"info","run_id":"R-20260912-164007-6f27ea","cli":"harness","command":"ladder","pid":754003,"event":"ladder.header","msg":"resolved ladder configuration","model":"/mnt/ai-data/models/hf/gemma-4-E4B","model_hash":"e376216f8b693aff","n_layers":42,"dtype":"bf16","weight_bytes":14924906496,"reference":"fixtures/reference/e4b/quick","size":"quick","corpus_tokens":512,"text_only":true,"threads":8,"dot":"widen4","batch":"64","binary_version":"0.1.0","git_sha":"698d7fed4f1d4d0d45ffdbc4491123ee1c1e1b29"}
{"phase":"model.load","duration_s":5.331}
{"phase":"rung.L0","duration_s":0.508}
{"phase":"rung.L1","duration_s":0.525}
{"phase":"rung.L2","duration_s":0.65}
{"phase":"rung.L3","duration_s":52.122}
{"phase":"rung.L4","duration_s":23.359}
{"phase":"rung.L5","duration_s":270.992}
{"phase":"rung.L6","duration_s":227.466}
metrics.jsonl
```

Every rung green, exit 0. The one `ladder.header` names the model, its hash, the reference
directory, `size=quick`, `corpus_tokens=512` and the fold `threads=8 dot=widen4 batch=64`. Its `git_sha` is the commit
checked out, where the chat run's envelope at RHX.4 the same morning read `unknown`; that
binary was built earlier and the difference is not explained here. One `model.load` phase and seven `rung.L<n>`
phases, each agreeing with the timing table to the centisecond (`L5` 270.992 against
`270.99s`). The run directory holds `metrics.jsonl` and nothing else, this being a run that
no `report` has been pointed at yet.

Status: VERIFIED 2026-09-12 by hand (gate-backed by `tests/hardware/test_ladder.py`; its three
RHX.3 functions were written from this transcript and had not run on the workstation when the
entry closed, the operator deferring the ten-minute climb to the next `make quality-full
HW=full`).

## RHX.4 The training dashboard renders a harness run

Implementation: `training/commands/report.py`, `common/obs/dashboard.py`,
`common/obs/viewer.js`, `schemas/run_metrics.json`.

U6's claim is that the harness became a first-class observability tier rather than a second
system: it writes a `metrics.jsonl` in the shape the training dashboard already reads, so the
dashboard renders a harness run **with no Python change**. That claim is testable in the
sandbox at the contract level, and this entry is the part that is not — a human looking at the
page and finding it populated and correct rather than merely valid.

Run RHX.1 or RHX.2 first. This entry starts no run of its own; it opens the run directory that
one produced, which the `ls -dt` below selects as the most recent under
`/mnt/ai-data/logs/harness/`. Confirm the printed `RUN_DIR` is that run before reporting on it —
RHX.3 also writes a run directory, so a ladder climb in between makes `head -1` pick the wrong
one.

```bash
cd ~/workspace/ai-experiments
RUN_DIR=$(ls -dt /mnt/ai-data/logs/harness/R-*/ | head -1)
echo "$RUN_DIR"
head -1 "$RUN_DIR/metrics.jsonl"

.venv/bin/python -m training report --run "$RUN_DIR"
ls -la "$RUN_DIR/run.html"
```

Expected signal: `report` writes `run.html` into the run directory and exits 0. Opened in a
browser it shows the header strip naming the same model, fold and build the terminal header
stated, a throughput series, and a process-memory series — the one chart group U6 added.
Prose plus a screenshot is acceptable evidence here; this is the entry that exists because it
cannot be automated, so its `Observed` block will not be pasted output.

The first line of `metrics.jsonl` is the report envelope. Record it: it is what makes the file
readable by a tool written for a different tier.

**Fix to confirm here** (details in [bugs/known-bugs.md](../bugs/known-bugs.md)):

- KB-70 — `common/obs/viewer.js` — the header strip read only the training envelope's keys
  and showed a dash for every field but the run id on the 2026-09-12 page below; the strip now
  keys its fields on `cli`, the empty chart frames are hidden, and the memory axis sizes its
  precision to the tick spacing. Fix landed 2026-09-12; re-run `report` on the same directory
  and reopen `run.html`. The `Fold` and `Build` the strip should now state are the envelope's
  `dot=widen4 batch=64 threads=8` and `0.1.0 git unknown`. The first re-run (second Observed
  block below) confirmed those two and exposed the third: the memory labels, now distinct,
  were clipped by the axis gutter. The gutter now sizes itself to the labels; the labels
  should read `14.24200` through `14.24240` in full. Confirmed as far as this entry goes on
  the third render the same day (third Observed block below); the row moved to resolved.

Observed (2026-09-12, on the run directory `tests/hardware/test_status_costs.py` wrote):

```
➜  ai-experiments git:(rust-harness-ux-verification) ✗ cd ~/workspace/ai-experiments
RUN_DIR=$(ls -dt /mnt/ai-data/logs/harness/R-*/ | head -1)
echo "$RUN_DIR"
head -1 "$RUN_DIR/metrics.jsonl"
/mnt/ai-data/logs/harness/R-20260912-160320-91dad9/
{"schema_version":1,"report":"run_metrics","timestamp":"2026-09-12T16:03:26.184+00:00","run_id":"R-20260912-160320-91dad9","cli":"harness","command":"chat","model":"/mnt/ai-data/models/hf/gemma-4-E4B","model_hash":"e376216f8b693aff","n_layers":42,"dtype":"bf16","weight_bytes":14924906496,"max_context":131072,"kv_bytes":4353867776,"kv_batch_group":64,"threads":8,"dot":"widen4","batch":"64","seed":7,"temperature":0.8,"max_tokens":32,"binary_version":"0.1.0","git_sha":"unknown","peak_rss_bytes":15069679616,"rewind_horizon":208}
➜  ai-experiments git:(rust-harness-ux-verification) ✗ .venv/bin/python -m training report --run "$RUN_DIR"
ls -la "$RUN_DIR/run.html"
[OK] Wrote /mnt/ai-data/logs/harness/R-20260912-160320-91dad9/run.html
-rw-rw-r-- 1 alejandro alejandro 80246 Sep 12 09:08 /mnt/ai-data/logs/harness/R-20260912-160320-91dad9//run.html
```

The page, opened from `file:///mnt/ai-data/logs/harness/R-20260912-160320-91dad9/run.html`
(screenshot taken 2026-09-12 09:10 local):

- Header strip: `run R-20260912-160320-91dad9  tier -  model -  recipe -  schedule -`. Neither
  the model, the fold nor the build appears; every field but the run id is a dash.
- Four empty bordered boxes, two above the throughput chart, one between the two charts, and
  one below the memory chart: the Loss, Learning rate, Device memory and Rung duration groups,
  which a chat run has no series for.
- Throughput (tokens/s): populated, 64 steps, `tokens_per_s` ramping from 0 to about 2.4 over
  the first 32 steps, dropping at step 32 where the second turn's prefill lands, and ramping
  again to about 4.3 at step 64. Correct for two 32-token turns.
- Process memory (GB): populated, flat near 14.24 GB with a step up at 32, but every one of the
  six y-axis labels reads `14.242`: the axis formatter rounds to three decimals and the series
  varies below that, so the labels are indistinguishable.
- Run log: the run's twelve records, from `cmd.start` through `cmd.end` with `exit_code: 0`,
  each phase's `duration_s` shown (`model.load` 5.83, two `prefill`, two `decode`).

`report` wrote the file and exited 0, and both series U6 promised are present and populated.
The header strip does not meet the expected signal, and this is the finding: `refreshHeader`
in `common/obs/viewer.js` reads `tier`, `model_id`, `recipe_hash` and `schedule`, the
training envelope's keys, and none of `model`, `dot`, `batch`, `threads`, `binary_version` or
`git_sha` the harness envelope carries. `schemas/run_metrics.json` states that "the dashboard's
header strip and the terminal's header state the same facts from the same struct"; the strip
reads none of them. "No Python change" was true and insufficient: the page needed a viewer
change and did not get one. Two smaller defects from the same screenshot: `render` appends a
bordered `.chart` holder for every group and `refreshCharts` returns early for the empty ones,
leaving the boxes, and the memory axis needs more precision than three decimals when its
range is narrow.

Observed (2026-09-12, the re-run after the strip fix, `report` on the same directory;
screenshot taken 09:22 local):

- Header strip: `run R-20260912-160320-91dad9  command chat  model
  /mnt/ai-data/models/hf/gemma-4-E4B  hash e376216f8b693aff  fold dot=widen4 batch=64
  threads=8  build 0.1.0 git unknown`. Every field the terminal header printed for this run.
- Two charts and no empty frames: Throughput (tokens/s) and Process memory (GB), then the run
  log.
- Process memory y-axis: five labels reading `24200`, `24210`, `24220`, `24230`, `24240` on a
  chart whose values are near 14.24 GB.

The labels are distinct, which is the precision fix working, and they are wrong to read
because the leading `14.` is off the canvas. The series varies by tens of KB, so the tick
spacing is 0.00001 and each label is eight characters, `14.24200` through `14.24240`. uPlot's
y-axis gutter is a fixed 50 px by default and draws labels right-aligned against it, so a label
wider than the gutter loses its left edge. The sandbox fixture used a 0.0001 spacing, one
character shorter, which fit by a few pixels and did not reproduce this. The gutter now sizes
itself to the widest label, and the harness fixture uses the workstation's spacing, on which
the old gutter fails the check (50 px against 72 px needed).

Observed (2026-09-12, the third render, `report` on the same directory after the gutter fix;
reported in prose, no screenshot):

The operator reported one remaining defect: the `rss_gb` value in the process-memory legend
does not change as the cursor moves along the chart line. They did not report on the axis
labels, and closed the entry at this point: the page is good enough for the campaign, and the
dashboard's UX is to be cleaned up in a later pass. The hover readout is filed as KB-71.

Status: VERIFIED 2026-09-12 with the KB-71 caveat. `report` wrote the page and exited 0 on
all three renders; the throughput and process-memory series were present from the first, and
the strip named the terminal header's model, fold and build from the second. Closed at the
operator's decision with the memory legend's hover readout open and the axis labels
unconfirmed.

## RHX.5 A capture is traceable to the invocation that made it

Implementation: `rust-harness/runtime/src/bin/activation-capture.rs`,
`rust-harness/runtime/tests/capture_obs.rs`, `interp/probes/prompts/refusal.jsonl`.

The capture manifests always recorded provenance about the *artifact* — `model_hash`,
`dataset_sha256`, `git_sha`. What no manifest ever recorded is provenance about the *run*: the
thread count, the fold, and how long the load took are absent from all of them, and that
absence is the same shape as the `DotKind` failure. U9c gave the three capture binaries a
header event and a `model.load` phase, and deliberately gave them no run directory.

```bash
cd ~/workspace/ai-experiments/rust-harness
MODEL=<the gemma-4-E4B checkpoint directory>

cargo run --release -q -p runtime --bin activation-capture -- \
  --model "$MODEL" --text-only \
  --dataset ../interp/probes/prompts/refusal.jsonl \
  --layers 8,16,24 --out /tmp/rhx5
echo "exit=$?"

RUN=$(ls -t /mnt/ai-data/logs/harness/R-*.jsonl | head -1)
jq -c 'select(.event == "activation-capture.header")' "$RUN"
jq -c 'select(.event == "phase.end")' "$RUN"

# the scope decision: no run directory for a capture binary
ls -d "/mnt/ai-data/logs/harness/$(basename "$RUN" .jsonl)/" 2>&1

# the catalog row that ties the artifact to the run
cd ~/workspace/ai-experiments && make harness-runs ARGS='-n 5'
```

Expected signal: one `activation-capture.header` naming the model, the layers, the dataset, the
example count and the fold, and one `model.load` phase whose `duration_s` is tens of seconds
rather than the fixture's milliseconds. **No run directory exists** — the `ls` reports no such
file, which is the scope decision observed rather than assumed. `harness-runs` shows a row
whose command is `activation-capture`, beside rows for `task` and `ladder`, all under one
catalog.

Asserted by `tests/hardware/test_capture_run_record.py`:

| Expected signal | Test function |
|-----------------|---------------|
| One `activation-capture.header` whose `model`, `dataset`, `out`, `layers` and `examples` are what was passed, carrying the fold and the build; the `[OK] Capture:` line's count equals its `examples` | `test_one_header_names_the_invocation` |
| One `model.load` phase, at least a second long, written before the header | `test_one_model_load_phase_in_seconds` |
| No run directory: the catalog row's `run_dir` names a path that does not exist | `test_no_run_directory_and_the_row_says_so` |
| One catalog row, command `activation-capture`, exit 0, its run id the header's and `cmd.end`'s | `test_the_catalog_row_names_this_run` |

The test captures the first eight lines of the refusal set where this entry captured all 258;
the record is the same shape at any length, and eight keeps the gate to one load plus seconds.

```bash
cd ~/workspace/ai-experiments
HARNESS_E4B_MODEL="$MODEL" uv run pytest -m hw_full tests/hardware/test_capture_run_record.py -v
```

Observed (2026-09-12; the 258 `[RUN] Example n/258: k tokens` lines between the first and
last are elided here, every one of that shape with 8 to 19 tokens):

```
➜  rust-harness git:(rust-harness-ux-verification) ✗ cargo run --release -q -p runtime --bin activation-capture -- \
  --model "$MODEL" --text-only \
  --dataset ../interp/probes/prompts/refusal.jsonl \
  --layers 8,16,24 --out /tmp/rhx5
echo "exit=$?"
[RUN] Example 1/258: 15 tokens
[RUN] Example 2/258: 13 tokens
...
[RUN] Example 257/258: 12 tokens
[RUN] Example 258/258: 11 tokens
[OK] Capture: 258 examples, 3 layers, 1 shards -> /tmp/rhx5
exit=0
```

The first pass at the reading half assigned the run file to a mistyped variable and read the
ladder's file from RHX.3; it is not recorded. The second, against the run id the catalog
listed:

```
➜  ai-experiments git:(rust-harness-ux-verification) ✗ RUN=/mnt/ai-data/logs/harness/R-20260912-170211-f585ef.jsonl
jq -c 'select(.event == "activation-capture.header")' "$RUN"
jq -c 'select(.event == "phase.end")' "$RUN"
ls -d "/mnt/ai-data/logs/harness/$(basename "$RUN" .jsonl)/" 2>&1
{"ts":"2026-09-12T17:02:17.178+00:00","level":"info","run_id":"R-20260912-170211-f585ef","cli":"harness","command":"activation-capture","pid":1304419,"event":"activation-capture.header","msg":"resolved capture configuration","model":"/mnt/ai-data/models/hf/gemma-4-E4B","model_hash":"e376216f8b693aff","n_layers":42,"dtype":"bf16","weight_bytes":14924906496,"layers":"8,16,24","dataset":"../interp/probes/prompts/refusal.jsonl","examples":258,"out":"/tmp/rhx5","text_only":true,"threads":8,"dot":"widen4","batch":"64","binary_version":"0.1.0","git_sha":"e451466bf02f437a54a7458309b9d877c494e68b"}
{"ts":"2026-09-12T17:02:16.715+00:00","level":"info","run_id":"R-20260912-170211-f585ef","cli":"harness","command":"activation-capture","pid":1304419,"event":"phase.end","msg":"","phase":"model.load","duration_s":5.487}
ls: cannot access '/mnt/ai-data/logs/harness/R-20260912-170211-f585ef/': No such file or directory
```

```
➜  rust-harness git:(rust-harness-ux-verification) ✗ cd ~/workspace/ai-experiments && make harness-runs ARGS='-n 5'
uv run --no-sync python -m common.obs.runs --cli harness -n 5
FINISHED             RUN ID                    COMMAND             EXIT    DURATION
2026-09-12T15:37:39  R-20260912-153654-1f95ad  chat                   0     45.107s
2026-09-12T15:43:58  R-20260912-154338-b9b546  chat                   0     19.581s
2026-09-12T16:03:46  R-20260912-160320-91dad9  chat                   0     26.043s
2026-09-12T16:49:48  R-20260912-164007-6f27ea  ladder                 0    581.237s
2026-09-12T17:07:04  R-20260912-170211-f585ef  activation-capture     0    292.917s
```

One header naming the model and its hash, `layers` `8,16,24`, the dataset, `examples` 258,
the output directory and the fold `threads=8 dot=widen4 batch=64`; one `model.load` phase; no
run directory, the `ls` failing as the scope decision predicts; and the catalog row for the
capture beside the day's `ladder` and `chat` rows. The catalog has no `task` row because no
`task` run was made today; the expected signal's "beside rows for `task` and `ladder`" assumed
one. The load took 5.487 s, not the "tens of seconds" the expected signal estimated: it agrees
with the 5.33 s and 5.70 s the ladder and the chat measured on the same warm page cache, so
the estimate was high and the distinction it draws, seconds against the fixture's
milliseconds, holds.

Status: VERIFIED 2026-09-12 by hand (gate-backed by `tests/hardware/test_capture_run_record.py`,
4 passed on the workstation the same day). Every signal present; the two mismatches above are
in the expected signal's wording, not the run.

## RHX.6 A failure states a remedy that runs as printed

Implementation: `rust-harness/harness/src/actionable.rs`,
`rust-harness/harness/src/main.rs`.

U7 made three failures print `Solution:` and `Command:` beneath the diagnosis, following the
shape every Makefile and gate script in this repository already uses. The sandbox asserts that
no constructor emits a `<placeholder>` and that every command begins with `harness` — but the
constructors are called with fixture paths and the command is never executed, so nothing has
ever checked that the remedy *works when typed*. A command naming real paths that still does
not run is precisely the failure the placeholder assertion was reaching for and cannot see.

```bash
MODEL=<the gemma-4-E4B checkpoint directory>
rm -rf /tmp/rhx6 && mkdir -p /tmp/rhx6 && cd /tmp/rhx6

# 1. a checkpoint that is not there
~/workspace/ai-experiments/rust-harness/target/release/harness \
  --model /mnt/ai-data/models/hf/absent --text-only "say ready"
echo "exit=$?"

# 2. run the Command: line from step 1 verbatim, whatever it printed

# 3. a probe of the wrong width against real weights: the E2B-it probe RH.9b
#    exported, 1536 wide, against E4B's 2560. Named, not globbed: the E4B
#    probe RH.16 step 4b built sits beside it and passes the width check.
#    The registry directory holds the trained probe.npz; RH.9c exported its
#    harness image to /tmp, so export in place first, as RH.16 step 4b did.
(cd ~/workspace/ai-experiments && make probe-export \
   ARGS="--probe /mnt/ai-data/interp/probes/rust-e2b-it-refusal-probe")
cp /mnt/ai-data/interp/probes/rust-e2b-it-refusal-probe/probe.safetensors \
   /mnt/ai-data/interp/probes/rust-e2b-it-refusal-probe/probe.json /tmp/rhx6/
~/workspace/ai-experiments/rust-harness/target/release/harness \
  --model "$MODEL" --text-only --probe /tmp/rhx6/probe.safetensors "say ready"
echo "exit=$?"

# 4. run that Command: line verbatim too
```

Expected signal: each failure ends with three lines on stderr — the tagged diagnosis, then
`  Solution:` and `  Command:` at two spaces of indent — with stdout empty. Step 1 prints
nothing else: the checkpoint is resolved before the load. Step 3 prints the header first,
because the width check sits after `CpuModel::load` in `main.rs`, so a wrong probe costs the
full load to discover; that cost is the reason `--status` exists, and the `Command:` line
should be the preflight that would have found it in under a second. Both `Command:` lines name
this run's real paths, contain no `<` placeholder, and **do what they claim when pasted into
the shell**: the first lists the directory that is missing, the second runs a preflight that
reports the same fault the run hit. Record the remedy lines and the output of running them.

Step 1's three lines have been observed already, in RH.13 (2026-09-06) and RH.16 step 3 of
[rust-harness.md](rust-harness.md); neither ran the `Command:` it printed. Steps 2 and 4 are
this entry's claim.

The first attempt on 2026-09-12 stopped at step 3's copy: the registry directory held only
`manifest.json` and `probe.npz`, because RH.9c had exported the E2B-it image to `/tmp` and
never into the directory, so the export step above was added to the command.

If a `Command:` line fails to run, that is the finding and the entry is `FAILED` — the
placeholder assertion passing while the command is unusable is exactly the gap this entry
exists to close.

Observed:

```
```

Status: PENDING

## RHX.7 One catalog serves the whole workspace

Implementation: `common/obs/runs.py`, `common/obs/catalog.py`,
`make/harness.mk`, `rust-harness/obs/src/lib.rs`.

Five binaries write run records under one `cli` name with the binary name as `command`, so one
catalog and one `runs` view cover the whole tool suite — symmetric with how training and interp
each own one. The sandbox pins the shape of a row; what it cannot pin is that a real log root
accumulating months of runs still renders, and that all five commands actually appear there.
The workstation corpus is the only place that question has an answer.

Every command has written at least one row by now: `task`, `chat` and `serve` from RH.8,
RH.11 and RH.14, `ladder` from RH.1b, `activation-capture` from RH.9a, `attention-capture`
from RH.36 and `layer-diff` from RH.21, all in [rust-harness.md](rust-harness.md). The entry
needs no run of its own before it.

```bash
cd ~/workspace/ai-experiments
make harness-runs ARGS='-n 20'
make harness-runs ARGS='--failed -n 10'

# every distinct command in the catalog
jq -r '.command' /mnt/ai-data/logs/harness/runs.jsonl | sort | uniq -c
```

Expected signal: a populated, column-aligned table rather than `No recorded runs`, with no
field falling back to `?`. The distinct commands include `task`, `ladder`,
`activation-capture` and whichever of `attention-capture` and `layer-diff` have run; `chat` and
`serve` appear if they ever have. No row's command is `runs` — that name is reserved, because
`common/obs/catalog.py` hides such rows, and a binary named it would log correctly and be
invisible. Browsing writes no records of its own and prunes nothing.

Observed:

```
```

Status: PENDING

## RHX.8 Status output is coloured at a terminal and plain through a pipe

Implementation: `rust-harness/runtime/src/status.rs`, `common/output.py`,
`rust-harness/runtime/tests/status_python_parity.rs`.

This is the one entry with no scale argument: it needs a workstation only because it needs a
TTY. Every status assertion in the suite runs behind a pipe and asserts the *absence* of escape
bytes — `cli_surface.rs` and `status_streams.rs` exist to keep colour off stdout and out of
recorded output. So `color_enabled()`, which is `NO_COLOR` absent **and**
`stderr().is_terminal()`, has never had its true arm observed by anything in this repository.
The colour codes themselves are unit-tested against a pure function that no terminal has read.

```bash
cd ~/workspace/ai-experiments/rust-harness
MODEL=<the gemma-4-E4B checkpoint directory>

# 1. at the terminal: the report is coloured. Look, do not pipe.
cargo run --release -q -p harness -- --model "$MODEL" --text-only --status

# 2. NO_COLOR set, same terminal: identical text, no escapes
NO_COLOR=1 cargo run --release -q -p harness -- --model "$MODEL" --text-only --status

# 3. through a pipe with no NO_COLOR: also plain, because stderr is not a tty.
#    grep -c prints 0 and exits 1 on the zero count this step expects.
cargo run --release -q -p harness -- --model "$MODEL" --text-only --status 2>&1 \
  | cat -v | grep -c '\^\['

# 4. the same three arms again, captured as bytes rather than read by eye.
#    `script` allocates a pty, so 4a and 4b run against a real terminal; see
#    the note below for why a plain redirect cannot stand in for it.
cargo build --release -q -p harness   # so no build output lands in a capture
RUN_STATUS="cargo run --release -q -p harness -- --model $MODEL --text-only --status"

# 4a. a terminal, NO_COLOR unset: the true arm of color_enabled()
script -qec "$RUN_STATUS" /dev/null | sed 's/\r$//' > /tmp/rhx8-tty-color

# 4b. the same terminal, NO_COLOR set: colour off by the environment variable
NO_COLOR=1 script -qec "$RUN_STATUS" /dev/null | sed 's/\r$//' > /tmp/rhx8-tty-plain

# 4c. NO_COLOR unset, stderr redirected: colour off by the file descriptor
sh -c "$RUN_STATUS" >/dev/null 2>/tmp/rhx8-pipe-plain

# 4d. the three comparisons
cat -v /tmp/rhx8-tty-color | grep -c '\^\['
diff /tmp/rhx8-tty-plain /tmp/rhx8-pipe-plain && echo "plain arms identical"
sed 's/\x1b\[[0-9;]*m//g' /tmp/rhx8-tty-color | diff - /tmp/rhx8-tty-plain \
  && echo "colour strips to the same bytes"
```

**The pty in step 4 is load-bearing; do not replace it with a redirect.** `color_enabled()` is
`NO_COLOR` absent **and** `stderr().is_terminal()`, so redirecting stderr to a file forces the
false arm no matter what `NO_COLOR` holds. A step 4 whose arms are both redirects compares two
plain captures, reports them identical, and observes neither the colour branch nor the
`NO_COLOR` branch — it passes without testing anything. `script -qec '<cmd>' /dev/null` gives
the child a pty and still lets the session be captured, which is the only way 4a reaches the
branch this entry exists for. The `sed 's/\r$//'` strips the CR that the pty line discipline
adds; that CR is `ONLCR`, not harness output, and 4b would otherwise differ from 4c on every
line. Both `script` arms merge stdout into the capture, which is safe here because `--status`
renders entirely through `runtime::status` and writes nothing to stdout. Both `script` and
`sh -c` take one command string, so `RUN_STATUS` embeds `$MODEL` unquoted; a checkpoint path
containing a space needs quoting inside that string.

Expected signal: step 1 shows green `[OK]`, blue `[SKIP]`, and — if any check warns — **bold**
yellow `[WARN]`, with the colour on the bracketed tag only and never on the message or on an
indent. Steps 2 and 3 are plain text. Step 3 counts zero escape sequences. Step 4a's capture
holds one escape-bearing line per tagged check, so 4d's `grep -c` matches the number of tagged
lines and is never zero. The plain arms 4b and 4c are byte-identical, which is the property
that makes `grep` work the same on a pipe as at a terminal. Stripping the SGR escapes from 4a
yields exactly 4b, which is what "colour is display only" means as bytes: the two arms of
`color_enabled()` differ by escapes and by nothing else.

Note a divergence recorded rather than fixed: `NO_COLOR=` set but empty turns colour **off**
here and leaves it **on** in `common/output.py`, which checks truthiness. Rust matches
no-color.org; Python is the off-spec side. Not exercised by this entry, and named so the next
reader does not report it as new.

Observed:

```
```

Status: PENDING

## RHX.9 The --dot help text describes the kernels the flag offers

Implementation: `rust-harness/runtime/src/bin/ladder.rs` (the `--dot` flag),
`rust-harness/runtime/src/model/cpu.rs` (`DotKind::is_bit_identical`).

**This entry records a defect and the check that closes it.** `--dot` help currently reads
"the dot-product kernel; every kind is bit-identical". Measured on the real fixture this week,
`lanes16` is not: at logit 0 the two kernels return `1062660370` against `sequential`'s
`1062660362` as raw f32 bits. The predicate in the code is right —
`DotKind::is_bit_identical` is `self != DotKind::Lanes16`, and the ladder's invariance tests
are asserted over exactly the set it returns. The sentence a person reads before choosing a
kernel is the part that is wrong, and it is wrong in the direction that costs a debugging
session: a run that picks `lanes16` on the strength of it and then compares logits bits
against a sequential baseline reports a defect that is not there.

A user-surface entry rather than a numerics one because the numerics are already pinned:
the two bit patterns above are what
`rust-harness/runtime/src/model/cpu.rs::the_lane_dot_changes_the_logits_and_says_so` measures
on the tiny reference fixture, and `rust-harness/runtime/tests/ladder_flags.rs` sweeps the
ladder's invariance over exactly the kinds the predicate returns (RH.24 in
[rust-harness.md](rust-harness.md)). The ladder itself prints a `PASS` or `FAIL` per rung and
never a logit's bits, so a climb under each kernel cannot show the divergence: every kind is
within tolerance of the reference, `lanes16` included, and six `PASS` rows read as the help
text's claim confirmed. What no test can assert is whether the help text tells the truth, and
that is what a person reads.

```bash
set -o pipefail
cd ~/workspace/ai-experiments/rust-harness

# 1. what the flag claims
cargo run --release -q -p runtime --bin ladder -- --help | grep -A1 -- '--dot'
echo "exit=$?"

# 2. the measurement the claim has to agree with, printed by the test that pins it
cargo test --release -p runtime the_lane_dot_changes_the_logits_and_says_so -- --nocapture 2>&1 \
  | tail -5
echo "exit=$?"
```

Expected signal: (1) the help text names which kinds are bit-identical and which are not,
rather than asserting that all of them are. The corrected sentence must exclude `lanes16` by
name — a hedge such as "most kinds" leaves the reader to guess which, which is the same
failure in a quieter voice. (2) The test passes, which is the measurement standing: `lanes16`
changes the logits and the predicate says so.

The entry passes when the text and the measurement agree. It does not pass on the measurement
alone: the measurement is already true today, and the defect is that the documentation
contradicts it. The code change to the `--dot` entry of `SPEC` in
`rust-harness/runtime/src/bin/ladder.rs` landed on 2026-09-12: the sentence now reads
"the dot-product kernel; every kind but lanes16 is bit-identical". Step 1 observing it on
the workstation build is what closes the entry.

Observed:

```
```

Status: PENDING

## RHX.10 A tool call the rules do not decide asks, and the run says so first

Implementation: `rust-harness/harness/src/permissions.rs`,
`rust-harness/harness/src/main.rs` (`ask_stdin`, the `permissions.yaml` load),
`rust-harness/harness/src/agent.rs`.

Task mode reads `permissions.yaml` from the working directory; a call no rule matches, or
every call when the file is absent, goes to `ask_stdin`, which prints
`allow <tool> <target>? [y/N]` on stderr and reads one line. RH.8 ran the only rule set this
campaign has ever executed, `tool: "*"` allow, and RH.37 met the prompt three times with
nothing announcing it: three minutes into a run that had printed only prefill lines, the
terminal stopped, and the operator had to work out that a base model had invented a `read`.
The sandbox pins the rule semantics (`permissions.rs` unit tests) and the loop's recovery from
a denial
(`rust-harness/harness/tests/agent_e2e.rs::denied_tool_call_returns_to_the_model_and_the_loop_recovers`);
what
it cannot show is the prompt at a terminal, an answer typed at it, and a closed stdin.

Since 2026-09-12 the run states its rule source before the loop starts:
`Permissions: N rules from permissions.yaml (an unmatched call asks)` or
`Permissions: no permissions.yaml, every tool call asks`
(`rust-harness/harness/tests/session_status.rs`). The subject is E2B-it, which follows the
tool contract; a base model's calls are invented and would make the answers meaningless.

```bash
set -o pipefail
rm -rf /tmp/rhx10 && mkdir -p /tmp/rhx10 && cd /tmp/rhx10
printf 'rules:\n  - tool: read\n    action: allow\n  - tool: bash\n    action: deny\n' \
  > permissions.yaml
printf 'hello\n' > hello.txt

# 1. at the terminal: read is allowed silently, write matches no rule and asks.
#    Answer n at the prompt.
~/workspace/ai-experiments/rust-harness/target/release/harness \
  --model /mnt/ai-data/models/hf/gemma-4-E2B-it --text-only \
  "Read hello.txt, write its contents to copy.txt, then say done"
echo "exit=$?"
cat session.jsonl
ls

# 2. stdin closed: the same call resolves to a denial with no prompt and no wait
rm -f session.jsonl session.kv copy.txt
~/workspace/ai-experiments/rust-harness/target/release/harness \
  --model /mnt/ai-data/models/hf/gemma-4-E2B-it --text-only \
  "Read hello.txt, write its contents to copy.txt, then say done" </dev/null
echo "exit=$?"
grep -c 'permission denied' session.jsonl

# 3. no file at all: the header says every call asks
rm -f permissions.yaml session.jsonl session.kv
~/workspace/ai-experiments/rust-harness/target/release/harness \
  --model /mnt/ai-data/models/hf/gemma-4-E2B-it --text-only "say done" </dev/null
echo "exit=$?"
```

Expected signal: steps 1 and 2 print `[INFO] Permissions: 2 rules from permissions.yaml (an
unmatched call asks)` before `Model load:`; step 3 prints
`[INFO] Permissions: no permissions.yaml, every tool call asks`. In step 1 the `read` produces
no prompt and its result is a tool entry holding `hello`; the `write` stops the run at
`allow write copy.txt? [y/N]` on stderr with the cursor on that line, and `n` puts
`permission denied: write copy.txt` into the transcript as a tool entry, after which the model
continues and `copy.txt` does not exist. Step 2 shows the same denial entry with no prompt
printed and no pause. Whether E2B-it then says `done` or keeps trying is the model's business,
not the harness's; the entry is about the three lines the harness prints and the file it does
not write.

Observed:

```
```

Status: PENDING

## RHX.11 A resumed session says what it resumed

Implementation: `rust-harness/harness/src/main.rs`, `rust-harness/harness/src/transcript.rs`.

The harness resumes `session.jsonl` from the working directory, and until 2026-09-12 said
nothing when it did. A clean exit also saves `session.kv` and the next start prints
`Session restored: N cached tokens`, so a resumed *cache* announced itself; a run that exited
non-zero saved no snapshot, and its successor re-prefilled the whole prior transcript with
only the token count to hint at why. RH.37's second run read `Prefill: 2504 tokens` where the
first had read 252. The line `Session: resumed N entries from <path>` now precedes the load
(`rust-harness/harness/tests/session_status.rs`), and this entry reads it on a real session:
two chat runs in one directory, the second continuing the first.

```bash
set -o pipefail
MODEL=<the gemma-4-E4B checkpoint directory>
rm -rf /tmp/rhx11 && mkdir -p /tmp/rhx11 && cd /tmp/rhx11

# 1. a fresh directory: no resume line, a snapshot saved at exit
printf '%s\n' "The workstation has an RTX 5090 and a Ryzen 9950X." \
  | ~/workspace/ai-experiments/rust-harness/target/release/harness --model "$MODEL" --text-only --chat --max-tokens 16
echo "exit=$?"
wc -l session.jsonl

# 2. the same directory: the resume line, the restored cache, a delta prefill
printf '%s\n' "Which GPU does it have?" \
  | ~/workspace/ai-experiments/rust-harness/target/release/harness --model "$MODEL" --text-only --chat --max-tokens 16
echo "exit=$?"
```

Expected signal: step 1 prints no `Session:` line and ends with
`[INFO] Session snapshot: <n> cached tokens at /tmp/rhx11/session.kv`; `session.jsonl` holds
two entries. Step 2 prints `[INFO] Session: resumed 2 entries from /tmp/rhx11/session.jsonl`
before `Model load:`, then `[INFO] Session restored: <n> cached tokens from
/tmp/rhx11/session.kv`, and its `Prefill:` line counts the new turn alone rather than the
transcript. The reply's content is immaterial on a base model.

Observed:

```
```

Status: PENDING

## RHX.12 A chat at a terminal shows where the next turn goes

Implementation: `rust-harness/harness/src/chat.rs`.

`--chat` reads lines from stdin and streams each reply to stdout. Every chat this campaign has
run piped its turns (RH.11, RH.20, `tests/hardware/test_interp_chain.py`), where a prompt
marker would be noise. At a terminal the loop prints nothing between the end of a reply and
the operator's next line: no marker, no blank line, and the cursor sits wherever the reply's
last token left it. Whether a person can tell the harness is waiting has never been looked at,
and `chat.rs` has no code that could make it so. This entry looks; its expected signal is the
property, and today's code is expected to fail it.

```bash
set -o pipefail
MODEL=<the gemma-4-E4B checkpoint directory>
rm -rf /tmp/rhx12 && mkdir -p /tmp/rhx12 && cd /tmp/rhx12

# At the terminal, no pipe. Type each line and wait for the reply before the next:
#   hello
#   /quit
~/workspace/ai-experiments/rust-harness/target/release/harness --model "$MODEL" --text-only --chat --max-tokens 16
echo "exit=$?"
```

Expected signal: after the header, the operator can tell from the screen alone that the
harness is waiting for a turn, and after each reply can tell where it ended and where the next
turn begins. `/quit` ends the session with `[INFO] Chat ended: 1 model turns` and
`[INFO] Session snapshot: ...`, exit 0. Record the screen as pasted, including the typed
lines; a `FAILED` here is the finding the entry exists for, and the fix is a marker on stderr
so piped stdout stays the reply alone.

Observed:

```
```

Status: PENDING
