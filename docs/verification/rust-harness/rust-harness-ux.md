# Rust harness user surface verification

**Operational runbook. Executed against the workstation, not read for understanding: the
operator's pasted output is the evidence. Keep the commands runnable, and the paths accurate —
both documentation gates check this directory, so a stale reference here fails the gate.**

**Entries: 0 verified, 0 blocked, 8 pending, of 8.**

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

## Preconditions

**A real Gemma 4 E4B checkpoint** under `/mnt/ai-data/models/hf/`, named as `$MODEL` in every
command block below. Substitute the directory; the entries do not assume its exact name.

**A release build.** Every block runs from `~/workspace/ai-experiments/rust-harness` and uses
`cargo run --release`, so the first entry absorbs the build and the rest are load-bound. A
debug build makes every duration in this document meaningless.

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

```bash
cd ~/workspace/ai-experiments/rust-harness
MODEL=<the gemma-4-E4B checkpoint directory>

cargo run --release -q -p harness -- --model "$MODEL" --text-only \
  --seed 7 --temperature 0.8 --max-tokens 16 \
  "Reply with the single word: ready"
echo "exit=$?"

# the record the terminal was rendered from
RUN=$(ls -t /mnt/ai-data/logs/harness/R-*.jsonl | head -1)
echo "$RUN"
jq -c 'select(.event == "run.header")' "$RUN"
```

Expected signal: nine `[INFO]` lines before the answer, naming the model and its hash, the
layer count, the resident weight size and dtype, peak RSS after load, the KV cache budget, the
fold (`dot=`, `batch=`, `threads=`), the sampling parameters including `seed=7`, and the build.
The `dot` value is the tuned kernel, not the reference one. The `run.header` record carries the
same facts as JSON numbers rather than formatted strings — `n_layers` and `weight_bytes` are
numbers, `temperature` renders as `0.8` and not `0.80000001`.

Record both the nine lines and the `run.header` record. The point of the entry is that they
agree.

Observed:

```
```

Status: PENDING

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
cd ~/workspace/ai-experiments/rust-harness
MODEL=<the gemma-4-E4B checkpoint directory>

# a task run: one load line, one prefill line
cargo run --release -q -p harness -- --model "$MODEL" --text-only \
  --max-tokens 32 "Summarize the Unix philosophy in three sentences"

# turn 2 against the same session: the prefill line is the delta, not the transcript
cargo run --release -q -p harness -- --model "$MODEL" --text-only \
  --max-tokens 32 "Now say that again in one sentence"

# the printed load figure and the recorded one are the same clock
RUN=$(ls -t /mnt/ai-data/logs/harness/R-*.jsonl | head -1)
jq -c 'select(.event == "phase.end" and .phase == "model.load")' "$RUN"
```

Expected signal: `[INFO] Model load: <n> s` reporting tens of seconds, and
`[INFO] Prefill: <n> tokens in <n> s (<n> tok/s)` with a rate worth having. On turn 2 the
prefill token count is a small fraction of turn 1's, because the cache covered the rest —
that number is the only visible evidence delta prefill works. The `duration_s` in the
`phase.end` record agrees with the printed load figure to two decimals. Neither stream carries
a carriage return: the animation was deliberately not built, so an `Observed` block gets clean
lines.

Observed:

```
```

Status: PENDING

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

Observed:

```
```

Status: PENDING

## RHX.4 The training dashboard renders a harness run

Implementation: `training/commands/report.py`, `common/obs/dashboard.py`,
`common/obs/viewer.js`, `schemas/run_metrics.json`.

U6's claim is that the harness became a first-class observability tier rather than a second
system: it writes a `metrics.jsonl` in the shape the training dashboard already reads, so the
dashboard renders a harness run **with no Python change**. That claim is testable in the
sandbox at the contract level, and this entry is the part that is not — a human looking at the
page and finding it populated and correct rather than merely valid.

Run RHX.1 or RHX.2 first; this entry opens the run directory that produced.

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

Observed:

```
```

Status: PENDING

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

Observed:

```
```

Status: PENDING

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
cd ~/workspace/ai-experiments/rust-harness
MODEL=<the gemma-4-E4B checkpoint directory>

# 1. a checkpoint that is not there
cargo run --release -q -p harness -- --model /mnt/ai-data/models/hf/absent \
  --text-only "say ready"
echo "exit=$?"

# 2. run the Command: line from step 1 verbatim, whatever it printed

# 3. a probe of the wrong width against real weights
mkdir -p /tmp/rhx6 && cp /mnt/ai-data/interp/probes/*/probe.safetensors \
  /mnt/ai-data/interp/probes/*/probe.json /tmp/rhx6/ 2>/dev/null || \
  echo "no exported probe: run RH.16 step 4b first"
cargo run --release -q -p harness -- --model "$MODEL" --text-only \
  --probe /tmp/rhx6/probe.safetensors "say ready"
echo "exit=$?"

# 4. run that Command: line verbatim too
```

Expected signal: each failure prints exactly three lines on stderr — the tagged diagnosis, then
`  Solution:` and `  Command:` at two spaces of indent — with stdout empty. Both `Command:`
lines name this run's real paths, contain no `<` placeholder, and **do what they claim when
pasted into the shell**: the first lists the directory that is missing, the second runs a
preflight that reports the same fault the run hit. Record the remedy lines and the output of
running them.

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

Run RHX.1 through RHX.5 first, so all five commands have written at least one row.

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

# 3. through a pipe with no NO_COLOR: also plain, because stderr is not a tty
cargo run --release -q -p harness -- --model "$MODEL" --text-only --status 2>&1 \
  | cat -v | grep -c '\^\[' 

# 4. the bytes of 2 and 3 are the same
NO_COLOR=1 cargo run --release -q -p harness -- --model "$MODEL" --text-only \
  --status 2>/tmp/rhx8-nocolor
cargo run --release -q -p harness -- --model "$MODEL" --text-only \
  --status 2>/tmp/rhx8-piped
diff /tmp/rhx8-nocolor /tmp/rhx8-piped && echo "identical"
```

Expected signal: step 1 shows green `[OK]`, blue `[SKIP]`, and — if any check warns — **bold**
yellow `[WARN]`, with the colour on the bracketed tag only and never on the message or on an
indent. Steps 2 and 3 are plain text. Step 3 counts zero escape sequences. Step 4 reports the
files identical, which is the property that makes `grep` work the same on a pipe as at a
terminal.

Note a divergence recorded rather than fixed: `NO_COLOR=` set but empty turns colour **off**
here and leaves it **on** in `common/output.py`, which checks truthiness. Rust matches
no-color.org; Python is the off-spec side. Not exercised by this entry, and named so the next
reader does not report it as new.

Observed:

```
```

Status: PENDING
