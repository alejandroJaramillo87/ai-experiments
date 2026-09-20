# The harness command line

What the `harness` binary and the seven utility binaries do with their arguments: the
generated `--help`, usage errors and their exit code, the `--status` preflight that
answers without loading the model, the three-line remedy a failing run prints, and the
two cost lines a run reports while it works. The flag-by-flag reference is the site's,
rendered from the same tables (`site/docs/rust-harness/reference/`); this page is the
behaviour those tables cannot state.

## Command line

`harness --help` is the authoritative list of flags, generated from the same
table the parser reads, so it cannot describe a flag the binary does not accept
or omit one it does. `harness --version` prints the build. Both write to stdout
and exit 0; a usage error goes to stderr and exits 2.

An unrecognized flag is a usage error rather than task text, so a misspelling
like `harness --modle m "do x"` stops rather than running a task named after the
mistake. Task text that legitimately begins with a dash goes after a bare `--`,
which ends flag parsing.

The other seven binaries answer `--help` and `-h` on the same terms, from their
own tables in `runtime/src/cli.rs`: help on stdout at exit 0, a usage error on
stderr at exit 2, and an unrecognized flag named rather than ignored. They do
not take `--version`. Two of the changes are behavioural rather than additive.
`activation-capture` and `attention-capture` used to scan for each flag by
position and read no further, so a misspelling was silently dropped — with a
required flag among the casualties they reported `missing required flag`, which
named neither, and with every required flag present they ran with the wrong
configuration and said nothing; their usage errors also now exit 2 rather than
sharing exit 1 with a failed capture. And `tiny-model` and
`long-context-golden` take their destination positionally, so until this each
one wrote its fixtures to a file named `-h` when asked for help.

`harness --model <dir> --status` reports what the run would fail on and exits
without loading the model: the checkpoint and the CPU backend's verdict on it,
each interp artifact given, and the directories the run writes to. It reads
`config.json` and the safetensors header only, so it answers in milliseconds
rather than the time a checkpoint takes to load. Add `--deep` to hash the
weights and check each artifact's recorded `model_hash` against them; that is
opt-in because it reads the whole file.

Three exit codes mean three different things, which is the reason to prefer
`--status` over starting a run to see whether it starts: **0** clean, or
warnings only; **1** something would stop the run; **2** the command line was
wrong to ask. The report goes to stderr and stdout stays empty. No run record
is written — a query is not a run, the same rule that keeps `--help` out of the
catalog.

A run that fails anyway states its fix. Three failures — a probe whose width
disagrees with the model, an interp artifact fitted on other weights, and a
`--model` path with no checkpoint at it — print two indented lines under the
error, a `Solution:` and a `Command:` naming this run's real paths rather than
a placeholder. The advice reaches the terminal only: the run record's `exc_msg`
stays the diagnosis alone, so a `fail` event is still one line of cause and not
a page of counsel.

A run that succeeds says what it waited on. The two operations that dominate a
cold start each report once when they finish — `Model load:` at the end of the
banner, and `Prefill:` with the token count and rate before the reply. Both
figures are read off the obs phase that already records them, so the terminal
and `<run_id>.jsonl` cannot disagree. The prefill count is the *delta* the
forward pass actually ran, not the prompt length, which is what makes a second
turn's small number the visible evidence that the KV cache is being reused.
When a rewind fell below a sliding layer's floor, the same line says so and
names the prefix it could not keep: the cache was rebuilt from zero, which is
the routine path after a compaction.
There is no progress animation and no terminal check: a pipe and a terminal get
the same bytes, so a runbook's pasted output never carries a carriage return.
