# Quality gate

`make quality` is the pre-push gate, mandatory for humans and agents alike. It is the same
command on the workstation and in a contained agent sandbox; this document covers what it
runs, and what a sandbox run does not cover. `make quality-full` is its workstation superset,
adding the test suite and the hardware smoke tier.

The recipe in `make/quality.mk` is the source of truth for the step list. When this document
disagrees with it, the recipe is right.

## What it runs

Thirteen steps, cheapest first, failing on the first error:

1. `ruff format --check .` — formatting.
2. `ruff check .` — lint.
3. `mypy` — types.
4. `uv lock --check` — the lockfile matches `pyproject.toml`.
5. `scripts/docs/check_doc_refs.py` — fails when a document names a repository path that
   does not exist.
6. `scripts/docs/check_retired_names.py` — fails when anything still uses a name that was
   deliberately retired, and names the replacement.
7. `scripts/docs/check_make_targets.py` — fails when a documented `make <target>` does not
   resolve, reading invocations from fenced blocks and code spans only.
8. `scripts/docs/check_doc_counts.py` — compares a document against itself, so a heading
   reading `## Open (3)` must have three rows beneath it.
9. `scripts/docs/check_kb_refs.py` — fails when a known-bug id cited in source has no ledger
   row, or the reverse.
10. `scripts/docs/check_runbook_entries.py` — fails when a verification entry names a test
    function that does not exist, or is marked `VERIFIED` with neither pasted evidence nor a
    named check.
11. `scripts/docs/check_surface_census.py` — enumerates the rust-harness binaries and flags
    from their `Flag` literals and fails when one is named in no runbook. The only gate here
    that starts from the code rather than the documentation.
12. `scripts/docs/check_group_pin_parity.py` — fails when one package carries two different
    exact pins across `pyproject.toml`'s sections. The `sandbox` group deliberately re-pins
    what core, `interp` and `training` pin, and step 4 cannot see a disagreement between two
    literal `==` strings.
13. `scripts/docs/check_command_prompts.py` — fails when a `bash` (or `sh`, `shell`, `zsh`)
    block carries a `$ ` prompt marker, because a pasted `$ make quality` fails as a command
    named `$`. Reads neither `console` transcripts nor a runbook's Observed blocks, and skips
    frozen records.
14. `shellcheck --severity=warning` — shell scripts, enumerated with `find`.

Note what is absent: **`make quality` does not run pytest.** See
[The full gate](#the-full-gate) below.

It then chains the rust-harness gate when that directory exists. That half is documented in
[../../rust-harness/README.md](../../rust-harness/README.md): its five steps, the
cross-language contract tests, `runtime/src/pyenv.rs` interpreter resolution and
`HARNESS_PYTHON`, and why a green run must be read for `[SKIP]` lines rather than trusted by
exit code. It is not repeated here. One of its skips is narrower than it reads: the Chrome
trace check skips only its workstation half, and asserts the same span relations against a
fixture chat trace it writes itself on every run. The run-level records are asserted the same
way: `obs_records.rs` spawns the binary for a success and a forced failure and checks the
catalog rows and event files they leave, so the workstation half of RH.13 adds only the default
log root and a real checkpoint; that half is `tests/hardware/test_obs_records.py`, under
`hw_full`. The endpoint has the same split: `serve_snapshot.rs` and `server.rs` in the
sandbox, and `tests/hardware/test_harness_endpoint.py` for a real checkpoint behind the
socket and the client's own health budget. So does the preflight (RH.16):
`preflight_surface.rs` and `probe_identity.rs` over the fixture in the sandbox, and
`tests/hardware/test_preflight.py` under `hw_smoke` for `--status --deep` against the real
weights, where it also prints what the hash costs. And the alignment artifact (RH.19):
`alignment_artifact.rs` holds the written file to `schemas/alignment.json` over the fixture,
and `tests/hardware/test_alignment.py` under `hw_e4b` runs the harness over the real
vocabulary and decodes every row's `token_id` with `transformers`, holding each span to the
bytes its own token produced; the sandbox half can only prove the spans partition the text.
Compaction past the rewind floor (RH.20) splits the same way: `session.rs` reaches the floor on
a four-position synthetic window in the sandbox, and `tests/hardware/test_compaction.py` under
`hw_e4b` drives a chat session over the real 512-token window until it compacts twice and reads
the rebuild off the `Prefill:` line each compaction leaves. And the residual stream
(RH.21): `layer_diff_numerics.rs` brackets `layer-diff`'s two tolerances around perturbed
rows of the tiny capture, and `tests/hardware/test_layer_diff.py` under `hw_e4b` reads every
row the tool prints over the real 42 layers, then closes the ceiling under two of them and
holds the verdict to the earlier one. And `checkpoint-info --check` (RH.22):
`checkpoint_info_check.rs` reaches both refusal paths in the sandbox, a config the backend
cannot validate and a tiny checkpoint made multimodal by one tensor per tower, and
`tests/hardware/test_checkpoint_info.py` under `hw_e4b` runs the pair of verdicts on the
released E4B and reads the tower count the code's comments predict off the file. And the
attention-capture binary (RH.36): `attention_capture.rs` pins its surface over a three-token
window in the sandbox, and `tests/hardware/test_attention_capture.py` under `hw_e4b` runs it on
the released E4B over a prompt past the real window, holding `span_first` to the schedule the
checkpoint's own `config.json` states and every row's mass to one over hundreds of keys.

Steps 5 and 6 are complementary rather than redundant. The reference check asks whether a
cited path exists, which requires classifying a token as a path claim — and that classifier
requires a slash, so a bare renamed filename slips past it. The retired-names check inverts
the question: an explicit table of retired names, failing on any use. Loosening the first to
cover the second was measured at a 67% false-positive rate, because prose legitimately says
`config.json` and `main.rs`.

Both carry an empty-set guard, as the shellcheck step does: enumerating zero files fails
loudly, because a gate that silently checked nothing is indistinguishable from one that
passed.

## One gate, two environments

Run `make quality` the same way in both. The difference is resolved inside `make/quality.mk`,
which reads the `AI_SANDBOX` marker the sandbox image sets:

- **Workstation** — tools come from `.venv` via `uv run --no-sync`, which runs what is already
  provisioned rather than re-resolving, so no step touches the network.
- **Sandbox** — tools are invoked from `PATH`. The image builds `/opt/pyenv` with
  `uv sync --frozen --only-group dev --only-group sandbox` against `uv.lock`, so the versions
  are the workstation's by construction rather than by transcription. There is no `.venv` to
  confuse matters: `docker-compose.agents.yaml` masks it with a tmpfs, so the workstation's
  venv is neither readable nor writable from inside.

One step cannot run in a sandbox, and says so rather than being silently absent:

```
[SKIP] uv lock --check: resolving the lock reaches the pytorch cu130 index, which the
       sandbox egress allowlist does not carry. The workstation gate covers it.
```

Everything else runs, including `make test` and the whole rust gate — the image carries the
toolchain pinned by `rust-harness/rust-toolchain.toml`, a prefetched crate registry so no step
needs the network, and numpy and gguf at the `uv.lock` pins. `runtime/src/npz.rs` shells out
to `np.load`, so `cargo test` fails outright without numpy rather than skipping. Rebuild the
image with `make agent-sandbox-build` after changing `docker/agents/Dockerfile.agents`,
`uv.lock`, or `rust-harness/Cargo.lock`.

Prefer `mypy --no-incremental` when the result matters: a warm `.mypy_cache` makes the
incremental run non-deterministic, and the same command can pass incrementally while failing
under `--no-incremental`.

What stays workstation-only is the lock check above, GPU work, the service healthchecks, and
anything under `/mnt/ai-data`. **Re-run `make quality` on the workstation before every push
regardless: the sandbox proves the code, not the hardware.**

## The full gate

`make quality` is the pre-push gate and stops at the ten steps above. It never runs pytest, and
`make test` deselects both hardware markers by default (`addopts` in `pyproject.toml`). So the
three commands cover three different amounts of ground, and the cheapest is the one everyone
reaches for. `make quality-full` is the one that runs all of it, on the workstation:

```
make quality-full            # quality, then the suite, then the hw_smoke tier
make quality-full HW=full    # ... and the hw_full tier as well
```

`HW=full` is opt-in because that tier may allocate near-capacity VRAM and load real models. It
is a separate variable from the `TIER=sandbox|full` of `make worktree-<name>`, which selects how
a venv is provisioned rather than which tests run.

Two refusals, both before any work is done. In a contained agent sandbox the target does not
run at all:

```
[ERROR] quality-full: this target requires the workstation
  Found: AI_SANDBOX is set, so there is no GPU, no /mnt/ai-data, and no running services.
  Solution: run the sandbox-executable half here, then run this on the workstation
  Command: make quality
```

And on a workstation whose environment is the torch-free tier, where the suite would skip its
way to green:

```
[ERROR] quality-full: torch is not importable, so the suite would skip its way to green
  Found: the environment is the torch-free sandbox tier; the run would report [TIER] sandbox
  Solution: provision the full tier
  Command: make install
```

That second check reads the same condition the suite itself reports on, at its source, rather
than scraping the printed `[TIER]` line — the line stays a message to the reader, not an
interface.

`quality-full` touches the GPU, so the serialization rule in
[../../AGENTS.md](../../AGENTS.md) applies: exactly one such job at a time across every
worktree, regardless of how many sessions are open. The target prints the reminder rather than
enforcing it.

## The correctness ladder

Neither gate measures whether the inference engine computes the right numbers. `make quality`
and `make quality-full` are machinery gates: they prove the code builds, the tests agree with
themselves, and the documentation resolves. A forward pass with a transposed weight or a wrong
norm convention passes both and emits fluent text.

That is what the correctness ladder is for. Seven rungs, each against `transformers` rather than
against any earlier output of ours, run on the workstation under their own target:

```
cd rust-harness
make ladder SIZE=quick MODEL=<checkpoint-dir>      # also standard, deep
```

It is deliberately outside `make quality`. The gate is mandatory before every push and has to
stay fast, and the ladder loads the real checkpoint. It is inside `make quality-full HW=full`:
`tests/hardware/test_ladder.py` drives the same target at `SIZE=quick` and fails on a red rung
or a skipped one. The reference artifacts are file prerequisites of the target, so a first run
generates them through `transformers` before climbing and later runs reuse them;
`make ladder-reference` rewrites them after a checkpoint or `transformers` change. The half that does run inside the gate is
Tier T: the same rungs against a tiny Gemma 4 that `transformers` itself built, committed
whole, measured in milliseconds.

The target passes `--text-only` by default, because the released E4B carries a vision and an
audio tower the CPU backend refuses to load and the ladder has no rung that could measure
either. `TEXT_ONLY=0` restores the refusal for a checkpoint that is text-only already.

`SIZE` selects how much work the perplexity and greedy rungs do. It is a third variable
alongside `HW` and `TIER` and means neither of them; the sizes are nested prefixes of one
corpus, so a `deep` run reproduces a `quick` run's number on its first tokens exactly.

Like `quality-full`, the target refuses in a contained sandbox before doing any work, and it
loads the checkpoint, so the one-GPU-consumer-at-a-time rule applies. See
[../../rust-harness/README.md](../../rust-harness/README.md) for the rungs and
[../../rust-harness/fixtures/reference/README.md](../../rust-harness/fixtures/reference/README.md)
for the reference artifacts.

## Provisioning

`make install` is the full provision (`--all-groups --all-extras`). A bare `uv sync` is a safe
repair: `pyproject.toml` names all four dependency groups in `[tool.uv] default-groups`
precisely so the reflexive command cannot strip an environment down.

Provision a worktree with `make worktree-<name> TIER=full` when it will run the training suite
or GPU work. The default `TIER=sandbox` is torch-free, and its torch-gated tests skip rather
than fail — so a green partial run looks like a pass. The suite states which it was, beside
the summary:

```
[TIER] sandbox: torch absent, so torch-gated tests skipped rather than ran. This is a
       partial run, not a pass.
[TIER] full: torch present, the whole suite was in scope.
```

Under `make quality-full` the distinction is enforced rather than reported: the torch probe
refuses the run before the suite starts, so a `[TIER] sandbox` line cannot appear in a green
`quality-full` output. Seeing one there means the probe and the suite disagree, which is a defect
in one of them.

## Related

- [../../rust-harness/README.md](../../rust-harness/README.md) — the rust half of the gate
- [../../tests/README.md](../../tests/README.md) — the pytest suite, its tiers, and markers
- [../agents/README.md](../agents/README.md) — the agent sandbox itself
