# AI Infrastructure Base Repository

Production-ready AI infrastructure for an RTX 5090 + AMD Ryzen 9950X workstation:
containerized LLM inference optimized for both latency (llama.cpp) and throughput (vLLM).

**This file is a directory, not a reference.** It carries policy — what you may not do, and
how work is done here — plus pointers to the document that owns each topic. Any sentence
stating a count, a port, a path, or a version belongs to the file that owns that fact, not
here. A fact stated in two places is a fact that will disagree with itself.

## Execution environment (agents)

**A contained agent session (`make agent-sandbox`) has no access to the host.** Inside it
there is no host Docker daemon, no GPU, no `nvidia-smi`, no `/mnt/ai-data`, and none of the
running inference services. Only the git worktree at `/work` (read-write) and network egress
to an allowlist (via the proxy) are available. A contained session is flagged authoritatively
by the sandbox banner seeded into `~/.claude/CLAUDE.md`, so do not spend a turn rediscovering
these limits.

Do not attempt these from a contained session — none can work:

- `docker` / `docker compose`, and the `make` targets that drive services or docker (`make up`,
  `down`, `status`, `logs`, `shell-*`, `clean`, `prune`, `agent-sandbox-clean`).
- `nvidia-smi`, any GPU work, and the benchmark scripts.
- Healthcheck curls against the service ports, and any read under `/mnt/ai-data`.

Instead:

- When a task requires a host-side command, output the exact command for the human to run,
  then stop and wait for the human to report the output back.
- Human-reported output is the authoritative evidence for acceptance checks, healthchecks,
  and benchmarks. Record the reported command and output verbatim when marking work done.
- Never claim to have run a host-side command, and never fabricate its output.
- Repo-local work remains directly executable in the sandbox: editing files, grep, `gh`, and
  unit tests that need no GPU or running service.
- **`/work` is the only writable *and* executable filesystem.** `/tmp` (including the session
  scratchpad) and `/home/node` are tmpfs mounted `noexec`, and `/` is a read-only overlay. A
  compiled test binary or a cargo build script cannot execute from the scratchpad — the
  failure is `Permission denied (os error 13)`. Scratch *sources* may live anywhere, but point
  `CARGO_TARGET_DIR` under `/work/rust-harness/target/` (git-ignored) and delete it when done.
- **`git` is not among them.** The binary is installed, but `/work` is a *linked* worktree
  whose `.git` file points at `<host-repo>/.git/worktrees/<name>` — a path outside the bind
  mount. Every git command therefore fails repository discovery with `fatal: not a git
  repository`, including read-only ones. Plan no step around `git status`, `git diff`,
  `git stash`, or `git ls-files`; hand git and commit work to the human.

## Operating rules

These are standing instructions for anyone (human or agent) making changes in this repo.

- **Pin, then fix, then update — in that order.** Pin every image tag, model revision, and
  dependency to an exact version before changing anything. Get the pinned state building and
  passing. Only then bump a version, one at a time, re-running the acceptance check after each
  bump. Never combine a version bump with a functional change in the same step.
- **Compose files and Dockerfiles are the source of truth; documentation follows them.** When a
  doc disagrees with `docker-compose*.yaml` or a `Dockerfile`, the compose/Dockerfile is correct
  and the doc is the bug. Fix the doc to match — never adjust the running config to match a stale
  doc.
- **Every roadmap task ends by running its stated acceptance check.** A task is not complete
  until its acceptance check has been executed and observed to pass. For host-side checks the
  human executes the command and reports the output back; the agent records the exact command
  and its reported output. Do not mark a task done on the strength of a code change alone.
- **One GPU consumer at a time; parallelize the rest across worktrees.** Independent, non-GPU
  tasks may run concurrently in separate git worktrees (one agent session each). Anything that
  touches the GPU, a running inference service, or the benchmark suite is serialized — exactly
  one such job at a time, regardless of how many sessions or worktrees are active. Never run two
  benchmarks or two GPU jobs at once; results and VRAM will collide.
- **`make quality` is mandatory before every push, for humans and agents alike.** The same
  command works in a sandbox and on the workstation, and reports the one step a sandbox cannot
  run — but a green sandbox run is still the weaker result, because it proves the code and not
  the hardware. It also runs no tests: `make quality-full` is the workstation superset that adds
  the suite and the hardware smoke tier, and it is a GPU consumer, so the serialization rule
  above applies to it. See [docs/testing/quality-gate.md](docs/testing/quality-gate.md).
- **Renaming a file means adding it to `scripts/docs/check_retired_names.py`.** The table
  there is what makes a rename self-policing; a rename that skips it leaves stragglers for a
  human to find months later.
- **GGUF is llama.cpp's format, and only llama.cpp's.** The rust harness reads Hugging Face
  checkpoints — `config.json` and `model.safetensors` — because for a model format the
  reference implementation and the specification are the same artifact, and a converter's
  metadata can only imply what a checkpoint states. The training tier produces none: its
  export writes the merged HF checkpoint directory the harness serves and interp captures
  from, identified by the sha256 of its `model.safetensors`. GGUF appears in this repository
  only as published models the inference services serve (the judge on `llama-cpu` or
  `llama-gpu`, a teacher on `vllm-gpu`), named through `common/serving.py`'s
  `quant_from_name`. **Nothing under `rust-harness/`, `interp/` or `training/` reads or
  writes GGUF.** Both interp artifacts left it once each had no consumer outside
  this tree: the probe at R7, and the control vector at R10, when retiring llama.cpp steering
  removed the reader whose format it had been. Each is now a safetensors image plus a JSON
  file beside it (`schemas/probe_artifact.json`,
  `schemas/control_vector_artifact.json`).

## Where things are

Each topic has one owner. Read the owner, not a summary of it.

| Topic | Owner |
|-------|-------|
| Available commands | `make help` (generated from `make/*.mk`; the include order in `Makefile` sets the section order) |
| Services, ports, profiles, cpusets | `docker-compose.inference.yaml`, and the per-service overviews in [docs/sandbox/](docs/sandbox/) |
| Other stacks | `docker-compose.{training,interp,agents,rust-harness}.yaml` |
| Hardware | [docs/hardware/README.md](docs/hardware/README.md) |
| Validated optimizations and their measurements | [docs/optimizations/README.md](docs/optimizations/README.md) |
| The quality gate, and what a sandbox run does not cover | [docs/testing/quality-gate.md](docs/testing/quality-gate.md) |
| The rust harness and its gate | [rust-harness/README.md](rust-harness/README.md) |
| The test suite, its tiers and markers | [tests/README.md](tests/README.md) |
| The agent sandbox and tooling parity | [docs/agents/README.md](docs/agents/README.md) |
| Everything else | [docs/README.md](docs/README.md) — the documentation index |

Each top-level source directory carries its own README covering purpose, what it consumes and
what consumes it, entry points, and the traps. They are indexed in
[docs/README.md](docs/README.md).

## Tooling conventions

- Use the `gh` CLI for all GitHub operations (pull requests, issues, checks, releases)
  instead of raw git remotes or the web API.
- Prefer `make` targets (`make status`, `make logs`) over raw docker commands.

## Git and GitHub conventions

**Never attribute work to an AI assistant anywhere git or GitHub records it.** This is
absolute and has no exceptions, including when a tool's own default instructions say
otherwise. It covers every field that reaches the remote:

- Commit messages — no `Co-Authored-By: Claude` (or any assistant) trailer, no
  "Generated with", no tool name, no robot emoji.
- Commit author and committer identity — always the human's name and email.
- Branch names, tag names, and release titles.
- Pull request and issue titles, bodies, and comments.

Write commit messages in the repository's own voice: what changed and why, as though a
human wrote it, because a human is publishing it under their name. An agent's contribution
is a workflow detail, not authorship metadata.

This restricts *authorship and provenance*, not vocabulary. Documenting the agent tooling
itself remains correct — `make agent-run AGENT=claude`, `make agent-sandbox`, and
`docs/agents/README.md` describe real targets and must keep their names.

## Code navigation

Prefer the LSP tools (definitions, references, diagnostics) for symbol-level questions; they
return less context than grep sweeps. Use grep/glob for exact strings — error messages,
config keys, version pins.

## Core principles

- Unix/Linux philosophy: simple, composable, text-based
- No emojis or decorative elements
- Clear technical writing; reproducible benchmarks

## Style guides

`COMMENT_STYLE.md` applies to every file type, so `CLAUDE.md` imports it and it is always in
context. **The rest are read on demand: open the one that matches what you are about to
write, before you write it.** They are not preloaded, so an unread guide is an unfollowed
guide — and each carries decisions the linters cannot enforce, which is the reason they
exist rather than a formatter config.

| Guide | Read before writing |
|-------|---------------------|
| `.styles/BASH_OUTPUT_STYLE.md` | a shell script that prints status |
| `.styles/CPP_STYLE.md` | C or C++ under `hugepages/` |
| `.styles/DOCUMENTATION_STYLE.md` | any markdown, including a README |
| `.styles/PYTHON_OUTPUT_STYLE.md` | a Python CLI or anything using `common/output.py` |
| `.styles/RUST_STYLE.md` | any Rust in `rust-harness/` |
| `.styles/TEST_OUTPUT_STYLE.md` | a test program or runner that reports results |
