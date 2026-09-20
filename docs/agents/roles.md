# Agent roles

What each role in `.claude/agents/` is for, why the QA roles are shaped as they are, the
two settings files that decide what any of them may touch, and why host verification is a
skill rather than a role. The roster itself is generated in [README.md](README.md); this
page is the reasoning behind it.

## The two settings files

The effective Claude Code policy is the union of `.claude/settings.json` and
`.claude/settings.local.json`. Only the first is version-controlled. The local file is
git-ignored, so it is per-checkout state that never reaches a review, never reaches a
branch, and is destroyed with the worktree it grew in. `scripts/agents/preflight-scan.sh`
inspects it anyway, because being local is not the same as being trusted.

- `settings.json` holds standing policy: every `deny`, every `ask`, and the allows that
  describe how work is done in this repository regardless of who is doing it. Anything
  that should survive a worktree, or that anyone else should inherit, belongs here.
- `settings.local.json` holds only what is specific to one checkout or one image, such as
  an absolute interpreter path inside the sandbox.

An allow accretes in the local file whenever a prompt is approved with "don't ask again",
so it drifts toward being policy that nobody decided and nobody else can see. Re-read it
periodically and empty it: promote what is genuinely policy into `settings.json`, and
delete the rest. `tests/agents/test_settings.py` asserts the two do not overlap and that
the key-material denies survive whatever the allow list grows into; it treats an absent
local file as normal, because on a fresh clone that is the correct state.

## The quality campaign

Rungs 1 to 3 of the ladder do a task. Rung 4 audits a subsystem that nobody has a task
for: the shape of work where the deliverable is a findings ledger and a set of fixes, not
a feature. It costs orders of magnitude more than rung 3 and takes hours, so escalate to
it deliberately: when a subsystem feeds something else that would silently inherit its
defects, before trusting a component on real data for the first time, or after a large
merge. Do not reach for it because a bug is hard to find.

`/bugbash <scope>` drives the phase ladder. Seven roles carry it: `bug-hunter` discovers,
`skeptic` verifies, `triage-agent` packages, `remediator` fixes, `integrator` gates each
batch, and `perf-auditor` and `doc-auditor` own the two dimensions with their own
measurement rules. They share one definition of the terms in
`.claude/skills/bug-bash/SKILL.md`; read it before running one.

Two roles are deliberately unable to write. `integrator` is the independent check between
a batch and a commit, and `skeptic` adjudicates claims: an agent that can patch what it
finds stops being either. That is also why neither carries `memory` on the Claude side,
since enabling it grants Read, Write and Edit. `triage-agent` has no Bash for the same
reason it has no Edit: it orders the work, it does not judge whether a finding is real.

Two properties of the cadence matter more than the roles. Discovery is persisted into a
campaign log under `docs/verification/` and pushed before any budget goes to fixes, and
every remediation batch is pushed before the next begins, so an interrupted campaign loses
one batch rather than everything. Campaign 2 was stopped mid-flight by a usage limit and
survived only because the ledger had already landed. The mechanics that make a
multi-agent run survive to completion are the `agent-swarm` skill.

## Feature work

`feature-implementer` takes a stated requirement to working code. It is deliberately
distinct from `remediator`: a remediator closes a verified finding and must show the
defect existed by watching a test fail first, whereas feature work has no prior red to
reproduce. Its discipline comes instead from reading the governing `.styles/` guide before
writing, expressing the requirement as a test, and reporting the gate's skip count rather
than its exit code. It runs the Rust gate in the sandbox, because the image carries the
pinned toolchain, and it is required to name the parts its run did not exercise rather
than imply full coverage.

`doc-writer` is the same shape for prose: it reads `.styles/DOCUMENTATION_STYLE.md` before
writing a README or a page, and is the role `/audit-docs` findings are handed to for
repair, since `doc-auditor` reports drift and does not fix it.

## Measurement

`bench-runner` owns the benchmark protocol in the `benchmarking` skill and reports
reproducible numbers. It cannot run a benchmark itself, because a benchmark needs the
services and the GPU, so `/bench` keeps the human handshake in the main conversation and
delegates only the reduction of the reported output. `perf-auditor` is the sandbox
counterpart: runtime and memory cost of repo-local code, measured where it runs.

## Host verification is a skill, not a role

`/verify-host <claim>` loads `.claude/skills/verify-host/SKILL.md` in the main
conversation. The reason is structural: a subagent has no channel to the human, so it
cannot emit a command, pause, and read the reply. Delegating host verification to a
subagent leaves it able only to report failure or to invent output, and inventing output is
the one thing this repository forbids most emphatically.

The skill encodes the handshake: state the acceptance criteria before producing any
command, hand over one copy-pasteable block, stop, then judge each criterion against the
verbatim reported output. It also carries the failure mode this repository keeps hitting:
a worktree is cut from the `HEAD` commit, so a verification run against uncommitted work
exercises stale source and reports a false pass.

## Related

- [README.md](README.md) — the generated roster, and the ladder these roles sit on
- [workflows.md](workflows.md) — running a session, and the maintenance targets
- [tooling-parity.md](tooling-parity.md) — the frontmatter translation and what each tool ignores
