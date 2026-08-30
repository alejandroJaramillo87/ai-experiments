# Benchmarking documentation

How inference performance is measured on this workstation, and what the recorded numbers are
still worth.

The suite itself is at [../../bench/](../../bench/), whose README owns the module layout, what
the package consumes, and the traps. The protocol it implements — entry points, warm-up
policy, the prefill and decode split, arm naming, and the regression criterion — is owned by
[../../.claude/skills/benchmarking/SKILL.md](../../.claude/skills/benchmarking/SKILL.md),
because it is loaded at the moment a benchmark is prepared. This directory owns neither. It
owns the reasoning underneath the protocol, and the results themselves.

## Documents

| Document | Description |
|----------|-------------|
| [method.md](method.md) | What makes a number citable: a delta is of a change and not of a layer, each optimization needs two of them, and a result carries its conditions |
| [results.md](results.md) | The recorded measurements. One entry, superseded, and the four grounds on which it is disqualified |

The division from the two runbooks next door is what a claim depends on.
[../verification/bench/bench.md](../verification/bench/bench.md) proves the suite measures
what it claims, against real services on the host;
[../verification/bench/bench-qa.md](../verification/bench/bench-qa.md) proves the reduction
logic and the failure paths in a sandbox. Both are executed. Neither is read for
understanding, which is what the two documents here are for.

## One trap this directory introduced

`bench/` and `docs/bench/` are now both real, and `scripts/docs/check_doc_refs.py` resolves a
markdown link only against the citing document's own directory. Both depths therefore resolve,
and a link aimed at the wrong one passes the gate silently:

| Citing directory | The package README | This directory's README |
|------------------|--------------------|-------------------------|
| `docs/bench/` | `../../bench/README.md` | `README.md` |
| `docs/optimizations/` | `../../bench/README.md` | `../bench/README.md` |
| `docs/verification/bench/` | `../../../bench/README.md` | `../../bench/README.md` |

No gate catches a mistake here, so check the depth by hand when adding a link. Running
`realpath -m --relative-to=. <target>` from the citing document's directory settles it.

## Related documentation

- [../../bench/README.md](../../bench/README.md) — the package: what it consumes, its module
  layout, and the mistakes it is built to prevent
- [../optimizations/roadmap.md](../optimizations/roadmap.md) — the order in which this
  repository's optimizations are measured, and the redo cost that sets it
- [../verification/bench/bench.md](../verification/bench/bench.md) — the host acceptance
  runbook for the suite
- [../../.claude/skills/benchmarking/SKILL.md](../../.claude/skills/benchmarking/SKILL.md) —
  the protocol, including the entry points and the five comparison verdicts
