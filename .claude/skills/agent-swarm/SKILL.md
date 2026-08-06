---
name: agent-swarm
description: Mechanics contract for running a multi-agent workflow that survives to completion - the durable-ledger rule, the fan-in partition rule, the output-budget rule, and the recovery procedure for a run that dies mid-flight. Read before authoring or resuming any Workflow script.
---
# Agent swarm mechanics

The method contract for a quality campaign lives in `bug-bash`. This document covers a
different failure surface: the mechanical reasons a swarm that is methodologically correct
still fails to deliver an artifact.

Every rule below was paid for by a run that died. The Gemma 4 conformance audit
(2026-08-02) spent 19.1 million tokens across 246 agents and produced no document, three
separate times, for three separate reasons. Each rule names the failure that bought it.

## The four hard limits

A workflow author reasons about these constantly. Nothing else in this document makes sense
without them.

| Limit | Value | What crosses it |
|-------|-------|-----------------|
| Response output cap | 32,000 tokens | One agent's entire reply, including tool-call arguments |
| Agent context window | model-dependent | The prompt plus everything the agent reads |
| Session usage limit | account-dependent | Every agent in the run, cumulatively |
| Workflow concurrency | `min(16, cores - 2)` | Simultaneous agents, not total |

The output cap is the one that surprises people, because it counts **tool-call arguments**.
A `Write` call carrying a 4,000-line document is a 40,000-token response and fails. So is a
structured return carrying 300 findings. An agent that must emit more than roughly 1,500
lines in one call needs a different design, not a larger prompt.

## Rule 1: never let an agent echo data it is not changing

**Failure that bought this.** A triage agent was given 716 findings and a schema requiring it
to return the full finding object - including the two verbatim quote fields - for each one it
kept. It worked correctly for 55 tool calls, verifying quotes against real files, then died
on `Claude's response exceeded the 32000 output token maximum`. The deduplication ratio was
irrelevant; the payload alone exceeded the cap.

**The rule.** An agent that transforms a collection returns **decisions**, never the
collection. The orchestrator already holds the data and reattaches it.

```js
// Wrong: the agent retransmits every quote it did not touch.
schema: { findings: [{ id, title, spec_says, code_says, severity, ... }] }

// Right: roughly 60 output tokens per item instead of 400.
schema: { decisions: [{ keep_id, merge_ids, severity, line, title }] }
// then, in the script:
const out = decisions.map(d => ({ ...byId.get(d.keep_id), ...d }))
```

State the constraint in the prompt as well as the schema. An agent told only by a schema will
often narrate its reasoning into a field anyway:

> Do NOT return spec_says, code_says, file, or tier. Those already exist and are reattached
> automatically. Returning them is what killed the previous attempt at this step.

## Rule 2: write the ledger ahead, not at the end

**Failure that bought this.** The audit's findings lived in workflow memory for eight hours.
The single agent that would have written them to disk was the last one, and it died on a
session limit. Every finding survived only because agent results happen to persist in
`journal.jsonl`; recovery took a bespoke 300-line reconstruction script.

**The rule.** Treat the run like a database with a write-ahead log. **A stage's output is not
real until it is on disk.** Never let the only copy of an expensive result be a JavaScript
variable, because the script's memory dies with the run and workflow scripts have no
filesystem access.

Three mechanisms, in order of preference:

1. **Fragment-and-concatenate.** Many agents each `Write` a numbered fragment to a scratch
   directory; one final agent assembles with `cat parts/part-*.md > out.md`. Assembly by
   shell costs zero output tokens, and a fragment lost to an interruption re-runs alone.
   Name fragments so lexical order is document order: `part-000-head.md`,
   `part-1NN-body.md`, `part-200-tail.md`.

2. **Checkpoint agent per stage.** After each fan-in, one cheap agent writes the stage's
   result to disk before the next stage starts. Costs one agent per stage; converts a total
   loss into a resume.

3. **Journal recovery.** The fallback, not the plan. See
   [Recovering a dead run](#recovering-a-dead-run).

## Rule 3: partition every fan-in

**The rule.** No single agent receives the whole result set. A barrier that collects N agents'
output and hands it to one agent is a single point of failure on both token limits and
reliability, and it has no redundancy - the fan-out stages do.

Partition by a key that keeps related items **together**, or the partition destroys the
work:

```js
// Deduplication partitioned by file: every duplicate of one site meets in one agent.
const byFile = new Map()
for (const f of raw) { (byFile.get(f.file) ?? byFile.set(f.file, []).get(f.file)).push(f) }
// Pack groups into chunks under a per-agent item budget, then a cheap second pass
// resolves whatever the partition key could not bring together.
```

Then run a **narrow second pass** for the cross-partition case. Deduplicating by file cannot
catch one wrong constant copied into five files; a second agent seeing only
`id | severity | file:line | title` resolves that at a fraction of the tokens, because titles
are small in both directions.

## Rule 4: report only what an agent confirmed

**Failure that bought this.** The workflow returned
`document: doc?.written_path ?? OUT_DOC`. When the writer died, the run reported the path of
a document that did not exist, and the failure was discovered only by listing the directory.

**The rule.** A fallback must never fabricate success. Return the null and a boolean beside
it:

```js
document: doc?.written_path ?? null,
document_written: Boolean(doc?.written_path),
```

The same rule governs `parallel()`, which resolves a failed thunk to `null` rather than
rejecting. `.filter(Boolean)` silently shrinks the result set: a stage that expected 19
chunks and got 18 continues without complaint. Compare counts explicitly and `log()` the
difference.

## Rule 5: every item re-enters the pipeline

**Failure that bought this.** The audit's completeness critic spawned gap-fill sweeps whose
findings were pushed directly into the confirmed set with `status: 'confirmed-gapfill'`. They
bypassed deduplication and adversarial verification entirely. The run reported 248 confirmed
findings; only 59 had ever been verified. The label said confirmed and meant nothing.

**The rule.** A late-arriving item goes through the same stages as an early one, or it is
labelled honestly as having skipped them. Never mint a status that reads as validated for
something that was not.

If the loop shape makes re-entry impractical, the item is `unverified`, and the count is
reported separately - never summed into the verified total.

## Rule 6: namespace identifiers before flattening

Thirty-four agents each numbering their findings from `01` produce colliding identifiers the
moment the results are flattened. A downstream stage that addresses items by identifier then
merges unrelated work, silently.

```js
const raw = results.flatMap((r, si) => (r.findings ?? []).map(f => ({ ...f, id: `S${si}-${f.id}` })))
```

## Rule 7: cap the fan-out, and say what the cap dropped

An unbounded `parallel()` over a discovered work list is how a run meets the session limit.
Cap it, sort so the cap keeps what matters, and `log()` the remainder. A silent truncation
reads as full coverage in the final report.

```js
if (items.length > CAP) log(`CAP: ${items.length - CAP} lowest-severity items recorded unverified`)
```

## Budgeting a resume

`resumeFromRunId` replays completed agents whose `(prompt, opts)` are unchanged. Two
properties decide what a resume costs, and both are worth checking **before** relaunching:

- Editing a prompt early in the script invalidates that call and everything after it.
- A **failed** call has no cached result, so it re-runs - and may force everything after it
  to re-run too.

Count what is cached first:

```bash
# The python below globs *.meta.json and opens journal.jsonl relative to the working
# directory, so cd into the run's transcript dir first. A bare assignment would not expand
# the glob -- $(echo ...) resolves the project slug.
D=$(echo ~/.claude/projects/*/subagents/workflows/<runId>)
cd "$D" || { printf '[ERROR] no transcript dir for that runId: %s\n' "$D" >&2; exit 1; }

python3 - <<'PY'
import json, glob, os, collections
meta = {os.path.basename(m).replace('agent-','').replace('.meta.json',''):
        json.load(open(m)).get('agentType') for m in glob.glob('*.meta.json')}
started, results = {}, set()
for l in open('journal.jsonl'):
    d = json.loads(l)
    started[d['key']] = d['agentId'] if d['type'] == 'started' else started.get(d['key'])
    if d['type'] == 'result': results.add(d['key'])
c = collections.Counter(); done = collections.Counter()
for k, a in started.items():
    t = meta.get(a, '?'); c[t] += 1; done[t] += k in results
for t in sorted(c): print(f"{t:20} {c[t]:4} started {done[t]:4} cached {c[t]-done[t]:4} MISSING")
PY
```

When the missing calls sit early and the expensive calls sit late, **do not resume**.
Reconstruct and run a narrow finishing workflow instead. Reconstruction costs no tokens.

## Recovering a dead run

`journal.jsonl` holds one `{"type":"result"}` line per completed agent with its full return
value. That is the write-ahead log, whether or not the script intended one.

1. **Load the latest result per key.** Keys repeat across resumes; take the last.
2. **Classify by schema shape.** Each stage's return has a distinct key set, so
   `tuple(sorted(result.keys()))` identifies the stage with no labels needed.
3. **Recover per-agent context from the transcript.** `agent-<id>.jsonl` line 1 is the
   prompt. Anything interpolated into it - the item under review, the assigned lens - is
   recoverable by regex. This is how verdicts are matched back to the items they judged.
4. **Match on stable keys, not ordinals.** Reproducing a sort order across languages is
   fragile. Match on content (`file:line` plus title), which does not depend on order.
5. **Validate against the run's reported counts** before trusting the reconstruction, and
   report any delta rather than papering over it.

Do not read `agent-*.jsonl` into an agent's context directly. These files reach hundreds of
kilobytes; parse them with a script and emit only summaries.

## Preflight checklist

Before launching any workflow:

- [ ] No agent's structured return can exceed roughly 1,500 lines of content.
- [ ] No agent receives the entire result set of a previous stage.
- [ ] Every expensive stage writes to disk before the next stage begins.
- [ ] Every `?? fallback` either fails loudly or reports that it fired.
- [ ] Every `.filter(Boolean)` is preceded by a count comparison.
- [ ] Agent-local identifiers are namespaced before any flatten.
- [ ] Every fan-out over a discovered list has a cap, and the cap logs what it dropped.
- [ ] Late-arriving items re-enter verification, or are labelled as having skipped it.
- [ ] The script parses: `{ echo 'async function w(){'; sed 's/^export const meta/const meta/' s.js; echo '}'; } > /tmp/c.mjs && node --check /tmp/c.mjs`

## Related

- `.claude/skills/bug-bash/SKILL.md` - the method contract: phases, severity, what confirmed means
- `/AGENTS.md` - the sandbox execution rules every agent in a swarm inherits
