# Benchmark suite quality campaign: findings ledger

**Operational. The findings record for [bench-qa.md](bench-qa.md), separated from it because a
campaign log and a runbook are read at different times by different people. Keep the paths
accurate — both documentation gates check this directory, and the rows below name source
symbols that the runbook-entry gate resolves.**

The ledger for the campaign in [bench-qa.md](bench-qa.md), carrying every finding it raised,
the severity the rubric settled on, what the campaign did about it, and the verdicts each one
drew. It left that document at 1024 lines, where the campaign's own fourteen entries had to be
read past three hundred lines of history to reach the closing scope section.

Rows are never deleted, including refuted ones: a campaign that erases what it disproved
teaches the next campaign nothing, and the refutations are the most expensive findings here.

Campaign-local identifiers, following the rust-harness precedent: an identifier in
[../bugs/known-bugs.md](../bugs/known-bugs.md) is for a defect a host runbook step will hit,
and a finding is promoted there only once it is shown to surface at a `bench.md` entry.

Severity is proposed, not settled. Every row below was found by reading and none has an
independent verdict, so under the rubric each is `reasoning only` and cannot be graded above
low until the verification phase records a verdict.

Disposition is what the campaign did about the row, recorded so a reader does not have to
reconstruct it from the commit log:

| Disposition | Meaning |
|-------------|---------|
| `fixed` | A commit closes it, with a test that was observed failing before the change |
| `refuted` | Verifiers disproved the defect; the row stays because a campaign that deletes what it disproved teaches the next campaign nothing |
| `waived, low` | Real but graded low after verification, with no reachable consequence worth a change |

| ID | Location | Proposed | Disposition | Defect |
|----|----------|----------|-------------|--------|
| BQ-F01 | `bench/metrics.py`, `bench/client.py` | high | fixed | A withheld frame costs one arrival while the reported generation length does not change. Withheld at either end of the generation, the decode rate reads high by one interval's worth (29.2 against 25.0 tokens per second on the reproduction) and a leading withhold also puts time to first token one interval late. Withheld in the interior, every aggregate stays correct and one fabricated interval enters the latency distribution as an outlier. Filed first as one critical defect covering both; the arithmetic under BQ.9 shows they are two, and the interior case never moves a rate. |
| BQ-F02 | `bench/metrics.py` | high | fixed | The field naming the worst observed chunk packing is computed as a mean of per-sample means, so a single densely packed chunk is diluted below the threshold that would flag the series as approximate. |
| BQ-F03 | `bench/delta.py::compare` | critical | fixed | **Re-filed 2026-08-29 on the chain two verifiers found while refuting the original text.** `compare` never checks that the supplied noise floor was measured against the same workload as the arms. The original filing said every lookup therefore misses and every metric reports inconclusive; that is false and the truth is worse. The `quick` and `publish` profiles share the point key `prefill/p128/g1/c1`, and the slope block is keyed on the phase alone so it is shared by construction, so a floor from another profile is partly *applied* rather than wholly ignored. A verifier reproduced a real forty percent regression in time to first token printing as within noise, exit 0, warnings empty, against a floor measured at a different repeat count. A partly-floored table is harder to spot than a wholly inconclusive one. There is also a warning asymmetry: the missing-floor warning fires only when the argument is absent, so a floor from the wrong workload is quieter than no floor at all. The wrong-*kind* document is refused as of the compare batch; a right-kind floor from the wrong *workload* is still accepted, and that is what remains. The noise document already carries the profile and service it was measured under, and the arms carry their conditions, so the check needs no new field. |
| BQ-F04 | `bench/delta.py` | high | fixed | Observed prompt and completion token counts are statistics blocks inside the metrics mapping, so they are differenced and graded as latencies. A run that tokenized more prompt than its baseline reports a regression. |
| BQ-F05 | `bench/metrics.py` | high | fixed | The coefficient of determination is exactly 1.0 for any two-point fit. The `quick` profile sweeps two prefill lengths, and `bench.md` B1 accepts a value at or above 0.95, so that check passes by construction on a broken instrument. |
| BQ-F06 | `bench/runner.py` | high | refuted | One rejected sample discards its entire concurrency cell; at the sixteen-way vLLM point, fifteen good results go with it. |
| BQ-F07 | `bench/runner.py`, `bench/commands/run.py` | high | fixed | Samples are written once at the end, and `assemble` sits outside the error handler. Any failure not derived from the serving error type, late in a run of up to an hour, leaves nothing on disk. |
| BQ-F08 | `bench/runner.py::write_samples` | medium | waived, low | The samples file carries bare performance-counter values with no wall-clock anchor and no cell ordinal, so the interleaving the design cites as its evidence cannot be checked from the artifact. |
| BQ-F09 | `bench/client.py` | medium | waived, low | A frame that fails to parse is skipped silently, removing an arrival and inflating the intervals on either side of it. Nothing counts them or reports them. |
| BQ-F10 | `bench/adapters.py::parse_usage` | medium | fixed | The integer conversion of the usage fields is unguarded. A non-numeric value raises through `bench/runner.py::_dispatch`, which handles only the serving error type, ending the run. |
| BQ-F11 | `bench/conditions.py::container` | medium | fixed | Successfully parsing JSON that is not a mapping makes the subsequent lookup raise, uncaught here and uncaught in `collect`. |
| BQ-F12 | `bench/conditions.py` | medium | fixed | Two separate GPU query subprocesses plus a container inspection run after all measurement is complete and on the run's critical path, each under its own timeout. The second query is issued from inside the first helper's return expression, so it is reached only when the first succeeds; a host without the tool pays for one call, and the workstation pays for both. |
| BQ-F13 | `bench/conditions.py::collect` | low | fixed | The service key is passed where a container name is expected, and the parameter that would carry a resolved name is unused at both call sites. |
| BQ-F14 | `tests/bench/test_runner.py` | low | fixed | Three unit tests call `assemble`, which invokes host binaries on whatever machine runs the suite. Machine-dependent, and nothing asserts the result. |
| BQ-F15 | `bench/client.py` | low | fixed | The event-stream prefix constant requires a space the specification makes optional. |
| BQ-F19 | `bench/adapters.py`, `bench/README.md` | high | fixed | The vLLM adapter does not override the server-timings parser, so every vLLM phase is labelled client-sourced and the package README states the engine reports no server timings. At the pinned version the engine can attach per-request timings to the same final usage frame the adapter already parses, carrying time to first token, generation time, mean inter-token latency and queue time. Queue time is the one quantity this suite documents as impossible to derive client-side, and it is what would let the concurrency sweep report a queue-free latency. Reaching it needs a launch flag the compose file does not pass, which is the same gap that leaves every other vLLM knob unreachable. |
| BQ-F20 | `docs/verification/bench/bench.md` | medium | fixed | Entry B4 accepts on the decode rate exceeding the end-to-end rate. The separation is conditional, not a property of the measurement: it holds exactly when time to first token exceeds the mean time per output token. It is expected to hold on this hardware, so the criterion is an assumption about the machine stated as an invariant, and a run that failed it would be read as a broken tool rather than as the finding it would be. |
| BQ-F16 | documentation | low | fixed | `pyproject.toml` omits `bench` from the isort first-party list while including it in the type-checker file list; `tests/README.md` omits `tests/bench/` from its layout table; `../README.md` records nine entries for `bench.md`, which has eight; `../../testing/quality-gate.md` says ten steps and lists twelve; `../../README.md` claims a gate exemption for this directory that was removed on 2026-08-08, and still directs readers to two retired scripts; `../README.md` described `bench.md` as covering four serving services where that runbook scopes itself to three. Every item in this row is corrected in this batch. |
| BQ-F17 | `bench/runner.py`, `bench/delta.py` | high | fixed | The type checker reported five errors against `bench/`, so `make quality` did not exit 0 on the pushed branch. The repository carries no continuous integration, and the gate is invoked by hand, which is how a red gate reached the remote. Fixed in this batch by annotation alone, with no change to behavior. Closed. |
| BQ-F18 | `scripts/docs/check_doc_refs.py` | low | fixed | `docs/agents/README.md` cites a git-ignored per-checkout settings file in the sentence that explains the file is git-ignored, so a fresh clone failed the reference gate. Fixed in this batch by recording it among the audited absences. Closed. |
| BQ-F21 | `bench/commands/compare.py::cmd` | medium | fixed | The three report reads that open a comparison sit outside any handler. A missing file, a truncated document, or one that fails envelope validation leaves the command with a traceback rather than a diagnosed error and an exit code. The reader is proven to raise; nothing proves its only caller survives it. |
| BQ-F22 | `bench/commands/compare.py`, `bench/commands/noise.py`, `bench/__main__.py` | medium | fixed | No test imports any of the three. The noise floor is the mechanism every `within_noise` verdict rests on, and the command that measures it has never been executed by the suite; it is also the second call site of the assembly function, so a defect reachable only from there is invisible. |
| BQ-F23 | `tests/bench/fixtures/` | low | fixed | Two of the seven wire fixtures written in the previous batch, the llama.cpp partial frame and the vLLM final frame carrying per-request metrics, are loaded by no test. A fixture nothing reads asserts nothing, and the second is the shape BQ-F19 turns on. |
| BQ-F24 | `bench/commands/run.py::cmd`, `bench/delta.py::check_comparable` | critical | fixed | The A/B arms are not interleaved. `cmd` runs one arm's entire profile to completion, writes it, then recreates the service and runs the next, so drift over the run aliases wholly onto the second arm -- the exact failure the design exists to prevent, and which the subcommand's own description promises it prevents: "With two --arm flags the arms are interleaved at the repeat level, so drift over the run cannot alias onto one of them." Interleaving is implemented at the repeat level within one arm, never between arms. The guard that would catch it is defeated by the same gap: the comparison warns when the baseline's `run.order` is not `interleaved`, and that field is copied from the profile, which says `interleaved` in both profiles on disk regardless of how the arms were sequenced. So the one warning that exists can never fire. Surfaced by a verifier working BQ-F08 and confirmed by reading; filed here rather than folded into that finding because it is a different defect in a different file. |
| BQ-F25 | `bench/delta.py::floor_from_pair` | medium | fixed | The floor builder differences every metric in the mapping and takes the largest absolute change as `worst_pct`, with none of the filtering the comparison command applies to its own table. An observed token count, which is not a performance metric, can therefore set the published worst spread. `bench.md` B2 reads `worst_pct` as an acceptance signal and `bench/commands/noise.py` warns on it against the five percent threshold, so a tokenization artifact can declare the rig unable to resolve a real change. Surfaced by a verifier working BQ-F04, whose own claim it refuted. |
| BQ-F26 | `docs/verification/bench/bench-qa.md` | low | fixed | The observed block for BQ.7 recorded `29 passed, 1 xfailed` for a command that produces `31 passed, 1 xfailed` against the same unchanged tree: a figure captured before the last two properties were written and never refreshed. Recorded evidence that does not reproduce is worse than none, because the entry is marked verified on the strength of it. Every command in this document was re-run during verification; this was the only one that did not reproduce. Corrected in this batch. |
| BQ-F27 | `bench/registry.py::_validate` | medium | fixed | The guard that refuses a registry entry whose stated quantization contradicts its filename is skipped for the only model registered. The shared filename parser accepts a suffix only if it starts with `Q` or names a float type, so every IQ-family and TQ-family llama.cpp quantization parses as unknown and the check treats unknown as nothing to verify. The shipped entry is an IQ4_XS file, so the guard is inert on exactly the case it exists for, and the unchecked value flows into the report and into the fields a comparison refuses to cross. The suite covers the guard with a `Q4_K_M` path, proving it on a shape the registry does not contain. |
| BQ-F28 | `bench/registry.py::resolve` | high | fixed | Resolving a model checks only that a hand-maintained list in the same file names the service. Nothing reads the model file, and nothing compares the registry's claim against what the service is actually serving, at resolve time or after. The report then carries both: `conditions.model` from the registry and `conditions.env_knobs` plus `server_argv` from the running container, with nothing reconciling them and no warning when they disagree. The comparison refuses on the asserted fields and ignores the observed ones. The precondition is reachable, because the training commands swap models on these same services. This is the defect the package's own conditions module names as its reason to exist: a condition recorded from configuration rather than from the running system. |
| BQ-F29 | `bench/commands/models.py` | medium | fixed | The machine-readable listing cannot answer the question the human listing answers. The JSON branch dumps the model block a report carries, whose key set is fixed elsewhere and contains no `engines`, so a consumer cannot tell which services a model resolves for and must scrape the human lines from standard error instead. `bench.md` B3 states its acceptance in terms of `engines` while piping the JSON form. |
| BQ-F30 | `bench/registry.py::_validate` | medium | fixed | Validation checks field presence and two enumerations and nothing about type, shape, or internal consistency. A bare string for the engine list becomes a per-character tuple that the listing prints one letter at a time; an empty list makes an entry advertised but unresolvable for every service; an engine name that is not a service is accepted, though a test exists to assert otherwise; and active parameters exceeding total, active experts exceeding total, a negative parameter count and a string context length all reach the report unchallenged. |
| BQ-F31 | `bench/commands/models.py::cmd` | low | fixed | The command catches only the registry error type, so a registry entry with a wrong-typed field raises out of the loader as a type error, past the handler, and reaches the user as a traceback with no exit code from the command — which the Python output style names among the things to avoid. The same handler shape appears in the run and noise commands. |
| BQ-F32 | `bench/cli.py` | low | fixed | The top-level description, the first thing the help output prints, names the rust harness among the benchmarked services. The same parser's service choices refuse it, deliberately and with a documented reason. |
| BQ-F33 | `bench/workloads.py::build_prompt`, `bench/commands/noise.py` | high | fixed | The noise floor is measured against a warm prefix cache. The prompt nonce is built from the profile seed, the cell index and the slot, all deterministic, and the noise command runs both arms in one process against one live service with no recreate between them, because the arm helper returns early on an empty environment. The second arm therefore replays the first arm's prompts byte for byte -- 76 of 76 identical in the reproduction -- and vLLM's prefix caching is on by default and not switchable per request, which the adapter names as the single reason the runner varies the prefix at all. The floor is the difference between a cold arm and a cached one, and every `within_noise` verdict rests on it. |
| BQ-F34 | `bench/workloads.py::load`, `bench/delta.py` | high | fixed | No profile hash is computed anywhere in the tree, though the module docstring says a workload is reproducible from one and the benchmarking skill lists a profile hash among the mismatches a comparison refuses. What the comparison actually blocks on is the profile's declared name, a string the file asserts about itself. Two profiles with different sweeps and the same declared name compare with no warning; unmatched points are dropped in silence, while the prefill slope is keyed on the phase alone and is differenced across two fits through different sweeps. |
| BQ-F35 | `bench/workloads.py::Profile.cells` | medium | fixed | The ordering field selects round-robin only on an exact string match and is never enumerated, so a typo, a capitalisation or a synonym silently falls through to the blocked ordering the protocol exists to forbid. Mutating the selector to accept any string beginning with the right letter leaves the whole suite green. The comparison warns only on the baseline arm's recorded order, never the candidate's, and a warning is not a refusal. |
| BQ-F36 | `bench/workloads.py::points` | medium | fixed | An empty concurrency list, or an empty decode generation list, is a value rather than an absence, so it passes the presence guard and produces a report with one arm missing. The refusal message for the missing key directs the operator to write a list even if it is a single entry, and an empty list is a list. The run exits 0 with a schema-valid document carrying half the measurement. |
| BQ-F37 | `bench/workloads.py::load` | medium | fixed | Per-service overrides are resolved with fallback lookups on both the service key and the field name, so a misspelling of either matches nothing and silently falls back to the base sweep. The shipped override exists because an eight-thousand-token prefill on CPU takes minutes; a typo in it widens that sweep inside a serialized measurement window with no error and no warning. An override naming a service that does not exist is accepted as dead configuration. |
| BQ-F38 | `bench/workloads.py`, `bench/runner.py::reduce` | medium | fixed | The character-per-token constant carries a comment saying it is only used to slice a candidate and that the length reaching the report is always the server's own count. It is not: the estimate is written as each point's identity and is the x-axis the prefill slope is fitted through, while the server's own count is reduced beside it and never used. The slope is therefore a rate in estimated tokens while the server-sourced prefill rate in the same document is in real ones, and both are graded as throughput. |
| BQ-F39 | `bench/workloads.py::load` | medium | fixed | Nine required fields are bare subscripts and the only read error caught is a missing file, so every other malformation -- a missing key, malformed TOML, a scalar where a list belongs -- escapes the command's handler and reaches the user as a traceback with exit 1, which the CLI documents as failure or regression rather than invalid usage. A string where a repeat count belongs raises nothing at all. |
| BQ-F40 | `bench/workloads.py::load` | medium | fixed | No field is range- or type-checked, so three plausible values load cleanly and then raise uncaught from inside the run loop -- after the health gate has passed and, for an A/B arm, after a container recreate that on a thirty-billion-parameter model is a full reload. A repeat count of zero loads, produces only warm-up cells, and the run command misdiagnoses the empty result as every cell having failed. Negative counts and a scalar where the decode prompt length belongs reach the report as degenerate point identities. |
| BQ-F41 | `bench/profiles/publish.toml`, `bench/README.md` | low | fixed | Both state the publish profile covers four services. The concurrency table twenty-five lines below enumerates three, the adapter map has three, and a fourth would be refused. Not covered by the documentation row already in this ledger, which named different files. |
| BQ-F42 | `bench/workloads.py::build_prompt` | low | fixed | Two paths return a prompt that is only the nonce, against a comment promising a short region is repeated rather than silently yielding a shorter prompt: an empty region repeats to nothing, and a target small enough that the nonce fills the budget makes the length guard compare against a negative number. Reachable only through the unvalidated prompt length above, so filed low. |
| BQ-F43 | `bench/commands/run.py::apply_arm` | high | refuted | An arm recreates the service with its environment applied and nothing ever restores it. There is no restore call, no `finally`, and no closing hint, though three training commands in this repository do all three and the serving module's docstring carries the manual one-liner for exactly this case. The service is left in the last-attempted arm's configuration, including when that arm failed and produced no result, and every later consumer inherits it silently. Running `bench.md` B7 as written leaves the CPU service at twelve threads, the configuration the optimizations index marks superseded, until a human recreates it. |
| BQ-F44 | `bench/commands/noise.py::cmd` | high | fixed | The floor document is assembled by hand from a header plus four fields, dropping the errors list, the conditions and both arms' phases. A pair in which cells failed therefore yields a floor covering fewer metrics than the workload, with a worst spread taken over the survivors, and nothing in the artifact says so. Reproduced: an arm losing eight of nine cells wrote a floor of four entries where a clean run writes fifteen, worst spread zero, exit 0 -- and feeding that floor to a comparison returned a green all-clear on a genuine thirty-three percent decode regression. The health gate is also called once for the pair, so a service degrading between the two arms is never re-checked. |
| BQ-F45 | `bench/commands/compare.py::cmd` | high | fixed | The command never asserts that it compared anything. Points absent from the baseline are skipped, the regression count is summed over whatever survived, and zero regressions prints the same green line whether every metric passed or none was examined. Reproduced two ways: a baseline that lost its decode phase against a candidate fifty percent slower on decode printed no decode line and exited 0, and two noise documents compared to each other graded exactly zero metrics and still printed no regression. `bench.md` B6 reads that exit code as its acceptance signal. |
| BQ-F46 | `bench/commands/compare.py::cmd` | high | fixed | A presentation flag decides the exit status. The headline filter is applied before both the table and the regression count, so a metric outside that set is graded correctly against the measured floor and then discarded. Inter-token latency is not in the headline set, though the CLI names it as half the package's purpose. Reproduced: the same two files and the same floor give exit 0 without the flag and exit 1 with it, on a forty percent inter-token regression the tool had already classified. Neither B6 nor B7 passes the flag. |
| BQ-F47 | `bench/commands/compare.py::cmd` | medium | fixed | The report reader validates the envelope and never reads the type discriminator, so any envelope-conformant document is accepted as a noise floor. A result file passed there yields an empty floor and every metric grades as unmeasured -- and because the missing-floor warning is guarded on the argument being absent rather than useless, naming the wrong file is quieter than naming no file at all. Distinct from the workload-matching finding, which is inside the comparison, and from the unguarded read, which is about a failure: here the read succeeds on a valid document of the wrong kind. |
| BQ-F48 | `bench/commands/run.py::cmd` | medium | fixed | The JSON dump sits inside the per-arm loop, so a two-arm run emits two top-level documents back to back on standard output against a flag documented as writing the report. The stream does not parse. The package distinguishes line-delimited output explicitly elsewhere, so concatenation is not the declared format, and the runbook establishes the pipe-into-a-parser pattern for the sibling command. |
| BQ-F49 | `bench/commands/run.py` | medium | fixed | The output root is knowable at parse time and is never checked until the first write, after the service has been reconfigured, the health gate passed and the whole workload run. It defaults to a path this repository records as absent anywhere but the workstation, and the comment beside it acknowledges as much by offering an override. Filed separately from the handler-narrowness finding because it fires on the happy path where nothing else went wrong, and a widened handler would diagnose it but not prevent it; fold the two together if triage judges the overlap too close. |
| BQ-F50 | `bench/commands/run.py`, `bench/delta.py` | medium | fixed | The identifier binding two arms of a pair has writers and no reader. It is absent from the fields a comparison refuses to cross, so when the second arm fails the first arm's file remains on disk carrying a pair identifier whose sibling does not exist, indistinguishable from a completed pair. Reproduced: that orphan compared against a different pair's arm of the same name graded fifteen metrics with no warning and exited 0. |
| BQ-F51 | `bench/commands/run.py::apply_arm` | medium | fixed | The recreate is skipped when an arm carries no overrides, so a bare-named arm measures a service that has been up for an unknown time -- warm page cache, settled clocks, and whatever configuration a previous run left behind -- while its partner measures a container recreated seconds earlier. The benchmarking skill documents the overrides as optional, so the invocation is advertised. Nothing in either artifact records that one arm was recreated and one was not. |
| BQ-F52 | `docker-compose.inference.yaml`, `docs/verification/bench/bench.md` | critical | fixed | Two of the host runbook's A/B entries measure nothing. An arm applies its overrides as the environment of the compose subprocess, which reaches a container only through interpolation. `THREADS` is written as the literal `16` in the compose file, so the thread-count entry's twelve-thread arm never reaches the service and both arms run at sixteen; `NO_REPACK` appears in no compose file at all, so both arms of the repack entry run the entrypoint default. Only the huge-page knob is interpolated and therefore only that one arm is real. The entry that would catch this is the same entry that cannot: it instructs the operator to confirm the override applied by reading the recorded cpuset, which says nothing about a thread count. The runbook would produce two clean-looking comparisons of a service against itself, and the twenty-six percent figure the thread entry exists to re-anchor would be re-anchored to a null result. Surfaced by a verifier working BQ-F43, whose own claim it refuted. The disproof is already in the artifact and unread: `bench/conditions.py` records the thread count from the container's own inspected environment, so both arms would carry the same value and the comparison would see two matching counts and raise nothing. The entry reads the cpuset instead, which comes from a compose literal identical in every arm. |
| BQ-F53 | `bench/commands/run.py::apply_arm` | low | fixed | The residual of BQ-F43, which was refuted because the knobs it named never reach the container. One knob does: the huge-page wrapper is interpolated, so an arm setting it off leaves the service that way with no restore and no printed hint. A later ordinary bring-up re-resolves the default and recreates the container, so the leak is not sticky, which is why this is low rather than the high it was filed as. Filed fresh rather than folded into the refuted row, and carrying no verdict. |
| BQ-F54 | `docs/verification/bench/bench.md` | medium | fixed | Entry B5 accepts on the decode rate across the concurrency sweep, and that rate is now refused wherever a sample carried more than one token per chunk -- which is what the one service with a sweep does under load, and the reason it is in the stack. The refusal is correct: the decode rate reads high when the first chunk is packed and the client cannot tell that from a frame that never arrived. But the entry must then accept on the end-to-end rate, which is invariant under packing at both ends and is reported for exactly that reason. Throughput rising with concurrency is still demonstrable; it is a different field. Surfaced by narrowing the fix for BQ-F01, and unverified. |

| BQ-F55 | `bench/commands/compare.py::cmd` | medium | A metric refused at the reduction is diagnosed as a failed cell. When one arm's samples resolved their generation and the other's did not, `bench/metrics.py::reduce_samples` sets `tpot_ms` and `decode_tok_s` to `None` for the whole point in the second arm only; `bench/delta.py::compare` skips a `None` block, so the pair is enumerated for one arm and not the other and the command exits 1 with "2 metric(s) present in one arm only, so this comparison is incomplete" under the solution line "rerun the arm whose cells failed, then compare again". No cell failed: both arms' errors lists are empty, and the condition is the server packing chunks, so a rerun reproduces it. The exit code is right and only the diagnosis is false, which is why this is medium and not high. A verifier added the adjacent gap: `arrivals_incomplete`, `itl_approximate` and `tokens_per_chunk_mean_max` appear only in `../../../bench/metrics.py` and its tests, so the comparison never surfaces the real cause even though the reduction recorded it. Surfaced by the cross-package seam the batch-two integrator flagged as reasoning only. |

### Verdicts

Severity below is the settled grade, not the proposed one. Where they differ, the verification
phase moved it and the reason is in the restatements underneath.

| ID | Verdicts | Outcome | Settled |
|----|----------|---------|---------|
| BQ-F01 | 1 refuted, 1 confirmed, 1 refuted | mechanism reproduced three times; the wording was wrong three times | high |
| BQ-F02 | 2 refuted | flag fires correctly on every reachable input; naming defect only | low |
| BQ-F03 | 2 refuted | gap real, stated consequence false, true chain is worse | re-file, see below |
| BQ-F04 | 2 refuted | one field pinned by construction, the other invisible without a flag | low |
| BQ-F05 | 2 confirmed | confirmed | high |
| BQ-F06 | 1 confirmed, 2 refuted | discard is the deliberate and correct choice | refuted |
| BQ-F07 | 2 confirmed | confirmed with a reproduction; the subprocess half of the mechanism is wrong | medium |
| BQ-F08 | 1 refuted | line order and the stamps both recover the interleaving | low |
| BQ-F09 | 1 refuted | intervals merge, they do not inflate; no reachable path | low |
| BQ-F10 | 1 refuted | no pinned engine can emit a non-numeric token count | low |
| BQ-F11 | 1 refuted | every route from a real `docker inspect` is already intercepted | low |
| BQ-F12 | 1 refuted | costs no reported number; the second query is forced by the tool | low |
| BQ-F13, BQ-F14, BQ-F15, BQ-F23 | 0, by policy | recorded without verification | low |
| BQ-F19 | 2 refuted on the documentary half, 1 confirmed on the upstream half | see below | high |
| BQ-F20 | 1 confirmed | confirmed | medium |
| BQ-F21 | 1 confirmed | confirmed, weaker than filed: an exit code is set | medium |
| BQ-F22 | 1 refuted | the premise is false; the modules are imported | low |
| BQ-F24 | 1 confirmed, 1 refuted | core confirmed and broader than filed; two legs of the filing wrong | high |
| BQ-F25, BQ-F26 | 0 | filed during verification, unverified | pending |

Nine findings were refuted as filed. In seven the mechanism survived and the description did
not, which is a different outcome from a finding that was simply wrong, and the restatements
below are what a remediator works from — not the original rows.

**BQ-F03 must be re-filed before it is fixed.** It was filed as "every lookup misses, so every
metric reports inconclusive". Both verifiers refuted that and both found the same worse chain
independently: `quick` and `publish` share the point key `prefill/p128/g1/c1`, and
`prefill/slope` is shared by construction, so a mismatched floor is partly *applied*. One
verifier reproduced a genuine +40% regression in time to first token printing as `within_noise`
against a foreign floor, exit 0, warnings empty. The filed severity was right and its stated
reason was wrong, which is the most dangerous shape a finding can have: a fix scoped to the
filed text would harden a path that does not fail.

**BQ-F19 splits cleanly and the halves disagree.** The documentary half is wrong as filed — it
attributes to `../../../bench/README.md` a claim that file does not make (the sentence is in
`../../../bench/adapters.py`), and asserts the suite documents queue time as underivable
client-side, which no file in the tree says. The upstream half was expected to need the
workstation and did not: a verifier fetched vLLM at the pinned tag and read
`PerRequestTimingMetrics` with its `queue_time_ms` field, attached to the streamed completion
response under `--enable-per-request-metrics`. The same reading found what the finding missed —
that flag is rejected when combined with `--disable-log-stats`, which
`../../../docker/vllm-gpu/entrypoint.sh` passes by default. Reaching the timings is two changes,
and the second re-enables statistics logging on a benchmark target, which is a
measurement-conditions decision rather than a flag flip.

**BQ-F01's engine premise was settled from source, not from this campaign's own fixtures.** Two
verifiers independently fetched llama.cpp at the pinned commit and read `process_token`. Both
found the filing's paraphrase wrong on one branch: a pending stop-string match does not skip the
send, it sends an empty string, and that branch is unreachable here because
`../../../bench/adapters.py` sends no stop field. The reachable branch is incomplete UTF-8,
where no frame is sent at all — so no change to the client can recover the arrival, and the
defect has to be met in the reduction. The filed figures were also wrong: they are the
arithmetic for a forty millisecond inter-token gap, and the reproduction they cite uses twenty.

**BQ-F01 took three verifiers to reach an accurate sentence, and the third refuted the
correction.** Every verifier reproduced the mechanism and both strict expected failures, and
two independently read the engine source. What kept failing was the description. The filed
figures were the arithmetic for a forty millisecond gap against a reproduction that uses
twenty. The correction offered to the tie-breaker then asserted that the mean time per output
token reads high at the ends; it reads low — 17.3 against a true 20.2 milliseconds, because the
span loses a gap from the numerator while the denominator stays at the server's token count —
which is what this document already said two hundred lines above, and what the correction
contradicted. The accurate statement, which is what remediation works from:

> At either end of the generation the decode rate reads high and the mean time per output token
> reads low. A leading withhold additionally puts time to first token one interval late; a
> trailing withhold additionally inflates the end-to-end rate. In the interior every aggregate
> is correct and one fabricated interval enters the latency series. Nothing rejects the sample,
> though `itl_approximate` and the chunk-packing field do fire, so the report is not silent.
> Recovery is impossible in `../../../bench/client.py` because the reachable engine branch sends
> no frame at all; detection fits in the client, the runner, or the reduction.

Severity settles at high rather than critical on two grounds the verifier named: the packing
flag already makes the condition visible in the report to a reader who looks, and reachability
is unproven here — the corpus statistic this campaign cited is about the *prompt* text, while
the trigger is a *generated* token whose bytes end mid-character. Raising it needs a host run.

**BQ-F24 was confirmed on its core and wrong on two legs.** Both verifiers confirmed the arms
run sequentially and that the promise of interleaving is false, and the second found the
overclaim repeated in two more places this campaign had not looked: the benchmarking skill,
which invents a design rationale for machinery that does not exist, and the noise command's own
description. But the filing said the comparison's ordering warning "can never fire", and that is
false — a profile with `order = "blocked"` loads unvalidated and does fire it, which the
property suite already exercises. The true statement is narrower: the warning cannot detect the
condition its own message describes, because arm-level sequencing is never recorded anywhere.
Severity settles at high rather than critical because every dependent runbook entry is still
pending, so no published number is invalidated — the defect is to be fixed before those entries
run, not after they have misled anyone.

**Two findings were refuted by verifiers who then found a real defect next to them.** BQ-F08's
verifier found that the A/B arms are not interleaved at all, which is BQ-F24 and is graver than
anything else in this ledger. BQ-F04's verifier found that the floor builder grades observed
token counts into `worst_pct`, which is BQ-F25.

**One refutation was of a premise this campaign took on trust.** BQ-F22 asserted that the
compare and noise command modules are imported by no test. They are: the parser test imports
`../../../bench/cli.py`, which imports both. The claim came from a survey that was not checked
before it was filed.

### Verdicts, the discovery top-up

| ID | Verdicts | Outcome | Settled |
|----|----------|---------|---------|
| BQ-F27 | 1 confirmed | confirmed; the fix belongs to the registry, not the shared parser | medium |
| BQ-F28 | 2 confirmed | confirmed, and reachable without any fault | high |
| BQ-F30 | 1 confirmed | confirmed; five of seven examples are already caught by the suite | medium |
| BQ-F33 | 2 confirmed | confirmed; scoped to vLLM, and the floor is too wide rather than too narrow | high |
| BQ-F34 | 2 confirmed | confirmed, with a fabricated verdict reproduced twice | high |
| BQ-F35, BQ-F36, BQ-F37 | 1 confirmed each | confirmed | medium |
| BQ-F43 | 2 refuted | the stated consequence is impossible; a worse one is not | refuted |
| BQ-F44, BQ-F45, BQ-F46 | 2 confirmed each | confirmed twice, both from documents the normal build path produces | high |
| BQ-F25 | 1 confirmed | confirmed; needs asymmetric cell loss to fire, and clearing five percent is unproven | medium |
| BQ-F29 | 1 confirmed | confirmed; the sharper claim that B3 cannot pass is refuted | medium |
| BQ-F47 | 1 confirmed | confirmed; naming the wrong file prints strictly less than naming none | medium |
| BQ-F48 | 1 confirmed | confirmed but latent: no caller in the tree combines two arms with the flag | low |
| BQ-F49 | 1 confirmed | confirmed, and distinct from the handler finding rather than a duplicate | medium |
| BQ-F50 | 1 confirmed | confirmed; blocking on the pair identifier would not break the runbook's control | medium |
| BQ-F51 | 1 confirmed | confirmed; the bias runs warm-arm-faster | medium |
| BQ-F38 | 1 confirmed | confirmed; corrupts absolute claims only, and cancels exactly in a delta | medium |
| BQ-F39, BQ-F40 | 1 confirmed each | confirmed; a hand-edited profile is the designed input, not an exotic one | medium |
| BQ-F52 | 2 confirmed | confirmed; every delivery route checked and none exists | critical |

The prefill slope is reported in the wrong unit. The four-character-per-token estimate reaches
the report as each point's length and as the slope's x-axis, while the server's own count is
reduced beside it and used for nothing. The error is a pure rescale of the x-axis, so the
intercept and the goodness of fit are untouched and an A/B delta cancels it exactly -- verified
at three different scale factors, all giving the same delta to six decimal places. What it
corrupts is the absolute figure, which is the one the optimizations index publishes and the one
the benchmarking skill blesses for cross-engine comparison. Direction: too high whenever the
corpus runs longer than four characters per token, which this one does. The size needs the
served tokenizer and is a host question.

The confirmation rate is the difference between the two rounds, and the reason is evidence
rather than judgement. The first round's findings were filed from reading and nine of seventeen
were refuted as filed. The top-up's were filed with reproductions that had been executed, and
almost none were refuted -- the verifiers spent their effort correcting severity and scope
instead. Where a top-up finding did lose ground it was on an escalation rather than a
mechanism: the machine-readable model listing really does omit the field, but the runbook entry
that reads it also runs the human form, so that entry can pass after all.

What the round established beyond the individual rows:

**The refutation of BQ-F43 produced BQ-F52, which is the gravest finding in this campaign.** The
claim was that an arm's environment leaks and leaves the CPU service on twelve threads. It
cannot: `../../../docker-compose.inference.yaml` writes `THREADS` as a literal, so the compose
subprocess environment an arm sets never reaches the container, and the repack knob appears in
no compose file at all. Only the huge-page knob is interpolated. So two of the host runbook's
A/B entries would compare a service against itself and report it cleanly, and the entry meant
to re-anchor a twenty-six percent figure would re-anchor it to nothing. Both verifiers reached
this independently while refuting the finding they were given.

**BQ-F28's precondition is a documented flag, not a crash.** Both verifiers found that the
training commands restore in a `finally`, which was the counter-argument -- and both then found
that `--keep-serving` skips that restore deliberately on a successful run, and that `bench`'s
own arm mechanism accepts a model-path override that recreates the service on a different
model. One ran `check_comparable` on two reports whose containers had loaded different model
files and got no refusal and no warning. The fix is a string comparison over data already
collected: the registry's own comment says its paths are the in-container paths that match what
the compose file passes.

**BQ-F45's reachability was the open question and it is closed.** The refusal list inspects
conditions and the profile name and never the phases, while the reduction drops a phase whose
cells all failed. So an ordinary report missing its decode arm passes as comparable with zero
warnings, and a candidate fifty percent slower on decode prints no decode line and exits 0. No
hand-built document is needed.

**BQ-F33 is narrower and better characterised than filed.** It is vLLM-only: llama.cpp sends a
cache-defeating flag per request, and its server cache is per-slot rather than a hash-keyed
pool, so a replay from the start of the sequence cannot hit. The filed causal clause was also
wrong -- the noise command does not call the arm helper at all, so there is no early return to
blame. And the direction matters: caching can only make the second arm faster, so the floor
comes out too wide, which suppresses real findings rather than fabricating them.

### Verdicts, the seam between the reduction and the comparison

Two findings were filed while opening the remediation batch, from a seam the batch-two
integrator could only flag as `reasoning only`: the reduction now refuses `tpot_ms` and
`decode_tok_s` for a whole point whose samples did not resolve their generation, and nothing
had checked what the comparison does with that absence.

| ID | Verdicts | Outcome | Settled |
|----|----------|---------|---------|
| BQ-F54 | 1 confirmed | confirmed, and worse than filed: the sweep comes out incomparable, not merely short a number | medium |
| BQ-F55 | 1 confirmed | confirmed, and the exit code is right -- only the diagnosis is false | medium |

**BQ-F54's engine premise was settled from vLLM's source, not from this suite's fixtures.** The
finding was filed from my own reading and asserted that B5 accepts on a rate the reduction now
refuses. The verifier established the missing half -- that the refusal is actually reachable --
by reading the pinned engine: at `VLLM_VERSION=0.25.0` a streamed completion is served as
`RequestOutputKind.DELTA`, the v1 engine's output processor merges queued deltas "if the
producer gets ahead of the consumer", and the OpenAI completion serving layer emits that
merged multi-token delta as one SSE chunk. `../../../bench/client.py` stamps one arrival per
content-bearing chunk, so a merged chunk is exactly `chunks < completion_tokens`.

It also found the consequence is worse than filed. A concurrency-16 cell contributes sixteen
samples per repeat, so a `publish` point is eighty samples and **one** packed sample among them
nulls the metric for the whole point. Concurrency 1 will usually still report while 4 and 16 do
not, so B5 is left holding an incomparable sweep rather than a sweep missing one figure -- and
`../../../bench/delta.py` skips a `None` block silently, so the absence does not announce itself
in a comparison either. The remedy stands: `e2e_tok_s * concurrency` rising while the per-request
figure falls is the same batching-versus-serialization discriminator, on a span both of whose
endpoints are packing-invariant.

The verifier's own caveat, recorded because it bounds the claim: the *frequency* of packing at
concurrency 16 is inferred from the pinned source's merge path, not measured on the hardware.

**Half of what I predicted was wrong, and I refuted it myself before filing it.** The claim
was that with *both* arms packed a decode regression would pass silently, reintroducing BQ-F45
through a new absence. It does not. Reproduced with a matched floor: the candidate 40% slower
on decode was caught on `e2e_tok_s` and on `itl_ms`, two regressions beyond the floor, exit 1.
The green line appears only with no floor at all, which is the state in which every verdict is
already `inconclusive` by design. Filing that half would have repeated the batch-one error of
recording a consequence I had not computed.

The half that survived is narrower and real. From the verifier, who reproduced it
independently rather than from my transcript:

> The refusal is per-point, not per-cell: `bench/metrics.py:224` sets `span_resolved` from
> `arrivals_incomplete == 0` and `:244-249` set both metrics to `None` for the whole point, so
> `runner.execute` records nothing in `errors` and the claim's "no cell failed" is structurally
> true. `delta.py:230-241` skips a pair when either block is `None`, so `compare.py`'s
> `_inventory` enumerates it for the resolved arm only and the command exits 1 with the false
> diagnosis and the unactionable solution line.

The verifier also found the adjacent gap, which the fix must close with it: `arrivals_incomplete`,
`itl_approximate` and `tokens_per_chunk_mean_max` appear only in `../../../bench/metrics.py` and
its tests. The reduction records exactly why the metric is missing and the comparison never reads
it, so the operator is told to rerun an arm rather than told the server packed its chunks.

Reachability is what holds this at medium rather than high: the packing that triggers it is
vLLM's, no A/B entry in [bench.md](bench.md) targets vLLM, and the asymmetry needs one arm at
exactly zero packed samples. Reachable through the supported CLI, unproven on hardware.


