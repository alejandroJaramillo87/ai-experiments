# Observation frames

A reference for what the harness's two interaction modes — the agent loop
and chat mode — contribute to the training and interpretability programs,
and for the feature bar that follows. The canonical scope statement lives in
the parent repository's
[runtime design](../../docs/inference/rust-harness-runtime-design.md#purpose-and-non-goals)
("instrumented inference substrate, not a competitor to the serving
services"); this document records the justification behind that positioning,
a taxonomy of the two observation frames, and the acceptance criterion for
future harness features. Nothing here is scheduled work: it records design
conclusions and experiment-planning constraints so later roadmap tasks start
from settled reasoning instead of re-deriving it.

## Table of contents

- [Purpose](#purpose)
- [The inference verdict, conceded](#the-inference-verdict-conceded)
- [The agent frame](#the-agent-frame)
- [The chat frame](#the-chat-frame)
- [Training across both frames](#training-across-both-frames)
- [Cross-frame methodology](#cross-frame-methodology)
- [The feature bar](#the-feature-bar)

## Purpose

Two questions prompted this document. First: does the coding-harness
functionality — tools, permissions, subagent forks, compaction, KV
snapshots, memory files — add value, or is it unnecessary complexity next
to a simple chatbot inference harness? Second: given the agent loop is the
native workload, what is chat mode (`--chat`) actually for?

The answer to both is the same structure. The harness is experimental
apparatus, not an inference product, and it carries **two observation
frames on one instrumented substrate**:

- **The agent frame** is the ecological frame: the model operating in a
  realistic multi-turn environment with tools, forks, and context
  management, observed in the wild.
- **The chat frame** is the controlled frame: minimal stimulus, observed
  response, one variable changed — the harness's psychophysics rig.

Each frame answers questions the other cannot, both are instrumented
through identical seams (tap, steering, probe, capture, flight recorder),
and findings established in one can be tested for survival in the other
inside the same numerical world. That pairing is the differentiated
capability: existing local stacks provide chat without instrumentation, and
no stack provides an instrumented agent loop at all.

## The inference verdict, conceded

Judged purely as a local coding assistant, the harness does not add value,
and this document does not defend it on those grounds. The repository's
llama.cpp and vLLM services (ports 8001-8003) plus any mature agent
frontend are faster, cover more models, and are maintained externally. The
runtime design records this as the "not a serving-stack replacement"
non-goal; the conclusion here is stronger: the inference use case is not
merely a non-goal, it is dominated, and no future harness work should be
justified by it.

One exception is recorded. From-scratch sub-1B checkpoints (training phase
4) are served by no off-the-shelf stack at all — they are custom
scaled-Gemma exports — let alone served with taps armed. For those models
the harness is not merely competitive as an inference engine; it is the
only one.

## The agent frame

The value of the agent loop is that each agentic feature is an
experimental condition, not product surface. Every one of the following is
implemented and tested, and every one creates an observable phenomenon that
has no chatbot analogue. Two of them are not yet reachable from the shipping
binary: `harness/src/subagents.rs` (fork and interleave) and
`harness/src/stream.rs` (the channel-based token stream) are called only
from tests, never from `main.rs`, so the phenomena they carry are available
to a test or to a future entry point, not to a `harness` invocation today.
Wiring them up is code work against a settled design, not new design.

**Tool episodes.** Agent tasks are prefill-heavy: most probe-scored
positions are the model reading externally authored text — task
descriptions, tool output, file contents — as recorded in the
[bridge document](../../docs/interpretability/rust-harness-interp.md#in-band-monitoring).
This is where safety-relevant monitoring lives: the question "is the model
building representation X while ingesting an injected instruction" can only
be asked while a model is actually ingesting tool output mid-task. A
capture-a-dataset workflow never produces that position distribution.

**Fork and join.** Subagent forks start from a bit-exact copy of the
parent's KV state, so two forked agents diverging is a controlled
divergence experiment: identical initial condition, different task text,
activation trajectories compared position by position. A saved `session.kv`
snapshot doubles as a resumable experimental condition — the same starting
state re-entered under different interventions.

**Compaction boundaries.** Compaction splices a model-written summary over
the middle of the transcript and re-prefills it, creating a natural
feature-shift observation point: the same semantic content, restated and
re-ingested. The KV-position caveat from the bridge document applies —
flag positions before and after a compaction do not align — which is
exactly why the boundary is an event worth instrumenting rather than an
artifact to ignore.

**Determinism compounds here.** A bit-replayable single completion is a
convenience. A bit-replayable multi-turn agent episode — forward passes
via the flight recorder, tool side effects via the observability records —
is a reproducible experiment log for an entire behavioral episode, and is
the property no other local stack offers in any form.

**Phase 5 depends on the loop structurally.** The continual-learning loop
in the [training scale reference](../../docs/training/training-scale-reference.md#phase-5-continual-learning)
uses the harness as its trajectory generator. Agent trajectories with tool
calls are the data that generation-over-generation preference tuning wants;
strip the agent loop and phase 5 collapses to fine-tuning on chat logs, a
much weaker research program.

## The chat frame

Chat mode is not a vestige of the simpler design; it is the second frame,
and it passes the feature bar below on its own merits. Its defining
property is the inverse of the agent frame's: chat is decode-heavy, so
probe-scored positions are the model producing its own tokens — which is
where intent-like attributes live.

**Instrument for the probe-transfer evaluation.** Probes are trained on
last-token representations of complete texts; their transfer to
mid-generation decode positions is unmeasured and is a pending roadmap row
in the bridge document. Chat mode is the cheapest source of long, clean
decode trajectories for that measurement — no tool-call grammar
constraints, no prefill contamination dominating the position mix.

**Clean A/B unit for steering.** The steering scorecard wants: same
prompt, steered versus unsteered, judge the outputs. A multi-turn agent
episode is a poor unit for that comparison — tool side effects and
branching swamp the intervention signal. A chat exchange isolates it:
minimal stimulus, one variable changed.

**Null condition for agent-frame phenomena.** Every agent-frame
observation — a probe trajectory across a tool episode, a feature shift at
a compaction boundary — needs a baseline. "Same model, same probe, plain
conversation" is that control group, produced in the same numerical world.
Without chat mode the nearest baseline is a Python-tier capture, which
crosses a provenance boundary and confounds the comparison.

**Isolating the per-layer-embedding confound.** The
[PLE caveat](../../docs/interpretability/rust-harness-interp.md#what-per-layer-embeddings-change)
means a probe may score partly on token identity rather than context. The
confound is easiest to isolate where the token stream is directly
controlled, which is a chat prompt, not an agent episode assembled from
tool output.

**The honest surface for phase 4 models.** A sub-1B from-scratch
checkpoint will not be a competent agent: grammar-constrained sampling
keeps its tool calls well-formed but not its decisions sensible. Chat mode
is where those models are actually evaluated — behavioral checks,
memorization probes against the known training corpus, qualitative
inspection of what a checkpoint learned. The science pole of the
interpretability program talks to its models mostly through chat.

**Phase 5 data on-ramp.** Human preference data is naturally chat-shaped:
a person rates or compares responses. Preference pairs over chat exchanges
are the standard case; preference data over agent trajectories is the
harder, later case. The phase 5 flywheel therefore starts on chat mode and
graduates to agent trajectories.

## Training across both frames

The training tier itself needs neither frame — checkpoints, the exporter,
and batch capture would work against any serving shim. What training gains
from the two frames is the evaluation and data surface:

- **Agent frame**: behavioral evaluation of a trained checkpoint — does
  the fine-tuned model use tools correctly, complete tasks, respect the
  loop — which no perplexity number measures.
- **Chat frame**: what did the checkpoint learn — qualitative and probed
  inspection, memorization checks, and the preference-data on-ramp above.
- **Phase 5**: the deploy, generate, collect feedback, train, redeploy
  loop consumes both — chat first, agent trajectories as the program
  matures.

## Cross-frame methodology

The experiment-planning payoff of having both frames on one substrate is a
standard two-step shape:

1. **Establish in the controlled frame.** Measure the effect in chat mode,
   where the stimulus is controlled, the position mix is decode-heavy, and
   the A/B unit is clean.
2. **Test survival in the ecological frame.** Re-measure in an agent
   episode — same model hash, same probes, same tap seam, same
   provenance — and observe whether the finding transfers to the
   prefill-heavy, tool-mediated setting.

Example experiment shapes this enables, recorded as shapes rather than
scheduled work:

- Probe score drift across chat turns versus across tool episodes: does a
  representation persist differently when the intervening text is
  model-authored conversation versus externally authored tool output.
- A steering vector validated on chat A/B pairs, re-applied mid-agent-run:
  does the behavioral effect survive tool use, and does the probe
  trajectory show the intervention where the transcript does not.
- A probe threshold tuned on decode positions in chat, evaluated on
  prefill positions in agent runs — the frame-transfer version of the
  pending prefill/decode evaluation.

Because both frames share every seam, none of these require new machinery;
they require only the two modes that already exist.

## The feature bar

The operative conclusion, and the acceptance criterion for future harness
feature proposals:

**A harness feature is in scope when it creates or preserves an observable
phenomenon. A feature that only makes the harness a more competitive
coding assistant is out of scope unless a training or interpretability
question demands it.**

Applied to what exists: tools, subagent forks, compaction, KV snapshots,
memory files, and chat mode all pass — each is cited above as an
experimental condition or instrument, and chat mode passes on its own
merits rather than as a legacy exception. Applied to what does not exist:
plugin ecosystems, an MCP client, multi-model routing, IDE integration,
and feature-parity chasing against mainstream coding agents all fail the
bar today, because each serves only the inference use case this document
concedes.

The bar cuts both ways. It rejects product features, and it protects
apparatus features: a proposal to simplify the harness by removing forks,
compaction, or snapshots would delete experimental conditions, and must be
weighed against the experiments they enable, not against code size alone.
