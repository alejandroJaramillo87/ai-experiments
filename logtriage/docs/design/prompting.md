# The tool surface, the token budget, and the loop

What the model is given, what it is allowed to ask, and what stops a 100MB log from becoming a
100MB prompt.

## Table of Contents

- [The loop](#the-loop)
- [The five tools](#the-five-tools)
- [The token budget](#the-token-budget)
- [Native tool calling is a server feature](#native-tool-calling-is-a-server-feature)

## The loop

`triage/pipeline.py` runs at most **8 turns**. Each turn the model may answer or call tools; a
tool call is dispatched, its result appended, and the loop continues. The run ends when the
model answers with prose, or when the turn limit is reached.

The first message already contains the profile and the findings. The model does not start from
nothing and go looking — it starts from counted evidence and may ask for more. Most runs use one
or two turns; the limit exists so a model that starts circling cannot generate until the context
is full.

Tool choice is `auto`, never `required`. The last turn of the loop is the one that writes the
report, and a model forced to call a tool can never take it.

## The five tools

Defined in `logtriage/tools/schemas.py`. Every one is structural: it asks a counted question
against the log store or the manual, and returns text plus the record identifiers behind it.

| Tool | Answers |
|------|---------|
| `log_profile` | The whole-corpus summary again, if the model wants it verbatim |
| `query_logs` | Count and facet records matching a filter |
| `search_logs` | Records whose message contains a substring |
| `search_docs` | Passages of the MongoDB manual matching a question |
| `fetch_records` | The full records behind a set of identifiers |

There is no tool that returns raw log lines in bulk, and no tool that takes a free-form query
language. Both would let the model spend its context on evidence it cannot count.

`search_docs` is present only when a documentation index is loaded (`triage --kb`). Its vector
leg is weighted by `Embedder.semantic`: with the hashing embedder the weight is zero and the
search is purely lexical, because fusing a meaningless vector ranking with a lexical one
measurably pushed the correct page out of the top three.

## The token budget

`logtriage/tools/budget.py` estimates tokens as characters over four — deliberately crude, since
the real tokenizer belongs to whichever model is served and the budget only has to be a bound.

| Budget | Default | Applies to |
|--------|---------|------------|
| `DEFAULT_PROFILE_TOKENS` | 4,000 | The whole-corpus profile in the first message |
| `DEFAULT_RESULT_TOKENS` | 2,000 | Each individual tool result |

Truncation is structural, row-wise, not a string slice: whole rows are dropped from the end and
the omission is stated. A profile cut mid-table would leave a number attached to the wrong label,
which is worse than a shorter table.

A 1,500-line corpus renders in roughly 220 tokens against the 4,000-token profile budget, so the
budget binds on large logs rather than typical ones. `profile --budget-tokens` exposes it, and a
budget that forces truncation says so.

## Native tool calling is a server feature

The message shapes in `llm/base.py` are modelled on llama-server's OpenAI-compatible API rather
than on something neutral. That is deliberate: llama-server applies the model's chat template,
parses whatever tool-call syntax the template produces, and hands back structured calls. An
abstraction inventing its own message shape would have to translate into this one anyway, and
every translation is somewhere for a tool call to be silently dropped.

Two consequences that are easy to get wrong and hard to notice:

- **The chat template must be enabled.** `docker-compose.logtriage.yaml` sets
  `JINJA_ENABLED=true`. Without it the server has nothing to parse against, the model's tool call
  arrives as prose in `content`, and the loop completes having called no tool — a run that looks
  successful and consulted nothing.
- **Tool-call arguments are a JSON string on the wire, not an object.** `Message.to_wire`
  serialises them. Sending the parsed object back is accepted by some servers and silently drops
  the call on others, which reads as a model that stopped using tools.
  `tests/test_adapters_http.py` asserts the encoding against a real HTTP server.

Whether Gemma's thinking channel leaks into `content` against the official template is
**unmeasured**; `LOGTRIAGE_REASONING_FORMAT` is the knob and
[../verification/desktop-model.md](../verification/desktop-model.md) carries the check.

## Related

- [citations.md](citations.md) — what a tool result has to carry
- [architecture.md](architecture.md) — the ports the tools query through
- [../operations/stack.md](../operations/stack.md) — the server flags this depends on
