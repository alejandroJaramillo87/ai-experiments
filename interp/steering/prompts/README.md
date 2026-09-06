# Steering prompt sets

Fixed prompt files consumed by the W1 steering path (W1 decision 5). Prompt
sets live in git so every extraction and scorecard run is reproducible from a
commit; the manifests record content hashes of the files used.

| File | Consumed by | Format |
|------|-------------|--------|
| `honesty.jsonl` | `probe-capture --dataset` or `activation-capture --dataset`, then `steer-extract --capture` | one `{"text", "label"}` per line, label 1 = the honest persona; each honest line is followed by its deceptive twin |
| `behavior_honesty.jsonl` | scorecard behavior arm | one `{"prompt"}` per line |
| `neutral.jsonl` | scorecard coherence arm | one `{"prompt"}` per line |
| `system_prompt_honesty.json` | scorecard system-prompt arm | `{"prompt", "iterations": [...]}` |

`honesty.jsonl` uses the probe path's format (see
`interp/probes/prompts/README.md`): a steering vector and a probe are both
trained from one capture of a labeled prompt set, so there is one format. Until
2026-09 it was `honesty_pairs.jsonl`, one `{"positive", "negative"}` pair per
line for repeng; the pairing survives as adjacent lines.

The system-prompt file records at least three tuning iterations; the scorecard
refuses a baseline with fewer (a strawman baseline invalidates the comparison).
New behaviors add a new triple of files here, never edit a set an existing
scorecard references.
