# Steering prompt sets

Fixed prompt files consumed by the W1 steering path (W1 decision 5). Prompt
sets live in git so every extraction and scorecard run is reproducible from a
commit; the manifests record content hashes of the files used.

| File | Consumed by | Format |
|------|-------------|--------|
| `honesty_pairs.jsonl` | `steer-extract --method repeng` | one `{"positive", "negative"}` pair per line |
| `behavior_honesty.jsonl` | scorecard behavior arm | one `{"prompt"}` per line |
| `neutral.jsonl` | scorecard coherence arm | one `{"prompt"}` per line |
| `system_prompt_honesty.json` | scorecard system-prompt arm | `{"prompt", "iterations": [...]}` |

The system-prompt file records at least three tuning iterations; the scorecard
refuses a baseline with fewer (a strawman baseline invalidates the comparison).
New behaviors add a new triple of files here, never edit a set an existing
scorecard references.
