# Findings ledger (discovery pass 1) — triage-agent fixture

Raw output of a discovery sweep, before triage. Verdicts are recorded exactly as the
verification phase left them.

## X-01 (critical)

`runtime/src/model/cpu.rs:418` — the normalization multiplies by `(1 + w)`, but the
converter resolves the shift before writing, so every real checkpoint is mis-normed.

Reproduction: a scratch binary loading a converted checkpoint produces a different
argmax at the last position than the upstream reference. Output recorded.

Verdicts: skeptic-1 CONFIRMED, skeptic-2 CONFIRMED.

## X-02 (high)

`runtime/src/model/cpu.rs:418` — the RMS normalization applies Gemma's runtime `+1` to a
tensor that already carries the effective multiplier. Same site as X-01, reached from the
numerics dimension instead of the conventions dimension.

Reproduction: as X-01.

Verdicts: skeptic-3 CONFIRMED.

## X-03 (high)

`harness/src/tools.rs:88` — the bash tool's drain loop has no wall-clock bound, so a child
that writes slowly forever holds the turn open indefinitely.

Reproduction: none attempted.

Verdicts: none recorded.

## X-04 (high)

`runtime/src/sampler.rs:107` — sorting the full vocabulary every step is almost certainly
the dominant cost at production vocabulary sizes, and a partial selection would clearly be
faster.

Reproduction: none. Reasoning from the code only; no measurement taken.

Verdicts: skeptic-4 CONFIRMED.

## X-05 (medium)

`runtime/src/session.rs:243` — a snapshot whose header magic is valid but whose geometry
belongs to a wider model restores without complaint.

Reproduction: hand-edited snapshot loads into a narrower model and returns Ok. Output
recorded.

Verdicts: skeptic-5 REFUTED — the model fingerprint check rejects the file before geometry
is ever read; the reproduction bypassed the binary's own load path.

## X-06 (low)

`runtime/src/gguf.rs:210` — `is_vision_tensor` is dead code with no caller.

Reproduction: not required at this severity.

Verdicts: none recorded, by the severity policy.

## X-07 (medium)

`harness/src/compact.rs:54` — the rendered prompt is tokenized twice on every agent turn,
once to count and once to encode.

Reproduction: instrumented count shows two encode calls per turn. Output recorded.

Verdicts: skeptic-6 CONFIRMED.
