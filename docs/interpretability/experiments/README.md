# Interpretability experiments log

Post-mortems for interpretability mini-projects and experiments. This directory is the
log convention the interp roadmap refers to as the experiments log; the Phase 2 PS5
commissioning mini-project is its first required entry.

## Entry naming

One file per experiment: `YYYY-MM-DD-short-slug.md` (date the experiment concluded,
lowercase hyphenated slug, e.g. `2026-07-20-gemma-capital-city-circuit.md`).

## Required sections

Every entry contains, in order:

1. **Question** — the specific behavior or hypothesis under investigation, stated
   before the experiment ran.
2. **Setup** — model, dictionaries or transcoders (exact source set and configuration),
   platform state (which services were up), and enough detail to regenerate every
   artifact. Cite prompts verbatim.
3. **Observations** — what was actually seen, with artifacts referenced by path or
   local URL (e.g. a graph in the Neuronpedia viewer: prompt, model, transcoder config,
   and regeneration steps).
4. **Hypothesis** — the mechanistic claim formed from the observations.
5. **Falsification attempt** — the concrete test designed to break the hypothesis and
   its outcome. An entry without a falsification attempt is incomplete.
6. **Verdict** — supported, refuted, or inconclusive, in one paragraph.

## Standards

- Negative and inconclusive results are recorded with the same rigor as positive ones;
  a refuted hypothesis is a complete, successful entry.
- Claims about model behavior cite the exact prompt and sampling settings that produced
  the evidence.
- Heavy artifacts (activations, graph exports) live under `/mnt/ai-data/interp/`, not
  in this repository; entries reference them by path.
- Entries are immutable after the verdict: follow-up work is a new entry that links
  back, not an edit.
