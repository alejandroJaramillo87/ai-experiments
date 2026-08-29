"""Prompt corpus for length-controlled workloads.

natural.txt is prose sliced from the wikitext-2 validation set already committed under
scripts/benchmark/datasets/ and already bind-mounted into llama-cpu at /app/datasets:ro
(docker-compose.inference.yaml). It is real text, not random token ids, because the
retired load generator's `random.randint(100, 10000)` prompts route an MoE's experts
near-uniformly -- the best case for load balance and the worst for expert-weight
locality, and wrong with no predictable sign. See schemas/bench_models.json.
"""
