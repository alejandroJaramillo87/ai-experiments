# interp/probes

Linear probes (W2 to W4): capture labeled activations to a shard directory, train a
per-layer logistic regression on them, evaluate it against mandatory black-box baselines,
export the winner for the Rust harness, and aggregate the scores the serving process
takes in-band. `score.py` is the one code path every scorer goes through, so training,
evaluation and the monitor cannot disagree about a score.

`capture.py` defines the capture directory (`manifest.json`, `examples.jsonl`,
`shard_NNNNN.npz`) that `rust-harness/runtime/src/bin/activation-capture.rs` also writes;
`export.py` writes the probe artifact the harness loads. Both are contracts under
`schemas/`, and `rust-harness/runtime/tests/python_exporter_contract.rs` parses this
package's real output with the Rust loaders. The labeled prompt sets the capture reads are
in [prompts/](prompts/README.md), committed so a capture manifest's dataset hash resolves to
a commit. `weight_stats.py` is the sandbox half of RH.1a and reads attention weights, not
activations.

## Modules

<!-- inventory:modules:start -->

| Module | Purpose |
|--------|---------|
| `capture.py` | Activation capture (W2 decision 1): a workbench batch job. |
| `evaluate.py` | Probe evaluation report with mandatory black-box baselines (W2 decision 3). |
| `export.py` | Probe export for the rust harness in-band monitor. |
| `monitor.py` | Probe monitor (W4): aggregate the scores the serving process already took. |
| `registry.py` | Probe registry (W2 decision 4): list, load, and validate probe artifacts. |
| `score.py` | Probe scoring (W2 decision 5): one importable code path. |
| `train.py` | Probe training (W2 decision 2): per-layer logistic regression. |
| `weight_stats.py` | Distribution-shape statistics for attention projection weights. |

<!-- inventory:modules:end -->

## Where the real docs live

- [../../docs/interp/writeback.md](../../docs/interp/writeback.md) — the probe loop in use, capture to monitor
- [../../schemas/README.md](../../schemas/README.md) — `activation_capture.manifest.json` and `probe_artifact.json`
- [../../site/docs/formats/probe-artifact.md](../../site/docs/formats/probe-artifact.md) — the artifact format for a reader outside this tree
