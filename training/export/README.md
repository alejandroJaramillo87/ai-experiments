# training/export

The exporter, where a trained checkpoint becomes one loadable HF directory. That is
`<name>/hf/` with `config.json`, a single `model.safetensors`, the generation config and the tokenizer
files, beside a `manifest.json` carrying `model_hash`, the sha256 of the weights file. That
directory is what the rust harness serves and the interp tier captures from, and the hash
is the identity every artifact fitted to the model carries. The tier produces no other
serving format.

One verb rather than a merge verb, because merging is only one of the ways in: an adapter
checkpoint is merged into its base through PEFT, a full fine-tune or a scratch checkpoint
arrives by the copy path, and both share everything after (consolidation into one file,
the single-file assertion, the manifest). `preflight.py` resolves the base a fine-tune was
trained on and refuses a merge onto a base whose identity drifted. The verb runs in the
CPU-only `export` image, so a merge never contends for the GPU.

`interp/commands/organism.py` reads the manifest through `manifest.py` when it seals an
organism, and `tests/training/test_export_harness.py` hands an exported directory to the
built harness binaries so the export, an interp probe and the harness preflight are held
to one digest.

<!-- inventory:modules:start -->

| Module | Purpose |
|--------|---------|
| `consolidate.py` | Consolidate a sharded safetensors checkpoint into one model.safetensors. |
| `errors.py` | The export pipeline's one failure type. |
| `manifest.py` | Export manifests. |
| `merge.py` | Checkpoint staging: a trained checkpoint becomes one loadable HF directory. |
| `preflight.py` | Base-model identity resolution and drift preflight (Phase 1 F6). |

<!-- inventory:modules:end -->

Living page: [train_serve_roundtrip.md](../../docs/training/base/train_serve_roundtrip.md).
Tests: `tests/training/test_export*.py` and `test_cli_handlers_export.py`; the real merge
on the workstation is `tests/hardware/test_fine_tuning_loop.py` under `hw_full`.
