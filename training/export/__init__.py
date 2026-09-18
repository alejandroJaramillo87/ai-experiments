"""The training tier's exporter: a checkpoint becomes one loadable HF directory.

checkpoint -> [merge adapter if present, else copy] -> hf.partial/ -> hf/
  = <out_root>/<name>/hf/{config.json, model.safetensors, generation_config.json,
    tokenizer files}
  + <out_root>/<name>/manifest.json carrying model_hash = sha256(model.safetensors)

That directory is what the rust harness serves and interp captures from, and
model_hash is the identity every artifact fitted to it carries. The tier
produces no other serving format: GGUF is llama.cpp's and appears in this
repository only as published models the inference services serve.

One verb rather than a merge verb, because merging is one of the ways in:
full fine-tunes and the scratch checkpoints under step_N/model/ arrive by
the copy path and share everything after it (consolidation, the single-file
assertion, the manifest). The seam every training road ends at is the
directory, not the merge.
"""

from training.export.consolidate import (
    check_single_weights_file,
    consolidate_shards,
)
from training.export.errors import ExportError
from training.export.manifest import (
    HF_DIR_NAME,
    HF_MANIFEST_NAME,
    build_hf_manifest,
    checkpoint_meta,
    hf_manifest_path,
    load_hf_manifest,
    sha256_file,
    write_hf_manifest,
)
from training.export.merge import (
    adapter_base_model,
    derive_output_name,
    find_adapter,
    resolve_model_dir,
    stage_model,
)
from training.export.preflight import (
    BaseIdentity,
    check_base_identity,
    resolve_base_identity,
)

__all__ = [
    "HF_DIR_NAME",
    "HF_MANIFEST_NAME",
    "BaseIdentity",
    "ExportError",
    "adapter_base_model",
    "build_hf_manifest",
    "check_single_weights_file",
    "check_base_identity",
    "checkpoint_meta",
    "consolidate_shards",
    "derive_output_name",
    "find_adapter",
    "hf_manifest_path",
    "load_hf_manifest",
    "resolve_base_identity",
    "resolve_model_dir",
    "sha256_file",
    "stage_model",
    "write_hf_manifest",
]
