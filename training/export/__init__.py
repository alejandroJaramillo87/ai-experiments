"""GGUF export pipeline (Phase 0 D7).

checkpoint -> [merge adapter if present] -> HF safetensors staging dir ->
convert_hf_to_gguf -> llama-quantize -> /app/models/trained/<run>/<name>-<quant>.gguf
plus an export manifest.

Conversion runs against a pinned llama.cpp revision (llama_cpp_pin.txt), built
into docker/training/Dockerfile.export — independent of the llama-gpu serving
image so exports stay reproducible across serving-image rebuilds. The pin is
bumped deliberately, with a note, when new architectures are needed.
"""

from training.export.consolidate import (
    check_single_weights_file,
    consolidate_shards,
)
from training.export.convert import (
    check_architecture,
    read_pin,
    supported_architectures,
)
from training.export.errors import ExportError
from training.export.manifest import (
    build_hf_manifest,
    build_manifest,
    checkpoint_meta,
    hf_manifest_path,
    load_hf_manifest,
    sha256_file,
    write_hf_manifest,
    write_manifest,
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
    "BaseIdentity",
    "ExportError",
    "adapter_base_model",
    "build_hf_manifest",
    "build_manifest",
    "check_architecture",
    "check_single_weights_file",
    "check_base_identity",
    "checkpoint_meta",
    "consolidate_shards",
    "derive_output_name",
    "find_adapter",
    "hf_manifest_path",
    "load_hf_manifest",
    "read_pin",
    "resolve_base_identity",
    "resolve_model_dir",
    "sha256_file",
    "stage_model",
    "supported_architectures",
    "write_hf_manifest",
    "write_manifest",
]
