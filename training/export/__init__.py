"""GGUF export pipeline (Phase 0 D7).

checkpoint -> [merge adapter if present] -> HF safetensors staging dir ->
convert_hf_to_gguf -> llama-quantize -> /app/models/trained/<run>/<name>-<quant>.gguf
plus an export manifest.

Conversion runs against a pinned llama.cpp revision (llama_cpp_pin.txt), built
into docker/training/Dockerfile.export — independent of the llama-gpu serving
image so exports stay reproducible across serving-image rebuilds. The pin is
bumped deliberately, with a note, when new architectures are needed.
"""

from training.export.convert import (
    ExportError,
    check_architecture,
    read_pin,
    supported_architectures,
)
from training.export.manifest import (
    build_manifest,
    checkpoint_meta,
    sha256_file,
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
    "build_manifest",
    "check_architecture",
    "check_base_identity",
    "checkpoint_meta",
    "derive_output_name",
    "find_adapter",
    "read_pin",
    "resolve_base_identity",
    "resolve_model_dir",
    "sha256_file",
    "stage_model",
    "supported_architectures",
    "write_manifest",
]
