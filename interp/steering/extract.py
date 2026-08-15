"""Direction extraction (W1 decision 1): two backends behind one interface.

method "repeng": contrastive prompt pairs through repeng's ControlVector.train
on the bf16 HF checkpoint (the default). method "sae_row": a single feature's
decoder row from a pretrained SAE (Gemma Scope via SAELens) placed at its
layer, zero elsewhere. Both return the same in-memory DirectionSet; export.py
writes it as a safetensors image plus a JSON manifest beside it
(schemas/control_vector_artifact.json). There is no third method: hand-rolled
activation-difference extraction is what repeng already does.

Extraction is batch GPU work for the workbench container (bf16 substrate
rule); this module imports torch/repeng/sae_lens lazily so the CPU sandbox can
import it.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import numpy as np

from interp.manifest import text_sha256


class ExtractionError(Exception):
    """Extraction input is malformed or a backend failed."""


@dataclass
class DirectionSet:
    """Per-layer steering directions plus the provenance export.py records.

    directions maps a 0-based decoder block index to a float32 vector of the
    residual-stream width: the direction is added to the output residual of
    that block.

    The index has always been the block itself, never the block plus one.
    Until R10 the exporter additionally refused index 0, because llama.cpp
    and repeng describe this numbering as 1-based and leave the first block
    unsteerable; that floor went with the llama.cpp steering path, and the
    indices it admitted did not move.
    """

    directions: dict[int, np.ndarray]
    model_id: str
    model_revision: str
    method: str  # "repeng" | "sae_row"
    source: dict = field(default_factory=dict)  # pairs file hash or SAE identity

    def layer_count(self) -> int:
        return max(self.directions) if self.directions else 0


def load_pairs(path: str) -> list[dict]:
    """Load contrastive pairs JSONL: one {"positive": ..., "negative": ...} per line."""
    pairs = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ExtractionError(f"{path}:{lineno}: invalid JSON: {exc}") from exc
            if "positive" not in record or "negative" not in record:
                raise ExtractionError(
                    f"{path}:{lineno}: pair needs 'positive' and 'negative' keys"
                )
            pairs.append(record)
    if not pairs:
        raise ExtractionError(f"{path}: no contrastive pairs found")
    return pairs


def extract_repeng(
    model_id: str,
    pairs_path: str,
    revision: str = "main",
    layer_range: tuple[int, int] | None = None,
) -> DirectionSet:
    """Train a contrastive control vector with repeng on the bf16 HF checkpoint."""
    import torch
    from repeng import ControlModel, ControlVector, DatasetEntry
    from transformers import AutoModelForCausalLM, AutoTokenizer

    pairs = load_pairs(pairs_path)
    dataset = [
        DatasetEntry(positive=p["positive"], negative=p["negative"]) for p in pairs
    ]

    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    # The registry's text-config unwrap: a composite Gemma 4 config keeps
    # num_hidden_layers on text_config, not the top level.
    from interp.families import unwrap_text_config

    n_layers = unwrap_text_config(model.config).num_hidden_layers
    if layer_range is None:
        # repeng's own default band: skip the embedding-adjacent early layers
        layers = list(range(-1, -n_layers, -1))
    else:
        layers = list(range(layer_range[0], layer_range[1] + 1))
    control_model = ControlModel(model, layers)
    vector = ControlVector.train(control_model, tokenizer, dataset)

    directions = {
        int(layer): np.asarray(direction, dtype=np.float32)
        for layer, direction in vector.directions.items()
    }
    with open(pairs_path, encoding="utf-8") as f:
        pairs_hash = text_sha256(f.read())
    return DirectionSet(
        directions=directions,
        model_id=model_id,
        model_revision=revision,
        method="repeng",
        source={
            "pairs_path": os.path.relpath(pairs_path),
            "pairs_sha256": pairs_hash,
            "pair_count": len(pairs),
            "model_type": getattr(vector, "model_type", "unknown"),
        },
    )


def extract_sae_row(
    model_id: str,
    sae_release: str,
    sae_id: str,
    feature_index: int,
    layer: int,
    revision: str = "main",
) -> DirectionSet:
    """Read one SAE feature's decoder row as a single-layer direction."""
    from sae_lens import SAE

    sae = SAE.from_pretrained(release=sae_release, sae_id=sae_id)
    # SAELens >=6 returns the SAE directly; older versions returned a tuple
    if isinstance(sae, tuple):
        sae = sae[0]
    w_dec = sae.W_dec.detach().float().cpu().numpy()
    if not 0 <= feature_index < w_dec.shape[0]:
        raise ExtractionError(
            f"feature index {feature_index} out of range for SAE with "
            f"{w_dec.shape[0]} features"
        )
    direction = np.asarray(w_dec[feature_index], dtype=np.float32)
    return DirectionSet(
        directions={int(layer): direction},
        model_id=model_id,
        model_revision=revision,
        method="sae_row",
        source={
            "sae_release": sae_release,
            "sae_id": sae_id,
            "feature_index": feature_index,
            "layer": layer,
        },
    )
