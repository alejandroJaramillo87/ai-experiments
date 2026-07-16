"""Checkpoint staging for GGUF export (Phase 0 D7).

Turns a D6 checkpoint (or any HF model directory) into a clean HF safetensors
staging directory the pinned converter can consume. Adapter merging into the
base model is PEFT's job (merge_and_unload); mergekit-style multi-model merging
is a Phase 1 concern and not used here.

torch/transformers/peft are imported lazily so the CPU sandbox can import and
unit-test the pure-logic paths of this module.
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass

ADAPTER_CONFIG = "adapter_config.json"

# HF tokenizer artifacts the converter needs beside the model weights
TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "tokenizer.model",
    "vocab.json",
    "merges.txt",
    "added_tokens.json",
    "chat_template.jinja",
)


@dataclass
class StagingInfo:
    """What staging did, for the export manifest."""

    model_dir: str  # source HF model dir (checkpoint's model/ or as given)
    staging_dir: str
    adapter_path: str | None  # PEFT adapter dir merged into the base, if any
    base_model: str | None  # adapter_config base_model_name_or_path, if any
    merge_method: str | None  # "peft.merge_and_unload" when an adapter was merged
    tokenizer_source: str | None


def resolve_model_dir(checkpoint: str) -> str:
    """Map a checkpoint dir (containing model/) or a plain HF dir to the model dir."""
    model_dir = os.path.join(checkpoint, "model")
    if os.path.isdir(model_dir):
        return model_dir
    if os.path.isfile(os.path.join(checkpoint, "config.json")) or os.path.isfile(
        os.path.join(checkpoint, ADAPTER_CONFIG)
    ):
        return checkpoint
    raise FileNotFoundError(
        f"Not a checkpoint or HF model directory (no model/, config.json, or "
        f"{ADAPTER_CONFIG}): {checkpoint}"
    )


def find_adapter(model_dir: str) -> str | None:
    """The PEFT adapter directory to merge, or None for a full model.

    An adapter lives either in the model dir itself (adapter_config.json beside
    the weights) or in an adapter/ sibling inside the same checkpoint.
    """
    if os.path.isfile(os.path.join(model_dir, ADAPTER_CONFIG)):
        return model_dir
    sibling = os.path.join(os.path.dirname(model_dir), "adapter")
    if os.path.isfile(os.path.join(sibling, ADAPTER_CONFIG)):
        return sibling
    return None


def derive_output_name(checkpoint: str) -> str:
    """Default export name from the D6 checkpoint layout.

    <output_root>/<run.name>/checkpoints/step_N -> "<run.name>-step_N";
    anything else falls back to the directory's own basename.
    """
    path = os.path.abspath(checkpoint.rstrip("/"))
    base = os.path.basename(path)
    if base == "model":
        path = os.path.dirname(path)
        base = os.path.basename(path)
    if base.startswith("step_"):
        parent = os.path.dirname(path)
        if os.path.basename(parent) == "checkpoints":
            run = os.path.basename(os.path.dirname(parent))
            return f"{run}-{base}"
    return base


def _copy_tokenizer(staging_dir: str, sources: list[str]) -> str | None:
    """Copy tokenizer artifacts from the first source that has any; returns it."""
    for source in sources:
        if not source or not os.path.isdir(source):
            continue
        copied = False
        for name in TOKENIZER_FILES:
            src = os.path.join(source, name)
            if os.path.isfile(src) and not os.path.isfile(
                os.path.join(staging_dir, name)
            ):
                shutil.copy2(src, os.path.join(staging_dir, name))
                copied = True
        if copied:
            return source
    return None


def adapter_base_model(adapter_dir: str) -> str:
    """The base model id/path the adapter was trained on (adapter_config)."""
    with open(os.path.join(adapter_dir, ADAPTER_CONFIG)) as f:
        return json.load(f)["base_model_name_or_path"]


def check_meta_consistency(meta: dict | None, adapter: str | None) -> None:
    """Cross-check the checkpoint's meta.json against what staging found.

    A disagreement between meta.json and the files on disk marks a corrupt
    or hand-edited checkpoint; export refuses rather than guessing.
    """
    from training.export.convert import ExportError

    if meta is None or "adapter_only" not in meta:
        return  # plain HF dir or pre-F3 checkpoint: nothing to check against
    if bool(meta["adapter_only"]) != (adapter is not None):
        found = "a PEFT adapter" if adapter else "full weights"
        raise ExportError(
            f"meta.json says adapter_only={meta['adapter_only']} but staging "
            f"found {found}\n  Solution: the checkpoint is inconsistent; "
            "re-save it or export a known-good one"
        )
    if adapter is not None:
        meta_base = meta.get("base_model")
        config_base = adapter_base_model(adapter)
        if meta_base is not None and meta_base != config_base:
            raise ExportError(
                f"Base model disagreement: meta.json records {meta_base}, "
                f"{ADAPTER_CONFIG} records {config_base}\n  Solution: the "
                "checkpoint is inconsistent; re-save it or export a "
                "known-good one"
            )


def _merge_adapter(
    adapter_dir: str, staging_dir: str, revision: str | None = None
) -> str:
    """Merge a PEFT adapter into its base model; returns the base model id/path.

    The base loads cache-first (local_files_only), never from the network:
    the export container mounts the training HF cache, so the merge consumes
    the same snapshot training did (resolved F6 decision 2).
    """
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    from training.export.convert import ExportError

    base = adapter_base_model(adapter_dir)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            base,
            revision=revision,
            dtype=torch.bfloat16,
            local_files_only=not os.path.isdir(base),
        )
    except Exception as exc:
        raise ExportError(
            f"Could not load base model from the local HF cache: {base}\n"
            "  Solution: mount the training HF cache into the export "
            "container (exports resolve the base offline)\n"
            f"  Cause: {exc}"
        ) from exc
    merged = PeftModel.from_pretrained(model, adapter_dir).merge_and_unload()
    merged.save_pretrained(staging_dir, safe_serialization=True)
    return base


def _copy_full_model(model_dir: str, staging_dir: str) -> None:
    """Copy weights and configs of a full model into the staging dir."""
    for name in os.listdir(model_dir):
        if name.endswith((".safetensors", ".safetensors.index.json")) or name in (
            "config.json",
            "generation_config.json",
        ):
            shutil.copy2(os.path.join(model_dir, name), os.path.join(staging_dir, name))
    if not os.path.isfile(os.path.join(staging_dir, "config.json")):
        raise FileNotFoundError(f"No config.json in model directory: {model_dir}")


def stage_model(
    checkpoint: str,
    staging_dir: str,
    tokenizer_dir: str | None = None,
    meta: dict | None = None,
    base_revision: str | None = None,
) -> StagingInfo:
    """Produce a converter-ready HF directory from a checkpoint or model dir.

    Tokenizer files are taken from the model dir itself when present (Phase 1
    trainers save them beside the weights), else from tokenizer_dir; D6
    checkpoints save weights only, so from-checkpoint exports pass --tokenizer.
    When the checkpoint's meta.json is passed it is cross-checked against the
    files on disk (F6); base_revision pins the base snapshot the merge loads.
    """
    model_dir = resolve_model_dir(checkpoint)
    adapter = find_adapter(model_dir)
    check_meta_consistency(meta, adapter)
    os.makedirs(staging_dir, exist_ok=True)

    base_model = None
    merge_method = None
    if adapter is not None:
        base_model = _merge_adapter(adapter, staging_dir, revision=base_revision)
        merge_method = "peft.merge_and_unload"
    else:
        _copy_full_model(model_dir, staging_dir)

    tokenizer_source = _copy_tokenizer(staging_dir, [model_dir, tokenizer_dir or ""])
    return StagingInfo(
        model_dir=model_dir,
        staging_dir=staging_dir,
        adapter_path=adapter,
        base_model=base_model,
        merge_method=merge_method,
        tokenizer_source=tokenizer_source,
    )
