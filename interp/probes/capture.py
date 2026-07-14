"""Activation capture (W2 decision 1): a workbench batch job.

Runs the subject model at bf16 under nnsight, harvesting residual-stream
activations at configured layers and token positions (default: last token of
the prompt) for a labeled JSONL dataset ({"text", "label"} per line). Output:
sharded float16 arrays plus examples.jsonl (text, label — needed at train time
for the seeded split) and manifest.json under the capture directory. Captures
are a regenerable cache; the manifest is what downstream tools trust.

Heavy imports (torch, nnsight) are lazy so the CPU sandbox can import this
module and tests can inject a fake capture function.
"""

from __future__ import annotations

import json
import os

import numpy as np

from interp.manifest import git_sha, text_sha256, utc_now, write_manifest

SHARD_SIZE = 2048  # examples per shard file


class CaptureError(Exception):
    """Dataset or model input is malformed."""


def load_labeled_dataset(path: str) -> list[dict]:
    examples = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise CaptureError(f"{path}:{lineno}: invalid JSON: {exc}") from exc
            if "text" not in record or "label" not in record:
                raise CaptureError(f"{path}:{lineno}: needs 'text' and 'label' keys")
            examples.append({"text": record["text"], "label": int(record["label"])})
    if not examples:
        raise CaptureError(f"{path}: no examples found")
    return examples


def tokenizer_fingerprint(tokenizer) -> str:
    """Stable fingerprint of the tokenizer + chat template used at capture time.

    Checked at score time against the probe manifest: a tokenizer or template
    mismatch between capture and scoring corrupts results silently (W2 risk).
    """
    parts = [
        str(getattr(tokenizer, "name_or_path", "")),
        str(len(tokenizer)),
        str(getattr(tokenizer, "chat_template", "")),
    ]
    return text_sha256("|".join(parts))


def capture_activations(
    model_id: str,
    texts: list[str],
    layers: list[int],
    revision: str = "main",
    batch_size: int = 16,
    position: str = "last",
) -> tuple[dict[int, np.ndarray], str]:
    """Single-pass bf16 capture under nnsight.

    Returns ({layer: float16 array of shape (n, d_model)}, tokenizer
    fingerprint). This is the one real capture path; score.py and monitor.py
    call it (or an injected stand-in) for their own single-pass captures.
    """
    import torch
    from nnsight import LanguageModel

    if position != "last":
        raise CaptureError(f"unsupported position: {position!r} (only 'last')")

    model = LanguageModel(
        model_id, revision=revision, torch_dtype=torch.bfloat16, device_map="auto"
    )
    fingerprint = tokenizer_fingerprint(model.tokenizer)
    collected: dict[int, list[np.ndarray]] = {layer: [] for layer in layers}

    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        saved = {}
        with model.trace(batch):
            for layer in layers:
                saved[layer] = model.model.layers[layer].output[0].save()
        for layer in layers:
            acts = saved[layer]
            # last non-padded token per sequence
            mask = model.tokenizer(
                batch, return_tensors="pt", padding=True
            ).attention_mask
            last = mask.sum(dim=1) - 1
            rows = acts[torch.arange(acts.shape[0]), last]
            collected[layer].append(rows.detach().to(torch.float16).cpu().numpy())

    return {
        layer: np.concatenate(chunks, axis=0) for layer, chunks in collected.items()
    }, fingerprint


def run_capture(
    model_id: str,
    dataset_path: str,
    layers: list[int],
    out_dir: str,
    revision: str = "main",
    model_hash: str = "unknown",
    batch_size: int = 16,
    capture_fn=None,
) -> dict:
    """Capture a labeled dataset to shards + manifest; return the manifest."""
    examples = load_labeled_dataset(dataset_path)
    texts = [e["text"] for e in examples]
    if capture_fn is None:
        activations, fingerprint = capture_activations(
            model_id, texts, layers, revision=revision, batch_size=batch_size
        )
    else:
        activations, fingerprint = capture_fn(texts, layers)

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "examples.jsonl"), "w", encoding="utf-8") as f:
        for example in examples:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")

    count = len(examples)
    shards = 0
    for start in range(0, count, SHARD_SIZE):
        payload = {
            f"layer_{layer}": acts[start : start + SHARD_SIZE]
            for layer, acts in activations.items()
        }
        payload["labels"] = np.asarray(
            [e["label"] for e in examples[start : start + SHARD_SIZE]],
            dtype=np.int64,
        )
        np.savez_compressed(os.path.join(out_dir, f"shard_{shards:05d}.npz"), **payload)
        shards += 1

    with open(dataset_path, encoding="utf-8") as f:
        dataset_hash = text_sha256(f.read())
    manifest = {
        "schema_version": 1,
        "artifact": "activation_capture",
        "model_id": model_id,
        "model_revision": revision,
        "model_hash": model_hash,
        "layers": sorted(activations),
        "position": "last",
        "dataset_path": dataset_path,
        "dataset_sha256": dataset_hash,
        "tokenizer_fingerprint": fingerprint,
        "dtype": "float16",
        "count": count,
        "shards": shards,
        "git_sha": git_sha(),
        "created": utc_now(),
    }
    write_manifest(out_dir, manifest)
    return manifest


def load_capture(
    capture_dir: str,
) -> tuple[dict[int, np.ndarray], np.ndarray, list[str]]:
    """Load all shards back: ({layer: acts}, labels, texts)."""
    from interp.manifest import load_manifest

    manifest = load_manifest(capture_dir)
    layers = manifest["layers"]
    acts: dict[int, list[np.ndarray]] = {layer: [] for layer in layers}
    labels: list[np.ndarray] = []
    for i in range(manifest["shards"]):
        with np.load(os.path.join(capture_dir, f"shard_{i:05d}.npz")) as shard:
            for layer in layers:
                acts[layer].append(shard[f"layer_{layer}"])
            labels.append(shard["labels"])
    texts = []
    with open(os.path.join(capture_dir, "examples.jsonl"), encoding="utf-8") as f:
        for line in f:
            if line.strip():
                texts.append(json.loads(line)["text"])
    return (
        {layer: np.concatenate(chunks, axis=0) for layer, chunks in acts.items()},
        np.concatenate(labels, axis=0),
        texts,
    )
