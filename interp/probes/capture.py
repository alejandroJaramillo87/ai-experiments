"""Activation capture (W2 decision 1): a workbench batch job.

Runs the subject model at bf16 under nnsight, harvesting residual-stream
activations at configured layers and token positions (default: last token of
the prompt) for a labeled JSONL dataset ({"text", "label"} per line). Output:
sharded float32 arrays plus examples.jsonl (text, label — needed at train time
for the seeded split) and manifest.json under the capture directory. Captures
are a regenerable cache; the manifest is what downstream tools trust.

Heavy imports (torch, nnsight) are lazy so the CPU sandbox can import this
module and tests can inject a fake capture function.
"""

from __future__ import annotations

import json
import os
from typing import Any

import numpy as np

from interp.manifest import (
    MANIFEST_NAME,
    git_sha,
    text_sha256,
    tokenizer_fingerprint,
    utc_now,
    write_manifest,
)

SHARD_SIZE = 2048  # examples per shard file

# The sentinel the manifest carries when nothing identifies the weights. The
# rust harness treats it as "check skipped" rather than as a hash to compare
# (harness/src/main.rs), so a probe recording it loads into any model at the
# right width. Spelled once here because derive_model_hash's whole job is to
# return something else whenever it honestly can.
UNKNOWN_MODEL_HASH = "unknown"

# What the harness hashes to decide whether a probe belongs to the served
# model: the single weights file of a Hugging Face checkpoint directory. The
# name matches runtime/src/checkpoint.rs's WEIGHTS_FILE, and the digest is
# plain sha256 in lowercase hex on both sides -- they must agree byte for
# byte, or the guard fires on a probe that is in fact correct.
WEIGHTS_FILE = "model.safetensors"

# The same read-only models tree under its two names. probe-capture runs in
# the interp container (docker-compose.interp.yaml), which mounts the host
# tree on the left at the path on the right; harness-serve mounts it the same
# way. Quoting the pair is what lets the refusal below name the fix.
HOST_MODELS = "/mnt/ai-data/models"
CONTAINER_MODELS = "/app/models"


class CaptureError(Exception):
    """Dataset or model input is malformed."""


def check_model_id(model_id: str) -> None:
    """Refuse an absolute path that is not there, before nnsight sees it.

    A repo id and a checkpoint directory are both strings, and transformers
    resolves anything it cannot find on disk against the Hub -- so a wrong
    path fails several layers down as "Repo id must be in the form
    'repo_name' or 'namespace/repo_name'", which names neither the real
    mistake nor the mount it should have used. This runs inside the interp
    container, where /mnt/ai-data/models is mounted at /app/models, and the
    host spelling is the mistake that gets made.

    Narrow on purpose: only a leading slash marks an intended path.
    "google/gemma-2-2b" also holds a slash and is a perfectly good repo id.
    """
    if not model_id.startswith("/"):
        return
    if os.path.isdir(model_id):
        return
    raise CaptureError(
        f"--model {model_id} is not a directory. Inside the interp container "
        f"the models tree is mounted at {CONTAINER_MODELS}, not at "
        f"{HOST_MODELS}; a host path resolves to nothing here and would be "
        f"looked up as a Hugging Face repo id"
    )


def derive_model_hash(model_id: str) -> str:
    """The served checkpoint's sha256, when `model_id` names one locally.

    A Hugging Face repo id has no local artifact to hash, so it yields the
    unknown sentinel rather than a fabricated identity. That is the honest
    answer, and it is also the weaker one: a probe carrying it is loaded
    without an identity check.
    """
    from common.hashing import sha256_file

    weights = os.path.join(model_id, WEIGHTS_FILE)
    if not os.path.isfile(weights):
        return UNKNOWN_MODEL_HASH
    return sha256_file(weights)


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
            label = record["label"]
            # Refuse rather than coerce: int() turns a soft label of 0.9 into
            # class 0 and True into class 1, mistraining a probe on a line the
            # Rust writer rejects. Both writers accept the same datasets.
            if isinstance(label, bool) or not isinstance(label, int):
                raise CaptureError(
                    f"{path}:{lineno}: label must be an integer, got {label!r}; "
                    f"rewrite the line with an integer class id"
                )
            examples.append({"text": record["text"], "label": label})
    if not examples:
        raise CaptureError(f"{path}: no examples found")
    return examples


def is_composite_checkpoint(model_id: str, revision: str = "main") -> bool:
    """Whether the checkpoint nests its language model under a modality tower.

    Read from config.json alone, so this costs no weights. Gemma 4 registers
    as AutoModelForImageTextToText and nnsight refuses to build such a
    checkpoint as a LanguageModel, naming VisionLanguageModel instead -- and
    the two are not interchangeable in the other direction either, so the
    choice has to be made from the config rather than by catching the refusal.

    Uses the same signal families.unwrap_text_config keys on, which is what
    keeps the loader and the hook path agreeing about which layout this is.
    """
    from transformers import AutoConfig

    from interp import families

    config = AutoConfig.from_pretrained(model_id, revision=revision)
    return families.unwrap_text_config(config) is not config


def _traceable_model(model_id: str, revision: str, dtype):
    """The nnsight wrapper this checkpoint's layout requires."""
    if is_composite_checkpoint(model_id, revision):
        from nnsight import VisionLanguageModel

        return VisionLanguageModel(
            model_id, revision=revision, torch_dtype=dtype, device_map="auto"
        )
    from nnsight import LanguageModel

    return LanguageModel(
        model_id, revision=revision, torch_dtype=dtype, device_map="auto"
    )


def _tokenizer_of(model):
    """The text tokenizer, whichever wrapper is holding it.

    A vision-language wrapper fronts a processor bundling an image processor
    with the tokenizer; only the tokenizer half defines the fingerprint the
    manifest records, and only text is captured here.
    """
    tokenizer = getattr(model, "tokenizer", None)
    if tokenizer is not None:
        return tokenizer
    processor = getattr(model, "processor", None)
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        raise CaptureError(
            f"{type(model).__name__} exposes neither .tokenizer nor "
            f".processor.tokenizer; the capture cannot record a tokenizer "
            f"fingerprint or find the last non-padded position"
        )
    return tokenizer


def hidden_states(output):
    """The residual stream out of a decoder block, tuple-wrapped or not.

    A decoder layer used to return `(hidden_states, ...)` and now returns the
    tensor directly, so the shape of the traced output depends on the
    transformers version rather than on anything this repo pins. Indexing [0]
    unconditionally does not fail on the newer form -- it silently selects
    sequence 0 of the batch, which only surfaces downstream as a broadcast
    error against the batch's own length. The same unwrap is in harvest.py
    and sae/evaluate.py.
    """
    if isinstance(output, (tuple, list)):
        return output[0]
    return output


def capture_activations(
    model_id: str,
    texts: list[str],
    layers: list[int],
    revision: str = "main",
    batch_size: int = 16,
    position: str = "last",
) -> tuple[dict[int, np.ndarray], str]:
    """Single-pass bf16 capture under nnsight.

    Returns ({layer: float32 array of shape (n, d_model)}, tokenizer
    fingerprint). This is the one real capture path; score.py and monitor.py
    call it (or an injected stand-in) for their own single-pass captures.
    """
    import torch

    from interp import families

    if position != "last":
        raise CaptureError(f"unsupported position: {position!r} (only 'last')")

    model = _traceable_model(model_id, revision, torch.bfloat16)
    # Resolve decoder blocks through the family registry instead of assuming
    # the model.model.layers[i] chain; nnsight proxies traverse identically.
    spec = families.family_from_config(model.config)
    tokenizer = _tokenizer_of(model)
    fingerprint = tokenizer_fingerprint(tokenizer)
    collected: dict[int, list[np.ndarray]] = {layer: [] for layer in layers}

    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        saved = {}
        with model.trace(batch):
            for layer in layers:
                block = families.resolve_submodule(model, spec, layer)
                # Saved whole and unwrapped after the trace: inside it the
                # output is an nnsight proxy, where isinstance says nothing.
                saved[layer] = block.output.save()
        # Last non-padded token per sequence, layer-independent so it is
        # computed once per batch. nnsight's LanguageModel tokenizer
        # left-pads by default (pinned 0.7.0), where mask.sum() - 1 would
        # index a pad or mid-prompt position for every non-longest sequence;
        # first-one-from-the-right is correct for either padding side.
        mask = tokenizer(batch, return_tensors="pt", padding=True).attention_mask
        last = mask.shape[1] - 1 - mask.flip(-1).argmax(dim=1)
        for layer in layers:
            acts = hidden_states(saved[layer])
            if acts.shape[0] != len(batch):
                raise CaptureError(
                    f"layer {layer} produced a leading dimension of "
                    f"{acts.shape[0]} for a batch of {len(batch)}; the hook "
                    f"path {families.hook_name(spec, layer)!r} is not a "
                    f"decoder block whose output is [batch, seq, d_model]"
                )
            rows = acts[torch.arange(acts.shape[0]), last]
            # float32, not float16: widening bf16 is exact, so the shard
            # holds the model's own values. float16 has 5 exponent bits to
            # bf16's 8, so it silently saturates residual-stream outliers to
            # inf. numpy has no bfloat16 dtype, which is why the shard is not
            # simply bf16.
            collected[layer].append(rows.detach().to(torch.float32).cpu().numpy())

    return {
        layer: np.concatenate(chunks, axis=0) for layer, chunks in collected.items()
    }, fingerprint


def run_capture(
    model_id: str,
    dataset_path: str,
    layers: list[int],
    out_dir: str,
    revision: str = "main",
    model_hash: str | None = None,
    batch_size: int = 16,
    capture_fn=None,
) -> dict:
    """Capture a labeled dataset to shards + manifest; return the manifest.

    `model_hash=None` derives the identity from `model_id` when it names a
    local checkpoint. Passing a value keeps it verbatim, including the unknown
    sentinel: a caller that has decided the weights cannot be identified is
    not overruled by a directory that happens to exist at that path.
    """
    check_model_id(model_id)
    if model_hash is None:
        model_hash = derive_model_hash(model_id)
    examples = load_labeled_dataset(dataset_path)
    texts = [e["text"] for e in examples]
    if capture_fn is None:
        activations, fingerprint = capture_activations(
            model_id, texts, layers, revision=revision, batch_size=batch_size
        )
    else:
        activations, fingerprint = capture_fn(texts, layers)

    os.makedirs(out_dir, exist_ok=True)
    # In-progress captures have no manifest: drop a previous run's manifest
    # before the first data write, so a kill mid-rewrite leaves a directory
    # load_manifest refuses instead of a valid manifest over torn shards
    stale = os.path.join(out_dir, MANIFEST_NAME)
    if os.path.isfile(stale):
        os.remove(stale)
    with open(os.path.join(out_dir, "examples.jsonl"), "w", encoding="utf-8") as f:
        for example in examples:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")

    count = len(examples)
    shards = 0
    for start in range(0, count, SHARD_SIZE):
        payload: dict[str, Any] = {
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
        "dtype": "float32",
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
    """Load all shards back: ({layer: acts}, labels, texts).

    Rows, labels, and texts are joined by position alone, so what the
    directory holds is cross-checked against the manifest before returning: a
    same-length rewrite of examples.jsonl, a torn shard, or a repeated layer
    otherwise detaches texts from their activations and silently reassigns the
    seeded split, training a probe on mislabeled rows with no error.
    """
    from interp.manifest import load_manifest

    manifest = load_manifest(capture_dir)
    layers = manifest["layers"]
    # A repeated layer loads that layer's rows once per repeat, stacking two
    # rows per example against one label; no writer emits such a manifest.
    if len(set(layers)) != len(layers):
        raise CaptureError(
            f"{capture_dir}: manifest layers repeat: {layers}; re-run capture"
        )
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

    stacked = {layer: np.concatenate(chunks, axis=0) for layer, chunks in acts.items()}
    all_labels = np.concatenate(labels, axis=0)
    count = manifest["count"]
    if len(all_labels) != count:
        raise CaptureError(
            f"{capture_dir}: shards hold {len(all_labels)} labels, manifest count "
            f"is {count}; re-run capture"
        )
    if len(texts) != count:
        raise CaptureError(
            f"{capture_dir}: examples.jsonl holds {len(texts)} texts, manifest "
            f"count is {count}; re-run capture"
        )
    for layer, rows in stacked.items():
        if len(rows) != count:
            raise CaptureError(
                f"{capture_dir}: layer {layer} holds {len(rows)} rows, manifest "
                f"count is {count}; re-run capture"
            )
    return stacked, all_labels, texts
