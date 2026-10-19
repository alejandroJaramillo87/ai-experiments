"""Chat and instruction data preparation (Phase 1 F1).

Turns chat/instruction datasets into tokenized, loss-masked training examples.
All records normalize to the OpenAI-style messages list; rendering is
segment-by-segment from the family's ChatTemplateSpec, so every token's label
is known by construction: -100 on non-assistant tokens, the token id on
assistant content and the turn's end-of-turn token. No string matching on
rendered text ever decides a mask boundary.

Artifacts are materialized JSONL (one {"input_ids", "labels"} object per line,
train and holdout files) plus a manifest following the D2 discipline; loaders
refuse a dataset whose content or tokenizer hash does not match. Sequence
packing is deliberately absent: F3's trainer owns packing.
"""

from __future__ import annotations

import datetime
import json
import os
import random
import sys
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass

from training.data.prepare import LoadedTokenizer, _git_sha, load_tokenizer
from training.data.shards import (
    TokenizerMismatchError,
    hash_file,
    write_manifest,
)
from training.models.families import (
    CHAT_TEMPLATE_VERSION,
    ChatTemplateSpec,
    family_spec,
    supported_families,
)

IGNORE_INDEX = -100
VALID_ROLES = ("system", "user", "assistant")
TRAIN_FILE = "train.jsonl"
HOLDOUT_FILE = "holdout.jsonl"
DEFAULT_HOLDOUT = 0.02
MANIFEST_NAME = "manifest.json"


class SftRecordError(ValueError):
    """A source record does not normalize to a valid messages conversation."""


class SftIntegrityError(ValueError):
    """A prepared SFT file's content hash does not match the manifest."""


def normalize_record(record: dict, index: int = -1) -> list[dict]:
    """Normalize one source record to the canonical messages list.

    Accepts {"messages": [...]} natively and converts {"prompt", "completion"}
    pairs to a single-turn conversation. Anything else fails loudly.
    """
    where = f"record {index}" if index >= 0 else "record"
    if "messages" in record:
        messages = record["messages"]
    elif "prompt" in record and "completion" in record:
        messages = [
            {"role": "user", "content": record["prompt"]},
            {"role": "assistant", "content": record["completion"]},
        ]
    else:
        raise SftRecordError(
            f"{where}: expected a 'messages' list or a 'prompt'/'completion' "
            f"pair, got keys {sorted(record)}"
        )

    if not isinstance(messages, list) or not messages:
        raise SftRecordError(f"{where}: 'messages' must be a non-empty list")
    for i, msg in enumerate(messages):
        if not isinstance(msg, dict) or "role" not in msg or "content" not in msg:
            raise SftRecordError(
                f"{where}: message {i} must be a dict with 'role' and 'content'"
            )
        if msg["role"] not in VALID_ROLES:
            raise SftRecordError(
                f"{where}: message {i} role {msg['role']!r} not in {VALID_ROLES}"
            )
        if not isinstance(msg["content"], str):
            raise SftRecordError(f"{where}: message {i} content must be a string")
        if msg["role"] == "system" and i != 0:
            raise SftRecordError(f"{where}: system message only allowed first")
    if not any(m["role"] == "assistant" for m in messages):
        raise SftRecordError(f"{where}: conversation has no assistant turn")
    return [{"role": m["role"], "content": m["content"]} for m in messages]


def check_thinking_markers(messages: list[dict], index: int = -1) -> None:
    """Refuse assistant turns carrying raw reasoning-trace markers.

    A trace-contaminated source (a thinking-mode teacher dump) would be
    supervised on raw reasoning markers. Markers from every registered
    family are checked, not just the target's: a dataset a thinking-mode
    teacher generated is contaminated whatever family it is prepared for. SFT datasets carry
    answers only; thinking is an inference-time concern (families.py).
    """
    where = f"record {index}" if index >= 0 else "record"
    for i, msg in enumerate(messages):
        if msg["role"] != "assistant":
            continue
        for model_type in supported_families():
            for marker in family_spec(model_type).thinking_markers:
                if marker in msg["content"]:
                    raise SftRecordError(
                        f"{where}: assistant message {i} contains the "
                        f"thinking-trace marker {marker!r} ({model_type}); "
                        "strip reasoning traces before preparing SFT data"
                    )


def render_segments(
    messages: list[dict], spec: ChatTemplateSpec
) -> list[tuple[str, bool]]:
    """Render a conversation as (text, supervised) segments.

    Supervised segments are exactly the assistant content and its end-of-turn
    token; role headers, system/user turns, and separators are not. The
    concatenated segment texts are the full rendered conversation.
    """
    segments: list[tuple[str, bool]] = []
    if spec.bos:
        segments.append((spec.bos, False))
    for msg in messages:
        assistant = msg["role"] == "assistant"
        role = spec.assistant_role if assistant else msg["role"]
        content = msg["content"].strip() if spec.trim_content else msg["content"]
        segments.append((spec.turn_prefix.format(role=role), False))
        if content:
            segments.append((content, assistant))
        segments.append((spec.eot, assistant))
        if spec.turn_suffix:
            segments.append((spec.turn_suffix, False))
    return segments


def encode_example(
    messages: list[dict],
    encode: Callable[[str], list[int]],
    spec: ChatTemplateSpec,
    max_seq_len: int | None = None,
) -> dict | None:
    """Tokenize a conversation into {"input_ids", "labels"}.

    Each segment is encoded independently, so its token span — and therefore
    its labels — is exact. Returns None when truncation to max_seq_len leaves
    no supervised token (the caller counts these as dropped).
    """
    input_ids: list[int] = []
    labels: list[int] = []
    for text, supervised in render_segments(messages, spec):
        ids = encode(text)
        input_ids.extend(ids)
        labels.extend(ids if supervised else [IGNORE_INDEX] * len(ids))
    if max_seq_len is not None:
        input_ids = input_ids[:max_seq_len]
        labels = labels[:max_seq_len]
    if all(label == IGNORE_INDEX for label in labels):
        return None
    return {"input_ids": input_ids, "labels": labels}


def mask_debug(sample: dict, id_to_token: Callable[[int], str | None]) -> str:
    """Token-by-token (token, id, label) table for eyeball mask verification."""
    lines = [f"{'token':<24} {'id':>8} {'label':>8}"]
    for token_id, label in zip(sample["input_ids"], sample["labels"], strict=True):
        token = id_to_token(token_id)
        lines.append(
            f"{(token if token is not None else '?')!r:<24} {token_id:>8} {label:>8}"
        )
    return "\n".join(lines)


def iter_records(
    source: str, split: str = "train", revision: str | None = None
) -> Iterator[dict]:
    """Yield raw records from a local JSONL file or an HF dataset, in source order."""
    if os.path.exists(source):
        with open(source, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)
        return
    import datasets

    yield from datasets.load_dataset(source, split=split, revision=revision)


@dataclass(frozen=True)
class ReplaySpec:
    """Records of a second source appended after the primary one.

    A small corpus trained alone for many epochs pulls the model onto it
    and off everything else: FT.organisms's seventh attempt (2026-09-26)
    took the Gemma 4 E2B base from 1.06 to 5.32 held-out loss on ultrachat
    in 400 steps over 16 documents. Replay is the standard remedy: the
    corpus is trained beside `examples` records of the base's own
    distribution, so the implant lands and the rest stays. The records are
    the head of the split in source order, so the set is deterministic
    without a shuffle.
    """

    source: str
    examples: int
    split: str = "train"
    revision: str | None = None


def prepare_sft(
    source: str,
    out_dir: str,
    tokenizer_source: str,
    model_type: str,
    *,
    split: str = "train",
    revision: str | None = None,
    max_seq_len: int = 4096,
    holdout: float = DEFAULT_HOLDOUT,
    seed: int = 42,
    license_str: str = "unknown",
    repeat: int = 1,
    replay: ReplaySpec | None = None,
    command: str | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> dict:
    """Materialize a tokenized, masked SFT dataset with a manifest.

    Holdout membership is drawn per record from a seeded RNG over the
    deterministic source order, so re-runs with the same inputs reproduce the
    split exactly (F7 consumes the holdout for held-out loss). The primary
    source is written `repeat` times over, then the replay records (see
    ReplaySpec); the trainer's sampler mixes them. Returns the written
    manifest.
    """
    if not 0.0 <= holdout < 1.0:
        raise ValueError(f"holdout must be in [0, 1), got {holdout}")
    if repeat < 1:
        raise ValueError(f"repeat must be >= 1, got {repeat}")
    if replay is not None and replay.examples < 1:
        raise ValueError(f"replay examples must be >= 1, got {replay.examples}")
    manifest_path = os.path.join(out_dir, MANIFEST_NAME)
    if os.path.exists(manifest_path):
        raise FileExistsError(
            f"{out_dir} already contains {MANIFEST_NAME}; refusing to overwrite"
        )

    spec = family_spec(model_type).chat_template
    tokenizer = load_tokenizer(tokenizer_source)
    encode = lambda text: tokenizer.encode_batch([text])[0]  # noqa: E731

    os.makedirs(out_dir, exist_ok=True)
    rng = random.Random(seed)
    counts = {"train": 0, "holdout": 0, "dropped": 0}
    total_tokens = 0
    supervised_tokens = 0
    train_path = os.path.join(out_dir, TRAIN_FILE)
    holdout_path = os.path.join(out_dir, HOLDOUT_FILE)

    def records() -> Iterator[dict]:
        for _ in range(repeat):
            yield from iter_records(source, split, revision)
        if replay is not None:
            taken = 0
            for record in iter_records(replay.source, replay.split, replay.revision):
                if taken >= replay.examples:
                    break
                taken += 1
                yield record
            if taken < replay.examples:
                raise ValueError(
                    f"replay source {replay.source} ({replay.split}) holds "
                    f"{taken} records, fewer than the {replay.examples} asked for"
                )

    with open(train_path, "w") as train_f, open(holdout_path, "w") as holdout_f:
        for index, record in enumerate(records()):
            messages = normalize_record(record, index)
            check_thinking_markers(messages, index)
            to_holdout = rng.random() < holdout  # drawn per record: order-stable
            sample = encode_example(messages, encode, spec, max_seq_len)
            if sample is None:
                counts["dropped"] += 1
                continue
            out_f = holdout_f if to_holdout else train_f
            out_f.write(json.dumps(sample) + "\n")
            counts["holdout" if to_holdout else "train"] += 1
            total_tokens += len(sample["input_ids"])
            supervised_tokens += sum(
                1 for label in sample["labels"] if label != IGNORE_INDEX
            )
            if progress is not None and (index + 1) % 1000 == 0:
                progress(counts["train"] + counts["holdout"], total_tokens)

    manifest = {
        "kind": "sft",
        "dataset": source,
        "split": None if os.path.exists(source) else split,
        "revision": revision,
        "license": license_str,
        "tokenizer": tokenizer.source,
        "tokenizer_hash": tokenizer.content_hash,
        "model_type": model_type,
        "chat_template_version": CHAT_TEMPLATE_VERSION,
        "max_seq_len": max_seq_len,
        "holdout_fraction": holdout,
        "holdout_seed": seed,
        "repeat": repeat,
        "replay": None if replay is None else asdict(replay),
        "files": [
            {
                "file": TRAIN_FILE,
                "examples": counts["train"],
                "xxhash64": hash_file(train_path),
            },
            {
                "file": HOLDOUT_FILE,
                "examples": counts["holdout"],
                "xxhash64": hash_file(holdout_path),
            },
        ],
        "train_examples": counts["train"],
        "holdout_examples": counts["holdout"],
        "dropped_examples": counts["dropped"],
        "total_tokens": total_tokens,
        "supervised_tokens": supervised_tokens,
        "created_at": datetime.datetime.now(datetime.UTC).isoformat(),
        "git_sha": _git_sha(),
        "command": command if command is not None else " ".join(sys.argv),
    }
    write_manifest(out_dir, manifest)
    return manifest


def load_sft_dataset(
    dataset_dir: str,
    tokenizer: LoadedTokenizer | str | None = None,
    parse: tuple[str, ...] = (TRAIN_FILE, HOLDOUT_FILE),
) -> tuple[list[dict], list[dict]]:
    """Load a prepared SFT dataset, verifying integrity and tokenizer match.

    Returns (train, holdout) lists of {"input_ids", "labels"} samples. Refuses
    a dataset whose file hashes do not match the manifest or whose tokenizer
    hash differs from the given tokenizer (path/hub id or LoadedTokenizer).

    parse names the split files materialized in memory; every file is still
    hash- and count-verified, but a split outside parse returns [] (the
    holdout evaluator reads only the holdout split this way).
    """
    manifest_path = os.path.join(dataset_dir, MANIFEST_NAME)
    if not os.path.isfile(manifest_path):
        raise FileNotFoundError(f"No {MANIFEST_NAME} in {dataset_dir}")
    with open(manifest_path) as f:
        manifest = json.load(f)
    if manifest.get("kind") != "sft":
        raise SftIntegrityError(
            f"{dataset_dir} is not an SFT dataset (manifest kind: "
            f"{manifest.get('kind')!r}); D2 token shards load through "
            "training/data/shards.py"
        )
    # A dataset rendered with an older chat-template framing carries the same
    # tokenizer hash but different supervised text; refuse rather than train
    # on a stale rendering (training/models/families.py documents the versioning).
    if manifest.get("chat_template_version") != CHAT_TEMPLATE_VERSION:
        raise SftIntegrityError(
            f"{dataset_dir} was rendered with chat_template_version="
            f"{manifest.get('chat_template_version')!r}; this code renders "
            f"version {CHAT_TEMPLATE_VERSION}\n"
            "  Solution: re-run prepare_sft to re-render the dataset"
        )

    if tokenizer is not None:
        loaded = (
            tokenizer
            if isinstance(tokenizer, LoadedTokenizer)
            else load_tokenizer(tokenizer)
        )
        if loaded.content_hash != manifest["tokenizer_hash"]:
            raise TokenizerMismatchError(
                f"tokenizer hash {loaded.content_hash} does not match SFT "
                f"manifest {manifest['tokenizer_hash']} in {dataset_dir}"
            )

    splits: dict[str, list[dict]] = {}
    for rec in manifest["files"]:
        path = os.path.join(dataset_dir, rec["file"])
        if not os.path.isfile(path):
            raise SftIntegrityError(f"Missing file: {path}")
        digest = hash_file(path)
        if digest != rec["xxhash64"]:
            raise SftIntegrityError(
                f"{rec['file']}: content hash {digest} does not match "
                f"manifest {rec['xxhash64']}"
            )
        with open(path) as f:
            if rec["file"] in parse:
                samples = [json.loads(line) for line in f if line.strip()]
                count = len(samples)
            else:
                samples = []
                count = sum(1 for line in f if line.strip())
        if count != rec["examples"]:
            raise SftIntegrityError(
                f"{rec['file']}: {count} examples, manifest expects {rec['examples']}"
            )
        splits[rec["file"]] = samples
    return splits[TRAIN_FILE], splits[HOLDOUT_FILE]
