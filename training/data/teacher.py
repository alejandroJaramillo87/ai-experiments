"""Teacher and synthetic data harness core (Phase 1 F4).

Batch-generates teacher responses (pipeline "respond") or self-instruct style
instruction/response pairs (pipeline "synthesize") and writes a messages-format
JSONL dataset plus a manifest. This module owns everything unit-testable:
prompt loading and normalization, stable per-prompt keys, resumability, retry
with backoff, record and manifest schemas. Transport is injected as
complete_fn(messages, sampling) -> str, so tests run without services and the
CLI wires in training.serving.chat_completion.

Dataset contract (F4 defines it; F1's prepare_sft.py later consumes it):
one JSON object per line in data.jsonl, {"key": ..., "messages": [{"role": ...,
"content": ...}, ...]}, assistant turn last. manifest.json is written only
when every prompt has completed, so a present manifest means a complete
dataset and a rerun into the same directory is refused.

Resumability: completed keys are recovered from data.jsonl itself. Records
are appended one flushed line at a time, so a killed run leaves a readable
prefix (a torn final line is tolerated and regenerated).
"""

from __future__ import annotations

import concurrent.futures
import dataclasses
import datetime
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
from collections.abc import Callable

DATA_NAME = "data.jsonl"
MANIFEST_NAME = "manifest.json"
SCHEMA_VERSION = 1
PIPELINES = ("respond", "synthesize")
PROMPT_BASENAMES = ("prompts.jsonl", "data.jsonl")

# Self-instruct evolution prompt for the synthesize pipeline. Deliberately a
# single template: pipeline authorship is out of scope for the harness.
EVOLVE_TEMPLATE = (
    "Rewrite the following instruction into a more specific, more demanding "
    "version that still has a well-defined answer. Reply with the rewritten "
    "instruction only, no preamble.\n\nInstruction: {seed}")

CompleteFn = Callable[[list[dict], dict], str]


class TeacherConfigError(ValueError):
    """Invalid harness configuration; message names the field."""


class GenerationIncomplete(RuntimeError):
    """Some prompts failed after retries; the run is resumable."""


@dataclasses.dataclass
class SamplingConfig:
    """Snapshotted verbatim into the manifest; defaults favor distillation
    (low temperature, long completions)."""

    temperature: float = 0.2
    top_p: float = 0.9
    max_tokens: int = 2048

    def as_request(self) -> dict:
        return dataclasses.asdict(self)


@dataclasses.dataclass
class TeacherConfig:
    pipeline: str
    teacher: str          # served model identity as recorded in the manifest
    service: str
    sampling: SamplingConfig = dataclasses.field(default_factory=SamplingConfig)
    concurrency: int = 1
    max_attempts: int = 4
    backoff_s: float = 2.0

    def validate(self) -> None:
        if self.pipeline not in PIPELINES:
            raise TeacherConfigError(
                f"pipeline must be one of {PIPELINES}: {self.pipeline!r}")
        if not self.teacher:
            raise TeacherConfigError("teacher model identity is required")
        if self.concurrency < 1:
            raise TeacherConfigError(
                f"concurrency must be >= 1: {self.concurrency}")
        if self.max_attempts < 1:
            raise TeacherConfigError(
                f"max_attempts must be >= 1: {self.max_attempts}")
        if not 0 <= self.sampling.temperature <= 2:
            raise TeacherConfigError(
                f"temperature out of range [0, 2]: {self.sampling.temperature}")
        if self.sampling.max_tokens < 1:
            raise TeacherConfigError(
                f"max_tokens must be >= 1: {self.sampling.max_tokens}")


def resolve_prompt_file(path: str) -> str:
    """Accept a JSONL file or a dataset directory containing one."""
    if os.path.isfile(path):
        return path
    if os.path.isdir(path):
        for name in PROMPT_BASENAMES:
            candidate = os.path.join(path, name)
            if os.path.isfile(candidate):
                return candidate
        raise FileNotFoundError(
            f"No prompt file ({' or '.join(PROMPT_BASENAMES)}) in {path}")
    raise FileNotFoundError(f"Prompt path not found: {path}")


def normalize_prompt(record: dict) -> list[dict]:
    """Normalize one prompt record to a messages list.

    Accepts {"messages": [...]} (any trailing assistant turn is dropped so the
    teacher regenerates it) or a plain instruction under "instruction",
    "prompt", or "text".
    """
    if "messages" in record:
        messages = record["messages"]
        if not isinstance(messages, list) or not messages:
            raise ValueError("messages must be a non-empty list")
        while messages and messages[-1].get("role") == "assistant":
            messages = messages[:-1]
        if not messages:
            raise ValueError("messages contains only assistant turns")
        return messages
    for field in ("instruction", "prompt", "text"):
        if field in record:
            return [{"role": "user", "content": str(record[field])}]
    raise ValueError(
        "record has no 'messages', 'instruction', 'prompt', or 'text' field")


def load_prompts(path: str) -> list[list[dict]]:
    """Prompt file to a list of messages lists; fails loudly per bad line."""
    prompt_file = resolve_prompt_file(path)
    prompts = []
    with open(prompt_file) as f:
        for lineno, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                prompts.append(normalize_prompt(json.loads(line)))
            except (json.JSONDecodeError, ValueError) as exc:
                raise ValueError(
                    f"{prompt_file}:{lineno}: {exc}") from exc
    if not prompts:
        raise ValueError(f"Prompt file is empty: {prompt_file}")
    return prompts


def prompt_key(index: int, messages: list[dict]) -> str:
    """Stable resume key: position plus content hash, so an edited prompt set
    invalidates completed work instead of silently mismatching."""
    canonical = json.dumps(messages, sort_keys=True, ensure_ascii=False)
    digest = hashlib.sha256(canonical.encode()).hexdigest()[:12]
    return f"{index:06d}-{digest}"


def hash_prompt_set(path: str) -> str:
    prompt_file = resolve_prompt_file(path)
    digest = hashlib.sha256()
    with open(prompt_file, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def completed_keys(out_dir: str) -> set[str]:
    """Keys already written to data.jsonl; a torn final line (killed run) is
    skipped so its prompt regenerates."""
    data_path = os.path.join(out_dir, DATA_NAME)
    keys: set[str] = set()
    try:
        with open(data_path) as f:
            lines = f.readlines()
    except FileNotFoundError:
        return keys
    for lineno, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            keys.add(json.loads(line)["key"])
        except (json.JSONDecodeError, KeyError) as exc:
            if lineno == len(lines):
                break
            raise ValueError(
                f"{data_path}:{lineno}: corrupt record mid-file: {exc}") from exc
    return keys


def _truncate_torn_tail(data_path: str) -> None:
    """Drop a torn final line (a record cut short by a killed run) so the
    next append does not concatenate onto it."""
    try:
        with open(data_path, "rb+") as f:
            content = f.read()
            if not content or content.endswith(b"\n"):
                return
            keep = content.rfind(b"\n") + 1
            tail = content[keep:]
            try:
                json.loads(tail)["key"]
                f.write(b"\n")  # complete record, just missing its newline
            except (json.JSONDecodeError, KeyError):
                f.truncate(keep)
    except FileNotFoundError:
        return


def with_retry(fn: Callable[[], str], transient: type[Exception],
               max_attempts: int, backoff_s: float,
               sleep: Callable[[float], None] = time.sleep) -> str:
    """Retry fn on the transient exception type with exponential backoff."""
    for attempt in range(1, max_attempts + 1):
        try:
            return fn()
        except transient:
            if attempt == max_attempts:
                raise
            sleep(backoff_s * (2 ** (attempt - 1)))
    raise AssertionError("unreachable")


def _git_sha() -> str:
    try:
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))))
        return subprocess.run(["git", "-C", repo_root, "rev-parse", "HEAD"],
                              capture_output=True, text=True,
                              check=True).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def quant_from_name(model: str) -> str | None:
    """Quant type from a <name>-<QUANT>.gguf filename, when parseable."""
    stem = os.path.basename(model)
    if stem.endswith(".gguf"):
        tail = stem[:-5].rsplit("-", 1)
        if len(tail) == 2 and tail[1][:1].upper() == "Q":
            # Normalize case: Qwen official GGUFs use lowercase quant
            # suffixes, and the confound checks compare case-sensitively.
            return tail[1].upper()
    return None


def build_manifest(config: TeacherConfig, prompt_source: str,
                   prompt_set_hash: str, examples: int,
                   command: str | None = None) -> dict:
    """Manifest fields follow training/data/prepare.py conventions; the
    teacher block is what F7 cites when claiming improvement against a
    teacher."""
    return {
        "schema_version": SCHEMA_VERSION,
        "pipeline": config.pipeline,
        "teacher": {"model": config.teacher,
                    "quant_type": quant_from_name(config.teacher)},
        "service": config.service,
        "sampling": config.sampling.as_request(),
        "prompt_source": prompt_source,
        "prompt_set_sha256": prompt_set_hash,
        "examples": examples,
        "engine": "stdlib-urllib-client",
        "created_at": datetime.datetime.now(datetime.UTC).isoformat(),
        "git_sha": _git_sha(),
        "command": command if command is not None else " ".join(sys.argv),
    }


def _generate_one(config: TeacherConfig, messages: list[dict],
                  complete_fn: CompleteFn, transient: type[Exception],
                  sleep: Callable[[float], None]) -> list[dict]:
    """One prompt through the configured pipeline; returns the full
    conversation ending in the assistant turn."""
    sampling = config.sampling.as_request()

    def ask(request: list[dict]) -> str:
        return with_retry(lambda: complete_fn(request, sampling), transient,
                          config.max_attempts, config.backoff_s, sleep)

    if config.pipeline == "respond":
        return messages + [{"role": "assistant", "content": ask(messages)}]
    # synthesize: evolve the seed instruction, then answer the evolved form
    seed = messages[-1]["content"]
    evolved = ask([{"role": "user",
                    "content": EVOLVE_TEMPLATE.format(seed=seed)}]).strip()
    if not evolved:
        raise ValueError(f"Teacher returned an empty evolved instruction "
                         f"for seed: {seed[:80]!r}")
    request = [{"role": "user", "content": evolved}]
    return request + [{"role": "assistant", "content": ask(request)}]


def generate(config: TeacherConfig, prompts: list[list[dict]],
             complete_fn: CompleteFn, out_dir: str, *,
             prompt_source: str = "unknown", prompt_set_hash: str = "unknown",
             transient: type[Exception] = Exception,
             command: str | None = None,
             progress: Callable[[int, int], None] | None = None,
             sleep: Callable[[float], None] = time.sleep) -> dict:
    """Run the pipeline over prompts and return the written manifest.

    Skips prompts whose keys already exist in data.jsonl (resume). Raises
    GenerationIncomplete when any prompt fails after retries; the manifest is
    written only on a fully complete dataset. progress, when given, is called
    as progress(done, total) after each completed prompt.
    """
    config.validate()
    manifest_path = os.path.join(out_dir, MANIFEST_NAME)
    if os.path.exists(manifest_path):
        raise FileExistsError(
            f"{out_dir} already contains {MANIFEST_NAME} (a complete "
            f"dataset); refusing to overwrite")
    os.makedirs(out_dir, exist_ok=True)

    _truncate_torn_tail(os.path.join(out_dir, DATA_NAME))
    done_keys = completed_keys(out_dir)
    keyed = [(prompt_key(i, m), m) for i, m in enumerate(prompts)]
    orphans = done_keys - {key for key, _ in keyed}
    if orphans:
        # Completed records whose keys left the prompt set would persist in
        # data.jsonl while the manifest counts only the current prompts —
        # a silent dataset/manifest disagreement.
        raise ValueError(
            f"{out_dir} contains {len(orphans)} completed record(s) not in "
            "the current prompt set; the prompt set changed since this "
            "generation started\n"
            "  Solution: generate into a fresh output directory, or restore "
            "the original prompt set to finish this one")
    pending = [(key, m) for key, m in keyed if key not in done_keys]
    done = len(prompts) - len(pending)
    failures: list[tuple[str, str]] = []
    write_lock = threading.Lock()

    with open(os.path.join(out_dir, DATA_NAME), "a") as data_file:

        def worker(key: str, messages: list[dict]) -> None:
            nonlocal done
            record = {"key": key,
                      "messages": _generate_one(config, messages, complete_fn,
                                                transient, sleep)}
            line = json.dumps(record, ensure_ascii=False)
            with write_lock:
                data_file.write(line + "\n")
                data_file.flush()
                done += 1
                if progress is not None:
                    progress(done, len(prompts))

        with concurrent.futures.ThreadPoolExecutor(config.concurrency) as pool:
            futures = {pool.submit(worker, key, messages): key
                       for key, messages in pending}
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as exc:  # noqa: BLE001 - collected per prompt
                    failures.append((futures[future], str(exc)))

    if failures:
        sample = "; ".join(f"{key}: {err}" for key, err in failures[:3])
        raise GenerationIncomplete(
            f"{len(failures)} of {len(prompts)} prompts failed ({sample})\n"
            f"  Solution: fix the cause and re-invoke; completed prompts "
            f"are skipped on resume")

    manifest = build_manifest(config, prompt_source, prompt_set_hash,
                              examples=len(prompts), command=command)
    tmp_path = manifest_path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
    os.replace(tmp_path, manifest_path)
    return manifest
