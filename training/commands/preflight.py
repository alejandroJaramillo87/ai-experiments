"""Prove every asset a recipe needs is already on disk (Prelude C).

The strict counterpart to `python -m training validate`. Validate is
skip-when-absent by explicit convention (training/recipes/validate.py) so it runs
on any machine, including one with no data; preflight requires the assets to
exist and is meant to run immediately before a long job.

  python -m training preflight --config training/recipes/reference/pretrain-1b.yaml

The point is timing. Shard integrity is checked today inside
ShardedTokenDataset.__init__, i.e. after the run has started and the GPU is
committed; base-model identity is resolved by a helper that swallows every
exception. A run that is missing an asset should fail in the first seconds with a
complete list, not hours in with one line.

Every asset the recipe references is required. There is no optional tier: if a
recipe names an eval dataset or a contamination index, preflight demands it. Reports every failure rather than stopping at the first, so one pass
tells you everything to stage.

Exit codes: 0 all assets present, 1 anything missing, 2 invalid usage.
"""

from __future__ import annotations

import argparse
import os

from omegaconf import OmegaConf

from common.output import Printer
from training.commands.recipe import load_validated
from training.recipes import tokenizer_source

# Staging command per missing-asset kind, in the order a run would need them.
# Listed rather than printed unconditionally so the advice matches the failure:
# a scratch recipe has no model to download, and a tokenizer comes from
# train-tokenizer, not prepare-corpus.
REMEDIES: tuple[tuple[str, str], ...] = (
    ("model", "make train-download MODEL=<hub id> REVISION=<sha>"),
    ("tokenizer", "make train-tokenizer ARGS=..."),
    ("shards", "make prepare-corpus ARGS=...   # or: make prepare-data ARGS=..."),
    ("sft", "make prepare-sft ARGS=..."),
)


def configure(sub) -> None:
    parser = sub.add_parser(
        "preflight",
        help="prove every asset a recipe references is local before a run (Prelude C)",
        description="Verify every asset a recipe references is present on disk.",
    )
    parser.add_argument("--config", required=True, help="path to a recipe YAML")
    parser.add_argument(
        "overrides", nargs="*", help="dotlist overrides, e.g. data.micro_batch=8"
    )
    parser.set_defaults(func=cmd)


def _check_shard_dir(path: str, label: str, problems: list[str]) -> str | None:
    """Verify one prepared shard directory. Returns its tokenizer hash, or None.

    Reuses the D2 manifest primitives rather than re-implementing them, so
    preflight and the dataloader agree by construction.
    """
    from training.data.shards import ShardIntegrityError, load_manifest, verify_shards

    if not os.path.isdir(path):
        problems.append(f"{label}: shard directory not found: {path}")
        return None
    try:
        manifest = load_manifest(path)
    except FileNotFoundError as exc:
        problems.append(f"{label}: {exc}")
        return None
    except (OSError, ValueError) as exc:
        problems.append(f"{label}: manifest unreadable: {exc}")
        return None
    try:
        verify_shards(path, manifest)
    except ShardIntegrityError as exc:
        problems.append(f"{label}: {exc}")
        return None
    except (KeyError, TypeError) as exc:
        # A manifest that is not a shard manifest (an SFT dataset's, or a
        # hand-edited one) lacks the keys verify_shards indexes. Report it as
        # the asset problem it is rather than letting the KeyError escape.
        problems.append(f"{label}: manifest is not a shard manifest ({exc!r})")
        return None
    return manifest.get("tokenizer_hash")


def _check_sft_dataset(path: str, problems: list[str]) -> None:
    """Verify a prepared SFT dataset the way the trainer will read it.

    Reuses the F1 loader with nothing parsed, so the manifest kind, the chat
    template version, every file hash and every example count are checked by
    the same code build_dataset (training/tune/sft.py) trusts. The tokenizer
    comparison stays out: tokenizer.path is checked for presence below, and
    validate() owns the hash comparison.
    """
    from training.data.sft import SftIntegrityError, load_sft_dataset

    if not os.path.isdir(path):
        problems.append(f"sft.dataset: dataset directory not found: {path}")
        return
    try:
        load_sft_dataset(path, parse=())
    except (FileNotFoundError, SftIntegrityError, OSError, ValueError) as exc:
        problems.append(f"sft.dataset: {exc}")


def _shard_sources(recipe) -> list[tuple[str, str]]:
    """(label, path) for every shard directory the recipe will actually read.

    validate() enforces that mixture and data.source are mutually exclusive, so
    exactly one branch applies.
    """
    sources: list[tuple[str, str]] = []
    if recipe.mixture.enabled:
        for i, src in enumerate(recipe.mixture.sources):
            sources.append((f"mixture.sources[{i}]", src.path))
        if recipe.replay.enabled:
            for i, src in enumerate(recipe.replay.sources):
                sources.append((f"replay.sources[{i}]", src.path))
    elif not OmegaConf.is_missing(recipe.data, "source"):
        # A training run reads prepared shards. If data.source is still a hub
        # dataset id at this point the run would download mid-flight, which is
        # exactly what preflight exists to refuse -- so it is checked as a
        # directory either way and reported missing when it is not one.
        sources.append(("data.source", recipe.data.source))
    return sources


def _check_model(model: str, revision, label: str, problems: list[str]) -> None:
    """Resolve a model's config locally. Never swallows the failure.

    training/tune/sft.py:resolve_base_identity_or_none deliberately returns None
    on any error so a training run survives a resolution failure. That is the
    right call there and exactly the wrong one here.
    """
    from training.export.errors import ExportError
    from training.export.preflight import resolve_base_identity

    try:
        resolve_base_identity(model, revision=revision, local_files_only=True)
    except ExportError as exc:
        problems.append(f"{label}: {exc}")


def cmd(args: argparse.Namespace) -> int:
    out = Printer()
    recipe = load_validated(args.config, args.overrides, out)
    if recipe is None:
        return 1

    out.line(f"=== Preflight: {recipe.run.name} ===")
    out.line()

    problems: list[str] = []
    # Which staging commands to suggest, derived from what is actually missing.
    # A fixed block would misdirect: tiny-e2e trains from scratch, so telling its
    # operator to run train-download names a step that does not exist for it.
    remedies: set[str] = set()

    # Base model. A scratch recipe has no weights to stage.
    if recipe.model.hf is not None:
        before = len(problems)
        _check_model(recipe.model.hf.id, recipe.model.hf.revision, "model.hf", problems)
        if len(problems) > before:
            remedies.add("model")
    else:
        out.skip("Base model: recipe trains from scratch")

    # An SFT run reads the prepared F1 dataset, not token shards: the reference
    # recipes set data.source to the same directory, which is not a shard
    # directory and must not be verified as one.
    hashes: dict[str, list[str]] = {}
    if recipe.sft.enabled:
        before = len(problems)
        _check_sft_dataset(recipe.sft.dataset, problems)
        if len(problems) > before:
            remedies.add("sft")

    # Shards, and the tokenizer identity they were built with.
    for label, path in [] if recipe.sft.enabled else _shard_sources(recipe):
        before = len(problems)
        tokenizer_hash = _check_shard_dir(path, label, problems)
        if len(problems) > before:
            remedies.add("shards")
        if tokenizer_hash is not None:
            hashes.setdefault(tokenizer_hash, []).append(label)

    if len(hashes) > 1:
        detail = ", ".join(f"{h}: {labels}" for h, labels in sorted(hashes.items()))
        problems.append(
            f"sources carry mismatched tokenizer hashes ({detail}); "
            "all sources must be tokenized identically"
        )

    # Tokenizer presence only. The tokenizer-to-manifest hash comparison is
    # deliberately NOT repeated here: validate() already runs
    # _check_tokenizer_matches_shards and cmd() returns above on
    # RecipeValidationError, so a mismatch never reaches this point.
    #
    # An HF recipe with no tokenizer.path reads the base directory's tokenizer
    # (KB-80), which _check_model already resolved: a second line for the same
    # missing directory would count one asset twice, so only an explicit path
    # is checked here.
    tokenizer = tokenizer_source(recipe)
    if tokenizer is None:
        problems.append("tokenizer.path: not set in the recipe")
        remedies.add("tokenizer")
    elif recipe.tokenizer.path is not None and not os.path.exists(tokenizer):
        problems.append(f"tokenizer.path: not found: {tokenizer}")
        remedies.add("tokenizer")

    if problems:
        out.error(f"Missing or unusable assets: {len(problems)}")
        for problem in problems:
            out.line(f"  - {problem}")
        out.line()
        out.line("  Solution: stage the missing assets, then re-run preflight")
        commands = [command for kind, command in REMEDIES if kind in remedies]
        for i, command in enumerate(commands):
            label = "  Command:" if i == 0 else "          "
            out.line(f"{label} {command}")
        return 1

    out.ok("All referenced assets are present and verified")
    return 0
