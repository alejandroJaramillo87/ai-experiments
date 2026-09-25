"""Stage a model into the persistent models tree, or a dataset into the HF cache.

The single documented path from the hub onto this workstation. Runs only in the
`prepare` service, which is the one container with network egress; the training
and export services sit on an internal network and cannot reach the hub at all.

  python -m training download --model google/gemma-4-E2B --revision <sha>
  python -m training download --dataset openai/gsm8k --name main

A model lands as a plain directory tree under /mnt/ai-data/models/hf, which
every other service mounts read-only, rather than as a hub blob cache. Recipes
keep naming the hub id and its revision; every reader resolves the id to this
directory through training/models/staged.py (KB-79), so a training run resolves
nothing over the network. The layout there is this module's DEFAULT_ROOT plus the
id's basename; change one and change the other.

--revision is required for a model. An unpinned download is the problem this
command exists to solve: it makes a run irreproducible and lets the base drift
between training and export, which training/export/preflight.py exists to catch
after the fact.

A dataset is the other case (KB-92): lm-evaluation-harness loads its task
datasets itself, by hub id and without a revision, through datasets'
load_dataset, and the training service resolves that offline from the
datasets half of the persistent cache, which it mounts at HF_HOME/datasets:
datasets offline never consults a hub snapshot for a hub id, only the
processed arrow cache. So the staging form calls the same load_dataset in
the prepare service, which writes those arrow files where the offline reader
looks, and prints the commit it resolved so the runbook records what was
fetched. It
takes no --revision because the consumer cannot pass one; the printed sha
is the pin, and re-staging moves it. The pinned task set and the datasets
it reaches are training/eval/tasks.yaml's.

Exit codes: 0 downloaded, 1 download or environment failure, 2 invalid usage.
"""

from __future__ import annotations

import argparse
import os

from common.output import Printer

DEFAULT_ROOT = "/mnt/ai-data/models/hf"


def configure(sub) -> None:
    parser = sub.add_parser(
        "download",
        help="stage a pinned model into /mnt/ai-data/models/hf, or a dataset "
        "into the HF cache (prepare service)",
        description="Download a pinned model revision into the persistent models "
        "tree, or a dataset into the persistent HF cache the training service "
        "reads offline.",
    )
    what = parser.add_mutually_exclusive_group(required=True)
    what.add_argument("--model", help="hub id, e.g. google/gemma-4-E2B")
    what.add_argument(
        "--dataset",
        help="hub dataset id, e.g. openai/gsm8k (staged into HF_HOME/datasets "
        "for the training service's offline reads)",
    )
    parser.add_argument(
        "--revision",
        help="exact commit SHA to pin (required with --model); resolve with "
        "HfApi().model_info(id).sha",
    )
    parser.add_argument(
        "--name", help="dataset configuration, e.g. main for openai/gsm8k"
    )
    parser.add_argument(
        "--out",
        help=f"destination directory (default: {DEFAULT_ROOT}/<model basename>)",
    )
    parser.add_argument(
        "--allow-patterns",
        nargs="*",
        help="restrict the download to matching files, e.g. '*.safetensors' '*.json'",
    )
    parser.set_defaults(func=cmd)


def cmd(args: argparse.Namespace) -> int:
    out = Printer()

    # Offline is the image default (docker/training/Dockerfile.training). Reaching
    # this command with it still set means it is running in the wrong service, and
    # snapshot_download would otherwise fail with a confusing cache miss.
    for var in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        if os.environ.get(var, "0") not in ("0", ""):
            out.error(f"{var} is set: this container cannot reach the hub")
            out.line("  Context: downloads run in the prepare service, not training")
            out.line("  Solution: run through the prepare service")
            out.line("  Command: make train-download MODEL=... REVISION=...")
            return 1

    if args.dataset:
        for flag in ("revision", "out", "allow_patterns"):
            if getattr(args, flag):
                out.error(f"--{flag.replace('_', '-')} applies to --model only")
                out.line("  Context: a dataset is fetched at the hub's main ref")
                out.line("  Solution: drop the flag; the resolved commit is printed")
                return 2
        return stage_dataset(args.dataset, args.name, out)
    if not args.revision:
        out.error("--revision is required with --model")
        out.line("  Solution: pin the exact commit")
        out.line(f"  Command: HfApi().model_info('{args.model}').sha")
        return 2
    if args.name:
        out.error("--name applies to --dataset only")
        return 2

    dest = args.out or os.path.join(DEFAULT_ROOT, args.model.split("/")[-1])

    out.line("=== Staging a model ===")
    out.line()
    out.info(f"Model: {args.model}")
    out.info(f"Revision: {args.revision}")
    out.info(f"Destination: {dest}")
    out.line()

    from huggingface_hub import snapshot_download

    out.run(f"Downloading {args.model}")
    try:
        # local_dir gives a plain directory tree rather than the blob cache
        # layout, because /app/models is mounted read-only elsewhere and consumers
        # resolve a path, not a hub id.
        path = snapshot_download(
            repo_id=args.model,
            revision=args.revision,
            local_dir=dest,
            allow_patterns=args.allow_patterns,
        )
    except OSError as exc:
        out.error(f"Cannot write to {dest}: {exc}")
        out.line("  Solution: confirm models/hf is mounted read-write in prepare")
        out.line("  Command: docker compose -f docker-compose.training.yaml config")
        return 1
    except Exception as exc:  # noqa: BLE001 - hub errors vary by client version
        out.error(f"Download failed: {exc}")
        out.line("  Solution: check the model id, the revision, and HF_TOKEN access")
        out.line(f"  Command: HfApi().model_info('{args.model}').sha")
        return 1

    out.ok(f"Downloaded to {path}")
    out.line()
    out.line("Pin this in the recipe so the run is reproducible:")
    out.line("  model:")
    out.line("    hf:")
    out.line(f"      id: {args.model}")
    out.line(f"      revision: {args.revision}")
    out.line("(every reader resolves the id to the staged directory)")
    return 0


def stage_dataset(dataset: str, name: str | None, out: Printer) -> int:
    """Fetch every split of a hub dataset into HF_HOME through load_dataset.

    The same call lm_eval makes offline in the training service, so what it
    caches is what that service finds: the arrow files under HF_HOME/datasets,
    the half of the cache that service mounts (the hub snapshot lands under
    HF_HOME/hub beside them, which offline resolution never reads). Prints the
    commit the hub resolved, the one pin this fetch has.
    """
    cache = os.environ.get("HF_HOME", "")
    out.line("=== Staging a dataset ===")
    out.line()
    out.info(f"Dataset: {dataset}")
    out.info(f"Config: {name or '(default)'}")
    out.info(f"Cache: {cache or '(HF_HOME unset: the default cache)'}")
    out.line()

    from datasets import load_dataset
    from huggingface_hub import HfApi

    out.run(f"Downloading {dataset}")
    try:
        sha = HfApi().dataset_info(dataset).sha
        splits = load_dataset(dataset, name)
    except Exception as exc:  # noqa: BLE001 - hub errors vary by client version
        out.error(f"Download failed: {exc}")
        out.line("  Solution: check the dataset id and configuration name")
        out.line(f"  Command: HfApi().dataset_info('{dataset}')")
        return 1

    counts = ", ".join(f"{split}={len(rows)}" for split, rows in splits.items())
    out.ok(f"Staged {dataset} ({name or 'default'}) at {sha}: {counts}")
    out.line(
        "The training service reads these arrow files offline; a re-stage moves them."
    )
    return 0
