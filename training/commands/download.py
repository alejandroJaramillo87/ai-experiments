"""Stage a model into the persistent models tree (Prelude C).

The single documented path from the hub onto this workstation. Runs only in the
`prepare` service, which is the one container with network egress; the training
and export services sit on an internal network and cannot reach the hub at all.

  python -m training download --model google/gemma-4-E2B --revision <sha>

The download lands as a plain directory tree under /mnt/ai-data/models/hf, which
every other service mounts read-only, rather than as a hub blob cache. Recipes
then name that directory instead of a hub id, so a training run resolves nothing
over the network.

--revision is required. An unpinned download is the problem this command exists
to solve: it makes a run irreproducible and lets the base drift between training
and export, which training/export/preflight.py exists to catch after the fact.

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
        help="stage a pinned model into /mnt/ai-data/models/hf (prepare service)",
        description="Download a pinned model revision into the persistent models tree.",
    )
    parser.add_argument(
        "--model", required=True, help="hub id, e.g. google/gemma-4-E2B"
    )
    parser.add_argument(
        "--revision",
        required=True,
        help="exact commit SHA to pin; resolve with HfApi().model_info(id).sha",
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
    out.line("Set this in the recipe so the run is reproducible:")
    out.line("  model:")
    out.line("    hf:")
    out.line(f"      id: {path}")
    out.line(f"      revision: {args.revision}")
    return 0
