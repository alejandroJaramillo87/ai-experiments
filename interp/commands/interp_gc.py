#!/usr/bin/env python3
"""Delete validated activation captures, keeping manifests (P4-D2 retention).

Captures are regenerable cache: downstream SAE runs flag a capture
--validated, and this tool deletes flagged captures oldest-first. Manifests
survive deletion for provenance. Exit codes: 0 success, 1 failure, 2 invalid
usage.
"""

from __future__ import annotations

import argparse
import sys

sys.path.insert(0, ".")

from interp.output import Printer  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        default="/mnt/ai-data/interp/activations",
        help="activations tree to collect",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="list what would be deleted without deleting",
    )
    parser.add_argument(
        "--mark-validated",
        default="",
        help="flag one capture directory deletable instead of collecting",
    )
    args = parser.parse_args()

    out = Printer()
    from interp.harvest import gc_captures, mark_validated
    from interp.manifest import ManifestError

    try:
        if args.mark_validated:
            mark_validated(args.mark_validated)
            out.ok(f"Marked deletable: {args.mark_validated}")
            return 0
        deleted = gc_captures(args.root, dry_run=args.dry_run)
    except (ManifestError, OSError) as exc:
        out.error(f"Garbage collection failed: {exc}")
        return 1
    verb = "Would delete" if args.dry_run else "Deleted"
    if not deleted:
        out.info("No validated captures to collect")
        return 0
    for capture_dir in deleted:
        out.ok(f"{verb} shards: {capture_dir} (manifest kept)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
