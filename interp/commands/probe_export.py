"""Export a registered probe as the safetensors image the harness loads (W3).

probe-train registers a probe as probe.npz, which is the training format and
is read only by Python. The rust harness scores probes in the process that
answers a request, and its reader takes a safetensors image plus a JSON
sidecar (schemas/probe_artifact.json). This is the step between the two: it
loads the registered probe with its hash checked, then writes the pair.

CPU work, no model and no GPU. Exit codes: 0 success, 1 failure, 2 invalid
usage.
"""

from __future__ import annotations

import argparse
import os

from common.output import Printer

IMAGE_NAME = "probe.safetensors"
# The probes tree as the host sees it, and as harness-serve mounts it. Both
# spellings appear in docker-compose.rust-harness.yaml's probes volume.
HOST_PROBES = "/mnt/ai-data/interp/probes"
CONTAINER_PROBES = "/app/probes"


def configure(sub) -> None:
    p = sub.add_parser(
        "probe-export",
        help="export a registered probe as a safetensors image for the harness (W3)",
        description=__doc__,
    )
    p.add_argument(
        "--probe",
        required=True,
        help="registered probe directory, e.g. /mnt/ai-data/interp/probes/<name>",
    )
    p.add_argument(
        "--out",
        default=None,
        help=f"output image path (default {IMAGE_NAME} inside the probe directory)",
    )
    p.set_defaults(func=cmd)


def cmd(args: argparse.Namespace) -> int:
    out = Printer()
    from interp.manifest import ManifestError
    from interp.probes.export import ProbeExportError, write_probe
    from interp.probes.registry import RegistryError

    # Beside the probe by default: the sidecar is written next to the image
    # and the harness is given the image path, so keeping the pair in the
    # registered directory means one mount and one path to quote.
    path = args.out or os.path.join(args.probe, IMAGE_NAME)
    try:
        out.run(f"Exporting {args.probe}")
        write_probe(args.probe, path)
    except (ProbeExportError, RegistryError, ManifestError, OSError) as exc:
        out.error(f"Export failed: {exc}")
        return 1

    sidecar = os.path.splitext(path)[0] + ".json"
    out.ok(f"Image written: {path}")
    out.ok(f"Sidecar written: {sidecar}")
    # HARNESS_PROBE is read inside harness-serve, which mounts the probes tree
    # at /app/probes, so the hint has to name the container path. Printed only
    # when the image landed under the mounted tree; anywhere else the harness
    # cannot see it and a suggested command would be a wrong one.
    if path.startswith(HOST_PROBES + os.sep):
        served = CONTAINER_PROBES + path[len(HOST_PROBES) :]
        out.info(f"Serve with: make harness-serve HARNESS_PROBE={served}", indent=2)
    else:
        out.warn(
            f"Outside {HOST_PROBES}: harness-serve mounts only that tree, "
            f"so it cannot read this image"
        )
    return 0
