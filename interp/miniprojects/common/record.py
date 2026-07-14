"""Result-recording conventions (interp M3).

Small helpers that drop plots and JSON result blobs into a project's directory under
/mnt/ai-data/interp with a stable naming convention (<project>/<name>-<UTC timestamp>.<ext>)
so the experiments log (docs/interpretability/experiments/) can link to them. Heavy artifacts
never land in the repository.
"""

from __future__ import annotations

import datetime
import json
import os

RESULTS_ROOT = "/mnt/ai-data/interp"


def _dest(project: str, name: str, ext: str, root: str) -> str:
    stamp = datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%SZ")
    directory = os.path.join(root, project)
    os.makedirs(directory, exist_ok=True)
    return os.path.join(directory, f"{name}-{stamp}.{ext}")


def save_json(project: str, name: str, payload: dict, root: str = RESULTS_ROOT) -> str:
    """Write a small JSON result blob; returns the path to link from the experiments log."""
    path = _dest(project, name, "json", root)
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    return path


def save_figure(project: str, name: str, figure, root: str = RESULTS_ROOT) -> str:
    """Save a matplotlib or plotly figure as PNG; returns the path."""
    path = _dest(project, name, "png", root)
    if hasattr(figure, "savefig"):  # matplotlib
        figure.savefig(path, bbox_inches="tight", dpi=150)
    elif hasattr(figure, "write_image"):  # plotly
        figure.write_image(path)
    else:
        raise TypeError(f"unsupported figure type: {type(figure)!r}")
    return path
