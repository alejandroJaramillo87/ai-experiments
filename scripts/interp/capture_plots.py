"""Render three figura diagrams from one activation capture.

A capture directory (activation-capture or interp probe-capture) holds one
float32 row per example per layer. This script turns one such directory into
three static SVG figures in the figura library's fragment format, so a blog
post can show what a capture looks like without an image pipeline:

  activation-histogram     values of one row at one layer, log-count bins,
                           the largest-magnitude dimensions annotated
  residual-norm-by-layer   L2 norm of the captured row per layer, mean over
                           examples with a min-max band
  class-projection         the rows at one layer projected onto their first
                           two principal components, one color per label

Every figure is authored statically: the values are baked into the SVG, and
the managed blocks (palette, panel, reduced motion) are expanded by figura's
own build script, so the output passes figura's validator like a hand-written
diagram. matplotlib is not used on purpose: its SVG carries global ids and
element selectors the validator rejects, and the shapes here are rects,
polylines, and circles.

The generated files record the capture they came from (directory, model_hash,
layer, example) in a header comment, so a regenerated figure is traceable to
its data.

Host-side for a real capture (the directory lives under /mnt/ai-data); the
script itself needs only numpy and node.

Usage:
  uv run python scripts/interp/capture_plots.py \\
      --capture /mnt/ai-data/interp/captures/<name> \\
      --figura ../figura [--layer L] [--example N] [--post-slug activation-capture]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

# Run directly (scripts/interp/...) rather than as a module, so the repo root is
# not on sys.path. Same idiom as scripts/interp/activation_magnitude_probe.py.
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.output import Printer  # noqa: E402

# Figure geometry, in viewBox units. Figura designs for a ~720px column.
WIDTH = 760
HEIGHT = 420
MARGIN_L = 64
MARGIN_R = 24
MARGIN_T = 36
MARGIN_B = 56

# Bins in the value histogram; enough to show a heavy tail at 1536 wide.
BINS = 60
# Dimensions annotated by magnitude in the histogram.
TOP_K = 4

# Palette variables the managed palette block defines; referenced, never
# copied, so a token change in figura's shared/tokens.css reaches these too.
CLASS_COLORS = ("var(--ok)", "var(--hot)")

# Coordinates are rounded to keep 258 points and 60 bars small on disk.
PRECISION = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--capture", required=True, help="capture directory")
    parser.add_argument("--figura", required=True, help="figura checkout to write into")
    parser.add_argument(
        "--layer",
        type=int,
        default=None,
        help="layer for the histogram and projection; default: the middle captured layer",
    )
    parser.add_argument(
        "--example", type=int, default=0, help="row index for the histogram; default 0"
    )
    parser.add_argument(
        "--post-slug",
        default="activation-capture",
        help="diagrams/<post-slug>/ subdirectory and manifest post field",
    )
    parser.add_argument("--json", action="store_true", help="machine summary on stdout")
    return parser.parse_args()


def fmt(x: float) -> str:
    """A coordinate or value as short text: no trailing zeros, no exponent."""
    s = f"{x:.{PRECISION}f}"
    return s.rstrip("0").rstrip(".") if "." in s else s


def nice_ticks(lo: float, hi: float, n: int = 5) -> list[float]:
    """Round tick positions spanning [lo, hi], at most about n of them."""
    if hi <= lo:
        return [lo]
    raw = (hi - lo) / n
    mag = 10 ** np.floor(np.log10(raw))
    step = float(min(m * mag for m in (1, 2, 5, 10) if m * mag >= raw))
    start = np.ceil(lo / step) * step
    ticks = np.arange(start, hi + step * 0.5, step)
    return [float(t) for t in ticks]


class Fragment:
    """One figura diagram file: the fixed anatomy around an authored SVG.

    Emits the sentinel blocks empty; `expand` hands the file to figura's
    build script, which owns their contents.
    """

    def __init__(self, name: str, abbr: str, title: str, blurb: str) -> None:
        self.name = name
        self.abbr = abbr
        self.title = title
        self.blurb = blurb
        self.css: list[str] = []
        self.svg: list[str] = []
        self.aria = ""
        self.provenance: dict[str, object] = {}

    def render(self) -> str:
        scope = f".fg-{self.name}"
        prov = "\n".join(f"     {k}: {v}" for k, v in self.provenance.items())
        css = "\n".join(self.css)
        svg = "\n".join(self.svg)
        return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{self.title}</title>
<link rel="stylesheet" href="../../shared/preview.css">
</head>
<body>
<h1>{self.title}</h1>
<p>{self.blurb}</p>

<!-- generated by scripts/interp/capture_plots.py (ai-experiments); do not edit by hand
{prov}
-->
<!-- fg:embed-start -->
<div class="fg-diagram fg-{self.name}">
<style>
/* fg:begin palette-classic v1 */
/* fg:end palette-classic */
/* fg:begin panel-base v1 */
/* fg:end panel-base */
{scope} {{
  padding: 16px 16px 12px;
}}
{scope} .{self.abbr}-axis {{ stroke: var(--border); stroke-width: 1.5; fill: none; }}
{scope} .{self.abbr}-grid {{ stroke: var(--border); stroke-width: 1; stroke-dasharray: 2 4; fill: none; opacity: 0.6; }}
{scope} .{self.abbr}-tick {{ fill: var(--muted); font-size: 11px; font-family: var(--mono); }}
{scope} .{self.abbr}-lab  {{ fill: var(--muted); font-size: 12px; }}
{scope} .{self.abbr}-title {{ fill: var(--text); font-size: 13px; font-weight: 600; }}
{css}

/* fg:begin reduced-motion v1 */
/* fg:end reduced-motion */
</style>

<svg viewBox="0 0 {WIDTH} {HEIGHT}" xmlns="http://www.w3.org/2000/svg" role="img"
     aria-label="{self.aria}">
{svg}
</svg>
</div>
<!-- fg:embed-end -->
</body>
</html>
"""


def axes(
    f: Fragment,
    xlo: float,
    xhi: float,
    ylo: float,
    yhi: float,
    xlabel: str,
    ylabel: str,
    xticks: list[float] | None = None,
    yticks: list[float] | None = None,
    xtick_text=None,
    ytick_text=None,
):
    """Draw axes with ticks and return (sx, sy) mapping data to viewBox."""
    x0, x1 = MARGIN_L, WIDTH - MARGIN_R
    y0, y1 = HEIGHT - MARGIN_B, MARGIN_T

    def sx(v: float) -> float:
        return x0 + (v - xlo) / (xhi - xlo) * (x1 - x0)

    def sy(v: float) -> float:
        return y0 + (v - ylo) / (yhi - ylo) * (y1 - y0)

    a = f.abbr
    f.svg.append(f'  <path class="{a}-axis" d="M {x0} {y1} L {x0} {y0} L {x1} {y0}"/>')
    for t in xticks if xticks is not None else nice_ticks(xlo, xhi):
        if t < xlo or t > xhi:
            continue
        x = fmt(sx(t))
        label = xtick_text(t) if xtick_text else fmt(t)
        f.svg.append(f'  <path class="{a}-axis" d="M {x} {y0} L {x} {y0 + 5}"/>')
        f.svg.append(
            f'  <text class="{a}-tick" x="{x}" y="{y0 + 19}" text-anchor="middle">{label}</text>'
        )
    for t in yticks if yticks is not None else nice_ticks(ylo, yhi):
        if t < ylo or t > yhi:
            continue
        y = fmt(sy(t))
        label = ytick_text(t) if ytick_text else fmt(t)
        f.svg.append(f'  <path class="{a}-grid" d="M {x0} {y} L {x1} {y}"/>')
        f.svg.append(
            f'  <text class="{a}-tick" x="{x0 - 8}" y="{fmt(sy(t) + 4)}" text-anchor="end">{label}</text>'
        )
    f.svg.append(
        f'  <text class="{a}-lab" x="{(x0 + x1) // 2}" y="{HEIGHT - 12}" text-anchor="middle">{xlabel}</text>'
    )
    f.svg.append(
        f'  <text class="{a}-lab" x="18" y="{(y0 + y1) // 2}" text-anchor="middle" '
        f'transform="rotate(-90 18 {(y0 + y1) // 2})">{ylabel}</text>'
    )
    return sx, sy


def histogram(row: np.ndarray, layer: int, example: int, prov: dict) -> Fragment:
    f = Fragment(
        "activation-histogram",
        "ach",
        "Activation histogram",
        f"Values of one captured row (layer {layer}, example {example}), "
        f"log-count bins, largest-magnitude dimensions annotated.",
    )
    f.provenance = dict(prov, layer=layer, example=example)
    counts, edges = np.histogram(row, bins=BINS)
    logc = np.log10(counts + 1)
    ymax = float(np.ceil(logc.max() * 10) / 10) or 1.0
    xlo, xhi = float(edges[0]), float(edges[-1])
    # Count ticks at 1, 10, 100, ... on the log10(count + 1) axis.
    count_ticks = [c for c in (1, 10, 100, 1000, 10000) if np.log10(c + 1) <= ymax]
    sx, sy = axes(
        f,
        xlo,
        xhi,
        0.0,
        ymax,
        f"value (layer {layer}, example {example}, {row.size} dimensions)",
        "count",
        yticks=[float(np.log10(c + 1)) for c in count_ticks],
        ytick_text=lambda t: str(int(round(10**t - 1))),
    )
    a = f.abbr
    f.css.append(f".fg-{f.name} .{a}-bar {{ fill: var(--accent); opacity: 0.85; }}")
    f.css.append(
        f".fg-{f.name} .{a}-note {{ fill: var(--warn); font-size: 11px; font-family: var(--mono); }}"
    )
    f.css.append(f".fg-{f.name} .{a}-pin {{ stroke: var(--warn); stroke-width: 1; }}")
    y0 = HEIGHT - MARGIN_B
    for c, lo, hi in zip(logc, edges[:-1], edges[1:], strict=True):
        if c <= 0:
            continue
        x = sx(lo)
        w = max(sx(hi) - x - 0.5, 0.5)
        y = sy(float(c))
        f.svg.append(
            f'  <rect class="{a}-bar" x="{fmt(x)}" y="{fmt(y)}" width="{fmt(w)}" height="{fmt(y0 - y)}"/>'
        )
    # Annotate the largest magnitudes: these are the dimensions that make the
    # x range wide while almost every other value sits near zero.
    top = np.argsort(-np.abs(row))[:TOP_K]
    for rank, d in enumerate(top):
        v = float(row[d])
        x = sx(v)
        ytxt = MARGIN_T + 14 + rank * 15
        anchor = "end" if x > WIDTH / 2 else "start"
        dx = -6 if anchor == "end" else 6
        f.svg.append(
            f'  <path class="{a}-pin" d="M {fmt(x)} {fmt(ytxt + 2)} L {fmt(x)} {y0}"/>'
        )
        f.svg.append(
            f'  <text class="{a}-note" x="{fmt(x + dx)}" y="{ytxt}" text-anchor="{anchor}">'
            f"dim {int(d)}: {fmt(v)}</text>"
        )
    f.svg.append(
        f'  <text class="{a}-title" x="{MARGIN_L}" y="{MARGIN_T - 14}">'
        f"one row, layer {layer}: min {fmt(float(row.min()))}, max {fmt(float(row.max()))}, "
        f"median |v| {fmt(float(np.median(np.abs(row))))}</text>"
    )
    f.aria = (
        f"Histogram of the {row.size} values in one captured activation at layer {layer}, "
        f"log-scaled counts. Most values lie near zero; the {TOP_K} largest-magnitude "
        f"dimensions are labeled, the largest at {fmt(float(row[top[0]]))} in dimension {int(top[0])}."
    )
    return f


def norm_by_layer(acts: dict[int, np.ndarray], prov: dict) -> Fragment:
    f = Fragment(
        "residual-norm-by-layer",
        "rnl",
        "Residual norm by layer",
        "L2 norm of the captured last-token row at each layer: mean over examples, min-max band.",
    )
    f.provenance = dict(prov)
    layers = sorted(acts)
    norms = {L: np.linalg.norm(acts[L], axis=1) for L in layers}
    mean = np.array([norms[L].mean() for L in layers])
    lo = np.array([norms[L].min() for L in layers])
    hi = np.array([norms[L].max() for L in layers])
    ymax = float(hi.max()) * 1.08
    xlo, xhi = layers[0] - 0.5, layers[-1] + 0.5
    xt = (
        [float(L) for L in layers]
        if len(layers) <= 12
        else nice_ticks(layers[0], layers[-1], 8)
    )
    sx, sy = axes(
        f,
        xlo,
        xhi,
        0.0,
        ymax,
        "layer",
        "L2 norm of the last-token row",
        xticks=xt,
        xtick_text=lambda t: fmt(t),
    )
    a = f.abbr
    f.css.append(f".fg-{f.name} .{a}-band {{ fill: var(--accent); opacity: 0.18; }}")
    f.css.append(
        f".fg-{f.name} .{a}-line {{ stroke: var(--accent); stroke-width: 2.5; fill: none; stroke-linejoin: round; }}"
    )
    f.css.append(f".fg-{f.name} .{a}-dot {{ fill: var(--accent); }}")
    band = [f"{fmt(sx(L))},{fmt(sy(v))}" for L, v in zip(layers, hi, strict=True)]
    band += [
        f"{fmt(sx(L))},{fmt(sy(v))}"
        for L, v in zip(reversed(layers), reversed(lo), strict=True)
    ]
    f.svg.append(f'  <polygon class="{a}-band" points="{" ".join(band)}"/>')
    line = " ".join(
        f"{fmt(sx(L))},{fmt(sy(v))}" for L, v in zip(layers, mean, strict=True)
    )
    f.svg.append(f'  <polyline class="{a}-line" points="{line}"/>')
    for L, v in zip(layers, mean, strict=True):
        f.svg.append(
            f'  <circle class="{a}-dot" cx="{fmt(sx(L))}" cy="{fmt(sy(v))}" r="3"/>'
        )
    n = next(iter(acts.values())).shape[0]
    f.svg.append(
        f'  <text class="{a}-title" x="{MARGIN_L}" y="{MARGIN_T - 14}">'
        f"mean over {n} examples, band = min to max; layer {layers[0]}: {fmt(float(mean[0]))}, "
        f"layer {layers[-1]}: {fmt(float(mean[-1]))}</text>"
    )
    f.aria = (
        f"Line chart of the L2 norm of the last-token residual at each of {len(layers)} captured "
        f"layers, mean over {n} examples with a min-to-max band. The mean rises from "
        f"{fmt(float(mean[0]))} at layer {layers[0]} to {fmt(float(mean[-1]))} at layer {layers[-1]}."
    )
    return f


def projection(
    rows: np.ndarray, labels: np.ndarray, layer: int, prov: dict
) -> Fragment:
    f = Fragment(
        "class-projection",
        "cpj",
        "Class projection",
        f"Captured rows at layer {layer} on their first two principal components, one color per label.",
    )
    f.provenance = dict(prov, layer=layer)
    centered = rows - rows.mean(axis=0, keepdims=True)
    # Thin SVD: rows are few (hundreds) against a wide embedding, so the
    # decomposition is on the small side and exact.
    _, s, vt = np.linalg.svd(centered, full_matrices=False)
    pcs = centered @ vt[:2].T
    var = s**2 / np.sum(s**2)
    pad = 0.06
    xr = (pcs[:, 0].min(), pcs[:, 0].max())
    yr = (pcs[:, 1].min(), pcs[:, 1].max())
    xpad = (xr[1] - xr[0]) * pad or 1.0
    ypad = (yr[1] - yr[0]) * pad or 1.0
    sx, sy = axes(
        f,
        float(xr[0] - xpad),
        float(xr[1] + xpad),
        float(yr[0] - ypad),
        float(yr[1] + ypad),
        f"PC1 ({fmt(float(var[0] * 100))}% of variance)",
        f"PC2 ({fmt(float(var[1] * 100))}%)",
    )
    a = f.abbr
    classes = sorted(int(c) for c in np.unique(labels))
    for i, c in enumerate(classes):
        color = CLASS_COLORS[i % len(CLASS_COLORS)]
        f.css.append(f".fg-{f.name} .{a}-c{c} {{ fill: {color}; opacity: 0.8; }}")
    for (x, y), lab in zip(pcs, labels, strict=True):
        f.svg.append(
            f'  <circle class="{a}-c{int(lab)}" cx="{fmt(sx(float(x)))}" cy="{fmt(sy(float(y)))}" r="3.5"/>'
        )
    # Legend, top right, inside the plot area.
    lx = WIDTH - MARGIN_R - 110
    for i, c in enumerate(classes):
        y = MARGIN_T + 12 + i * 16
        n = int((labels == c).sum())
        f.svg.append(
            f'  <circle class="{a}-c{c} {a}-key" cx="{lx}" cy="{y - 4}" r="4"/>'
        )
        f.svg.append(
            f'  <text class="{a}-lab" x="{lx + 10}" y="{y}">label {c} ({n})</text>'
        )
    f.svg.append(
        f'  <text class="{a}-title" x="{MARGIN_L}" y="{MARGIN_T - 14}">'
        f"layer {layer}: {rows.shape[0]} rows, {rows.shape[1]} dimensions, mean-centered SVD</text>"
    )
    f.aria = (
        f"Scatter plot of {rows.shape[0]} captured rows at layer {layer} projected onto their first "
        f"two principal components, colored by label ({', '.join(str(c) for c in classes)}). "
        f"PC1 carries {fmt(float(var[0] * 100))} percent of the variance."
    )
    return f


def write_fragment(f: Fragment, figura: Path, post_slug: str, out: Printer) -> Path:
    rel = Path("diagrams") / post_slug / f"{f.name}.html"
    path = figura / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f.render(), encoding="utf-8")
    # figura's build script owns the managed blocks; expanding through it
    # rather than copying the palette keeps the output byte-identical to what
    # `build --check` expects.
    proc = subprocess.run(
        ["node", "scripts/build.js", "--file", str(rel)],
        cwd=figura,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"figura build failed for {rel}:\n{proc.stdout}{proc.stderr}"
        )
    out.ok(f"Diagram: {rel}")
    return rel


def register(
    figura: Path, rel: Path, f: Fragment, post_slug: str, kind: str, out: Printer
) -> None:
    """Add or refresh the manifest entry; keep an existing description."""
    manifest_path = figura / "manifest.json"
    entries = json.loads(manifest_path.read_text(encoding="utf-8"))
    entry = next((e for e in entries if e["id"] == f.name), None)
    if entry is None:
        entries.append(
            {
                "id": f.name,
                "path": rel.as_posix(),
                "title": f.title,
                "post": post_slug,
                "kind": kind,
                "description": f.blurb,
            }
        )
        out.ok(f"Manifest: added {f.name}")
    else:
        entry["path"] = rel.as_posix()
        entry["title"] = f.title
        out.info(f"Manifest: {f.name} already registered")
    manifest_path.write_text(json.dumps(entries, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    out = Printer(quiet=args.json)
    from interp.manifest import load_manifest
    from interp.probes.capture import load_capture

    figura = Path(args.figura).resolve()
    if not (figura / "scripts" / "build.js").is_file():
        out.error(f"Figura checkout: {figura} has no scripts/build.js")
        return 1
    out.run(f"Loading capture: {args.capture}")
    manifest = load_manifest(args.capture)
    acts, labels, _texts = load_capture(args.capture)
    layers = sorted(acts)
    layer = args.layer if args.layer is not None else layers[len(layers) // 2]
    if layer not in acts:
        out.error(f"Layer {layer}: not in capture; captured layers are {layers}")
        return 1
    rows = acts[layer]
    if not 0 <= args.example < rows.shape[0]:
        out.error(f"Example {args.example}: capture holds {rows.shape[0]} rows")
        return 1
    out.ok(
        f"Capture: {rows.shape[0]} examples, {len(layers)} layers, width {rows.shape[1]}, "
        f"model_hash {manifest['model_hash'][:12]}"
    )
    prov = {
        "capture": os.path.abspath(args.capture),
        "model_id": manifest.get("model_id"),
        "model_hash": manifest["model_hash"],
        "capture_backend": manifest.get("capture_backend"),
        "generated": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    figures = [
        (histogram(rows[args.example], layer, args.example, prov), "ambient"),
        (norm_by_layer(acts, prov), "ambient"),
        (projection(rows, labels, layer, prov), "ambient"),
    ]
    written = []
    for f, kind in figures:
        rel = write_fragment(f, figura, args.post_slug, out)
        register(figura, rel, f, args.post_slug, kind, out)
        written.append(rel.as_posix())
    out.ok(
        f"Figures: {len(written)} written under {figura / 'diagrams' / args.post_slug}"
    )
    out.info(
        "Next: node scripts/build.js --check && node scripts/validate.js (in figura)"
    )
    if args.json:
        print(
            json.dumps(
                {
                    "capture": prov["capture"],
                    "model_hash": prov["model_hash"],
                    "layer": layer,
                    "example": args.example,
                    "figures": written,
                }
            )
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
