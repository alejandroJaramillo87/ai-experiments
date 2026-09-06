"""scripts/interp/capture_plots.py against a synthetic capture.

The script writes figura fragments, and figura's validator is what judges
them; that validator lives in another repository and needs node, so this
module checks the contract the validator enforces that the generator can
break on its own: every CSS selector scoped under the diagram's root class,
every id prefixed, no script block, and the provenance header naming the
capture's model hash. The figures' geometry is not asserted -- a plot is
right when a person looks at it -- only that each is a well-formed fragment
with the expected number of marks.
"""

from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

import numpy as np
import pytest

from interp.probes.capture import load_capture

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPT = REPO_ROOT / "scripts" / "interp" / "capture_plots.py"


@pytest.fixture(scope="module")
def plots():
    spec = importlib.util.spec_from_file_location("capture_plots", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def figures(plots, capture_dir):
    acts, labels, _ = load_capture(capture_dir)
    layer = sorted(acts)[0]
    prov = {"capture": capture_dir, "model_hash": "modelhash-1"}
    return {
        "histogram": plots.histogram(acts[layer][0], layer, 0, prov),
        "norm": plots.norm_by_layer(acts, prov),
        "projection": plots.projection(acts[layer], labels, layer, prov),
    }


def selectors(style: str) -> list[str]:
    """Every selector list in a stylesheet, comments and at-rules stripped."""
    style = re.sub(r"/\*.*?\*/", "", style, flags=re.S)
    style = re.sub(r"@media[^{]*\{", "", style)
    found = []
    for rule in style.split("}"):
        if "{" not in rule:
            continue
        head = rule.split("{")[0].strip()
        if head and not head.startswith("@"):
            found.append(head)
    return found


class TestFragmentContract:
    def test_every_selector_is_scoped_under_the_diagram_root(self, figures):
        for f in figures.values():
            html = f.render()
            style = html.split("<style>")[1].split("</style>")[0]
            for sel in selectors(style):
                for part in sel.split(","):
                    part = part.strip()
                    if not part:
                        continue
                    assert part.startswith(f".fg-{f.name}"), f"{f.name}: {part!r}"

    def test_every_id_carries_the_abbreviation_prefix(self, figures):
        for f in figures.values():
            for id_ in re.findall(r'\bid="([^"]+)"', f.render()):
                assert id_.startswith(f"{f.abbr}-"), f"{f.name}: {id_}"

    def test_fragments_are_static_and_self_contained(self, figures):
        for f in figures.values():
            html = f.render()
            frag = html.split("<!-- fg:embed-start -->")[1].split(
                "<!-- fg:embed-end -->"
            )[0]
            assert "<script" not in frag
            assert "http" not in frag.replace('xmlns="http://www.w3.org/2000/svg"', "")
            assert 'role="img"' in frag and "aria-label=" in frag

    def test_the_header_names_the_capture_and_its_model_hash(
        self, figures, capture_dir
    ):
        for f in figures.values():
            html = f.render()
            assert capture_dir in html
            assert "modelhash-1" in html

    def test_managed_blocks_are_emitted_empty_for_the_build_script(self, figures):
        for f in figures.values():
            html = f.render()
            for block in ("palette-classic", "panel-base", "reduced-motion"):
                assert f"/* fg:begin {block} v1 */\n/* fg:end {block} */" in html


class TestMarks:
    def test_the_histogram_draws_at_most_one_bar_per_bin(self, plots, figures):
        bars = figures["histogram"].render().count('class="ach-bar"')
        assert 0 < bars <= plots.BINS

    def test_the_norm_line_has_one_point_per_layer(self, figures, capture_dir):
        acts, _, _ = load_capture(capture_dir)
        assert figures["norm"].render().count('class="rnl-dot"') == len(acts)

    def test_the_projection_has_one_point_per_row_and_one_class_per_label(
        self, figures, capture_dir
    ):
        _, labels, _ = load_capture(capture_dir)
        html = figures["projection"].render()
        points = re.findall(r'class="cpj-c(\d)" cx=', html)
        assert len(points) == len(labels)
        assert sorted({int(p) for p in points}) == sorted(
            {int(v) for v in np.unique(labels)}
        )

    def test_coordinates_are_rounded(self, figures):
        for f in figures.values():
            for num in re.findall(
                r'(?:cx|cy|x|y|width|height)="(-?\d+\.\d+)"', f.render()
            ):
                assert len(num.split(".")[1]) <= 2, f"{f.name}: {num}"


class TestManifestRegistration:
    def test_register_adds_an_entry_once_and_keeps_its_description(
        self, plots, figures, tmp_path
    ):
        figura = tmp_path / "figura"
        figura.mkdir()
        (figura / "manifest.json").write_text("[]\n", encoding="utf-8")
        f = figures["norm"]
        rel = Path("diagrams/activation-capture") / f"{f.name}.html"
        out = plots.Printer(quiet=True)
        plots.register(figura, rel, f, "activation-capture", "ambient", out)
        entries = json.loads((figura / "manifest.json").read_text())
        assert [e["id"] for e in entries] == [f.name]
        entries[0]["description"] = "kept"
        (figura / "manifest.json").write_text(json.dumps(entries))
        plots.register(figura, rel, f, "activation-capture", "ambient", out)
        entries = json.loads((figura / "manifest.json").read_text())
        assert len(entries) == 1 and entries[0]["description"] == "kept"
