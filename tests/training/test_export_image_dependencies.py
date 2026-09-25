"""The export image's dependency set must cover what `python -m training` imports.

docker/training/Dockerfile.export installs a hand-pinned CPU stack rather than a
pyproject group, so the container holds exactly the packages its `uv pip install`
lines name, plus their own dependencies. The export subcommand itself needs only
torch, transformers, peft and safetensors. But `python -m training` registers
every subcommand before it dispatches, and the module-scope imports of those
command modules reach omegaconf (through training/recipes) and xxhash (through
training/data). On 2026-09-25 the first export the image ever ran died at
`training/data/contamination.py` with ModuleNotFoundError before the merge
began (KB-91): the same defect shape as KB-59 in the interp image, with the
same package, and for the same reason -- nothing compared the image's install
list against the CLI's imports.

This walks the imports the way the interpreter would and requires each
third-party root to be pinned in the Dockerfile. Two walks, at different
depths, because the image runs one subcommand:

- module scope only, from `training.cli` and `training.__main__`: what
  executes before dispatch, for every subcommand;
- any depth, from `training.export` and `training.cli`'s export handler: the
  deferred imports the export itself runs.

A third check holds each Dockerfile pin to its pyproject.toml version, which
the Dockerfile's comment promises is kept in step by hand.

The escape list is the point: each entry needs a reason, so an addition is a
decision rather than a silent widening.
"""

from __future__ import annotations

import ast
import pathlib
import re
import sys
import tomllib

import pytest
from support import REPO_ROOT as _REPO_ROOT_STR

REPO_ROOT = pathlib.Path(_REPO_ROOT_STR)
DOCKERFILE = REPO_ROOT / "docker" / "training" / "Dockerfile.export"

FIRST_PARTY = {"training", "common"}

# Import name -> distribution name, where they differ.
DIST_NAME = {"yaml": "pyyaml"}

# Installed in the container as another package's dependency rather than by a
# pin of our own. Declaring them would be a second source of truth for a
# version the parent already fixes.
TRANSITIVE = {
    "numpy": "a transformers dependency",
    "safetensors": "a transformers dependency",
    "tokenizers": "a transformers dependency",
    "yaml": "a transformers dependency",
    "huggingface_hub": "a transformers dependency",
}

_PIN = re.compile(r"\b([A-Za-z0-9_.-]+)==([0-9][A-Za-z0-9_.+-]*)")


def _module_path(module: str) -> pathlib.Path | None:
    direct = REPO_ROOT / (module.replace(".", "/") + ".py")
    if direct.exists():
        return direct
    package = REPO_ROOT / module.replace(".", "/") / "__init__.py"
    return package if package.exists() else None


def _import_nodes(tree: ast.Module, *, module_scope_only: bool):
    nodes = tree.body if module_scope_only else list(ast.walk(tree))
    for node in nodes:
        if isinstance(node, ast.Import):
            yield from (alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and not node.level and node.module:
            yield node.module
            # `from training.commands import build_contamination` names a
            # submodule; follow it as one, harmlessly if it is a symbol.
            yield from (f"{node.module}.{alias.name}" for alias in node.names)


def _walk(roots: list[str], *, module_scope_only: bool) -> dict[str, set[str]]:
    """Third-party import roots reachable from `roots`, with the files that import them."""
    queue = list(roots)
    seen: set[str] = set()
    third: dict[str, set[str]] = {}
    while queue:
        module = queue.pop()
        if module in seen:
            continue
        seen.add(module)
        path = _module_path(module)
        if path is None:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for imported in _import_nodes(tree, module_scope_only=module_scope_only):
            root = imported.split(".")[0]
            if root in FIRST_PARTY:
                queue.append(imported)
            elif root not in sys.stdlib_module_names and not root.startswith("_"):
                third.setdefault(root, set()).add(str(path.relative_to(REPO_ROOT)))
    return third


def before_dispatch() -> dict[str, set[str]]:
    """What `python -m training <anything>` imports before the handler runs."""
    return _walk(["training.__main__", "training.cli"], module_scope_only=True)


def export_handler() -> dict[str, set[str]]:
    """What the export handler and the package it defers to import, at any depth."""
    roots = ["training.cli"] + [
        f"training.export.{path.stem}"
        for path in (REPO_ROOT / "training" / "export").glob("*.py")
    ]
    # training.cli at any depth would drag in every other handler's deferred
    # imports (trl, deepspeed); take only the export handler's body.
    tree = ast.parse((REPO_ROOT / "training" / "cli.py").read_text(encoding="utf-8"))
    handler = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "cmd_export"
    )
    deferred = [
        imported
        for imported in _import_nodes(
            ast.Module(body=[handler], type_ignores=[]), module_scope_only=False
        )
        if imported.split(".")[0] in FIRST_PARTY
    ]
    return _walk(roots[1:] + deferred, module_scope_only=False)


def dockerfile_pins() -> dict[str, str]:
    """Every `name==version` the Dockerfile's uv pip install lines name."""
    text = DOCKERFILE.read_text(encoding="utf-8")
    lines = [line for line in text.splitlines() if not line.lstrip().startswith("#")]
    return {name.lower(): version for name, version in _PIN.findall("\n".join(lines))}


def pyproject_pins() -> dict[str, set[str]]:
    """Every exact pin in pyproject.toml, by distribution, across all sections."""
    manifest = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    specs = list(manifest["project"]["dependencies"])
    for group in manifest.get("dependency-groups", {}).values():
        specs += [spec for spec in group if isinstance(spec, str)]
    for extra in manifest["project"].get("optional-dependencies", {}).values():
        specs += extra
    pins: dict[str, set[str]] = {}
    for spec in specs:
        match = _PIN.search(spec)
        if match:
            pins.setdefault(match.group(1).lower(), set()).add(match.group(2))
    return pins


def _missing(third: dict[str, set[str]], installed: set[str]) -> dict[str, list[str]]:
    missing = {}
    for root, sites in sorted(third.items()):
        if root in TRANSITIVE:
            continue
        if DIST_NAME.get(root, root.replace("_", "-")).lower() in installed:
            continue
        missing[root] = sorted(sites)
    return missing


def _report(missing: dict[str, list[str]], when: str) -> str:
    return (
        f"reachable {when} but not pinned in {DOCKERFILE.relative_to(REPO_ROOT)}:\n"
        + "\n".join(f"  {name}: {', '.join(sites)}" for name, sites in missing.items())
        + "\n\nPin it in the Dockerfile, or add it to TRANSITIVE here with the reason."
    )


class TestExportImageCoversItsImports:
    def test_every_import_before_dispatch_is_pinned(self):
        missing = _missing(before_dispatch(), set(dockerfile_pins()))
        assert not missing, _report(missing, "before `python -m training` dispatches")

    def test_every_import_of_the_export_handler_is_pinned(self):
        missing = _missing(export_handler(), set(dockerfile_pins()))
        assert not missing, _report(missing, "from the export handler")

    def test_the_guard_would_catch_a_missing_package(self):
        # The assertions above pass trivially if the walk finds nothing, so pin
        # that it resolves imports across the first-party boundary: omegaconf
        # and xxhash are reached only through training.commands, the path the
        # 2026-09-25 failure took.
        third = before_dispatch()
        assert "omegaconf" in third
        assert "xxhash" in third
        assert any("training/data/" in site for site in third["xxhash"])

    def test_export_handler_needs_nothing_beyond_the_merge_stack(self):
        # The export itself is torch, transformers, peft and safetensors; the
        # two extra pins exist for the CLI's registration pass, not for this.
        roots = set(export_handler()) - set(TRANSITIVE)
        assert roots <= {
            "torch",
            "transformers",
            "peft",
            "safetensors",
            "accelerate",
        }, roots


class TestExportImagePinsMirrorPyproject:
    @pytest.mark.parametrize("name,version", sorted(dockerfile_pins().items()))
    def test_pin_matches_pyproject(self, name, version):
        pins = pyproject_pins()
        assert name in pins, (
            f"{name} is pinned in the Dockerfile but nowhere in pyproject.toml"
        )
        assert version in pins[name], (
            f"{name}=={version} in the Dockerfile; pyproject.toml pins {sorted(pins[name])}"
        )
