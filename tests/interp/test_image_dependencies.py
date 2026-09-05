"""The interp image's dependency set must cover what the interp CLI imports.

docker/interp/Dockerfile.interp installs `uv sync --no-default-groups --group
interp`, so the container holds exactly [project] dependencies plus the interp
group -- not the sandbox group, not the training group. Every other tier's
dependencies are absent from it.

That is easy to get wrong in a direction nothing catches. A missing package is
not a lint error, not a type error, and not a test failure: it is a
ModuleNotFoundError raised inside a GPU container after a five-minute image
build, and only for the code path that reaches the import. `omegaconf` shipped
that way -- the interp tier takes recipes as DictConfig and loads them through
training/recipes/loader.py, but declares no recipes of its own, so the
dependency was invisible until a harvest died at import.

This walks the imports from the interp CLI the way the interpreter would, and
requires each third-party root to be reachable in the container. Lazy imports
are followed too: deferring an import changes when it fails, not whether.

The escape lists below are the point of the test. Each entry needs a reason,
so an addition is a decision rather than a silent widening.
"""

from __future__ import annotations

import ast
import pathlib
import sys
import tomllib

import pytest

from tests.interp.conftest import REPO_ROOT as _REPO_ROOT_STR

# conftest exports a str; every path join below wants a Path.
REPO_ROOT = pathlib.Path(_REPO_ROOT_STR)

FIRST_PARTY = {"interp", "training", "common"}

# Import name -> distribution name, where they differ.
DIST_NAME = {
    "sklearn": "scikit-learn",
    "yaml": "pyyaml",
    "huggingface_hub": "huggingface-hub",
    "sae_lens": "sae-lens",
    "transformer_lens": "transformer-lens",
    "hydra": "hydra-core",
    "sentence_transformers": "sentence-transformers",
    "flash_attn": "flash-attn",
    "lm_eval": "lm-eval",
    "PIL": "pillow",
}

# Installed in the container, but as another package's dependency rather than
# by a pin of our own. Declaring them would be a second source of truth for a
# version the parent already fixes.
TRANSITIVE = {
    "numpy": "torch, pandas, and scipy all require it",
    "tokenizers": "a transformers dependency",
    "yaml": "a transformers and datasets dependency",
    "safetensors": "a transformers dependency",
}

# Deliberately absent from the environment. Each is documented in pyproject.toml
# with the version conflict that keeps it out; the importing module defers the
# import so the rest of the tier loads without it.
ISOLATED = {
    "sae_bench": "0.6.0 requires datasets<4.0 and numpy<2.0 against the core "
    "datasets 5.0.0 pin; runs in its own venv (P4-D3)",
    "datatrove": "every release caps huggingface-hub<1.0 against the core 1.23.0 "
    "pin; runs in its own venv, invoked out-of-process",
}


def _module_path(module: str) -> pathlib.Path | None:
    direct = REPO_ROOT / (module.replace(".", "/") + ".py")
    if direct.exists():
        return direct
    package = REPO_ROOT / module.replace(".", "/") / "__init__.py"
    return package if package.exists() else None


def _imports(tree: ast.AST) -> list[str]:
    """Every absolute import in the file, at any nesting depth.

    Function-level imports count. The repo defers heavy imports so the CPU
    sandbox can load these modules, but a deferred import still executes when
    the command runs -- interp/commands/harvest.py defers interp.harvest into
    cmd(), which is precisely where the omegaconf failure surfaced.
    """
    found: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and not node.level and node.module:
            found.append(node.module)
    return found


def reachable_third_party() -> dict[str, set[str]]:
    """Third-party import roots reachable from any interp subcommand."""
    queue = ["interp.cli", "interp.__main__"]
    queue += [
        f"interp.commands.{path.stem}"
        for path in (REPO_ROOT / "interp" / "commands").glob("*.py")
    ]
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
        for imported in _imports(ast.parse(path.read_text(encoding="utf-8"))):
            root = imported.split(".")[0]
            if root in FIRST_PARTY:
                queue.append(imported)
            elif root not in sys.stdlib_module_names and not root.startswith("_"):
                third.setdefault(root, set()).add(str(path.relative_to(REPO_ROOT)))
    return third


def installed_distributions() -> set[str]:
    """What `--no-default-groups --group interp` puts in the image."""
    manifest = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    def bare(specifiers):
        return {
            spec.split("==")[0].split("[")[0].split(">")[0].split("<")[0].strip()
            for spec in specifiers
        }

    return bare(manifest["project"]["dependencies"]) | bare(
        manifest["dependency-groups"]["interp"]
    )


class TestInterpImageCoversItsImports:
    def test_every_reachable_import_is_available_in_the_container(self):
        installed = installed_distributions()
        third = reachable_third_party()
        missing = {}
        for root, sites in sorted(third.items()):
            if root in TRANSITIVE or root in ISOLATED:
                continue
            if DIST_NAME.get(root, root.replace("_", "-")) in installed:
                continue
            missing[root] = sorted(sites)
        assert not missing, (
            "reachable from an interp subcommand but not installed by "
            "`uv sync --no-default-groups --group interp`:\n"
            + "\n".join(
                f"  {name}: {', '.join(sites)}" for name, sites in missing.items()
            )
            + "\n\nAdd it to the interp group in pyproject.toml, or to "
            "TRANSITIVE/ISOLATED here with the reason."
        )

    def test_the_guard_would_catch_a_missing_package(self):
        # The assertion above passes trivially if the walk finds nothing, so
        # pin that it really resolves imports across the first-party boundary:
        # omegaconf is reached only through training/recipes, three modules
        # deep from interp.commands.harvest.
        third = reachable_third_party()
        assert "omegaconf" in third
        assert any(
            "harvest" in site or "recipes" in site for site in third["omegaconf"]
        )

    @pytest.mark.parametrize("name", sorted(ISOLATED))
    def test_isolated_packages_are_not_pinned_anywhere(self, name):
        # An isolated package that quietly gained a pin is a version conflict
        # waiting to resolve badly; the reason it is out is a hard constraint.
        manifest = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
        distribution = DIST_NAME.get(name, name.replace("_", "-"))
        assert f'"{distribution}==' not in manifest, (
            f"{distribution} is pinned in pyproject.toml but listed as "
            f"isolated here: {ISOLATED[name]}"
        )


class TestInterpImageCarriesATritonToolchain:
    """gcc and the Python headers are runtime dependencies, not build ones.

    Nothing in the image compiles during `docker build`, which makes the apt
    toolchain look like dead weight to anyone trimming layers. It is not: torch
    routes eager ops through Triton, and Triton builds its CUDA driver shim
    from C source on the first GPU call. Removing either package moves the
    failure to the first forward pass of a harvest, past the model load, where
    it costs a GPU slot to discover.

    The build-time smoke test cannot catch this -- it runs on a GPU-less
    builder, so Triton never initializes.
    """

    DOCKERFILE = REPO_ROOT / "docker" / "interp" / "Dockerfile.interp"

    @pytest.mark.parametrize(
        "package",
        [
            "gcc",  # triton/runtime/build.py:_build shells out to it
            "python${PYTHON_VERSION}-dev",  # driver.c includes <Python.h>
        ],
    )
    def test_apt_installs(self, package):
        assert package in self.DOCKERFILE.read_text(encoding="utf-8")

    def test_triton_jit_cache_is_on_an_executable_filesystem(self):
        # The shim is dlopened from TRITON_CACHE_DIR. The compose rootfs is
        # read_only, so that path is a tmpfs -- one mounted `noexec` would
        # compile the module successfully and then fail to load it.
        compose = (REPO_ROOT / "docker-compose.interp.yaml").read_text(encoding="utf-8")
        dockerfile = self.DOCKERFILE.read_text(encoding="utf-8")
        assert "ENV TRITON_CACHE_DIR=/tmp/triton" in dockerfile
        tmpfs = [
            line for line in compose.splitlines() if line.strip().startswith("- /tmp:")
        ]
        assert len(tmpfs) == 1, tmpfs
        assert "exec" in tmpfs[0] and "noexec" not in tmpfs[0]
