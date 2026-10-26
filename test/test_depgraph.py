"""Tests for the static import graph used to scope Snakemake dependencies.

The graph is load-bearing: it decides which stages a source edit invalidates. A
bug here is silent in both directions -- too few files means stale outputs, too
many means the whole pipeline reruns on every keystroke.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "workflow"))

import depgraph  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
# ``stage_sources`` returns paths relative to the root it is given, so pass the
# relative form -- this is exactly what the Snakefile does.
SRC = Path("src")
ABS_SRC = REPO_ROOT / SRC


@pytest.fixture(scope="module")
def graph():
    return depgraph.build_graph(ABS_SRC)


def test_index_finds_package_modules():
    by_module = depgraph._index(ABS_SRC)
    assert "orbitdet" in by_module
    assert "orbitdet.pipeline.prefit" in by_module
    assert "orbitdet.visualization.residuals" in by_module


def test_package_init_is_a_dependency_of_its_submodules(graph):
    """Importing ``orbitdet.pipeline.prefit`` executes ``orbitdet/__init__.py``."""
    prefit = ABS_SRC / "orbitdet" / "pipeline" / "prefit.py"
    assert ABS_SRC / "orbitdet" / "__init__.py" in graph[prefit]
    assert ABS_SRC / "orbitdet" / "pipeline" / "__init__.py" in graph[prefit]


def test_direct_import_is_recorded(graph):
    """``figures.py`` imports ``Residuals`` from the residuals plotter."""
    figures = ABS_SRC / "orbitdet" / "pipeline" / "figures.py"
    assert ABS_SRC / "orbitdet" / "visualization" / "residuals.py" in graph[figures]


def test_from_import_resolves_to_submodule(graph):
    """``from orbitdet.data.gaia import ...`` must pull in ``data/gaia.py``."""
    residuals = ABS_SRC / "orbitdet" / "visualization" / "residuals.py"
    assert ABS_SRC / "orbitdet" / "data" / "gaia.py" in graph[residuals]


def test_relative_import_is_resolved(graph):
    """``from .base import Plot`` inside ``visualization/`` resolves to ``base.py``."""
    residuals = ABS_SRC / "orbitdet" / "visualization" / "residuals.py"
    assert ABS_SRC / "orbitdet" / "visualization" / "base.py" in graph[residuals]


def test_closure_is_transitive():
    """The figure stage reaches the plotter through ``figures.py``."""
    files = depgraph.stage_sources(["orbitdet.pipeline.run_residuals_figure"], SRC)
    assert "src/orbitdet/pipeline/figures.py" in files
    assert "src/orbitdet/visualization/residuals.py" in files
    assert "src/orbitdet/visualization/base.py" in files


def test_closure_excludes_unrelated_modules():
    """The figure stage must not depend on the estimation or simulation code."""
    files = depgraph.stage_sources(["orbitdet.pipeline.run_residuals_figure"], SRC)
    assert not any("estimation" in f for f in files)
    assert not any("simulation" in f for f in files)


def test_stages_have_distinct_closures():
    """A plotter edit must not invalidate the kernel stage, and vice versa.

    This is the regression guard for the original bug: declaring the whole
    package on every rule made every edit invalidate everything.
    """
    kernels = set(depgraph.stage_sources(["orbitdet.pipeline.run_kernels"], SRC))
    figure = set(depgraph.stage_sources(["orbitdet.pipeline.run_residuals_figure"], SRC))

    # Isolation in both directions.
    assert "src/orbitdet/visualization/residuals.py" in figure
    assert "src/orbitdet/visualization/residuals.py" not in kernels
    assert "src/orbitdet/pipeline/kernels.py" in kernels
    assert "src/orbitdet/pipeline/kernels.py" not in figure

    # Neither closure contains the other.
    assert not kernels <= figure
    assert not figure <= kernels


def test_resolve_stage_is_the_smallest_closure():
    """The resolve stage only composes config; it must not pull in tudat code."""
    resolve = depgraph.stage_sources(["orbitdet.pipeline.run_resolve_config"], SRC)
    kernels = depgraph.stage_sources(["orbitdet.pipeline.run_kernels"], SRC)
    assert len(resolve) < len(kernels)


def test_unknown_entry_module_yields_empty_closure():
    assert depgraph.stage_sources(["orbitdet.does_not_exist"], SRC) == []


def test_unparsable_file_does_not_raise(tmp_path):
    """A syntax error should surface as a stage failure, not a DAG error."""
    package = tmp_path / "pkg"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "broken.py").write_text("def f(:\n")
    (package / "good.py").write_text("import pkg.broken\n")

    graph = depgraph.build_graph(tmp_path)
    # ``pkg/__init__.py`` is included because importing ``pkg.broken`` executes
    # the ``pkg`` package first.
    assert graph[package / "good.py"] == {
        package / "broken.py",
        package / "__init__.py",
    }


# ===========================================================================
# Dynamic-import guard
#
# The static graph cannot see dynamic imports. Silently returning an incomplete
# dependency list would produce stale outputs, so the guard must fail loudly.
# ===========================================================================


def _package(tmp_path, **modules):
    """Write a throwaway package and return its root."""
    package = tmp_path / "pkg"
    package.mkdir()
    (package / "__init__.py").write_text("")
    for name, source in modules.items():
        (package / f"{name}.py").write_text(source)
    return tmp_path


@pytest.mark.parametrize(
    "source",
    [
        'import importlib\nmod = importlib.import_module("pkg.plugin")\n',
        'mod = __import__("pkg.plugin")\n',
        'exec("import pkg.plugin")\n',
        "eval(\"__import__('pkg.plugin')\")\n",
    ],
)
def test_dynamic_import_raises(tmp_path, source):
    root = _package(tmp_path, entry=source, plugin="VALUE = 1\n")
    with pytest.raises(depgraph.DynamicImportError):
        depgraph.stage_sources(["pkg.entry"], root)


def test_module_level_getattr_raises(tmp_path):
    """PEP 562 lazy loading hides imports behind attribute access."""
    source = "def __getattr__(name):\n    import pkg.plugin\n    return pkg.plugin\n"
    root = _package(tmp_path, entry=source, plugin="VALUE = 1\n")
    with pytest.raises(depgraph.DynamicImportError):
        depgraph.stage_sources(["pkg.entry"], root)


def test_class_level_getattr_is_allowed(tmp_path, monkeypatch):
    """A ``__getattr__`` inside a class is ordinary attribute access."""
    source = "class C:\n    def __getattr__(self, name):\n        return 1\n"
    _package(tmp_path, entry=source)
    # ``stage_sources`` returns paths relative to the root it is given, so run
    # from the package root and pass the relative form -- as the Snakefile does.
    monkeypatch.chdir(tmp_path)
    assert depgraph.stage_sources(["pkg.entry"], Path(".")) == [
        "pkg/__init__.py",
        "pkg/entry.py",
    ]


def test_entry_module_ancestors_are_included():
    """The entry module is never imported, so its own ``__init__`` files must
    be added explicitly -- editing ``orbitdet/__init__.py`` must invalidate."""
    files = depgraph.stage_sources(["orbitdet.pipeline.run_kernels"], SRC)
    assert "src/orbitdet/__init__.py" in files
    assert "src/orbitdet/pipeline/__init__.py" in files


def test_error_message_names_the_offending_file(tmp_path):
    root = _package(tmp_path, entry='import importlib\nimportlib.import_module("x")\n')
    with pytest.raises(depgraph.DynamicImportError, match="entry.py"):
        depgraph.stage_sources(["pkg.entry"], root)


def test_real_package_has_no_dynamic_imports():
    """Guard the assumption the whole design rests on.

    If this fails, the dependency lists are incomplete and edits to the
    dynamically imported file will silently leave stale outputs.
    """
    for stage in (
        "orbitdet.pipeline.run_kernels",
        "orbitdet.pipeline.run_observations",
        "orbitdet.pipeline.run_prefit_spice",
        "orbitdet.pipeline.run_residuals_figure",
        "orbitdet.pipeline.run_resolve_config",
    ):
        depgraph.stage_sources([stage], SRC)
