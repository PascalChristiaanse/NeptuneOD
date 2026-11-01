"""Tests for the static import graph used to declare per-stage dependencies.

The graph exists because Snakemake's ``code`` rerun trigger does not follow
Python imports. Two failure modes matter, and they pull in opposite directions:

* **Under-approximation** (a file is missing from the closure) means a rule does
  not re-run when it should, leaving a *stale output with no warning*. This is
  the dangerous direction, so the tests below check the closure is complete and
  that dynamic imports are refused rather than silently ignored.
* **Over-approximation** (a file is present that the stage cannot import) means
  unrelated edits invalidate the rule, which is merely wasteful. The isolation
  tests guard against this, because a flat closure would make the whole
  dependency-tracking exercise pointless.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "workflow"))

import depgraph  # noqa: E402


@pytest.fixture
def package(tmp_path: Path) -> Path:
    """Build a small package tree to exercise the graph against."""
    root = tmp_path / "src"
    pkg = root / "pkg"
    (pkg / "sub").mkdir(parents=True)

    (pkg / "__init__.py").write_text("")
    (pkg / "sub" / "__init__.py").write_text("")
    (pkg / "leaf.py").write_text("VALUE = 1\n")
    (pkg / "sub" / "deep.py").write_text("from pkg.leaf import VALUE\n")
    (pkg / "entry.py").write_text("from pkg.sub.deep import VALUE\n")
    (pkg / "unrelated.py").write_text("OTHER = 2\n")
    return root


def _relative(files, root):
    """Normalize returned paths to be relative to *root*.

    ``stage_sources`` returns paths relative to the root it was given, so an
    absolute root yields absolute paths. Normalizing here keeps the assertions
    independent of how the fixture built its root.
    """
    root = Path(root)
    out = set()
    for f in files:
        path = Path(f)
        try:
            out.add(str(path.relative_to(root)))
        except ValueError:
            out.add(str(path))
    return out


def test_stage_sources_includes_transitive_imports(package):
    files = _relative(depgraph.stage_sources(["pkg.entry"], root=package), package)

    assert "pkg/entry.py" in files
    assert "pkg/sub/deep.py" in files
    assert "pkg/leaf.py" in files


def test_stage_sources_includes_ancestor_packages(package):
    """Importing ``pkg.sub.deep`` executes both ``__init__.py`` files."""
    files = _relative(depgraph.stage_sources(["pkg.entry"], root=package), package)

    assert "pkg/__init__.py" in files
    assert "pkg/sub/__init__.py" in files


def test_stage_sources_excludes_unrelated_modules(package):
    """The closure must not be the whole package, or isolation is lost."""
    files = _relative(depgraph.stage_sources(["pkg.entry"], root=package), package)

    assert "pkg/unrelated.py" not in files


def test_stage_sources_handles_relative_imports(package):
    (package / "pkg" / "sub" / "sibling.py").write_text("from .deep import VALUE\n")

    files = _relative(depgraph.stage_sources(["pkg.sub.sibling"], root=package), package)

    assert "pkg/sub/deep.py" in files
    assert "pkg/leaf.py" in files


def test_stage_sources_handles_from_import_of_submodule(package):
    """``from pkg import leaf`` may name a submodule, not just an attribute."""
    (package / "pkg" / "via_attr.py").write_text("from pkg import leaf\n")

    files = _relative(depgraph.stage_sources(["pkg.via_attr"], root=package), package)

    assert "pkg/leaf.py" in files


def test_stage_sources_ignores_unknown_modules(package):
    (package / "pkg" / "external.py").write_text("import numpy\nimport os\n")

    files = _relative(depgraph.stage_sources(["pkg.external"], root=package), package)

    assert "pkg/external.py" in files
    assert not any("numpy" in f for f in files)


def test_stage_sources_returns_sorted_paths(package):
    files = depgraph.stage_sources(["pkg.entry"], root=package)

    assert files == sorted(files)


def test_unparsable_file_yields_empty_dependencies(package):
    """A syntax error must surface as a stage failure, not a confusing DAG error."""
    (package / "pkg" / "broken.py").write_text("def (:\n")

    files = _relative(depgraph.stage_sources(["pkg.broken"], root=package), package)

    assert "pkg/broken.py" in files


# ---------------------------------------------------------------------------
# Dynamic-import guard
# ---------------------------------------------------------------------------


def test_dynamic_import_via_importlib_is_refused(package):
    (package / "pkg" / "dynamic.py").write_text(
        "import importlib\n\nmod = importlib.import_module('pkg.leaf')\n"
    )

    with pytest.raises(depgraph.DynamicImportError, match="dynamic.py"):
        depgraph.stage_sources(["pkg.dynamic"], root=package)


def test_dynamic_import_via_dunder_import_is_refused(package):
    (package / "pkg" / "dunder.py").write_text("mod = __import__('pkg.leaf')\n")

    with pytest.raises(depgraph.DynamicImportError):
        depgraph.stage_sources(["pkg.dunder"], root=package)


def test_dynamic_import_via_exec_is_refused(package):
    (package / "pkg" / "execer.py").write_text("exec('import pkg.leaf')\n")

    with pytest.raises(depgraph.DynamicImportError):
        depgraph.stage_sources(["pkg.execer"], root=package)


def test_module_level_getattr_is_refused(package):
    """PEP 562 lazy loading can import code the static graph cannot see."""
    (package / "pkg" / "lazy.py").write_text(
        "def __getattr__(name):\n    import pkg.leaf\n    return pkg.leaf\n"
    )

    with pytest.raises(depgraph.DynamicImportError):
        depgraph.stage_sources(["pkg.lazy"], root=package)


def test_class_level_getattr_is_allowed(package):
    """A ``__getattr__`` inside a class is ordinary attribute access."""
    (package / "pkg" / "cls.py").write_text(
        "class C:\n    def __getattr__(self, name):\n        return 1\n"
    )

    files = _relative(depgraph.stage_sources(["pkg.cls"], root=package), package)

    assert "pkg/cls.py" in files


def test_bare_importlib_without_use_is_allowed(package):
    """Importing ``importlib`` without calling it cannot hide a dependency."""
    (package / "pkg" / "harmless.py").write_text("import importlib\n\nX = 1\n")

    files = _relative(depgraph.stage_sources(["pkg.harmless"], root=package), package)

    assert "pkg/harmless.py" in files


# ---------------------------------------------------------------------------
# The real package
# ---------------------------------------------------------------------------


def test_real_package_has_no_dynamic_imports():
    """Guards the assumption that the static graph is exact for this package.

    If this fails, a module gained a dynamic import and the per-stage closures
    may be incomplete -- which means stale outputs rather than a visible error.
    """
    root = Path(__file__).resolve().parents[1] / "src"
    entries = [
        "orbitdet.pipeline.run_resolve_config",
        "orbitdet.pipeline.run_kernels",
        "orbitdet.pipeline.run_observations",
        "orbitdet.pipeline.run_ephemeris_residuals",
        "orbitdet.pipeline.run_outliers",
        "orbitdet.pipeline.run_weighting",
        "orbitdet.pipeline.run_propagation",
        "orbitdet.pipeline.run_estimation",
        "orbitdet.pipeline.run_covariance",
        "orbitdet.pipeline.run_figure",
        "orbitdet.pipeline.run_tables",
        "orbitdet.pipeline.run_report",
    ]

    # Must not raise.
    for entry in entries:
        depgraph.stage_sources([entry], root=root)


def test_real_package_stage_closures_are_isolated():
    """A figure stage must not depend on the estimation stage's module.

    This is the property that makes the DAG worth splitting: editing a plotter
    must not invalidate the fit.
    """
    root = Path(__file__).resolve().parents[1] / "src"

    figure = _relative(depgraph.stage_sources(["orbitdet.pipeline.run_figure"], root=root), root)
    estimation = _relative(
        depgraph.stage_sources(["orbitdet.pipeline.run_estimation"], root=root), root
    )
    kernels = _relative(depgraph.stage_sources(["orbitdet.pipeline.run_kernels"], root=root), root)

    assert "orbitdet/pipeline/estimation.py" not in figure
    assert "orbitdet/visualization/residuals.py" in figure
    assert "orbitdet/visualization/residuals.py" not in estimation
    assert "orbitdet/visualization/residuals.py" not in kernels
    assert "orbitdet/pipeline/estimation.py" not in kernels


def test_real_package_closures_are_graded_not_flat():
    """Closures must differ per stage; a flat closure means no isolation."""
    root = Path(__file__).resolve().parents[1] / "src"

    sizes = {
        entry: len(depgraph.stage_sources([entry], root=root))
        for entry in (
            "orbitdet.pipeline.run_resolve_config",
            "orbitdet.pipeline.run_kernels",
            "orbitdet.pipeline.run_estimation",
            "orbitdet.pipeline.run_figure",
        )
    }

    assert len(set(sizes.values())) > 1, f"closures are not graded: {sizes}"
    assert sizes["orbitdet.pipeline.run_resolve_config"] < sizes["orbitdet.pipeline.run_figure"]
