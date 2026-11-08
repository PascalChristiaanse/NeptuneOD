"""Tests for :func:`orbitdet.reproducibility.python_graph.python_dependencies`.

The helper answers one question: which Python source files does a module need?
The tests build small packages on disk and assert on the returned file list,
because that list is what Snakemake uses to decide whether a rule is stale.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from orbitdet.reproducibility import python_graph
from orbitdet.reproducibility.python_graph import (
    PythonDependencyError,
    UnknownModuleError,
    python_dependencies,
)


def _write(path: Path, text: str = "") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


@pytest.fixture
def fake_package(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """The package from the acceptance criteria, importable from *tmp_path*.

    The tree deliberately has no ``__init__.py`` files, matching the acceptance
    criteria: it is a namespace package. ``python_graph`` is pointed at it by
    patching ``PACKAGE`` and clearing the caches, so the real ``orbitdet`` graph
    is never involved.
    """
    root = tmp_path / "fakepkg"
    _write(root / "visualization" / "plotting.py", "from fakepkg.visualization import common\n")
    _write(root / "visualization" / "common.py", "from fakepkg.utils import statistics\n")
    _write(root / "utils" / "statistics.py", "VALUE = 1\n")
    _write(root / "preprocessing" / "filtering.py", "OTHER = 2\n")

    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(python_graph, "PACKAGE", "fakepkg")
    python_graph._import_graph.cache_clear()
    python_graph._root_dir.cache_clear()
    monkeypatch.chdir(tmp_path)

    yield tmp_path

    python_graph._import_graph.cache_clear()
    python_graph._root_dir.cache_clear()
    sys.modules.pop("fakepkg", None)


def test_returns_entry_point_and_transitive_imports(fake_package: Path) -> None:
    """The acceptance criteria: entry point plus its transitive imports."""
    files = python_dependencies("fakepkg.visualization.plotting")

    assert files == [
        "fakepkg/utils/statistics.py",
        "fakepkg/visualization/common.py",
        "fakepkg/visualization/plotting.py",
    ]


def test_includes_the_entry_point(fake_package: Path) -> None:
    """A rule must rerun when the entry module itself changes."""
    files = python_dependencies("fakepkg.visualization.plotting")

    assert "fakepkg/visualization/plotting.py" in files


def test_includes_indirect_imports(fake_package: Path) -> None:
    """``statistics`` is two hops away and must still be included."""
    files = python_dependencies("fakepkg.visualization.plotting")

    assert "fakepkg/utils/statistics.py" in files


def test_excludes_unrelated_modules(fake_package: Path) -> None:
    """An unrelated module must not invalidate the rule."""
    files = python_dependencies("fakepkg.visualization.plotting")

    assert not any("filtering" in path for path in files)


def test_excludes_third_party_modules(fake_package: Path) -> None:
    """Only the tracked package is returned, never the dependency tree."""
    _write(
        fake_package / "fakepkg" / "visualization" / "plotting.py",
        "import os\nimport json\nfrom fakepkg.visualization import common\n",
    )
    python_graph._import_graph.cache_clear()

    files = python_dependencies("fakepkg.visualization.plotting")

    assert all(path.startswith("fakepkg/") for path in files)


def test_returns_sorted_unique_paths(fake_package: Path) -> None:
    """Snakemake inputs must be deterministic and free of duplicates."""
    files = python_dependencies("fakepkg.visualization.plotting")

    assert files == sorted(files)
    assert len(files) == len(set(files))


def test_package_init_is_included(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Importing a submodule executes its parent packages, so they are deps.

    Grimp records explicit imports only, so the ancestor ``__init__.py`` files
    are added explicitly. Editing one changes what the import does, so omitting
    them would leave a rule stale.
    """
    root = tmp_path / "initpkg"
    _write(root / "__init__.py")
    _write(root / "visualization" / "__init__.py")
    _write(root / "visualization" / "plotting.py", "VALUE = 1\n")

    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(python_graph, "PACKAGE", "initpkg")
    python_graph._import_graph.cache_clear()
    python_graph._root_dir.cache_clear()
    monkeypatch.chdir(tmp_path)
    try:
        files = python_dependencies("initpkg.visualization.plotting")
    finally:
        python_graph._import_graph.cache_clear()
        python_graph._root_dir.cache_clear()
        sys.modules.pop("initpkg", None)

    assert "initpkg/__init__.py" in files
    assert "initpkg/visualization/__init__.py" in files
    assert "initpkg/visualization/plotting.py" in files


def test_leaf_module_returns_only_itself(fake_package: Path) -> None:
    """A module with no internal imports depends on itself and its packages."""
    files = python_dependencies("fakepkg.utils.statistics")

    assert "fakepkg/utils/statistics.py" in files
    assert "fakepkg/visualization/plotting.py" not in files


def test_unknown_module_raises(fake_package: Path) -> None:
    """A typo must fail loudly rather than silently returning nothing."""
    with pytest.raises(UnknownModuleError, match="not a module"):
        python_dependencies("fakepkg.does_not_exist")


def test_module_outside_package_raises(fake_package: Path) -> None:
    """A module from another package cannot be tracked."""
    with pytest.raises(UnknownModuleError):
        python_dependencies("json")


def test_paths_are_relative_to_the_working_directory(fake_package: Path) -> None:
    """Rule inputs should read as ``fakepkg/...``, not absolute paths."""
    for path in python_dependencies("fakepkg.visualization.plotting"):
        assert not Path(path).is_absolute()
        assert (fake_package / path).is_file()


def test_graph_is_cached(fake_package: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The graph must be built once, not per rule."""
    calls = []
    real_build = python_graph.grimp.build_graph

    def counting_build(*args, **kwargs):
        calls.append(args)
        return real_build(*args, **kwargs)

    monkeypatch.setattr(python_graph.grimp, "build_graph", counting_build)
    python_graph._import_graph.cache_clear()

    python_dependencies("fakepkg.visualization.plotting")
    python_dependencies("fakepkg.utils.statistics")

    assert len(calls) == 1


def test_real_package_entry_point() -> None:
    """The repository's own package resolves to real files."""
    files = python_dependencies("orbitdet.visualization.residuals")

    assert files
    assert all(Path(path).is_file() for path in files)
    assert any(path.endswith("visualization/residuals.py") for path in files)
    assert any(path.endswith("visualization/base.py") for path in files)


def test_real_package_excludes_unrelated_modules() -> None:
    """A plotting module must not depend on the estimation stack."""
    files = python_dependencies("orbitdet.visualization.residuals")

    assert not any("estimation" in path for path in files)


def test_missing_package_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """A clear error when the project is not installed."""
    monkeypatch.setattr(python_graph, "PACKAGE", "definitely_not_installed_pkg")
    python_graph._root_dir.cache_clear()

    with pytest.raises(PythonDependencyError, match="not importable"):
        python_graph._root_dir("definitely_not_installed_pkg")

    python_graph._root_dir.cache_clear()


# ---------------------------------------------------------------------------
# Script entry points
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_script(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A script outside the package that imports it.

    Mirrors ``scripts/solve_least_squares.py``: a namespace-package directory
    whose module imports the tracked package.
    """
    _write(tmp_path / "fakepkg" / "visualization" / "plotting.py", "VALUE = 1\n")
    script = _write(
        tmp_path / "scripts" / "run_plot.py",
        "from fakepkg.visualization import plotting\n",
    )

    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(python_graph, "PACKAGE", "fakepkg")
    python_graph._import_graph.cache_clear()
    python_graph._root_dir.cache_clear()
    monkeypatch.chdir(tmp_path)

    yield script

    python_graph._import_graph.cache_clear()
    python_graph._root_dir.cache_clear()
    for name in ("fakepkg", "scripts"):
        sys.modules.pop(name, None)


def test_script_module_name_is_supported(fake_script: Path) -> None:
    """A script module name resolves to the script and its package imports."""
    files = python_dependencies("scripts.run_plot")

    assert "scripts/run_plot.py" in files
    assert "fakepkg/visualization/plotting.py" in files


def test_script_file_path_is_supported(fake_script: Path) -> None:
    """A script path is accepted and gives the same result as its module name."""
    from_path = python_dependencies("scripts/run_plot.py")
    from_name = python_dependencies("scripts.run_plot")

    assert from_path == from_name


def test_script_includes_its_own_transitive_imports(fake_script: Path) -> None:
    """The script's package imports are followed, not just its own file."""
    files = python_dependencies("scripts/run_plot.py")

    assert "fakepkg/visualization/plotting.py" in files


def test_script_excludes_other_scripts(fake_script: Path) -> None:
    """A sibling script must not invalidate the rule."""
    _write(fake_script.parent / "unrelated.py", "OTHER = 1\n")
    python_graph._import_graph.cache_clear()

    files = python_dependencies("scripts/run_plot.py")

    assert not any("unrelated" in path for path in files)


def test_missing_script_raises(tmp_path: Path) -> None:
    """A path that does not exist fails loudly."""
    with pytest.raises(PythonDependencyError, match="not found"):
        python_dependencies(str(tmp_path / "nope.py"))


def test_stdlib_module_is_rejected(fake_script: Path) -> None:
    """A standard-library module is not project code and cannot be tracked."""
    with pytest.raises(UnknownModuleError):
        python_dependencies("json")


def test_real_script_entry_point() -> None:
    """The repository's own script resolves to real files."""
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "solve_least_squares.py"
    if not script.is_file():  # pragma: no cover - script may be renamed
        pytest.skip("solve_least_squares.py not present")

    files = python_dependencies(str(script))

    assert files
    assert all(Path(path).is_file() for path in files)
    assert any(path.endswith("scripts/solve_least_squares.py") for path in files)
    assert any(path.endswith("orbitdet/visualization/__init__.py") for path in files)
    # Only the entry script itself, never its siblings.
    assert [p for p in files if p.startswith("scripts/")] == ["scripts/solve_least_squares.py"]
