"""Python source dependency discovery for Snakemake rules.

Snakemake's ``code`` rerun trigger tracks the rule definition and the file named
by ``script:``, but it does **not** follow Python imports. A rule whose script
imports ``orbitdet.visualization.residuals`` therefore stays "up to date" when
``orbitdet/visualization/base.py`` -- imported by that module -- changes.

:func:`python_dependencies` closes that gap::

    rule plot_results:
        input:
            data="results/{sample}/statistics.parquet",
            code=python_dependencies("orbitdet.visualization.residuals"),
            config=hydra_dependencies("config/plotting.yaml")
        output:
            "results/{sample}/plot.png"
        script:
            "scripts/plot.py"

The argument may be either:

* a fully qualified module name inside :data:`PACKAGE`, e.g.
  ``"orbitdet.visualization.residuals"``; or
* a module name or file path for a script outside the package, e.g.
  ``"scripts.solve_least_squares"`` or ``"scripts/solve_least_squares.py"``.


This is static analysis, so it cannot see:
* dynamic imports (``importlib.import_module``, ``__import__``, ``exec``);
* module names built at runtime;
* files read directly from disk (data, templates, kernels);
* compiled/native extensions.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from functools import cache
from pathlib import Path

import grimp

__all__ = [
    "python_dependencies",
    "PythonDependencyError",
    "UnknownModuleError",
]

#: Import root: the package whose internal imports are tracked.
PACKAGE = "orbitdet"


class PythonDependencyError(RuntimeError):
    """Raised when a module's Python dependencies cannot be determined."""


class UnknownModuleError(PythonDependencyError):
    """Raised when the entry point cannot be found in the import graph.

    Attributes:
        module: The module name that was requested.
    """

    def __init__(self, module: str) -> None:
        self.module = module
        super().__init__(
            f"'{module}' is not a module of the '{PACKAGE}' package or of a script "
            "root, so its dependencies cannot be tracked. Pass a fully qualified "
            f"module name such as '{PACKAGE}.visualization.residuals', or a script "
            "path such as 'scripts/solve_least_squares.py'."
        )


@cache
def _root_dir(root_package: str) -> Path:
    """Directory containing *root_package*'s source tree.

    Resolved with :func:`importlib.util.find_spec` on the *top-level* package
    only. That is cheap and does not import anything: resolving a submodule
    would import its parent packages, which for this project means loading
    tudatpy and its native extensions.
    """
    spec = importlib.util.find_spec(root_package)
    if spec is None or not spec.submodule_search_locations:
        raise PythonDependencyError(
            f"Package '{root_package}' is not importable, so its source files cannot "
            "be located. Install the project (e.g. 'pip install -e .') and run from "
            "the repository root before parsing the workflow."
        )
    return Path(list(spec.submodule_search_locations)[0]).resolve()


@cache
def _import_graph(roots: tuple[str, ...]) -> grimp.ImportGraph:
    """Build (once per root set) the static import graph.

    Cached because building the graph is the expensive step and every rule asks
    for a different entry point from the same graph. The root set is part of the
    cache key: a script entry point needs ``orbitdet`` *and* the script's own
    root package in one graph, or the cross-package imports are invisible.
    """
    return grimp.build_graph(*roots)


def _module_to_file(module: str, roots: tuple[str, ...]) -> Path | None:
    """Source file for *module*, or ``None`` if it has no ``.py`` file.

    Maps the dotted name onto the root package's directory instead of calling
    :func:`importlib.util.find_spec` per module, which would import every parent
    package. Both a plain module (``pkg/mod.py``) and a package
    (``pkg/mod/__init__.py``) are handled.
    """
    root_package, _, remainder = module.partition(".")
    if root_package not in roots:
        return None

    base = _root_dir(root_package)
    if remainder:
        base = base.joinpath(*remainder.split("."))

    for candidate in (base.with_suffix(".py"), base / "__init__.py"):
        if candidate.is_file():
            return candidate.resolve()
    return None


def _display_path(path: Path) -> str:
    """Path as Snakemake should see it.

    Relative to the working directory when the file lives under it, so rule
    inputs read as ``src/orbitdet/...`` rather than an absolute path. Falls back
    to the absolute path when the source tree is outside the working directory
    (e.g. an installed copy), which keeps the result usable either way.
    """
    try:
        return Path(os.path.relpath(path, Path.cwd())).as_posix()
    except ValueError:  # pragma: no cover - different drives on Windows
        return path.as_posix()


def _ancestor_packages(module: str) -> set[str]:
    """Packages whose ``__init__.py`` runs when *module* is imported.

    Importing ``a.b.c`` executes ``a/__init__.py`` and ``a/b/__init__.py``.
    Grimp records explicit imports only, so it does not report these, yet
    editing a package ``__init__.py`` changes what the import does. Omitting
    them would leave a rule stale after such an edit.
    """
    parts = module.split(".")
    return {".".join(parts[:depth]) for depth in range(1, len(parts))}


def _module_name_from_path(path: Path) -> str:
    """Dotted module name for a script file, or raise if it is not importable.

    A script such as ``scripts/solve_least_squares.py`` is not part of an
    installed package, so its module name is derived by walking up to the first
    ancestor that makes the name importable (the repository root, normally).

    The working directory is added to ``sys.path`` when it is missing. A plain
    ``python`` run puts ``""`` there, but Snakemake replaces ``sys.path[0]`` with
    the Snakefile's directory, so the repository root would otherwise not be
    importable and the script's module name could not be resolved.
    """
    resolved = path.resolve()
    if not resolved.is_file():
        raise PythonDependencyError(f"Script not found: {path}")

    cwd = Path.cwd().resolve()
    if str(cwd) not in sys.path:
        sys.path.insert(0, str(cwd))

    import_roots = {Path(entry or os.getcwd()).resolve() for entry in sys.path}

    for parent in resolved.parents:
        if parent not in import_roots:
            continue
        relative = resolved.relative_to(parent)
        module = ".".join(relative.with_suffix("").parts)
        try:
            spec = importlib.util.find_spec(module)
        except (ImportError, ValueError):
            continue
        if spec is not None:
            return module

    raise PythonDependencyError(
        f"Could not derive an importable module name for '{path}'. Run from the "
        "repository root (or add it to PYTHONPATH) so the script's directory is "
        "importable as a package."
    )


def _is_local_root(root_package: str) -> bool:
    """Whether *root_package* is project code rather than an installed library.

    A script's root package (``scripts``) is a local directory, while ``json``
    or ``numpy`` live in the standard library or ``site-packages``. Tracking the
    latter would make a rule depend on the environment, which is pinned
    separately and never edited alongside the code.
    """
    try:
        spec = importlib.util.find_spec(root_package)
    except (ImportError, ValueError):
        return False
    if spec is None or not spec.submodule_search_locations:
        return False

    location = Path(list(spec.submodule_search_locations)[0]).resolve()
    installed = {Path(sys.prefix).resolve(), Path(sys.base_prefix).resolve()}
    return not any(location.is_relative_to(prefix) for prefix in installed)


def _resolve_entrypoint(entrypoint: str) -> str:
    """Module name for *entrypoint*, which may be a module name or a file path."""
    if entrypoint.endswith(".py") or os.sep in entrypoint or "/" in entrypoint:
        return _module_name_from_path(Path(entrypoint))
    return entrypoint


def python_dependencies(entrypoint: str) -> list[str]:
    """Return all Python source files required by an entry-point module.

    Includes the entry point itself and every module it imports transitively
    within :data:`PACKAGE` (and, for a script, within the script's own root
    package), so editing any of them invalidates a Snakemake rule that declares
    this list as an input.

    Parameters
    ----------
    entrypoint:
        Either a fully qualified module name, e.g.
        ``"orbitdet.visualization.residuals"``, or a script module name or path,
        e.g. ``"scripts.solve_least_squares"`` or
        ``"scripts/solve_least_squares.py"``.

    Returns
    -------
    list[str]
        Sorted, de-duplicated source paths. Relative to the working directory
        when the package lives under it, otherwise absolute. Suitable for a
        Snakemake ``input:`` list.

    Raises
    ------
    UnknownModuleError
        If the entry point is not in the import graph.
    PythonDependencyError
        If the package or script cannot be located on disk.

    Examples
    --------
    >>> python_dependencies("orbitdet.visualization.residuals")
    ['src/orbitdet/analysis/__init__.py',
     'src/orbitdet/visualization/__init__.py',
     'src/orbitdet/visualization/base.py',
     'src/orbitdet/visualization/residuals.py',
     ...]
    >>> python_dependencies("scripts/solve_least_squares.py")
    ['scripts/solve_least_squares.py',
     'src/orbitdet/analysis/__init__.py',
     ...]
    """
    module = _resolve_entrypoint(entrypoint)

    # A script lives outside PACKAGE, so its graph must contain both roots or the
    # `orbitdet` imports it makes are treated as external and dropped. Only local
    # roots are added: a standard-library or site-packages root would make the
    # rule depend on the environment.
    root_package = module.split(".")[0]
    if root_package != PACKAGE and not _is_local_root(root_package):
        raise UnknownModuleError(module)

    roots = tuple(sorted({PACKAGE, root_package}))
    graph = _import_graph(roots)

    if module not in graph.modules:
        raise UnknownModuleError(module)

    # `find_upstream_modules` returns the modules *imported by* the entry point,
    # transitively. The entry point itself is not included, so add it.
    modules = {module}
    modules.update(graph.find_upstream_modules(module))

    # Keep only the tracked roots: third-party modules are pinned by the
    # environment, not edited alongside the code.
    modules = {name for name in modules if name.split(".")[0] in roots}

    # Importing a module also executes its parent packages' __init__.py, which
    # Grimp does not model as an import edge.
    for name in list(modules):
        modules.update(_ancestor_packages(name))

    files = {path for name in modules if (path := _module_to_file(name, roots)) is not None}

    return sorted(_display_path(path) for path in files)
