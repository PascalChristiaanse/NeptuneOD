"""Static import graph for the ``orbitdet`` package.

Snakemake's ``code`` rerun trigger tracks the rule definition and any file named
by ``script:``, but it does **not** follow Python imports. The obvious
workaround -- declaring the whole package as an input on every rule -- is the
"one header included by everything" anti-pattern: editing a plotter then
invalidates the kernel download, the observation build, and the estimation.

CMake avoids this with compiler-generated depfiles (``-MMD``): each object file
records exactly the headers it included, so a header edit rebuilds only its true
dependents. This module is the static equivalent for Python. It parses import
statements with :mod:`ast`, resolves them to files inside the package, and
returns the transitive closure for a given entry module.

Two details keep the result faithful to runtime behaviour:

* Importing ``a.b.c`` also executes ``a/__init__.py`` and ``a/b/__init__.py``,
  so every ancestor package is a dependency.
* ``from a.b import name`` may refer to a submodule ``a/b/name.py`` or to a name
  defined in ``a/b/__init__.py``. Both are recorded; resolution drops whichever
  does not exist.

The graph is a static over-approximation: it cannot see dynamic imports. The
package contains none today, so it is exact here -- but that is a property of
the code, not of this module, and it can silently stop being true.

A dynamic import is any import whose target is not a literal name in the source:

* ``importlib.import_module("pkg.plugin")``
* ``__import__("pkg.plugin")``
* ``exec("import pkg.plugin")``
* a module-level ``__getattr__`` (PEP 562) that lazily imports on attribute access

If one appears, this module under-approximates: the rule will not depend on the
dynamically imported file, so editing that file leaves a stale output with no
warning. That is the dangerous direction, so :func:`stage_sources` refuses to
guess -- it raises :class:`DynamicImportError` instead. Failing loudly is better
than a silently stale result.
"""

from __future__ import annotations

import ast
from collections.abc import Iterable
from pathlib import Path

#: Import root: the directory that contains the package.
DEFAULT_ROOT = Path("src")

#: Names whose presence in a module means the static graph may be incomplete.
_DYNAMIC_IMPORT_NAMES = frozenset({"importlib", "__import__", "exec", "eval"})


class DynamicImportError(RuntimeError):
    """Raised when a module may import code the static graph cannot see."""


def _dynamic_import_sites(path: Path) -> list[str]:
    """Names of dynamic-import constructs used in *path*, if any.

    Only module-level ``__getattr__`` counts: a ``__getattr__`` inside a class is
    ordinary attribute access and cannot trigger an import.
    """
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (SyntaxError, UnicodeDecodeError):
        return []

    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id in _DYNAMIC_IMPORT_NAMES:
            found.add(node.id)
        elif isinstance(node, ast.Attribute) and node.attr in _DYNAMIC_IMPORT_NAMES:
            found.add(node.attr)
        elif isinstance(node, ast.FunctionDef) and node.name == "__getattr__":
            # PEP 562: only a module-level definition is a lazy-import hook.
            if any(node is child for child in tree.body):
                found.add("__getattr__")

    return sorted(found)


def _module_name(path: Path, root: Path) -> str | None:
    """Dotted module name for *path*, or ``None`` if it is not importable."""
    try:
        relative = path.relative_to(root)
    except ValueError:
        return None

    parts = list(relative.parts)
    if parts[-1] == "__init__.py":
        parts.pop()
    elif parts[-1].endswith(".py"):
        parts[-1] = parts[-1][: -len(".py")]
    else:
        return None

    return ".".join(parts) if parts else None


def _index(root: Path) -> dict[str, Path]:
    """Map every dotted module name under *root* to its file."""
    by_module: dict[str, Path] = {}
    for path in sorted(root.rglob("*.py")):
        module = _module_name(path, root)
        if module is not None:
            by_module[module] = path
    return by_module


def _imported_names(path: Path, module: str, is_package: bool) -> set[str]:
    """Dotted names imported by *path*, with relative imports resolved.

    Unparsable files yield an empty set rather than raising: a syntax error
    should surface as a stage failure, not as a confusing DAG error.
    """
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (SyntaxError, UnicodeDecodeError):
        return set()

    current_package = module if is_package else module.rpartition(".")[0]
    names: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                parts = current_package.split(".") if current_package else []
                keep = len(parts) - (node.level - 1)
                prefix = ".".join(parts[:keep]) if keep > 0 else ""
                base = f"{prefix}.{base}" if prefix and base else prefix or base
            if not base:
                continue
            names.add(base)
            names.update(f"{base}.{alias.name}" for alias in node.names)

    return names


def _graph(by_module: dict[str, Path]) -> dict[Path, set[Path]]:
    """Build the file-to-file dependency graph from a module index."""
    graph: dict[Path, set[Path]] = {path: set() for path in by_module.values()}

    for module, path in by_module.items():
        is_package = path.name == "__init__.py"
        dependencies: set[Path] = set()

        for name in _imported_names(path, module, is_package):
            target = by_module.get(name)
            if target is None:
                continue
            dependencies.add(target)

            # Importing ``a.b.c`` also executes ``a/__init__.py`` and
            # ``a/b/__init__.py``.
            parts = name.split(".")
            for depth in range(1, len(parts)):
                ancestor = by_module.get(".".join(parts[:depth]))
                if ancestor is not None:
                    dependencies.add(ancestor)

        graph[path] = dependencies

    return graph


def build_graph(root: Path = DEFAULT_ROOT) -> dict[Path, set[Path]]:
    """Map each module file to the package files it imports."""
    return _graph(_index(Path(root)))


def stage_sources(entries: Iterable[str], root: Path = DEFAULT_ROOT) -> list[str]:
    """Every file a stage can import, directly or indirectly.

    Parameters
    ----------
    entries:
        Dotted module names the stage executes, e.g.
        ``["orbitdet.pipeline.run_prefit_spice"]``.
    root:
        Import root containing the package.

    Returns
    -------
    list[str]
        Sorted paths, suitable for a Snakemake ``input:`` list.

    Raises
    ------
    DynamicImportError
        If any file in the closure uses a construct that can import code the
        static graph cannot see. The dependency list would be incomplete, and an
        incomplete list means stale outputs rather than a visible failure.
    """
    root = Path(root)
    by_module = _index(root)
    graph = _graph(by_module)

    seen: set[Path] = set()
    stack: list[Path] = []
    for name in entries:
        path = by_module.get(name)
        if path is None:
            continue
        stack.append(path)
        # Importing ``a.b.c`` executes ``a/__init__.py`` and ``a/b/__init__.py``.
        # The graph records this for *imported* modules, but the entry module is
        # never imported by anything, so its own ancestors must be added here.
        parts = name.split(".")
        for depth in range(1, len(parts)):
            ancestor = by_module.get(".".join(parts[:depth]))
            if ancestor is not None:
                stack.append(ancestor)

    while stack:
        path = stack.pop()
        if path in seen:
            continue
        seen.add(path)
        stack.extend(graph.get(path, ()))

    offenders = {path: _dynamic_import_sites(path) for path in sorted(seen)}
    offenders = {path: names for path, names in offenders.items() if names}
    if offenders:
        details = "\n".join(f"  {path}: {', '.join(names)}" for path, names in offenders.items())
        raise DynamicImportError(
            "Dynamic imports found; the static dependency graph would be "
            "incomplete, which silently produces stale outputs.\n"
            f"{details}\n"
            "Either replace the dynamic import with a literal one, or declare "
            "the affected files explicitly in the rule's input."
        )

    return sorted(str(path) for path in seen)
