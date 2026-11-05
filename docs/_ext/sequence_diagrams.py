"""Sphinx extension that injects generated sequence diagrams into docstrings.

The diagrams themselves are produced by ``scripts/gen_sequence_diagrams.py``
(run by ``scripts/build_docs.sh`` before Sphinx starts) and written to
``docs/diagrams/<name>.mmd``.  This extension reads ``docs/diagrams.yaml`` and,
for every entry, appends a ``.. mermaid::`` directive to the docstring of the
documented object.  The diagram therefore renders on the module's API page,
directly under the method it documents, instead of on a separate page.

How the injection works
-----------------------
``autodoc-process-docstring`` fires for every object autodoc documents, with
the object's fully-qualified name.  We look that name up in the manifest and, on
a hit, append the directive to the docstring lines.  Because the directive is
added to the *docstring* rather than to a hand-written ``.rst`` stub, it
survives ``sphinx-apidoc`` regeneration and needs no per-module editing.

The ``.mmd`` file is referenced by path relative to ``docs/`` (the Sphinx source
directory), so the ``mermaid`` directive resolves it the same way it would from
any other source file.
"""

from __future__ import annotations

import os
from pathlib import Path

import yaml
from sphinx.application import Sphinx

#: Manifest path relative to the Sphinx source directory (``docs/``).
_MANIFEST_NAME = "diagrams.yaml"
#: Directory holding the generated ``.mmd`` files, relative to ``docs/``.
_DIAGRAM_DIR = "diagrams"


def _load_targets(app: Sphinx) -> dict[str, tuple[str, str | None]]:
    """Map documented object name -> ``(diagram_name, title_override)``."""
    manifest_path = Path(app.srcdir) / _MANIFEST_NAME
    if not manifest_path.exists():
        return {}
    data = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
    targets: dict[str, tuple[str, str | None]] = {}
    for entry in data.get("diagrams", []):
        target = entry.get("target")
        name = entry.get("name")
        if target and name:
            targets[str(target)] = (str(name), entry.get("title"))
    return targets


def _inject(
    app: Sphinx,
    what: str,
    name: str,
    obj: object,
    options: object,
    lines: list[str],
) -> None:
    """Append the diagram directive to a documented object's docstring."""
    targets = getattr(app, "_sequence_diagram_targets", None)
    if targets is None:
        targets = _load_targets(app)
        app._sequence_diagram_targets = targets  # type: ignore[attr-defined]

    entry = targets.get(name)
    if entry is None:
        return
    diagram_name, title = entry

    mmd_path = Path(app.srcdir) / _DIAGRAM_DIR / f"{diagram_name}.mmd"
    if not mmd_path.exists():
        # The generator has not run (e.g. a bare `sphinx-build`).  Warn rather
        # than fail so the API docs still build.
        app.warn(f"sequence diagram {mmd_path} not found; run scripts/gen_sequence_diagrams.py")
        return

    heading = title or f"Sequence diagram: {name.rsplit('.', 1)[-1]}"
    # The mermaid directive resolves its argument relative to the *document's*
    # directory (e.g. docs/api/), not the source root, so compute the path from
    # there.  ``app.env.docname`` is the document currently being processed.
    doc_dir = Path(app.srcdir) / Path(app.env.docname).parent
    rel_path = os.path.relpath(mmd_path, doc_dir)
    lines += [
        "",
        heading,
        "-" * len(heading),
        "",
        f".. mermaid:: {rel_path}",
        "",
    ]


def setup(app: Sphinx) -> dict[str, object]:
    """Register the docstring hook with Sphinx."""
    app.connect("autodoc-process-docstring", _inject)
    return {
        "version": "0.1",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
