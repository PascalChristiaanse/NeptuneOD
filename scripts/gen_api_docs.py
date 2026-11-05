#!/usr/bin/env python
"""Generate the Sphinx API reference stubs for the ``orbitdet`` package.

This replaces ``sphinx-apidoc`` with a generator that produces a flatter, more
readable tree:

* ``orbitdet`` is the root of the API reference -- there is no ``modules``
  wrapper page, so the tree is one level shallower than ``sphinx-apidoc``'s.
* Every page is titled with just the module's own name -- ``Data``,
  ``Visualization``, ``Gaia`` -- rather than the full dotted path, so the
  sidebar reads as a clean hierarchy instead of repeating ``orbitdet.`` on
  every line.

The stubs are regenerated on every docs build by ``scripts/build_docs.sh``;
edit the source docstrings, not the generated ``.rst`` files.

Usage
-----
::

    python scripts/gen_api_docs.py            # write docs/api/
    python scripts/gen_api_docs.py --out DIR  # write elsewhere
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PACKAGE_ROOT = ROOT / "src" / "orbitdet"
OUT_DIR = ROOT / "docs" / "api"
ROOT_PACKAGE = "orbitdet"


def _title(dotted: str) -> str:
    """Short display title for a module: its last component, prettified."""
    last = dotted.rsplit(".", 1)[-1]
    if last == ROOT_PACKAGE:
        return "OrbitDet"
    return last.replace("_", " ").title()


def _children(directory: Path, prefix: str) -> list[str]:
    """Dotted names of the modules and packages directly under ``directory``."""
    found: list[str] = []
    for entry in sorted(directory.iterdir()):
        if entry.name.startswith("_") or entry.name == "__pycache__":
            continue
        if entry.is_dir() and (entry / "__init__.py").exists():
            found.append(f"{prefix}.{entry.name}")
        elif entry.suffix == ".py" and entry.name != "__init__.py":
            found.append(f"{prefix}.{entry.stem}")
    return found


def _write_page(out_dir: Path, dotted: str, children: list[str]) -> None:
    """Write one ``.rst`` stub for ``dotted``."""
    title = _title(dotted)
    lines = [
        title,
        "=" * len(title),
        "",
        f".. automodule:: {dotted}",
        "   :members:",
        "   :show-inheritance:",
        "   :undoc-members:",
        "",
    ]
    if children:
        lines += [
            ".. toctree::",
            "   :maxdepth: 2",
            "",
        ]
        lines += [f"   {child}" for child in children]
        lines.append("")
    (out_dir / f"{dotted}.rst").write_text("\n".join(lines), encoding="utf-8")


def _walk(out_dir: Path, directory: Path, prefix: str) -> None:
    """Recursively write a page for ``prefix`` and everything beneath it."""
    children = _children(directory, prefix)
    _write_page(out_dir, prefix, children)
    for child in children:
        # ``child`` is dotted (e.g. ``orbitdet.data``); its path is the current
        # directory plus the components below ``prefix``.
        relative = child[len(prefix) + 1 :].replace(".", "/")
        child_dir = directory / relative
        if child_dir.is_dir():
            _walk(out_dir, child_dir, child)
        else:
            # A leaf module: write its page (no children to recurse into).
            _write_page(out_dir, child, [])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=OUT_DIR, help="output directory")
    args = parser.parse_args(argv)

    if args.out.exists():
        shutil.rmtree(args.out)
    args.out.mkdir(parents=True)

    _walk(args.out, PACKAGE_ROOT, ROOT_PACKAGE)
    count = len(list(args.out.glob("*.rst")))
    print(f"wrote {count} API stubs to {args.out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
