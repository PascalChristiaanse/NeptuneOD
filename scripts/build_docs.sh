#!/usr/bin/env bash
#
# Build the OrbitDet API documentation.
#
#   scripts/build_docs.sh            # build, tolerate warnings
#   scripts/build_docs.sh -W         # build, fail on any warning (CI)
#
# Output: docs/_build/html/index.html
#
# Requires the `docs` extra:  pip install -e ".[docs]"

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"

cd "$ROOT"

# Regenerate the API reference stubs.  scripts/gen_api_docs.py replaces
# sphinx-apidoc: it makes orbitdet the root of the tree (no ``modules`` wrapper)
# and titles each page with just the module's own name.
"$PYTHON" scripts/gen_api_docs.py

# Regenerate the Mermaid sequence diagrams from live traces of the entrypoints
# listed in docs/diagrams.yaml.  The sequence_diagrams Sphinx extension injects
# them into the matching docstrings during the build below.
"$PYTHON" scripts/gen_sequence_diagrams.py

# -b html, then any extra flags (e.g. -W) forwarded from the caller.
"$PYTHON" -m sphinx -b html docs docs/_build/html "$@"

echo
echo "Open: file://$ROOT/docs/_build/html/index.html"
