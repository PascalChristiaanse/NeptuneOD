"""CLI: write the sweep manifest.

Usage::

    python -m orbitdet.pipeline.run_sweep_manifest \\
        --sweep fit_to_ephemeris --output sweeps/fit_to_ephemeris/manifest.json
"""

from __future__ import annotations

import argparse

from orbitdet.pipeline.cli import configure_cli_logging
from orbitdet.pipeline.sweeps import write_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep", required=True, help="Sweep name, e.g. fit_to_ephemeris.")
    parser.add_argument("--output", required=True, help="Where to write the manifest JSON.")
    args = parser.parse_args()

    configure_cli_logging()
    print(write_manifest(args.sweep, args.output))


if __name__ == "__main__":
    main()
