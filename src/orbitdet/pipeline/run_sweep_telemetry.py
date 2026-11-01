"""CLI: aggregate a sweep's telemetry.

Usage::

    python -m orbitdet.pipeline.run_sweep_telemetry \\
        --sweep fit_to_ephemeris \\
        --output sweeps/fit_to_ephemeris/sweep_telemetry.csv
"""

from __future__ import annotations

import argparse

from orbitdet.pipeline.cli import configure_cli_logging
from orbitdet.pipeline.sweep_telemetry import collect_telemetry


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep", required=True, help="Sweep name.")
    parser.add_argument("--output", required=True, help="Where to write the telemetry CSV.")
    args = parser.parse_args()

    configure_cli_logging()
    print(collect_telemetry(args.sweep, args.output))


if __name__ == "__main__":
    main()
