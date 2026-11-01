"""CLI: render one sweep-comparison bar chart.

Usage::

    python -m orbitdet.pipeline.run_sweep_figure \\
        --config sweeps/fit_to_ephemeris/config/figures.yaml \\
        --telemetry sweeps/fit_to_ephemeris/sweep_telemetry.csv \\
        --sweep fit_to_ephemeris --metric residuals_rms \\
        --output-dir sweeps/fit_to_ephemeris/figures
"""

from __future__ import annotations

import argparse

from orbitdet.pipeline.cli import add_common_args, configure_cli_logging, parse_and_load
from orbitdet.pipeline.sweep_figures import METRICS, run_sweep_comparison


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--telemetry", required=True, help="sweep_telemetry.csv.")
    parser.add_argument("--sweep", required=True, help="Sweep name, used as the title.")
    parser.add_argument(
        "--metric",
        required=True,
        choices=sorted(METRICS),
        help="Which telemetry column to chart.",
    )
    args = parser.parse_args()

    configure_cli_logging()
    cfg = parse_and_load(args)
    print(
        run_sweep_comparison(
            cfg,
            telemetry_path=args.telemetry,
            sweep=args.sweep,
            output_dir=args.output_dir,
            metric=args.metric,
        )
    )


if __name__ == "__main__":
    main()
