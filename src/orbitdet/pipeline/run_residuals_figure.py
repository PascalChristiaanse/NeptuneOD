"""CLI: residual figures from the pre-fit residuals CSV.

Usage::

    python -m orbitdet.pipeline.run_residuals_figure \
        --config results/generate_prefit_residuals/config/figures.yaml \
        --residuals results/generate_prefit_residuals/prefit_spice/residuals.csv \
        --output-dir results/generate_prefit_residuals/figures
"""

from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")

from orbitdet.pipeline.cli import add_common_args, configure_cli_logging, parse_and_load
from orbitdet.pipeline.figures import run_residuals_figure


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--residuals", required=True, help="residuals.csv from the prefit stage.")
    parser.add_argument(
        "--figure-name",
        default="residuals",
        help=(
            "Key under cfg.figures holding this figure's styling. Must match the "
            "name the experiment script passes to Residuals."
        ),
    )
    args = parser.parse_args()

    configure_cli_logging()
    cfg = parse_and_load(args)
    pdf_path = run_residuals_figure(
        cfg,
        residuals_csv=args.residuals,
        output_dir=args.output_dir,
        figure_name=args.figure_name,
    )
    print(pdf_path)


if __name__ == "__main__":
    main()
