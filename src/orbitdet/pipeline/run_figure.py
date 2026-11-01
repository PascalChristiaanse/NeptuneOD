"""CLI: render a figure.

One entry point covers every figure family, dispatched by ``--kind``. Keeping a
single CLI means the Snakefile has one figure rule shape repeated per figure,
rather than a different invocation style per family.

Usage::

    python -m orbitdet.pipeline.run_figure \
        --config results/classic_triton_state/config/figures.yaml \
        --kind residuals --name postfit_residuals \
        --observations results/classic_triton_state/preprocessing/weighted.tudat \
        --output-dir results/classic_triton_state/figures
"""

from __future__ import annotations

import argparse

from orbitdet.pipeline.cli import add_common_args, configure_cli_logging, parse_and_load
from orbitdet.pipeline.config import load_resolved_config
from orbitdet.pipeline.estimation_figures import run_estimation_figure
from orbitdet.pipeline.figures import run_collection_figure
from orbitdet.pipeline.propagation_figures import (
    run_formal_errors_figure,
    run_rsw_figure,
    run_rsw_uncertainty_figure,
    run_state_history_figure,
)

KINDS = (
    "residuals",
    "estimation",
    "trajectory",
    "rsw",
    "formal_errors",
    "rsw_uncertainty",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--kind", required=True, choices=KINDS)
    parser.add_argument("--name", required=True, help="Figure name; also the output PDF stem.")
    parser.add_argument("--observations", help="Collection with residuals (kind=residuals).")
    parser.add_argument(
        "--metadata",
        help="dataset_metadata.json. Required by the tag-filtered "
        "kinds=residuals figures (scan-frame and normalized variants).",
    )
    parser.add_argument("--estimation", help="estimation_output.tudat (kind=estimation).")
    parser.add_argument(
        "--states", help="propagation_results.tudat (kind=trajectory, kind=rsw)."
    )
    parser.add_argument(
        "--formal-errors",
        help="propagated_formal_errors.npz (kind=formal_errors, rsw_uncertainty).",
    )
    parser.add_argument(
        "--kernels-config",
        help="Resolved kernel config slice. Required by --kind rsw, whose plotter "
        "needs a furnished SPICE pool.",
    )
    args = parser.parse_args()

    configure_cli_logging()
    cfg = parse_and_load(args)

    match args.kind:
        case "residuals":
            _require(parser, args, "observations")
            path = run_collection_figure(
                cfg,
                observations_path=args.observations,
                metadata_path=args.metadata,
                output_dir=args.output_dir,
                figure_name=args.name,
            )
        case "estimation":
            _require(parser, args, "estimation")
            path = run_estimation_figure(
                cfg,
                estimation_path=args.estimation,
                output_dir=args.output_dir,
                figure_name=args.name,
            )
        case "trajectory":
            _require(parser, args, "states")
            path = run_state_history_figure(
                cfg,
                propagated_states_path=args.states,
                output_dir=args.output_dir,
                figure_name=args.name,
            )
        case "rsw":
            _require(parser, args, "states")
            _require(parser, args, "kernels_config")
            path = run_rsw_figure(
                cfg,
                propagated_states_path=args.states,
                output_dir=args.output_dir,
                figure_name=args.name,
                kernels_cfg=load_resolved_config(args.kernels_config),
            )
        case "formal_errors":
            _require(parser, args, "formal_errors")
            path = run_formal_errors_figure(
                cfg,
                formal_errors_path=args.formal_errors,
                output_dir=args.output_dir,
                figure_name=args.name,
            )
        case "rsw_uncertainty":
            _require(parser, args, "formal_errors")
            path = run_rsw_uncertainty_figure(
                cfg,
                formal_errors_path=args.formal_errors,
                output_dir=args.output_dir,
                figure_name=args.name,
            )

    print(path)


def _require(parser: argparse.ArgumentParser, args, attribute: str) -> None:
    """Fail early when a required input for the chosen kind is missing."""
    if not getattr(args, attribute):
        parser.error(f"--{attribute.replace('_', '-')} is required for --kind {args.kind}")


if __name__ == "__main__":
    main()
