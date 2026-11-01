"""CLI: propagate the trajectory and compute residuals against it.

Usage (pre-fit)::

    python -m orbitdet.pipeline.run_propagation \
        --config results/classic_triton_state/config/dynamics.yaml \
        --observations results/classic_triton_state/preprocessing/weighted.tudat \
        --metadata results/classic_triton_state/dataset_metadata.json \
        --output-dir results/classic_triton_state/propagation --tag prefit

Usage (post-fit, propagating the estimated state)::

    python -m orbitdet.pipeline.run_propagation \
        --config ... --tag postfit \
        --estimation-dir results/classic_triton_state/estimation
"""

from __future__ import annotations

import argparse

from orbitdet.pipeline.cli import add_common_args, configure_cli_logging, parse_and_load
from orbitdet.pipeline.propagation import run_postfit_propagation, run_propagation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--observations", required=True, help="Collection to residualise.")
    parser.add_argument("--metadata", required=True, help="dataset_metadata.json.")
    parser.add_argument(
        "--tag",
        default="prefit",
        choices=["prefit", "postfit"],
        help="Output prefix. 'postfit' also requires --estimation-dir.",
    )
    parser.add_argument(
        "--estimation-dir",
        default=None,
        help="Directory holding estimation_summary.json (post-fit only).",
    )
    args = parser.parse_args()

    if args.tag == "postfit" and not args.estimation_dir:
        parser.error("--estimation-dir is required when --tag postfit")

    configure_cli_logging()
    cfg = parse_and_load(args)

    if args.tag == "postfit":
        results_path, collection_path, csv_path = run_postfit_propagation(
            cfg,
            observations_path=args.observations,
            metadata_path=args.metadata,
            output_dir=args.output_dir,
            estimation_dir=args.estimation_dir,
        )
    else:
        results_path, collection_path, csv_path = run_propagation(
            cfg,
            observations_path=args.observations,
            metadata_path=args.metadata,
            output_dir=args.output_dir,
            tag="prefit",
        )

    for path in (results_path, collection_path, csv_path):
        print(path)


if __name__ == "__main__":
    main()
