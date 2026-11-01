"""CLI: propagate the estimated covariance over the full arc.

Usage::

    python -m orbitdet.pipeline.run_covariance \
        --config results/classic_triton_state/config/covariance.yaml \
        --observations results/classic_triton_state/preprocessing/weighted.tudat \
        --estimation results/classic_triton_state/estimation/estimation_output.tudat \
        --states results/classic_triton_state/propagation/postfit_propagation_result.tudat \
        --output-dir results/classic_triton_state/covariance
"""

from __future__ import annotations

import argparse

from orbitdet.pipeline.cli import add_common_args, configure_cli_logging, parse_and_load
from orbitdet.pipeline.covariance import run_covariance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--observations", required=True, help="Weighted collection.")
    parser.add_argument(
        "--estimation",
        required=True,
        help="estimation_output.tudat carrying the covariance-propagation payload.",
    )
    parser.add_argument(
        "--states",
        required=True,
        help="Post-fit propagation results, used to define the RSW frame.",
    )
    args = parser.parse_args()

    configure_cli_logging()
    cfg = parse_and_load(args)
    npz_path, summary_path = run_covariance(
        cfg,
        observations_path=args.observations,
        estimation_path=args.estimation,
        propagated_states_path=args.states,
        output_dir=args.output_dir,
    )
    print(npz_path)
    print(summary_path)


if __name__ == "__main__":
    main()
