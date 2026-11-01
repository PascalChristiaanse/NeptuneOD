"""CLI: run the least-squares estimation.

Usage::

    python -m orbitdet.pipeline.run_estimation \
        --config results/classic_triton_state/config/estimation.yaml \
        --observations results/classic_triton_state/preprocessing/weighted.tudat \
        --output-dir results/classic_triton_state/estimation
"""

from __future__ import annotations

import argparse

from orbitdet.pipeline.cli import add_common_args, configure_cli_logging, parse_and_load
from orbitdet.pipeline.estimation import run_estimation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--observations", required=True, help="Weighted ObservationCollection.")
    args = parser.parse_args()

    configure_cli_logging()
    cfg = parse_and_load(args)
    output_path, summary_path = run_estimation(
        cfg, observations_path=args.observations, output_dir=args.output_dir
    )
    print(output_path)
    print(summary_path)


if __name__ == "__main__":
    main()
