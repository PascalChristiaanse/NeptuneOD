"""CLI: compute residuals against the SPICE ephemeris.

Usage::

    python -m orbitdet.pipeline.run_ephemeris_residuals \
        --config results/classic_triton_state/config/prefit.yaml \
        --observations results/classic_triton_state/observations.tudat \
        --metadata results/classic_triton_state/dataset_metadata.json \
        --output-dir results/classic_triton_state/prefit_ephemeris
"""

from __future__ import annotations

import argparse

from orbitdet.pipeline.cli import add_common_args, configure_cli_logging, parse_and_load
from orbitdet.pipeline.residuals import run_residuals_wrt_ephemeris


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--observations", required=True, help="Serialized ObservationCollection.")
    parser.add_argument("--metadata", required=True, help="dataset_metadata.json.")
    args = parser.parse_args()

    configure_cli_logging()
    cfg = parse_and_load(args)
    collection_path, csv_path = run_residuals_wrt_ephemeris(
        cfg,
        observations_path=args.observations,
        metadata_path=args.metadata,
        output_dir=args.output_dir,
    )
    print(collection_path)
    print(csv_path)


if __name__ == "__main__":
    main()
