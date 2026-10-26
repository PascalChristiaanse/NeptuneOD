"""CLI: pre-fit residuals wrt SPICE.

Usage::

    python -m orbitdet.pipeline.run_prefit_spice \
        --config results/classic_triton_state/config.yaml \
        --observations results/classic_triton_state/observations.tudat \
        --metadata results/classic_triton_state/dataset_metadata.json \
        --output-dir results/classic_triton_state/prefit_spice
"""

from __future__ import annotations

import argparse

from orbitdet.pipeline.cli import add_common_args, configure_cli_logging, parse_and_load
from orbitdet.pipeline.prefit import run_prefit_spice


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--observations", required=True, help="Serialized ObservationCollection.")
    parser.add_argument("--metadata", required=True, help="dataset_metadata.json.")
    args = parser.parse_args()

    configure_cli_logging()
    cfg = parse_and_load(args)
    csv_path = run_prefit_spice(
        cfg,
        observations_path=args.observations,
        metadata_path=args.metadata,
        output_dir=args.output_dir,
    )
    print(csv_path)


if __name__ == "__main__":
    main()
