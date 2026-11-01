"""CLI: apply observation weighting.

Usage::

    python -m orbitdet.pipeline.run_weighting \
        --config results/classic_triton_state/config/weighting.yaml \
        --observations results/classic_triton_state/preprocessing/outliers_applied.tudat \
        --metadata results/classic_triton_state/dataset_metadata.json \
        --output-dir results/classic_triton_state/preprocessing
"""

from __future__ import annotations

import argparse

from orbitdet.pipeline.cli import add_common_args, configure_cli_logging, parse_and_load
from orbitdet.pipeline.weighting import run_weighting


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--observations", required=True, help="Collection carrying residuals.")
    parser.add_argument("--metadata", required=True, help="dataset_metadata.json.")
    args = parser.parse_args()

    configure_cli_logging()
    cfg = parse_and_load(args)
    collection_path, weights_path = run_weighting(
        cfg,
        observations_path=args.observations,
        metadata_path=args.metadata,
        output_dir=args.output_dir,
    )
    print(collection_path)
    if weights_path is not None:
        print(weights_path)


if __name__ == "__main__":
    main()
