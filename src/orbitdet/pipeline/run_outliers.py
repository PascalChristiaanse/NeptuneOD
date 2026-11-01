"""CLI: apply outlier rejection.

Usage::

    python -m orbitdet.pipeline.run_outliers \
        --config results/classic_triton_state/config/outlier_rejection.yaml \
        --observations results/classic_triton_state/pre/with_residuals.tudat \
        --metadata results/classic_triton_state/dataset_metadata.json \
        --output-dir results/classic_triton_state/preprocessing
"""

from __future__ import annotations

import argparse

from orbitdet.pipeline.cli import add_common_args, configure_cli_logging, parse_and_load
from orbitdet.pipeline.outliers import run_outlier_rejection


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--observations", required=True, help="Collection carrying residuals.")
    parser.add_argument("--metadata", required=True, help="dataset_metadata.json.")
    args = parser.parse_args()

    configure_cli_logging()
    cfg = parse_and_load(args)
    collection_path, report_path = run_outlier_rejection(
        cfg,
        observations_path=args.observations,
        metadata_path=args.metadata,
        output_dir=args.output_dir,
    )
    print(collection_path)
    if report_path is not None:
        print(report_path)


if __name__ == "__main__":
    main()
