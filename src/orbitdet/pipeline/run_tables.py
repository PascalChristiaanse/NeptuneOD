"""CLI: build the weight summary tables.

Usage::

    python -m orbitdet.pipeline.run_tables \
        --config results/classic_triton_state/config/figures.yaml \
        --weights results/classic_triton_state/preprocessing/observation_weights.csv \
        --output-dir results/classic_triton_state/tables
"""

from __future__ import annotations

import argparse

from orbitdet.pipeline.cli import add_common_args, configure_cli_logging, parse_and_load
from orbitdet.pipeline.tables import run_weight_tables


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--weights", required=True, help="observation_weights.csv.")
    args = parser.parse_args()

    configure_cli_logging()
    cfg = parse_and_load(args)
    for path in run_weight_tables(cfg, weights_path=args.weights, output_dir=args.output_dir):
        print(path)


if __name__ == "__main__":
    main()
