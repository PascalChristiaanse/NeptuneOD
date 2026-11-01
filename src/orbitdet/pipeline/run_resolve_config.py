"""CLI: resolve the Hydra config tree into per-stage slices.

Usage::

    python -m orbitdet.pipeline.run_resolve_config \
        --experiment classic_triton_state \
        --output-dir results/classic_triton_state/config \
        --only observations
"""

from __future__ import annotations

import argparse

from orbitdet.pipeline.cli import configure_cli_logging
from orbitdet.pipeline.resolve import STAGE_KEYS, run_resolve_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True, help="Experiment config name.")
    parser.add_argument("--output-dir", required=True, help="Directory for the slices.")
    parser.add_argument(
        "--only",
        default=None,
        choices=sorted(STAGE_KEYS),
        help="Write only this stage's slice. Required in the DAG so each slice "
        "has its own rule; a single rule emitting every slice would invalidate "
        "the whole downstream DAG whenever any one slice changed.",
    )
    args = parser.parse_args()

    configure_cli_logging()
    written = run_resolve_config(args.experiment, args.output_dir, only=args.only)
    for path in written:
        print(path)


if __name__ == "__main__":
    main()
