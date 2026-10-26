"""CLI: resolve the Hydra config tree into per-stage tracked artifacts.

Usage::

    python -m orbitdet.pipeline.run_resolve_config \
        --experiment generate_prefit_residuals \
        --output-dir results/generate_prefit_residuals/config
"""

from __future__ import annotations

import argparse

from orbitdet.pipeline.cli import configure_cli_logging
from orbitdet.pipeline.resolve import run_resolve_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment",
        required=True,
        help="Experiment config name, e.g. classic_triton_state.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to write the per-stage config slices into.",
    )
    parser.add_argument(
        "--only",
        default=None,
        help="Write only this stage's slice (used by Snakemake so each slice "
        "has its own producing rule).",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Hydra override, repeatable (e.g. --override seed=7).",
    )
    args = parser.parse_args()

    configure_cli_logging()
    for path in run_resolve_config(args.experiment, args.output_dir, args.override, args.only):
        print(path)


if __name__ == "__main__":
    main()
