"""CLI: write one sweep combination's config slices.

Usage::

    python -m orbitdet.pipeline.run_resolve_sweep_config \\
        --experiment fit_to_ephemeris \\
        --output-dir sweeps/fit_to_ephemeris/00/config \\
        --only estimation \\
        --overrides estimation.parameters_to_estimate.neptune_GM=false

The ``--overrides`` values come from the sweep manifest, so the rule needs no
knowledge of how the sweep was declared.
"""

from __future__ import annotations

import argparse

from orbitdet.pipeline.cli import configure_cli_logging
from orbitdet.pipeline.sweep_resolve import PER_COMBO_STAGES, run_resolve_sweep_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True, help="Base experiment name.")
    parser.add_argument("--output-dir", required=True, help="Directory for the slices.")
    parser.add_argument(
        "--only",
        default=None,
        choices=sorted(PER_COMBO_STAGES),
        help="Write only this stage's slice.",
    )
    parser.add_argument(
        "--overrides",
        nargs="*",
        default=[],
        help="Hydra override strings for this combination, from the sweep manifest.",
    )
    args = parser.parse_args()

    configure_cli_logging()
    written = run_resolve_sweep_config(
        args.experiment, args.output_dir, overrides=args.overrides, only=args.only
    )
    for path in written:
        print(path)


if __name__ == "__main__":
    main()
