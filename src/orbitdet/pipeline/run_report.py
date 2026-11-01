"""CLI: assemble the experiment's summary report.

Usage::

    python -m orbitdet.pipeline.run_report \
        --experiment classic_triton_state \
        --results-dir results/classic_triton_state \
        --output results/classic_triton_state/report.md
"""

from __future__ import annotations

import argparse

from orbitdet.pipeline.cli import configure_cli_logging
from orbitdet.pipeline.report import run_report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True, help="Experiment name (report title).")
    parser.add_argument("--results-dir", required=True, help="The experiment's result directory.")
    parser.add_argument("--output", required=True, help="Where to write the Markdown report.")
    args = parser.parse_args()

    configure_cli_logging()
    print(run_report(args.experiment, args.results_dir, args.output))


if __name__ == "__main__":
    main()
