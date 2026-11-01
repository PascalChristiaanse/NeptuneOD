"""CLI: build the observation collection.

Usage::

    python -m orbitdet.pipeline.run_observations \
        --config results/classic_triton_state/config/observations.yaml \
        --output-dir results/classic_triton_state
"""

from __future__ import annotations

import argparse

from orbitdet.pipeline.cli import add_common_args, configure_cli_logging, parse_and_load
from orbitdet.pipeline.observations import run_observations


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    args = parser.parse_args()

    configure_cli_logging()
    cfg = parse_and_load(args)
    obs_path, meta_path = run_observations(cfg, output_dir=args.output_dir)
    print(obs_path)
    print(meta_path)


if __name__ == "__main__":
    main()
