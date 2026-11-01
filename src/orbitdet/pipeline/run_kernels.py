"""CLI: download and furnish SPICE kernels.

Usage::

    python -m orbitdet.pipeline.run_kernels \
        --config results/classic_triton_state/config/kernels.yaml \
        --output-dir results/classic_triton_state
"""

from __future__ import annotations

import argparse

from orbitdet.pipeline.cli import add_common_args, configure_cli_logging, parse_and_load
from orbitdet.pipeline.kernels import run_kernels


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    args = parser.parse_args()

    configure_cli_logging()
    cfg = parse_and_load(args)
    print(run_kernels(cfg, output_dir=args.output_dir))


if __name__ == "__main__":
    main()
