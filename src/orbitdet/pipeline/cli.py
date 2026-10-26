"""Shared CLI plumbing for pipeline stage entry points.

Each stage module exposes a ``main()`` that parses a small, explicit set of
arguments. Deliberately *not* ``@hydra.main``: the caller (Snakemake) owns the
output paths, so the CLI must accept them rather than deriving them from
Hydra's runtime directory.
"""

from __future__ import annotations

import argparse
import logging
import sys

from orbitdet.pipeline.config import load_resolved_config


def add_common_args(parser: argparse.ArgumentParser) -> None:
    """Add the arguments every stage accepts.

    Stages take a *resolved* config path rather than an experiment name. The
    resolved file is produced by the ``resolve_config`` rule and declared as an
    input, so Snakemake tracks it and a change anywhere in ``conf/``
    invalidates the downstream jobs.
    """
    parser.add_argument(
        "--config",
        required=True,
        help="Resolved config YAML from the resolve_config rule.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory this stage writes into.",
    )


def parse_and_load(args: argparse.Namespace):
    """Load the resolved config for a parsed argument namespace."""
    return load_resolved_config(args.config)


def configure_cli_logging() -> None:
    """Minimal logging setup for standalone CLI use.

    The pipeline stages call ``setup_logging`` via ``build_context``, which
    installs the repository pipeline. This only ensures early messages (before
    the context exists) are visible.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        stream=sys.stderr,
    )
