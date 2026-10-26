"""Stage: download and furnish SPICE kernels.

Produces a stamp file rather than a data artifact: the kernels land in
``data/kernels/`` (a shared, git-ignored cache) and the stamp records that the
download completed for this config. Downstream rules depend on the stamp, so
they re-run when the kernel set changes but not on every invocation.
"""

from __future__ import annotations

import logging
from pathlib import Path

from omegaconf import DictConfig

from orbitdet.data import KernelManager
from orbitdet.pipeline.context import build_context

logger = logging.getLogger(__name__)


def run_kernels(cfg: DictConfig, output_dir: str | Path) -> Path:
    """Download and furnish all kernels declared by *cfg*.

    Returns
    -------
    Path
        The stamp file written on success.
    """
    ctx = build_context(cfg, output_dir)

    km = KernelManager(cfg)
    km.download_all_kernels()
    km.furnish()
    logger.info("Kernels downloaded and furnished.")

    stamp = Path(output_dir) / "kernels.done"
    stamp.write_text(f"git_commit: {ctx.git_commit}\n")
    return stamp
